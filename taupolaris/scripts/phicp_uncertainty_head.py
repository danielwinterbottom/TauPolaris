"""A per-event phiCP uncertainty for the transformer regressor, without a flow.

The trained regressor is frozen; a small head learns the concentration kappa(x) of a von Mises distribution
around the regressor's own phiCP estimate phi_hat:

    p(phi_true | phi_hat, kappa) = exp(kappa cos(phi_true - phi_hat)) / (2 pi I0(kappa))
    NLL = -kappa cos(phi_true - phi_hat) + log I0(kappa) + log 2pi,   log I0(k) = log(i0e(k)) + k

kappa is large for events whose phiCP is well determined; 1/sqrt(kappa) plays the role of an angular uncertainty
(only its ordering matters for a cut); the results store the von Mises circular std sqrt(-2 ln(I1/I0)) as
pred_phiCP_err, the same quantity as the flow's (circular std of its samples). The head sees the regressor's encoder output (64) and its 18 standardised
predictions. Training uses the same per-event weights as the regressor (sum of the TauSpinner alpha 0 and 90
weights). phi_hat and the true phiCP are computed exactly as evaluate_regressor.py does.

Stages (each reads what the previous one wrote to --workdir):
  features  run the frozen regressor over a training subset, a validation subset, the Jul20 test file and (optional)
            a background test file; store encoder output, predictions, phi_hat, phi_true, decay modes, weights
  train     fit the head (von Mises NLL), early-stopped on the validation subset
  apply     write results parquets for the test and background samples with pred_phiCP and pred_phiCP_err
            (von Mises circular std), in the format weighted_significance/uncertainty_cut_scan.py reads

    python taupolaris/scripts/phicp_uncertainty_head.py -c <regressor config> --workdir <dir> \\
        --stage features [--bkg_test <Z test parquet>] ; --stage train ; --stage apply
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.getcwd())
from taupolaris.python.DataProcessing import _resolve_polvec_feature_level, _resolve_frame_aligned_level
from taupolaris.python.NN_Tools import load_model
from taupolaris.scripts.evaluate_regressor import to_cartesian, unit
from taupolaris.scripts.evaluate_polvec import phicp_from_cart, add_DM, M_TAU

TL = ('taup', 'taun')


def load_regressor(cfg):
    D, NN = cfg['Data'], cfg['SetupNN']
    out = D['Features']['output_features']['onorm']
    model = load_model(NN['TransformerBaseline_hyperparams'], D['Features']['input_features'], out, useTransformerMLP=True,
                       leptonic_mode=D.get('leptonic_mode', 0), polvec_feature_level=_resolve_polvec_feature_level(D),
                       frame_aligned_level=_resolve_frame_aligned_level(D))
    od = f"outputs_{NN['model_name']}"
    model.load_state_dict(torch.load(os.path.join(od, 'plots', 'best_model.pth'), map_location='cpu', weights_only=True))
    model.float().eval()
    return model, np.load(os.path.join(od, 'normalization_params.npz')), out


def needed_columns(cfg, out):
    cols = set(cfg['Data']['Features']['input_features']) | set(out)
    for t in TL:
        cols |= {f'reco_{t}_{p}_p{c}' for p in ('charged', 'pizero1') for c in 'xyz'}
        cols |= {f'undecayed_{t}_p{c}' for c in 'xyz'} | {f'undecayed_{t}_e'} | {f'ts_hh_{t}_{c}' for c in 'xyz'}
        cols |= {f'reco_{t}_{f}' for f in ('ishadronic', 'npizero', 'is3prong')}
    return sorted(cols | {'tauspinner_wt_alpha0', 'tauspinner_wt_alpha45', 'tauspinner_wt_alpha90'})


def block_features(model, norm, out, cfg, df):
    inp = cfg['Data']['Features']['input_features']
    X = torch.from_numpy(((df[inp].to_numpy(np.float32) - norm['input_mean']) / norm['input_std']).astype(np.float32))
    with torch.no_grad():
        ctx = torch.cat([model.encoder(X[s:s + 8192]) for s in range(0, len(X), 8192)])
        pred_std = torch.cat([model.head(ctx[s:s + 8192]) for s in range(0, len(ctx), 8192)])
    native = pd.DataFrame(pred_std.numpy() * norm['output_std'].reshape(-1) + norm['output_mean'].reshape(-1), columns=out)
    h1, h2, p1, p2 = to_cartesian(native, df, TL, lambda t, c: f'ts_hh_{t}_{c}', lambda t, c: f'undecayed_{t}_{c}')
    cols = [f'ts_hh_{t}_{c}' for t in TL for c in 'xyz'] + [f'undecayed_{t}_p{c}' for t in TL for c in 'xyz']
    pred_cart = pd.DataFrame(np.concatenate([unit(h1), unit(h2), p1, p2], axis=1), columns=cols)
    phi_hat = phicp_from_cart(pred_cart, np.sqrt((p1 ** 2).sum(1) + M_TAU ** 2), np.sqrt((p2 ** 2).sum(1) + M_TAU ** 2), TL)
    true_cart = pd.DataFrame(np.concatenate([unit(df[[f'ts_hh_{t}_{c}' for c in 'xyz']].to_numpy(float)) for t in TL] +
                                            [df[[f'undecayed_{t}_p{c}' for c in 'xyz']].to_numpy(float) for t in TL], axis=1), columns=cols)
    phi_true = phicp_from_cart(true_cart, df['undecayed_taup_e'].to_numpy(float), df['undecayed_taun_e'].to_numpy(float), TL)
    dm = add_DM(df, TL, dm_prefix='reco')
    return dict(ctx=ctx.numpy().astype(np.float32), pred=pred_std.numpy().astype(np.float32),
                phi_hat=np.asarray(phi_hat, np.float32), phi_true=np.asarray(phi_true, np.float32),
                dm_p=np.asarray(dm[0], np.int16), dm_n=np.asarray(dm[1], np.int16),
                w0=df['tauspinner_wt_alpha0'].to_numpy(np.float32), w45=df['tauspinner_wt_alpha45'].to_numpy(np.float32),
                w90=df['tauspinner_wt_alpha90'].to_numpy(np.float32))


def features(args, cfg):
    model, norm, out = load_regressor(cfg)
    cols = needed_columns(cfg, out)
    D = cfg['Data']
    split = lambda kind, d: os.path.join(D['output_dir'], d, f'{kind}_dataframe_onorm_leptonic_mode_{D["leptonic_mode"]}_transformer.parquet')
    test = D['test_dataset'][0] if isinstance(D['test_dataset'], list) else D['test_dataset']
    samples = {'train': [(split('train', d), args.n_train_per_dataset) for d in D['datasets']],
               'val': [(split('val', d), args.n_val_per_dataset) for d in D['datasets']],
               'test': [(test, args.n_test)]}
    if args.bkg_test:
        samples['bkg'] = [(args.bkg_test, args.n_test)]
    for name, files in samples.items():
        parts, t0 = [], time.time()
        for path, nmax in files:
            n = 0
            for b in pq.ParquetFile(path).iter_batches(batch_size=250_000, columns=[c for c in cols if c in pq.read_schema(path).names]):
                df = b.to_pandas()
                if nmax and n + len(df) > nmax:
                    df = df.iloc[:nmax - n]
                parts.append(block_features(model, norm, out, cfg, df)); n += len(df)
                if nmax and n >= nmax:
                    break
            print(f'  {name}: {os.path.basename(os.path.dirname(path))} {n} events ({time.time() - t0:.0f}s)', flush=True)
        arr = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(os.path.join(args.workdir, f'features_{name}.npz'), **arr)
        print(f'>> features_{name}.npz: {len(arr["phi_hat"])} events', flush=True)


class KappaHead(nn.Module):
    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)            # log kappa


def von_mises_nll(log_kappa, dphi):
    k = torch.exp(log_kappa.clamp(-8, 6))
    return -k * torch.cos(dphi) + torch.log(torch.special.i0e(k)) + k


def head_inputs(f):
    return torch.from_numpy(np.concatenate([f['ctx'], f['pred']], axis=1).astype(np.float32))


def train(args, cfg):
    tr, va = np.load(os.path.join(args.workdir, 'features_train.npz')), np.load(os.path.join(args.workdir, 'features_val.npz'))
    had = lambda f: (f['dm_p'] >= 0) & (f['dm_p'] <= 11) & (f['dm_n'] >= 0) & (f['dm_n'] <= 11)
    sel = lambda f, m: {k: f[k][m] for k in f.files}
    tr, va = sel(tr, had(tr)), sel(va, had(va))
    Xtr, Xva = head_inputs(tr), head_inputs(va)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    d_tr = torch.from_numpy((tr['phi_true'] - tr['phi_hat']).astype(np.float32))
    d_va = torch.from_numpy((va['phi_true'] - va['phi_hat']).astype(np.float32))
    w_tr = torch.from_numpy(tr['w0'] + tr['w90']); w_va = torch.from_numpy(va['w0'] + va['w90'])
    head = KappaHead(Xtr.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    # reference: one global kappa (no per-event information) -- the head must beat this to be useful
    with torch.no_grad():
        best_g, gl = None, None
        for lk in np.linspace(-3, 3, 121):
            v = ((w_va * von_mises_nll(torch.full_like(d_va, float(lk)), d_va)).sum() / w_va.sum()).item()
            if gl is None or v < gl:
                gl, best_g = v, lk
    print(f'>> {len(d_tr)} training / {len(d_va)} validation hadronic events; constant-kappa reference: kappa {np.exp(best_g):.3f}, val NLL {gl:.5f}')
    best, best_state, bad = np.inf, None, 0
    for ep in range(1, args.epochs + 1):
        head.train(); perm = torch.randperm(len(d_tr))
        for s in range(0, len(perm), args.batch_size):
            i = perm[s:s + args.batch_size]
            loss = (w_tr[i] * von_mises_nll(head(Xtr[i]), d_tr[i])).sum() / w_tr[i].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            v = torch.cat([head(Xva[s:s + 65536]) for s in range(0, len(Xva), 65536)])
            vl = ((w_va * von_mises_nll(v, d_va)).sum() / w_va.sum()).item()
        print(f'  epoch {ep}: val NLL {vl:.5f} ({vl - gl:+.5f} vs constant kappa)', flush=True)
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: t.clone() for k, t in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    torch.save({'state': best_state, 'mu': mu, 'sd': sd, 'n_in': Xtr.shape[1]}, os.path.join(args.workdir, 'kappa_head.pth'))
    print(f'>> saved kappa_head.pth (best val NLL {best:.5f}, {best - gl:+.5f} vs constant kappa)')


def apply(args, cfg):
    ck = torch.load(os.path.join(args.workdir, 'kappa_head.pth'), weights_only=False)
    head = KappaHead(ck['n_in']); head.load_state_dict(ck['state']); head.eval()
    for name in ('test', 'bkg'):
        p = os.path.join(args.workdir, f'features_{name}.npz')
        if not os.path.exists(p):
            continue
        f = np.load(p)
        X = (head_inputs(f) - ck['mu']) / ck['sd']
        with torch.no_grad():
            lk = torch.cat([head(X[s:s + 65536]) for s in range(0, len(X), 65536)]).numpy()
        kappa = np.exp(np.clip(lk, -8, 6))
        kt = torch.from_numpy(kappa.astype(np.float64))
        # circular std of a von Mises distribution, sqrt(-2 ln(I1/I0)): same ordering as 1/sqrt(kappa), but the
        # same quantity as the flow's pred_phiCP_err (circular std of its samples), so the two are comparable
        circ_std = torch.sqrt(-2 * torch.log(torch.special.i1e(kt) / torch.special.i0e(kt))).numpy()
        res = pd.DataFrame({'pred_phiCP': f['phi_hat'], 'true_phiCP': f['phi_true'], 'pred_phiCP_kappa': kappa,
                            'pred_phiCP_err': circ_std, 'reco_taup_DM': f['dm_p'], 'reco_taun_DM': f['dm_n'],
                            'tauspinner_wt_alpha0': f['w0'], 'tauspinner_wt_alpha45': f['w45'], 'tauspinner_wt_alpha90': f['w90']})
        out = os.path.join(args.workdir, f'results_{name}.parquet'); res.to_parquet(out)
        had = res.reco_taup_DM.between(0, 11) & res.reco_taun_DM.between(0, 11)
        r = ((res.true_phiCP - res.pred_phiCP + np.pi) % (2 * np.pi) - np.pi)[had].to_numpy(); e = res.pred_phiCP_err[had].to_numpy()
        q = np.quantile(e, np.linspace(0, 1, 6))
        rel = [(e[(e >= a) & (e <= b)].mean(), np.sqrt(-2 * np.log(np.clip(np.abs(np.exp(1j * r[(e >= a) & (e <= b)]).mean()), 1e-9, 1))))
               for a, b in zip(q[:-1], q[1:])]
        print(f'>> {out}: median predicted circular std {np.median(e):.3f} rad; quintiles, predicted -> actual circular std of the residual: '
              + ', '.join(f'{a:.2f}->{b:.2f}' for a, b in rel))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c', required=True)
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--stage', choices=('features', 'train', 'apply'), required=True)
    ap.add_argument('--bkg_test', default=None)
    ap.add_argument('--n_train_per_dataset', type=int, default=500_000)
    ap.add_argument('--n_val_per_dataset', type=int, default=200_000)
    ap.add_argument('--n_test', type=int, default=0, help='events of the test/background samples (0 = all)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    torch.manual_seed(0)
    cfg = yaml.safe_load(open(args.config))
    {'features': features, 'train': train, 'apply': apply}[args.stage](args, cfg)


if __name__ == '__main__':
    main()
