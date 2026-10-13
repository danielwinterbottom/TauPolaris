"""Per-event uncertainties for the spin observables of the transformer regressor, two ways, compared with the
phiCP von Mises head (phicp_uncertainty_head.py).

The regressor is frozen throughout. Observables (canonical helicity basis, kinematic_helpers.compute_spin_angles:
ditau rest frame, k = tau+ direction, n/r from the beam axis), computed identically from the predicted and from the
true taus and polarimetric vectors:
    cos_i(+), cos_i(-)  for i = n, r, k           (6, the B coefficients)
    cos_i(+) * cos_j(-) for i, j = n, r, k        (9, the C matrix)
    and phiCP.

Option 1 ('cov')  a head predicts a full 12x12 covariance (Cholesky factor, standardised output space) of the
                  pol-vec and tau-momentum outputs around the regressor's prediction; multivariate Gaussian NLL.
                  Any observable's uncertainty then follows by sampling N(mu, Sigma) and recomputing it (32 samples).
Option 2 ('obs')  a head predicts one log sigma per observable (15), independent Gaussian NLLs on the observable
                  residuals (true - predicted). No correlations: one uncertainty, and one cut, per observable.
Reference ('phicp') the phiCP von Mises head's uncertainty, used as a single cut for every observable.

Stages:
  features  frozen-regressor pass over training/validation subsets and the test file; stores encoder output,
            predictions, truth targets, visible momenta, the observables (pred/true) and phiCP
  train     fits both heads (early-stopped on validation)
  apply     per-event uncertainties on the test file from both heads (option 1 by sampling)
  compare   calibration, ranking power and the per-observable measurement figure of merit vs a cut

    python taupolaris/scripts/spin_uncertainty_heads.py -c <regressor config> --workdir <dir> --stage features|train|apply|compare
        [--phicp_results <phicp_uncertainty_head results_test.parquet>]
"""
import argparse
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
from taupolaris.scripts.phicp_uncertainty_head import load_regressor, needed_columns, TL
from taupolaris.scripts.evaluate_regressor import unit
from taupolaris.scripts.evaluate_polvec import phicp_from_cart, add_DM, M_TAU
from taupolaris.utils.coordinate_conversions import ConvertFromOrthonormalNRK_Predictions_PolVec
from taupolaris.utils.kinematic_helpers import compute_spin_angles, boost, boost_vector

COS = [f'cos{i}_{s}' for s in ('plus', 'minus') for i in 'nrk']
OBS = COS + [f'cos{i}+cos{j}-' for i in 'nrk' for j in 'nrk']                     # 6 + 9
NOUT = 12                                                                          # h(6) + tau(6)


def out_index(out):
    """Indices of the pol-vec and tau outputs in the regressor's 18 outputs, in onorm_output_order."""
    names = [f'ts_hh_{t}_{c}' for t in TL for c in 'nrk'] + [f'undecayed_{t}_{c}' for t in TL for c in 'nrk']
    return [list(out).index(n) for n in names]


def observables(native12, vis):
    """native12: (N,12) onorm h(+), h(-), tau(+), tau(-); vis: (N,12) reco charged/pizero p for the two legs.
    Returns phiCP (N,) and the 15 observables (N,15)."""
    cart = ConvertFromOrthonormalNRK_Predictions_PolVec(native12, vis[:, 0:3], vis[:, 3:6], vis[:, 6:9], vis[:, 9:12])
    h1, h2, p1, p2 = unit(cart[:, 0:3]), unit(cart[:, 3:6]), cart[:, 6:9], cart[:, 9:12]
    E1, E2 = np.sqrt((p1 ** 2).sum(1) + M_TAU ** 2), np.sqrt((p2 ** 2).sum(1) + M_TAU ** 2)
    cols = [f'ts_hh_{t}_{c}' for t in TL for c in 'xyz'] + [f'undecayed_{t}_p{c}' for t in TL for c in 'xyz']
    phi = phicp_from_cart(pd.DataFrame(np.concatenate([h1, h2, p1, p2], 1), columns=cols), E1, E2, TL)
    t1, t2 = np.column_stack([E1, p1]), np.column_stack([E2, p2])
    bv = boost_vector(t1 + t2)
    sa = compute_spin_angles(boost(t1, -bv), boost(t2, -bv), h1, h2)
    c = np.stack([sa[k] for k in COS], 1)
    prods = np.stack([c[:, i] * c[:, 3 + j] for i in range(3) for j in range(3)], 1)
    return np.asarray(phi), np.concatenate([c, prods], 1)


# ---------------------------------------------------------------------------------------------- features
def features(args, cfg):
    model, norm, out = load_regressor(cfg)
    idx = out_index(out)
    cols = needed_columns(cfg, out)
    inp = cfg['Data']['Features']['input_features']
    D = cfg['Data']
    split = lambda kind, d: os.path.join(D['output_dir'], d, f'{kind}_dataframe_onorm_leptonic_mode_{D["leptonic_mode"]}_transformer.parquet')
    test = D['test_dataset'][0] if isinstance(D['test_dataset'], list) else D['test_dataset']
    samples = {'train': [(split('train', d), args.n_train_per_dataset) for d in D['datasets']],
               'val': [(split('val', d), args.n_val_per_dataset) for d in D['datasets']],
               'test': [(test, args.n_test)]}
    om, osd = norm['output_mean'].reshape(-1), norm['output_std'].reshape(-1)
    for name, files in samples.items():
        parts, t0 = [], time.time()
        for path, nmax in files:
            n = 0
            for b in pq.ParquetFile(path).iter_batches(batch_size=250_000, columns=[c for c in cols if c in pq.read_schema(path).names]):
                df = b.to_pandas()
                if nmax and n + len(df) > nmax:
                    df = df.iloc[:nmax - n]
                X = torch.from_numpy(((df[inp].to_numpy(np.float32) - norm['input_mean']) / norm['input_std']).astype(np.float32))
                with torch.no_grad():
                    ctx = torch.cat([model.encoder(X[s:s + 8192]) for s in range(0, len(X), 8192)])
                    pred = torch.cat([model.head(ctx[s:s + 8192]) for s in range(0, len(ctx), 8192)]).numpy()
                true_std = (df[list(out)].to_numpy(np.float64) - om) / osd
                vis = np.concatenate([df[[f'reco_{t}_{p}_p{c}' for c in 'xyz']].to_numpy(float) for t in TL for p in ('charged', 'pizero1')], 1)
                phi_p, obs_p = observables((pred[:, idx] * osd[idx] + om[idx]).astype(float), vis)
                phi_t, obs_t = observables(true_std[:, idx] * osd[idx] + om[idx], vis)
                dm = add_DM(df, TL, dm_prefix='reco')
                parts.append(dict(ctx=ctx.numpy().astype(np.float32), pred=pred.astype(np.float32), true=true_std.astype(np.float32),
                                  vis=vis.astype(np.float32), phi_p=phi_p.astype(np.float32), phi_t=phi_t.astype(np.float32),
                                  obs_p=obs_p.astype(np.float32), obs_t=obs_t.astype(np.float32),
                                  dm_p=np.asarray(dm[0], np.int16), dm_n=np.asarray(dm[1], np.int16),
                                  w0=df['tauspinner_wt_alpha0'].to_numpy(np.float32), w90=df['tauspinner_wt_alpha90'].to_numpy(np.float32)))
                n += len(df)
                if nmax and n >= nmax:
                    break
            print(f'  {name}: {os.path.basename(os.path.dirname(path))} {n} events ({time.time() - t0:.0f}s)', flush=True)
        arr = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(os.path.join(args.workdir, f'spin_features_{name}.npz'), **arr)
        print(f'>> spin_features_{name}.npz: {len(arr["phi_p"])} events', flush=True)


# ---------------------------------------------------------------------------------------------- heads
class Head(nn.Module):
    def __init__(self, n_in, n_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


TRIL = torch.tril_indices(NOUT, NOUT)


def chol(raw):
    """raw (B, 78) -> lower-triangular L (B,12,12) with positive diagonal (exp, clamped)."""
    L = torch.zeros(raw.shape[0], NOUT, NOUT, dtype=raw.dtype)
    L[:, TRIL[0], TRIL[1]] = raw
    d = torch.arange(NOUT)
    L[:, d, d] = torch.exp(L[:, d, d].clamp(-7, 4))
    return L


def nll_cov(raw, r):
    """multivariate Gaussian NLL (up to a constant) with Sigma = L L^T; r = y - mu (B,12)."""
    L = chol(raw)
    z = torch.linalg.solve_triangular(L, r.unsqueeze(-1), upper=False).squeeze(-1)
    return 0.5 * (z ** 2).sum(1) + torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)


def nll_obs(logsig, r):
    ls = logsig.clamp(-7, 3)
    return (0.5 * (r * torch.exp(-ls)) ** 2 + ls).sum(1)


def head_inputs(f):
    return torch.from_numpy(np.concatenate([f['ctx'], f['pred']], 1).astype(np.float32))


def fit(name, n_out, loss_fn, target_fn, tr, va, args):
    Xtr, Xva = head_inputs(tr), head_inputs(va)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    Rtr, Rva = target_fn(tr), target_fn(va)
    Wtr, Wva = torch.from_numpy(tr['w0'] + tr['w90']), torch.from_numpy(va['w0'] + va['w90'])
    head = Head(Xtr.shape[1], n_out)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    best, state, bad = np.inf, None, 0
    for ep in range(1, args.epochs + 1):
        head.train(); perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), args.batch_size):
            i = perm[s:s + args.batch_size]
            loss = (Wtr[i] * loss_fn(head(Xtr[i]), Rtr[i])).sum() / Wtr[i].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            v = sum((Wva[s:s + 65536] * loss_fn(head(Xva[s:s + 65536]), Rva[s:s + 65536])).sum().item() for s in range(0, len(Xva), 65536)) / Wva.sum().item()
        print(f'  [{name}] epoch {ep}: val NLL {v:.5f}', flush=True)
        if v < best - 1e-5:
            best, state, bad = v, {k: t.clone() for k, t in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    torch.save({'state': state, 'mu': mu, 'sd': sd, 'n_in': Xtr.shape[1], 'n_out': n_out},
               os.path.join(args.workdir, f'head_{name}.pth'))
    print(f'>> head_{name}.pth: best val NLL {best:.5f}', flush=True)


def train(args, cfg):
    _, _, out = load_regressor(cfg); idx = out_index(out)
    tr = dict(np.load(os.path.join(args.workdir, 'spin_features_train.npz')))
    va = dict(np.load(os.path.join(args.workdir, 'spin_features_val.npz')))
    cov_target = lambda f: torch.from_numpy((f['true'][:, idx] - f['pred'][:, idx]).astype(np.float32))
    obs_target = lambda f: torch.from_numpy((f['obs_t'] - f['obs_p']).astype(np.float32))
    for nm, f in (('train', tr), ('val', va)):
        r = (f['obs_t'] - f['obs_p']); print(f'>> {nm}: {len(r)} events; observable residual rms ' + ' '.join(f'{v:.3f}' for v in np.sqrt((r ** 2).mean(0))))
    if args.which in ('both', 'obs'):
        fit('obs', len(OBS), nll_obs, obs_target, tr, va, args)
    if args.which in ('both', 'cov'):
        fit('cov', len(TRIL[0]), nll_cov, cov_target, tr, va, args)


def load_head(path):
    ck = torch.load(path, weights_only=False)
    h = Head(ck['n_in'], ck['n_out']); h.load_state_dict(ck['state']); h.eval()
    return h, ck


def apply(args, cfg):
    _, norm, out = load_regressor(cfg); idx = out_index(out)
    om, osd = norm['output_mean'].reshape(-1)[idx], norm['output_std'].reshape(-1)[idx]
    f = dict(np.load(os.path.join(args.workdir, 'spin_features_test.npz')))
    res = {'phi_p': f['phi_p'], 'phi_t': f['phi_t'], 'dm_p': f['dm_p'], 'dm_n': f['dm_n'], 'w0': f['w0'], 'w90': f['w90']}
    for k, o in enumerate(OBS):
        res[f'pred_{o}'], res[f'true_{o}'] = f['obs_p'][:, k], f['obs_t'][:, k]
    X = head_inputs(f)
    h, ck = load_head(os.path.join(args.workdir, 'head_obs.pth'))
    with torch.no_grad():
        ls = torch.cat([h((X[s:s + 65536] - ck['mu']) / ck['sd']) for s in range(0, len(X), 65536)]).clamp(-7, 3).numpy()
    for k, o in enumerate(OBS):
        res[f'sig_obs_{o}'] = np.exp(ls[:, k])
    h, ck = load_head(os.path.join(args.workdir, 'head_cov.pth'))
    rng = np.random.default_rng(1)
    sig_cov = np.empty((len(X), len(OBS)), np.float32); phi_cov = np.empty(len(X), np.float32)
    t0 = time.time()
    for s in range(0, len(X), args.apply_chunk):
        e = min(s + args.apply_chunk, len(X))
        with torch.no_grad():
            L = chol(h((X[s:e] - ck['mu']) / ck['sd'])).numpy().astype(float)
        z = rng.standard_normal((e - s, args.n_samples, NOUT))
        smp = f['pred'][s:e, idx][:, None, :].astype(float) + np.einsum('bij,bkj->bki', L, z)        # (C, K, 12) standardised
        nat = (smp * osd + om).reshape(-1, NOUT)
        vis = np.repeat(f['vis'][s:e].astype(float), args.n_samples, axis=0)
        phi_s, obs_s = observables(nat, vis)
        phi_s = phi_s.reshape(e - s, args.n_samples); obs_s = obs_s.reshape(e - s, args.n_samples, -1)
        sig_cov[s:e] = obs_s.std(1)
        R = np.abs(np.exp(1j * phi_s).mean(1)); phi_cov[s:e] = np.sqrt(-2 * np.log(np.clip(R, 1e-9, 1)))
        if (s // args.apply_chunk) % 5 == 0:
            print(f'  cov sampling {e}/{len(X)} ({time.time() - t0:.0f}s)', flush=True)
    for k, o in enumerate(OBS):
        res[f'sig_cov_{o}'] = sig_cov[:, k]
    res['sig_cov_phiCP'] = phi_cov
    if args.phicp_results:
        p = pd.read_parquet(args.phicp_results, columns=['pred_phiCP', 'pred_phiCP_err'])
        assert len(p) == len(X) and np.allclose(np.cos(p.pred_phiCP.to_numpy() - f['phi_p']), 1, atol=1e-4), 'phiCP head results not row-aligned'
        res['sig_phicp'] = p.pred_phiCP_err.to_numpy(np.float32)
    pd.DataFrame(res).to_parquet(os.path.join(args.workdir, 'spin_uncertainties_test.parquet'))
    print('>> spin_uncertainties_test.parquet written')


# ---------------------------------------------------------------------------------------------- compare
def fom(pred, true, w):
    """1 / variance of the true mean estimated from the predicted values with a linear response correction:
    N_eff * cov(p,t)^2 / (var(t)^2 var(p)), with weighted moments and N_eff = (sum w)^2 / sum w^2."""
    W = w.sum(); mp, mt = (w * pred).sum() / W, (w * true).sum() / W
    vp, vt = (w * (pred - mp) ** 2).sum() / W, (w * (true - mt) ** 2).sum() / W
    cv = (w * (pred - mp) * (true - mt)).sum() / W
    return W ** 2 / (w ** 2).sum() * cv ** 2 / (vt ** 2 * vp), mt


def compare(args, cfg):
    d = pd.read_parquet(os.path.join(args.workdir, 'spin_uncertainties_test.parquet'))
    had = (d.dm_p.between(0, 11) & d.dm_n.between(0, 11)).to_numpy()
    d = d[had].reset_index(drop=True); w = d.w0.to_numpy(float)        # Standard Model (CP-even) spin correlations
    print(f'>> {len(d)} hadronic test events; figures of merit use the CP-even (SM) TauSpinner weights')
    sources = [('obs', 'own-observable head (option 2)'), ('cov', 'covariance head (option 1)'), ('phicp', 'phiCP head (same cut for all)')]
    sources = [s for s in sources if (f'sig_{s[0]}_{OBS[0]}' in d or (s[0] == 'phicp' and 'sig_phicp' in d))]
    sig = lambda src, o: d['sig_phicp'].to_numpy() if src == 'phicp' else d[f'sig_{src}_{o}'].to_numpy()
    fr = np.array([1.0, .9, .8, .7, .6, .5, .4, .3, .2])
    rows = []
    for o in OBS + ['phiCP']:
        if o == 'phiCP':
            r_abs = np.abs((d.phi_t - d.phi_p + np.pi) % (2 * np.pi) - np.pi).to_numpy()
        else:
            p, t = d[f'pred_{o}'].to_numpy(float), d[f'true_{o}'].to_numpy(float); r_abs = np.abs(t - p)
            f0, m0 = fom(p, t, w)
        for src, _ in sources:
            if o == 'phiCP' and src == 'obs':
                continue
            s = d['sig_cov_phiCP'].to_numpy() if (o == 'phiCP' and src == 'cov') else sig(src, o)
            q = np.quantile(s, np.linspace(0, 1, 6)); b = np.clip(np.searchsorted(q, s, side='right') - 1, 0, 4)
            if o == 'phiCP':
                cal = [(s[b == i].mean(), np.sqrt(-2 * np.log(np.clip(np.abs(np.exp(1j * (d.phi_t - d.phi_p).to_numpy()[b == i]).mean()), 1e-9, 1)))) for i in range(5)]
            else:
                cal = [(s[b == i].mean(), np.sqrt(((t - p)[b == i] ** 2).mean())) for i in range(5)]
            row = {'observable': o, 'source': src, 'Spearman(sig,|res|)': pd.Series(s).rank().corr(pd.Series(r_abs).rank()),
                   'calib. pred/actual (lowest, highest quintile)': f'{cal[0][0]:.3f}/{cal[0][1]:.3f}, {cal[4][0]:.3f}/{cal[4][1]:.3f}'}
            if o != 'phiCP':
                best = (1.0, 1.0, 0.0)
                for f in fr[1:]:
                    m = s <= np.quantile(s, f)
                    fm, mt = fom(p[m], t[m], w[m])
                    if fm / f0 > best[1]:
                        best = (f, fm / f0, mt - m0)
                row.update({'best keep': best[0], 'FOM gain': best[1] - 1, 'true-mean shift': best[2],
                            'true sd': np.sqrt((w * (t - m0) ** 2).sum() / w.sum())})
            rows.append(row)
    t = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
    print('\nFOM = 1/variance of the true mean estimated from the prediction (linear response); "FOM gain" at the best')
    print('keep fraction (cut on the source\'s uncertainty for that observable); "true-mean shift" = selection bias of the cut.')
    print(t.to_string(index=False, float_format=lambda v: f'{v:+.3f}' if abs(v) < 10 else f'{v:.0f}'))
    t.to_csv(os.path.join(args.workdir, 'spin_uncertainty_comparison.csv'), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c', required=True); ap.add_argument('--workdir', required=True)
    ap.add_argument('--stage', choices=('features', 'train', 'apply', 'compare'), required=True)
    ap.add_argument('--which', choices=('both', 'obs', 'cov'), default='both')
    ap.add_argument('--phicp_results', default=None)
    ap.add_argument('--n_train_per_dataset', type=int, default=500_000)
    ap.add_argument('--n_val_per_dataset', type=int, default=200_000)
    ap.add_argument('--n_test', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8192); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n_samples', type=int, default=32); ap.add_argument('--apply_chunk', type=int, default=50_000)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True); torch.manual_seed(0)
    cfg = yaml.safe_load(open(args.config))
    {'features': features, 'train': train, 'apply': apply, 'compare': compare}[args.stage](args, cfg)


if __name__ == '__main__':
    main()
