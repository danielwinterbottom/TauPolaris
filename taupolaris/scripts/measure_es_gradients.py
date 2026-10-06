"""Measure how hard the phiCP energy-score term pulls on a trained flow relative to its NLL,
to set lambda for  loss = NLL + lambda * ES  (taupolaris/python/phicp_loss.py).

For each of --n_batches batches (same batch size and event weights as training):
  g_NLL = d NLL / d theta,   g_ES = d ES / d theta     over all trainable parameters
Reported:
  * rho = mean ||g_ES|| / mean ||g_NLL|| per batch -- what a single optimiser step sees.
    lambda = r / rho makes the ES term pull with r times the NLL's strength (at this point of
    training), so the lambda scan is a scan in r.
  * the same ratio for the batch-AVERAGED gradients (noise removed), and separately for the
    conditioner (condition_net.*) and the flow (flow.*) parameters.
  * cos(avg g_ES, avg g_NLL): ~1 = the NLL already pushes phiCP the same way (the ES term would
    add little); ~0 = the ES pushes in directions the NLL ignores; < 0 = trade-off. Reference
    values: cos between the NLL gradients averaged over two disjoint halves of the batches (what
    "same direction" looks like at this noise level), and the same for ES.
  * the mean ES, against 0.637 for a flow whose phiCP samples are uniform (no information).

Reads the first events of the config's test file (Jul20, never used in training) -- only the
model's own standardisation is needed, not the training split. Weights the model the same
way evaluate_polvec does (its frozen eval snapshot if present, else best_model.pth).

    python taupolaris/scripts/measure_es_gradients.py -c <config> [--n_batches 20] [--n_samples 2] [--useCPU]
"""
import argparse
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

sys.path.insert(0, os.getcwd())
from taupolaris.python.DataProcessing import _resolve_polvec_feature_level, _resolve_frame_aligned_level
from taupolaris.python.NN_Tools import load_model, get_device, is_legacy_pizero_proj_checkpoint
from taupolaris.python.phicp_loss import PhiCPEnergyScore

ES_UNIFORM = 2 / np.pi      # E|X-y| - E|X-X'|/2 for uniform phiCP samples: 4/pi - 2/pi


def flat(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--n_batches', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=None, help='default: the config\'s hyperparams batch_size')
    ap.add_argument('--n_samples', type=int, default=2)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--useCPU', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    cfg = yaml.safe_load(open(args.config))
    D, NN = cfg['Data'], cfg['SetupNN']
    hp = NN['hyperparams']
    bs = args.batch_size or hp['batch_size']
    inp = list(D['Features']['input_features'])
    if D.get('inc_reco_taus'):
        inp += D['Features']['input_reco_tau_features']
    out = D['Features']['output_features'][D['coordinates']]
    outdir = f"outputs_{NN['model_name']}"
    norm = np.load(f'{outdir}/normalization_params.npz')
    device = torch.device('cpu') if args.useCPU else get_device()

    ckpt = args.checkpoint
    if ckpt is None:
        for c in (f'{outdir}/_cache/eval_model.pth', f'{outdir}/plots/best_model.pth', f"{outdir}/{NN['model_name']}.pth"):
            if os.path.exists(c):
                ckpt = c; break
    sd = torch.load(ckpt, map_location='cpu')
    model = load_model(hp, inp, out, useTransformer=NN.get('use_transformer', True), leptonic_mode=D.get('leptonic_mode', 0),
                       legacy_pizero_proj=is_legacy_pizero_proj_checkpoint(sd),
                       polvec_feature_level=_resolve_polvec_feature_level(D), frame_aligned_level=_resolve_frame_aligned_level(D),
                       context_hidden_layer=any(k.startswith('condition_net.context_hidden.') for k in sd))
    model.load_state_dict(sd)
    model.float().to(device).train()     # train mode, as in training (dropout on)
    print(f'>> {ckpt} on {device}; batch size {bs}, {args.n_batches} batches, {args.n_samples} samples/event')

    test = D['test_dataset'][0] if isinstance(D['test_dataset'], list) else D['test_dataset']
    wcols = D.get('training_weight_columns') or []
    n_ev = bs * args.n_batches
    pf = pq.ParquetFile(test)
    rows, got = [], 0
    for b in pf.iter_batches(batch_size=min(n_ev, 200_000), columns=list(dict.fromkeys(inp + out + wcols))):
        rows.append(b.to_pandas()); got += len(rows[-1])
        if got >= n_ev:
            break
    import pandas as pd
    df = pd.concat(rows).iloc[:n_ev]
    X = torch.from_numpy(((df[inp].to_numpy(np.float32) - norm['input_mean']) / norm['input_std']).astype(np.float32))
    Y = torch.from_numpy(((df[out].to_numpy(np.float32) - norm['output_mean']) / norm['output_std']).astype(np.float32))
    W = torch.from_numpy(df[wcols].to_numpy(np.float32).sum(1)) if wcols else torch.ones(len(df))
    es_fn = PhiCPEnergyScore(inp, out, *(torch.from_numpy(norm[k]) for k in ('input_mean', 'input_std', 'output_mean', 'output_std')),
                             n_samples=args.n_samples)

    names, params = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
    is_cond = torch.cat([torch.full((p.numel(),), n.startswith('condition_net.'), dtype=torch.bool) for n, p in zip(names, params)]).to(device)
    sums = {k: [torch.zeros(is_cond.numel(), device=device) for _ in range(2)] for k in ('nll', 'es')}
    norms = {'nll': [], 'es': []}; vals = {'nll': [], 'es': []}
    for b in range(args.n_batches):
        sl = slice(b * bs, (b + 1) * bs)
        x, y, w = X[sl].to(device), Y[sl].to(device), W[sl].to(device)
        emb = model.condition_net(x)
        nll = -(w * model.flow.log_prob(inputs=y, context=emb)).sum() / w.sum()
        g_nll = flat(torch.autograd.grad(nll, params, retain_graph=True, allow_unused=True), params)
        for attempt in range(5):
            try:
                es_ev = es_fn.per_event(model.flow, emb, x, y)
                break
            except AssertionError:      # rare nflows spline-inverse instability, as in evaluate_polvec
                if attempt == 4:
                    raise
        es = (w * es_ev).sum() / w.sum()
        g_es = flat(torch.autograd.grad(es, params, allow_unused=True), params)
        half = b % 2
        for k, g, v in (('nll', g_nll, nll), ('es', g_es, es)):
            sums[k][half] += g; norms[k].append(g.norm().item()); vals[k].append(v.item())
        print(f'   batch {b + 1}/{args.n_batches}: NLL {nll.item():.3f} ||g|| {norms["nll"][-1]:.3g} | '
              f'ES {es.item():.4f} ||g|| {norms["es"][-1]:.3g}', flush=True)

    cos = lambda a, c: (a @ c / (a.norm() * c.norm()).clamp_min(1e-30)).item()
    avg = {k: (s[0] + s[1]) / args.n_batches for k, s in sums.items()}
    rho = np.mean(norms['es']) / np.mean(norms['nll'])
    print('\n=== results ===')
    print(f'mean NLL {np.mean(vals["nll"]):.3f};  mean ES {np.mean(vals["es"]):.4f}  (uniform-phiCP flow: {ES_UNIFORM:.4f}; lower is better)')
    print(f'per-batch gradient norms: NLL {np.mean(norms["nll"]):.4g}, ES {np.mean(norms["es"]):.4g}  ->  rho = {rho:.4g}')
    for part, m in (('all', torch.ones_like(is_cond)), ('conditioner', is_cond), ('flow', ~is_cond)):
        a_n, a_e = avg['nll'][m], avg['es'][m]
        print(f'  [{part:11s}] batch-averaged ||g_NLL|| {a_n.norm().item():.4g}  ||g_ES|| {a_e.norm().item():.4g}  '
              f'ratio {a_e.norm().item() / max(a_n.norm().item(), 1e-30):.4g}   cos(ES, NLL) {cos(a_e, a_n):+.3f}')
    print(f'reference (two disjoint halves of the batches): cos(NLL_A, NLL_B) {cos(sums["nll"][0], sums["nll"][1]):+.3f}, '
          f'cos(ES_A, ES_B) {cos(sums["es"][0], sums["es"][1]):+.3f}')
    print('\nlambda for a target ratio r (ES pulls with r x the NLL strength per step):')
    for r in (0.03, 0.1, 0.3, 1.0):
        print(f'  r = {r:<5}  lambda = {r / rho:.4g}')


if __name__ == '__main__':
    main()
