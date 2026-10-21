"""MAP minimiser study for a trained conditional flow (referee request: robustness of the MAP search).

All methods maximise the same objective as Evaluation_Tools.flow_map_predict(method='gradient'): the flow density
expressed in the latent space, log p(x(z)|c) = log p_Z(z) - log|det dx/dz|, and the final value of every method is
recomputed with the same function at its end point, so they are directly comparable.

  grad200     Adam (lr 1e-2), 200 steps from z = 0                          -- the default used in the paper
  grad1000    Adam, 1000 steps from z = 0
  multi200    z = 0 plus N_RANDOM random starts z ~ N(0, 1), 200 steps each; best of all (best of the random
              starts alone is stored too)
  multi1000   the same with 1000 steps per start
  minuit      MIGRAD (iminuit) from z = 0, one event at a time. Default --minuit_mode gradient64: the model in float64
              and the exact gradient from autograd -- MIGRAD then converges (valid minimum, ~250 calls, ~7 s/event on 8
              threads). --minuit_mode numeric is the original flow_map_predict version (float32, numerical derivatives),
              which never reaches MIGRAD's convergence criterion on this objective (~2500 calls, ~56 s/event).
  minuit_numeric  always the original float32 / numerical-derivative MIGRAD from z = 0. It never converges, but its noisy
              path explores: on some events it finds a much higher peak (e.g. log p 11.8 vs 6.0) that z = 0 and the random
              starts all miss, so it is kept as part of the per-event 'best maximum found'.
  minuit_multi  MIGRAD from z = 0 plus the same N_RANDOM random starts; best of all (11x the cost of minuit, so it
              can be run on fewer events with --n_events_slow, as can minuit_numeric)

Stored per event (row index event_index in the test file): the final log p of every start (logp_starts; NLL = -log p),
the latent end points, log p at the convergence checkpoints (gradient methods), the decoded MAP of the best start
(x_best in physical output units, x_best_std standardised), and for Minuit the call count, validity and time per event.

Each method runs in its own process on CPU with a fixed number of threads, so the timing and the memory (peak resident
memory during the method, sampled every 20 ms, minus the baseline after loading model and data) are clean. Results
per method go to <outdir>/<method>.npz; '--method compare' writes the comparison table and plots.

The gradient methods can also run on a GPU (--device cuda), for the timing at large chunk sizes: one warm-up chunk is run
before the clock starts, the peak GPU memory (max_memory_allocated minus the model and data) is stored, and
--gpu_mem_gb caps what the process may use (e.g. 8, to emulate a smaller card) -- a chunk that does not fit then
stops with 'OOM' in the summary line and exit status 3.

    python taupolaris/scripts/map_minimizer_study.py -c <model config.yaml> --outdir <dir> --n_events N --method all
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

sys.path.insert(0, os.getcwd())

METHODS = ['grad200', 'grad1000', 'multi200', 'multi1000', 'minuit', 'minuit_multi', 'minuit_numeric']
N_RANDOM = 10
CHECKPOINTS = [0, 10, 20, 50, 100, 200, 500, 1000]


def rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024.0
    return float('nan')


class PeakRSS:
    def __init__(self):
        self.peak, self._stop = rss_mb(), False

    def __enter__(self):
        def run():
            while not self._stop:
                self.peak = max(self.peak, rss_mb()); time.sleep(0.02)
        self.t = threading.Thread(target=run, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self._stop = True; self.t.join(); self.peak = max(self.peak, rss_mb())


def load(cfg_path, model_path, test_file, n_events, device='cpu'):
    from taupolaris.python.NN_Tools import load_model, is_legacy_pizero_proj_checkpoint
    cfg = yaml.safe_load(open(cfg_path))
    D, NN = cfg['Data'], cfg['SetupNN']
    inp, out = D['Features']['input_features'], D['Features']['output_features'][D['coordinates']]
    sd = torch.load(model_path, map_location='cpu', weights_only=True)
    model = load_model(NN['hyperparams'], inp, out, useTransformer=NN.get('use_transformer', True),
                       leptonic_mode=D.get('leptonic_mode', 0), legacy_pizero_proj=is_legacy_pizero_proj_checkpoint(sd))
    model.load_state_dict(sd); model.float().eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    norm = np.load(os.path.join(os.path.dirname(os.path.dirname(model_path)) if model_path.endswith('plots/best_model.pth')
                                else os.path.dirname(model_path), 'normalization_params.npz'))
    df = next(pq.ParquetFile(test_file).iter_batches(batch_size=n_events, columns=inp)).to_pandas()
    X = torch.from_numpy(((df[inp].to_numpy(np.float32) - norm['input_mean']) / norm['input_std']).astype(np.float32)).to(device)
    return model, X, norm, out


def random_starts(n_events, d):
    """The N_RANDOM random latent starts per event, identical for every multi-start method (fixed seed, filled event
    by event, so the first n events get the same starts whatever n_events is)."""
    return np.random.default_rng(12345).standard_normal((n_events, N_RANDOM, d)).astype(np.float32)


def logp_z(model, z, ctx):
    _, logabsdet = model.flow._transform.inverse(z, context=ctx)
    return -0.5 * (z ** 2).sum(-1) - logabsdet


def adam_ascent(model, z0, ctx, n_steps, lr=1e-2, record=None):
    """Adam on z (batch), exactly as flow_map_predict(method='gradient'); record[k] = log p at step k."""
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for step in range(n_steps + 1):
        if record is not None and step in CHECKPOINTS:
            with torch.no_grad():
                record[step] = logp_z(model, z, ctx).cpu().numpy().copy()
        if step == n_steps:
            break
        opt.zero_grad()
        (-logp_z(model, z, ctx).sum()).backward()
        opt.step()
    with torch.no_grad():
        return z.detach(), logp_z(model, z, ctx)


def grad_chunk(model, Xc, z_rand, n_starts, n_steps, d, record=None):
    """One chunk of a gradient method: Adam from z = 0 (plus the random starts z_rand for multi-start)."""
    C = len(Xc)
    with torch.no_grad():
        ctx = model.condition_net(Xc)
    z0 = torch.zeros(C, n_starts, d, device=Xc.device)
    if n_starts > 1:
        z0[:, 1:] = z_rand
    z, lp = adam_ascent(model, z0.reshape(C * n_starts, d), ctx.repeat_interleave(n_starts, 0), n_steps, record=record)
    return z.reshape(C, n_starts, d).cpu().numpy(), lp.reshape(C, n_starts).cpu().numpy()


def run_method(args):
    torch.set_num_threads(args.threads); torch.manual_seed(1)
    dev = torch.device(args.device)
    gpu = dev.type == 'cuda'
    if gpu and dev.index is None:   # the torch.cuda memory functions need an explicit index
        dev = torch.device('cuda', 0)
    if gpu and args.method.startswith('minuit'):
        sys.exit('--device cuda is for the gradient methods only (Minuit evaluates one event at a time)')
    if gpu and args.gpu_mem_gb:
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.gpu_mem_gb * 1024 ** 3 / torch.cuda.get_device_properties(dev).total_memory), dev)
    model, X, norm, out = load(args.config, args.model_path, args.test_file, args.n_events, dev)
    d = model.flow._distribution._shape[0]
    res = {'loadavg_start': np.array(os.getloadavg()), 'host': np.array(os.uname().nodename), 'device': np.array(args.device)}
    where = res['host']
    if gpu:
        res['gpu_name'] = np.array(torch.cuda.get_device_name(dev))
        where = f"{res['host']} ({res['gpu_name']}, limit {args.gpu_mem_gb or 'none'} GB)"
    try:
        if gpu:   # warm-up: CUDA context, kernels and the caching allocator, outside the timing
            Xw = X[:args.chunk]
            grad_chunk(model, Xw, torch.randn(len(Xw), N_RANDOM, d, device=dev), 1 + N_RANDOM * args.method.startswith('multi'), 5, d)
            torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
            res['baseline_gpu_mem_mb'] = torch.cuda.memory_allocated(dev) / 2 ** 20
        base = rss_mb()
        t0 = time.perf_counter()
        with PeakRSS() as mem:
            if args.method in ('grad200', 'grad1000', 'multi200', 'multi1000'):
                n_steps = 200 if args.method.endswith('200') else 1000
                n_starts = 1 + (N_RANDOM if args.method.startswith('multi') else 0)
                lp_all, z_all, traj = [], [], {k: [] for k in CHECKPOINTS if k <= n_steps}
                rs = torch.from_numpy(random_starts(len(X), d))
                for s in range(0, len(X), args.chunk):
                    Xc = X[s:s + args.chunk]
                    rec = {}
                    z, lp = grad_chunk(model, Xc, rs[s:s + len(Xc)].to(dev), n_starts, n_steps, d, record=rec)
                    lp_all.append(lp); z_all.append(z)
                    for k in traj:
                        traj[k].append(rec[k].reshape(len(Xc), n_starts))
                res['logp_starts'] = np.concatenate(lp_all)                       # (N, n_starts); column 0 = z=0 start
                res['z_starts'] = np.concatenate(z_all)
                for k in traj:
                    res[f'traj_{k}'] = np.concatenate(traj[k])
            elif args.method in ('minuit', 'minuit_multi', 'minuit_numeric'):
                from iminuit import Minuit
                exact = args.minuit_mode == 'gradient64' and args.method != 'minuit_numeric'
                if exact:
                    model.double(); X = X.double()
                dtp = torch.float64 if exact else torch.float32
                n_starts = 1 + (N_RANDOM if args.method == 'minuit_multi' else 0)
                starts = np.zeros((len(X), n_starts, d))
                if n_starts > 1:
                    starts[:, 1:] = random_starts(len(X), d)
                lp, nfcn, valid = np.zeros((len(X), n_starts)), np.zeros((len(X), n_starts)), np.zeros((len(X), n_starts))
                zs = np.zeros((len(X), n_starts, d))
                t_ev = np.zeros(len(X))
                for i in range(len(X)):
                    ti = time.perf_counter()
                    with torch.no_grad():
                        ctx = model.condition_net(X[i:i + 1])
                        f = lambda zz: -logp_z(model, torch.as_tensor(zz, dtype=dtp).unsqueeze(0), ctx).item()

                    def g(zz, ctx=ctx):
                        zt = torch.as_tensor(zz, dtype=dtp).unsqueeze(0).requires_grad_(True)
                        (-logp_z(model, zt, ctx).sum()).backward()
                        return zt.grad[0].double().numpy()
                    for j in range(n_starts):
                        m = Minuit(f, starts[i, j], grad=g) if exact else Minuit(f, starts[i, j])
                        m.errordef = Minuit.LIKELIHOOD; m.migrad()
                        zz = torch.tensor(np.array(m.values), dtype=dtp).unsqueeze(0)
                        with torch.no_grad():
                            lp[i, j] = logp_z(model, zz, ctx).item()
                        nfcn[i, j] = m.nfcn; valid[i, j] = m.valid; zs[i, j] = np.array(m.values)
                    t_ev[i] = time.perf_counter() - ti
                    if (i + 1) % 10 == 0 and i + 1 < len(X):
                        # partial results, so a job that runs out of time still leaves the events it finished
                        n = i + 1
                        np.savez(os.path.join(args.outdir, f'{args.method}_partial.npz'), logp_starts=lp[:n], z_starts=zs[:n],
                                 nfcn=nfcn[:n], valid=valid[:n].astype(bool), time_per_event_s=t_ev[:n], event_index=np.arange(n),
                                 time_s=time.perf_counter() - t0, n_events=n, peak_mem_mb=mem.peak - base, threads=args.threads,
                                 host=res['host'], loadavg_start=res['loadavg_start'])
                        print(f'  {args.method}: {n}/{len(X)} events, {t_ev[:n].mean():.1f} s/event', flush=True)
                res['logp_starts'], res['z_starts'], res['nfcn'], res['valid'] = lp, zs, nfcn, valid.astype(bool)
                res['time_per_event_s'] = t_ev
        if gpu:
            torch.cuda.synchronize(dev)
    except torch.OutOfMemoryError:
        print(f'>> {args.method} on {where}: chunk {args.chunk}: OOM', flush=True)
        sys.exit(3)
    dt = time.perf_counter() - t0
    # per event: row in the test file, and the decoded MAP (best start) in standardised and physical output units
    res['event_index'] = np.arange(len(X))
    best = res['logp_starts'].argmax(1)
    zb = torch.tensor(res['z_starts'][np.arange(len(X)), best], dtype=next(model.parameters()).dtype, device=dev)
    with torch.no_grad():
        xb = torch.cat([model.flow._transform.inverse(zb[s:s + 1000], context=model.condition_net(X[s:s + 1000]))[0]
                        for s in range(0, len(X), 1000)]).cpu().numpy()
    res['x_best_std'] = xb.astype(np.float32)
    res['x_best'] = xb * norm['output_std'].reshape(-1) + norm['output_mean'].reshape(-1)
    res['output_names'] = np.array(out)
    res['loadavg_end'] = np.array(os.getloadavg())
    res['time_s'] = dt; res['peak_mem_mb'] = mem.peak - base; res['baseline_mem_mb'] = base
    res['threads'] = args.threads; res['n_events'] = len(X); res['chunk'] = args.chunk
    gmem = ''
    if gpu:
        res['peak_gpu_mem_mb'] = torch.cuda.max_memory_allocated(dev) / 2 ** 20 - res['baseline_gpu_mem_mb']
        gmem = f", peak extra GPU memory {res['peak_gpu_mem_mb']:.0f} MB (baseline {res['baseline_gpu_mem_mb']:.0f} MB)"
    np.savez(os.path.join(args.outdir, f'{args.method}.npz'), **res)
    print(f'>> {args.method} on {where}: load average {res["loadavg_start"][0]:.1f} -> {res["loadavg_end"][0]:.1f}; '
          f'{len(X)} events, chunk {args.chunk}, {dt:.1f} s ({1e3 * dt / len(X):.3f} ms/event), '
          f'peak extra memory {mem.peak - base:.0f} MB (baseline {base:.0f} MB){gmem}', flush=True)


def deficit_stats(d):
    """Per-event deficit d = (best log p found by any method) - (this method's), in nats: its mean and quantiles (the
    spread over events), and how often it exceeds 0.01 / 0.1 / 1 nat (density lower than the best peak's by ~1% /
    ~10% / a factor e)."""
    q = np.quantile(d, [0.5, 0.68, 0.95, 0.99])
    return {'mean deficit': d.mean(), 'median': q[0], '68%': q[1], '95%': q[2], '99%': q[3],
            'frac > 0.01': (d > 0.01).mean(), 'frac > 0.1': (d > 0.1).mean(), 'frac > 1': (d > 1).mean()}


def compare(args):
    import pandas as pd
    R = {}
    for m in METHODS:                         # a finished method, else the partial results of one still running / timed out
        for f in (f'{m}.npz', f'{m}_partial.npz'):
            if os.path.exists(os.path.join(args.outdir, f)):
                R[m] = dict(np.load(os.path.join(args.outdir, f)))
                if f.endswith('_partial.npz'):
                    print(f'>> {m}: partial results, {int(R[m]["n_events"])} events')
                break
    n = min(r['logp_starts'].shape[0] for r in R.values())           # events common to every method (the first n)
    best_of = {m: r['logp_starts'][:n].max(1) for m, r in R.items()}
    if 'minuit_multi' in R:
        best_of['minuit_multi (random starts only)'] = R['minuit_multi']['logp_starts'][:n, 1:].max(1)
    if 'multi200' in R:
        best_of['multi200 (random starts only)'] = R['multi200']['logp_starts'][:n, 1:].max(1)
    if 'multi1000' in R:
        best_of['multi1000 (random starts only)'] = R['multi1000']['logp_starts'][:n, 1:].max(1)
    ref = np.max(np.stack(list(best_of.values())), 0)                       # best value found by any method, per event
    rows = []
    for name, lp in best_of.items():
        m = name.split()[0]; d = ref - lp
        rows.append({'method': name, 'mean log p': lp.mean(), **deficit_stats(d),
                     'ms/event': 1e3 * R[m]['time_s'] / R[m]['n_events'], 'peak extra MB': float(R[m]['peak_mem_mb'])})
    t = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(f"\n=== {len(ref)} events, {int(R[next(iter(R))]['threads'])} CPU threads; deficit = best log p found by any method minus this method's ===")
    print(t.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    t.to_csv(os.path.join(args.outdir, 'comparison.csv'), index=False)
    big = {m: r for m, r in R.items() if r['logp_starts'].shape[0] > n}
    if len(big) > 1:
        nb = min(r['logp_starts'].shape[0] for r in big.values())
        bo = {m: r['logp_starts'][:nb].max(1) for m, r in big.items()}
        for m in ('multi200', 'multi1000'):
            if m in big:
                bo[f'{m} (random starts only)'] = big[m]['logp_starts'][:nb, 1:].max(1)
        refb = np.max(np.stack(list(bo.values())), 0)
        tb = pd.DataFrame([{'method': k, 'mean log p': v.mean(), **deficit_stats(refb - v)} for k, v in bo.items()])
        print(f'\n=== the methods run on more events: first {nb} events (deficit relative to the best of these methods) ===')
        print(tb.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        tb.to_csv(os.path.join(args.outdir, f'comparison_{nb}events.csv'), index=False)
    for m in ('multi200', 'multi1000', 'minuit_multi'):
        if m in R:
            lp = R[m]['logp_starts'][:n]; gain = lp[:, 1:].max(1) - lp[:, 0]
            print(f'>> {m}: a random start beats z=0 by > 0.01 nats in {(gain > 0.01).mean():.1%} of events, > 0.1 in {(gain > 0.1).mean():.1%}, '
                  f'> 1 in {(gain > 1).mean():.1%}; z=0 is the best start in {(lp.argmax(1) == 0).mean():.1%}')
    if 'minuit' in R:
        print(f">> minuit: valid minimum in {R['minuit']['valid'].mean():.1%} of events, median {np.median(R['minuit']['nfcn']):.0f} function calls")
    print(f'>> table on the first {n} events (common to all methods); ms/event and memory from each method\'s own run')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for m, col in (('multi1000', '#2a78d6'), ('grad1000', '#eb6834')):
        if m not in R:
            continue
        ks = [k for k in CHECKPOINTS if f'traj_{k}' in R[m]]
        z0 = [np.mean(ref - R[m][f'traj_{k}'][:, 0]) for k in ks]
        ax.plot([max(k, 1) for k in ks], z0, 'o-', color=col, label=f'{m}: start z = 0')
        if m == 'multi1000':
            rnd = [np.mean(ref - R[m][f'traj_{k}'][:n, 1:].max(1)) for k in ks]
            ax.plot([max(k, 1) for k in ks], rnd, 's--', color='#1baf7a', label='multi1000: best of 10 random starts')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Adam step'); ax.set_ylabel('mean deficit to best log p found [nats]')
    ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'convergence.pdf')); plt.close(fig)
    print(f">> wrote {args.outdir}/comparison.csv, convergence.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c', required=True)
    ap.add_argument('--model_path', default=None, help='default: <outputs dir>/plots/best_model.pth')
    ap.add_argument('--test_file', default='prepared_LHC_data_sep01/ppToHToTauTau_AllDM_UnCorr_Jul20/test_dataframe__leptonic_mode_0_transformer.parquet')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--method', default='all', choices=METHODS + ['all', 'compare'])
    ap.add_argument('--n_events', type=int, default=1000)
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--device', default='cpu', help="'cuda' for the gradient methods on a GPU")
    ap.add_argument('--gpu_mem_gb', type=float, default=None, help='with --device cuda: cap on the GPU memory this process may use')
    ap.add_argument('--n_events_slow', type=int, default=None,
                    help='events for the slow methods minuit_multi and minuit_numeric (the FIRST n of the same file, so they are '
                         'a subset of the events the other methods use; default: --n_events)')
    ap.add_argument('--minuit_mode', choices=('gradient64', 'numeric'), default='gradient64')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.model_path is None:
        args.model_path = os.path.join(f"outputs_{yaml.safe_load(open(args.config))['SetupNN']['model_name']}", 'plots', 'best_model.pth')
    if args.method == 'all':
        for m in METHODS:                                            # one process per method: clean time and memory
            subprocess.run([sys.executable, '-u', __file__, '-c', args.config, '--model_path', args.model_path, '--test_file', args.test_file,
                            '--outdir', args.outdir, '--method', m,
                            '--n_events', str(args.n_events_slow or args.n_events) if m in ('minuit_multi', 'minuit_numeric') else str(args.n_events), '--chunk', str(args.chunk),
                            '--threads', str(args.threads), '--minuit_mode', args.minuit_mode], check=False)
        compare(args)
    elif args.method == 'compare':
        compare(args)
    else:
        run_method(args)


if __name__ == '__main__':
    main()
