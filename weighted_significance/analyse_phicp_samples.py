"""MAP vs flow-sample phiCP, and calibration of the flow's phiCP uncertainty.

Needs the raw per-event phiCP samples cached by
    evaluate_polvec.py -c <config> --resume --save_phicp_samples --sampling_only --max_events N
(outputs_<model>/_cache/<test>/phicpsamples_*.npy) plus that model's full results parquet
(pred_phiCP = MAP, true_phiCP, decay modes, TauSpinner weights), which it reads row-aligned
for the events the samples cover.

1. phiCP estimators, per channel (ordered decay-mode pairs, 20 bins, same metric as
   make_asymmetry_yaml.py / plot_phiCP.py):
      MAP              the evaluation's usual estimate
      circular mean    atan2(<sin>, <cos>) of the flow samples
      single sample    the first sample (what a sampling-based estimate looks like)
   plus <cos(pred - true)> per channel and N-weighted totals.

2. Calibration: the rank of the true phiCP among the n samples of its event. phiCP is
   periodic, so the values are unrolled around a centre computed from samples AND truth
   together (a symmetric function of all n+1 values, which keeps the truth exchangeable
   with the samples): if the flow is calibrated the rank is uniform on 0..n. Also the
   coverage of central intervals (68%, 95%) built from the samples' spread about that
   centre, overall, per channel and per quartile of the uncertainty.

    python weighted_significance/analyse_phicp_samples.py --model_dir outputs_<model> [--label NAME]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_asymmetry_yaml import HADRONIC, asymmetry
from total_asymmetry import load_asymmetry_yaml

TWO_PI = 2 * np.pi
wrap = lambda x: (x + np.pi) % TWO_PI - np.pi          # to (-pi, pi]


def contiguous(cache_dir, kind):
    found = {}
    for f in glob.glob(os.path.join(cache_dir, f'{kind}_*.npy')):
        a, b = (int(x) for x in os.path.basename(f)[:-4].split('_')[-2:])
        found[a] = (b, f)
    files, at = [], 0
    while at in found:
        b, f = found[at]; files.append(f); at = b
    return files, at


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--test_output_name', default='output_results_UnCorr')
    ap.add_argument('--results', default=None, help='default: <model_dir>/polvec_eval_results_<test_output_name>.parquet')
    ap.add_argument('--label', default=None)
    ap.add_argument('--n_samples', type=int, default=50, help='which cached sample set: phicpsamples<n>_*.npy')
    ap.add_argument('--lep', default=os.path.join(HERE, 'asymmetry_lep_Polaris.yaml'))
    ap.add_argument('--ref_had', default=os.path.join(HERE, 'asymmetry_had_Polaris.yaml'))
    args = ap.parse_args()
    label = args.label or os.path.basename(os.path.normpath(args.model_dir))
    outdir = os.path.join(HERE, f'phicp_samples_{label}_n{args.n_samples}'); os.makedirs(outdir, exist_ok=True)

    files, n = contiguous(os.path.join(args.model_dir, '_cache', args.test_output_name), f'phicpsamples{args.n_samples}')
    if not files:
        raise SystemExit(f'no phicpsamples{args.n_samples}_*.npy cached -- run evaluate_polvec.py with --save_phicp_samples --n_flow_samples {args.n_samples} first')
    S = np.concatenate([np.load(f) for f in files]).astype(np.float32)     # (n, n_fs); float32 keeps 200 samples x 500k events in memory
    results = args.results or os.path.join(args.model_dir, f'polvec_eval_results_{args.test_output_name}.parquet')
    cols = ['true_phiCP', 'pred_phiCP', 'pred_phiCP_err', 'reco_taup_DM', 'reco_taun_DM', 'tauspinner_wt_alpha0', 'tauspinner_wt_alpha90']
    df = pd.read_parquet(results, columns=cols)
    if len(df) < n:
        raise SystemExit(f'results file has {len(df)} rows but samples cover {n} events')
    df = df.iloc[:n].reset_index(drop=True)
    true, MAP = df.true_phiCP.to_numpy(float), df.pred_phiCP.to_numpy(float)
    dm_p, dm_n = df.reco_taup_DM.to_numpy(), df.reco_taun_DM.to_numpy()
    w_e, w_o = df.tauspinner_wt_alpha0.to_numpy(float), df.tauspinner_wt_alpha90.to_numpy(float)
    good = np.isfinite(S).all(axis=1) & np.isfinite(true) & np.isfinite(MAP)
    n_fs = S.shape[1]
    print(f'>> {label}: {n} events with {n_fs} samples each ({len(files)} chunks); usable {good.sum()} '
          f'({(~good).sum()} dropped: failed sampling or undefined true phiCP)')

    # consistency: the circular std of the cached samples vs the stored pred_phiCP_err
    R = np.hypot(np.sin(S).mean(1), np.cos(S).mean(1))
    cstd = np.sqrt(-2 * np.log(np.clip(R, 1e-12, 1)))
    err = df.pred_phiCP_err.to_numpy(float)
    ok = good & np.isfinite(err)
    print(f'   circular std of the new samples vs the stored pred_phiCP_err: median ratio '
          f'{np.median(cstd[ok] / err[ok]):.3f} (1 expected; independent random draws)')

    cmean = np.arctan2(np.sin(S).mean(1), np.cos(S).mean(1)) % TWO_PI
    est = {'MAP': MAP, 'circular mean': cmean, 'single sample': S[:, 0] % TWO_PI}

    # ---- 1. estimators per channel
    rows = []
    for p, q in HADRONIC:
        m = good & (dm_p == p) & (dm_n == q)
        if m.sum() < 20:
            continue
        r = {'channel': f'DM{p}DM{q}', 'N': int(m.sum())}
        for k, v in est.items():
            r[f'asym {k}'] = asymmetry(v[m], w_e[m], w_o[m])
        for k, v in est.items():
            r[f'<cos d> {k}'] = float(np.mean(np.cos(v[m] - true[m])))
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab['circ mean / MAP'] = tab['asym circular mean'] / tab['asym MAP'] - 1
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print('\n=== phiCP estimators per channel (ordered pairs) ===')
    print(tab.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    tab.to_csv(os.path.join(outdir, 'estimators_per_channel.csv'), index=False)

    # totals: subset N-weighted, and combined with the leptonic channels using FULL-sample N
    lep = load_asymmetry_yaml(args.lep); ref = load_asymmetry_yaml(args.ref_had)
    S2_lep = sum(v['N'] * v['asymmetry'] ** 2 for v in lep.values()); N_lep = sum(v['N'] for v in lep.values())
    S_ref = np.sqrt((sum(v['N'] * v['asymmetry'] ** 2 for v in ref.values()) + S2_lep) / (sum(v['N'] for v in ref.values()) + N_lep))
    S_old = S_ref / 1.18
    print('\n=== totals ===')
    for k in est:
        had = np.sqrt((tab.N * tab[f'asym {k}'] ** 2).sum() / tab.N.sum())
        full_N = tab.channel.map(lambda c: ref[c]['N'] if c in ref else 0)
        comb = np.sqrt(((full_N * tab[f'asym {k}'] ** 2).sum() + S2_lep) / (full_N.sum() + N_lep))
        print(f'   {k:14s} hadronic (subset) {had:.4f}   had+lep {comb:.4f}  ({comb / S_old - 1:+.1%} vs old method)')

    # sample-count scan: the circular mean from the first k samples, for every k = 1..n_fs, from running
    # sums of sin and cos (one pass). Combined total uses full-sample N per channel, as the totals above.
    chans = [(f'DM{p_}DM{q_}', good & (dm_p == p_) & (dm_n == q_)) for p_, q_ in HADRONIC]
    chans = [(c, m) for c, m in chans if m.sum() >= 20 and c in ref]
    cs, cc = np.cumsum(np.sin(S), axis=1), np.cumsum(np.cos(S), axis=1)
    map_comb = np.sqrt((sum(ref[c]['N'] * asymmetry(MAP[m], w_e[m], w_o[m]) ** 2 for c, m in chans) + S2_lep)
                       / (sum(ref[c]['N'] for c, _ in chans) + N_lep))
    scan_rows = []
    for k in range(1, n_fs + 1):
        cm_k = np.arctan2(cs[:, k - 1], cc[:, k - 1]) % TWO_PI
        per = {c: asymmetry(cm_k[m], w_e[m], w_o[m]) for c, m in chans}
        had = np.sqrt(sum(m.sum() * per[c] ** 2 for c, m in chans) / sum(m.sum() for _, m in chans))
        comb = np.sqrt((sum(ref[c]['N'] * per[c] ** 2 for c, _ in chans) + S2_lep) / (sum(ref[c]['N'] for c, _ in chans) + N_lep))
        scan_rows.append({'k': k, 'hadronic (subset)': had, 'had+lep': comb, 'vs old method': comb / S_old - 1, **per})
    del cs, cc
    scan = pd.DataFrame(scan_rows); scan.to_csv(os.path.join(outdir, 'circular_mean_vs_nsamples.csv'), index=False)
    print('\n=== circular mean vs number of samples used (had+lep, vs old method) ===')
    show = sorted({1, 2, 5, 10, 20, 25, 50, 75, 100, 150, 200, n_fs} & set(range(1, n_fs + 1)))
    for _, r in scan[scan.k.isin(show)].iterrows():
        print(f"   k = {int(r.k):3d}   {r['had+lep']:.4f}   {r['vs old method']:+.1%}")
    print(f'   MAP (reference)  {map_comb:.4f}   {map_comb / S_old - 1:+.1%};  circular mean overtakes the MAP at k = '
          + (str(int(scan.k[scan['had+lep'] > map_comb].min())) if (scan['had+lep'] > map_comb).any() else 'never'))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(scan.k, 100 * scan['vs old method'], '-', color='#1E2761', lw=1.6, label='circular mean of the first k samples')
    ax.axhline(100 * (map_comb / S_old - 1), color='#B02A2A', ls='--', lw=1.2, label=f'MAP ({map_comb / S_old - 1:+.1%})')
    ax.axhline(18.0, color='0.4', ls=':', lw=1.2, label='nu-method, MAP (+18.0%)')
    ax.set_xlabel('number of flow samples per event, k'); ax.set_ylabel('combined φCP sensitivity vs old method [%]')
    ax.set_title(f'{label}: {n} events, had + lep (leptonic: nu-method)', fontsize=10)
    ax.set_xlim(1, n_fs); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'asymmetry_vs_nsamples.pdf')); plt.close(fig)

    # ---- 2. calibration
    allv = np.concatenate([S, true[:, None]], axis=1)
    centre = np.arctan2(np.sin(allv).mean(1), np.cos(allv).mean(1))       # symmetric in samples + truth
    ds = wrap(S - centre[:, None]); dt = wrap(true - centre)
    rank = (ds < dt[:, None]).sum(1)
    hist = np.bincount(rank[good], minlength=n_fs + 1)
    expct = good.sum() / (n_fs + 1)
    chi2 = float(((hist - expct) ** 2 / expct).sum())
    abs_s = np.abs(ds); abs_t = np.abs(dt)
    cov = {c: float(np.mean((abs_t <= np.quantile(abs_s, c, axis=1))[good])) for c in (0.68, 0.95)}
    print(f'\n=== calibration (rank of true phiCP among {n_fs} samples; uniform if calibrated) ===')
    print(f'   chi2/ndf {chi2:.0f}/{n_fs} ;  lowest+highest rank bins hold {(hist[0] + hist[-1]) / good.sum():.3f} '
          f'of events (expected {2 / (n_fs + 1):.3f}) ;  middle 20% of ranks hold '
          f'{hist[int(0.4 * n_fs):int(0.6 * n_fs) + 1].sum() / good.sum():.3f}')
    print(f'   coverage of central intervals: 68% -> {cov[0.68]:.3f},  95% -> {cov[0.95]:.3f}')
    print('   (too few in the middle / too many at the edges = uncertainties too small; the reverse = too large)')
    crow = []
    q = np.nanquantile(err[good], [0, .25, .5, .75, 1])
    for lo, hi in zip(q[:-1], q[1:]):
        m = good & (err >= lo) & (err <= hi)
        crow.append({'slice': f'err {lo:.2f}-{hi:.2f}', 'N': int(m.sum()),
                     'cov68': float(np.mean(abs_t[m] <= np.quantile(abs_s[m], .68, axis=1))),
                     'cov95': float(np.mean(abs_t[m] <= np.quantile(abs_s[m], .95, axis=1)))})
    for p, qq in HADRONIC:
        m = good & (dm_p == p) & (dm_n == qq)
        if m.sum() < 200:
            continue
        crow.append({'slice': f'DM{p}DM{qq}', 'N': int(m.sum()),
                     'cov68': float(np.mean(abs_t[m] <= np.quantile(abs_s[m], .68, axis=1))),
                     'cov95': float(np.mean(abs_t[m] <= np.quantile(abs_s[m], .95, axis=1)))})
    ctab = pd.DataFrame(crow)
    print(ctab.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    ctab.to_csv(os.path.join(outdir, 'coverage.csv'), index=False)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(np.arange(n_fs + 1), hist / good.sum(), width=1.0, color='#4C72B0', edgecolor='white')
    axes[0].axhline(1 / (n_fs + 1), color='k', ls='--', lw=1, label='calibrated')
    axes[0].set_xlabel(f'rank of true phiCP among {n_fs} flow samples'); axes[0].set_ylabel('fraction of events'); axes[0].legend()
    x = np.arange(len(tab)); w = 0.27
    for i, k in enumerate(est):
        axes[1].bar(x + (i - 1) * w, tab[f'asym {k}'], w, label=k)
    axes[1].set_xticks(x); axes[1].set_xticklabels(tab.channel, rotation=60, fontsize=7); axes[1].set_ylabel('CP asymmetry'); axes[1].legend(fontsize=8)
    fig.suptitle(label); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'rank_and_estimators.pdf')); plt.close(fig)
    np.savez(os.path.join(outdir, 'per_event.npz'), circ_mean=cmean, rank=rank, good=good)
    print(f'>> outputs in {outdir}')


if __name__ == '__main__':
    main()
