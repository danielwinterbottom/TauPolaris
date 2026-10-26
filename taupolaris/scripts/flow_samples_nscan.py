"""How many flow samples per event the sample-based estimates need: the phiCP circular mean, and the posterior means
of the spin projections cos theta_i^+- and of their products. Reads evaluate.py outputs made with
--inc_phiCP_error --inc_spin_means --save_flow_samples (the per-sample arrays in <dir>/flow_samples/) and recomputes
every estimate from the first N samples of each event, for each N in --n-list.

phiCP: per-channel asymmetry A = sqrt(sum_bins (odd - even)^2) as in plot_phiCP.py (20 bins, unit-normalised
TauSpinner-weighted histograms), for the circular mean of N samples, the MAP (map_pred_phiCP) and the Run-3 approximate
methods on the same events; the combined A = sqrt(sum N_ch A_ch^2 / sum N_ch) over the hadronic and semileptonic channels
(total_asymmetry.py), and the gains over Run-3. Each A is also given with the expected statistical noise subtracted,
A_corr^2 = A^2 - sum_b var(odd_b - even_b) with var from the weights in each bin: the raw A of a 200k-event subset is
biased upwards by the noise, by about the same amount for every estimator.

Spin variables: for each of the 9 products cos theta_i^+ cos theta_j^- and cos_n cos_n - cos_r cos_r, two per-event
estimators from N samples, the mean of the product ('meanprod') and the product of the means ('prodmean'), and the MAP's
own value. Each is scored on how well it separates two spin hypotheses from the TauSpinner weights, CP-even vs CP-odd
(alpha 0 vs 90) and CP-even vs no spin correlation (alpha 0 vs weight 1): D = the difference of the weighted means,
S = D / (unweighted std), the per-event separation of a measurement from the mean (significance ~ sqrt(N_events) S),
R = D / D_true, the response relative to the generator-level value, and the histogram asymmetry (20 bins on [-1, 1]).

    python taupolaris/scripts/flow_samples_nscan.py --had <hadronic dir> --sl <semileptonic dir> --outdir <dir>
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from taupolaris.utils.acoplanarity_tools import get_R_P_vectors_all, compute_aco_classic, compute_aco_classic_a1a1

BINS_PHI = np.linspace(0, 2 * np.pi, 21)
BINS_COS = np.linspace(-1, 1, 21)
CHANNELS = {  # as plot_phiCP.py; the Run-3 methods have no DM 11
    'had': [(0, 0), (0, 1), (1, 1), (2, 2), (1, 2), (0, 2), (10, 10), (0, 10), (1, 10), (2, 10),
            (0, 11), (1, 11), (2, 11), (10, 11), (11, 11)],
    'sl': [(100, 0), (100, 1), (100, 2), (100, 10), (100, 11)],
}
COS = [f'cos{i}_{s}' for s in ('plus', 'minus') for i in 'nrk']
PRODUCTS = [f'cos{i}_plus_cos{j}_minus' for i in 'nrk' for j in 'nrk'] + ['cosnn_minus_cosrr']
# the columns get_R_P_vectors_all and compute_aco_classic_a1a1 read (plot_phiCP.py's _COLUMN_PREFIXES)
RUN3_PREFIXES = ('reco_taup_', 'reco_taun_', 'FastMTT_', 'map_pred_tau_', 'pred_tau_')


def run3_phiCP(df):
    R1, P1, dp1 = get_R_P_vectors_all(df, tau_prefix='taup', use_map=True)
    R2, P2, dp2 = get_R_P_vectors_all(df, tau_prefix='taun', use_map=True)
    phi = np.asarray(compute_aco_classic(R1, P1, R2, P2, dp1, dp2))
    return np.where((df['taup_DM'] == 10) & (df['taun_DM'] == 10), np.asarray(compute_aco_classic_a1a1(df)), phi)


def hist_asym(x, wa, wb, bins):
    """sqrt(sum_b (a_b - b_b)^2) of the unit-normalised histograms of x with weights wa and wb, and the same with the
    expected statistical noise sum_b var(a_b - b_b) subtracted from its square."""
    ua, ub = wa / wa.sum(), wb / wb.sum()
    a, _ = np.histogram(x, bins, weights=ua)
    b, _ = np.histogram(x, bins, weights=ub)
    noise, _ = np.histogram(x, bins, weights=(ua - ub) ** 2)
    a2 = ((a - b) ** 2).sum()
    return np.sqrt(a2), np.sqrt(max(a2 - noise.sum(), 0.)), a2, a2 - noise.sum()


def prefix_estimates(arrs, n_list, chunk=10000):
    """{name: [n_events, len(n_list)]} estimates from the first N samples of each [n_events, n_samples] array:
    the circular mean of phiCP, the means of the cosines, and of each product, and the products of the means."""
    n_events = len(arrs['phiCP'])
    idx = np.asarray(n_list) - 1
    out = {'circmean': np.empty((n_events, len(n_list)), np.float32)}
    spin = 'cosn_plus' in arrs
    if spin:
        for v in COS:
            out[f'mean_{v}'] = np.empty((n_events, len(n_list)), np.float32)
        for v in PRODUCTS:
            out[f'meanprod_{v}'] = np.empty((n_events, len(n_list)), np.float32)
            out[f'prodmean_{v}'] = np.empty((n_events, len(n_list)), np.float32)
    for s in range(0, n_events, chunk):
        e = min(s + chunk, n_events)
        phi = np.asarray(arrs['phiCP'][s:e], dtype=np.float64)
        out['circmean'][s:e] = np.arctan2(np.sin(phi).cumsum(1)[:, idx], np.cos(phi).cumsum(1)[:, idx]) % (2 * np.pi)
        if not spin:
            continue
        # a rare sample has a NaN cosine (undefined spin axis): leave it out of the running means of that event
        cos = {v: np.asarray(arrs[v][s:e], dtype=np.float64) for v in COS}
        good = np.all([np.isfinite(cos[v]) for v in COS], axis=0)
        n_good = np.maximum(good.cumsum(1)[:, idx], 1)
        cos = {v: np.where(good, cos[v], 0.) for v in COS}
        means = {v: cos[v].cumsum(1)[:, idx] / n_good for v in COS}
        for v in COS:
            out[f'mean_{v}'][s:e] = means[v]
        for i in 'nrk':
            for j in 'nrk':
                v = f'cos{i}_plus_cos{j}_minus'
                out[f'meanprod_{v}'][s:e] = (cos[f'cos{i}_plus'] * cos[f'cos{j}_minus']).cumsum(1)[:, idx] / n_good
                out[f'prodmean_{v}'][s:e] = means[f'cos{i}_plus'] * means[f'cos{j}_minus']
        for kind in ('meanprod', 'prodmean'):
            out[f'{kind}_cosnn_minus_cosrr'][s:e] = (out[f'{kind}_cosn_plus_cosn_minus'][s:e]
                                                     - out[f'{kind}_cosr_plus_cosr_minus'][s:e])
    return out


def point_products(df, prefix):
    """The products of a point estimate's (or the generator's) own cosines, or None if the columns are missing."""
    if f'{prefix}cosn_plus' not in df.columns:
        return None
    out = {f'cos{i}_plus_cos{j}_minus': df[f'{prefix}cos{i}_plus'].values * df[f'{prefix}cos{j}_minus'].values
           for i in 'nrk' for j in 'nrk'}
    out['cosnn_minus_cosrr'] = out['cosn_plus_cosn_minus'] - out['cosr_plus_cosr_minus']
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--had', help='hadronic evaluate.py output dir')
    ap.add_argument('--sl', help='semileptonic evaluate.py output dir')
    ap.add_argument('--name', default='output_results', help='test_output_name of the evaluate.py output')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--n-list', default='1,2,3,5,10,20,30,50,75,100,150,200,300,500,750,1000')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    phi_rows, spin_rows, timings = [], [], {}
    for group, d in (('had', a.had), ('sl', a.sl)):
        if d is None:
            continue
        path = f'{d}/{a.name}.parquet'
        names = pq.ParquetFile(path).schema_arrow.names
        keep = ('taup_DM', 'taun_DM', 'map_pred_phiCP', 'pred_phiCP_circmean')
        df = pd.read_parquet(path, columns=[c for c in names if c.startswith(RUN3_PREFIXES + ('tauspinner_wt_', 'true_cos', 'map_pred_cos'))
                                            or c in keep])
        arrs = {v: np.load(f'{d}/flow_samples/{a.name}_{v}.npy', mmap_mode='r')
                for v in ['phiCP'] + COS if os.path.exists(f'{d}/flow_samples/{a.name}_{v}.npy')}
        n_fs = arrs['phiCP'].shape[1]
        n_list = [n for n in map(int, a.n_list.split(',')) if n <= n_fs]
        with open(f'{d}/{a.name}_sampling_timing.json') as f:
            timings[group] = json.load(f)
        print(f'{group}: {len(df)} events, {n_fs} samples/event, from {d}')

        est = prefix_estimates(arrs, n_list)
        # the stored circular mean (all samples) must match the one recomputed from the saved samples
        if n_list[-1] == n_fs:
            dphi = np.angle(np.exp(1j * (est['circmean'][:, -1] - df['pred_phiCP_circmean'].values)))
            print(f'  max |stored - recomputed circular mean| = {np.nanmax(np.abs(dphi)):.2e} rad')

        w0, w90 = df['tauspinner_wt_alpha0'].values, df['tauspinner_wt_alpha90'].values
        phis = {'run3': run3_phiCP(df), 'map': df['map_pred_phiCP'].values}
        phis.update({f'circmean_{n}': est['circmean'][:, k] for k, n in enumerate(n_list)})
        for p, n in CHANNELS[group]:
            mask = (((df['taup_DM'] == p) & (df['taun_DM'] == n)) | ((df['taup_DM'] == n) & (df['taun_DM'] == p))).values
            for name, phi in phis.items():
                if name == 'run3' and 11 in (p, n):
                    continue
                A, A_corr, A2, A2_corr = hist_asym(phi[mask], w0[mask], w90[mask], BINS_PHI)
                phi_rows.append(dict(group=group, channel=f'DM{p}DM{n}', estimator=name, N_events=int(mask.sum()),
                                     A=A, A_corr=A_corr, A2=A2, A2_corr=A2_corr))

        # spin variables: the hypotheses as event weights
        hyps = {'alpha0': w0, 'alpha90': w90, 'uncorr': np.ones(len(df))}
        truth = point_products(df, 'true_')
        mapv = point_products(df, 'map_pred_')
        if 'mean_cosn_plus' not in est:
            continue
        for v in PRODUCTS:
            variants = {'map': (None, mapv[v] if mapv else None)}
            for kind in ('meanprod', 'prodmean'):
                variants.update({f'{kind}_{n}': (n, est[f'{kind}_{v}'][:, k]) for k, n in enumerate(n_list)})
            for (ha, hb) in (('alpha0', 'alpha90'), ('alpha0', 'uncorr')):
                D_true = (np.average(truth[v], weights=hyps[ha]) - np.average(truth[v], weights=hyps[hb])) if truth else np.nan
                for name, (n, x) in variants.items():
                    if x is None:
                        continue
                    D = np.average(x, weights=hyps[ha]) - np.average(x, weights=hyps[hb])
                    A, A_corr, _, _ = hist_asym(x, hyps[ha], hyps[hb], BINS_COS)
                    spin_rows.append(dict(group=group, variable=v, pair=f'{ha}_vs_{hb}', estimator=name.split('_')[0], N=n,
                                          D=D, S=D / x.std(), R=D / D_true, A_hist=A, A_hist_corr=A_corr))

    phi_t = pd.DataFrame(phi_rows)
    phi_t.to_csv(f'{a.outdir}/phiCP_asymmetry_per_channel.csv', index=False)

    # combined over channels, per group and over both
    def combined(t, col):
        return np.sqrt((t['N_events'] * t[col]).sum() / t['N_events'].sum())
    comb = []
    for est_name, t in phi_t.groupby('estimator', sort=False):
        row = {'estimator': est_name}
        for g, tg in list(t.groupby('group')) + [('total', t)]:
            row[f'A_{g}'] = combined(tg, 'A2')
            row[f'A_corr_{g}'] = combined(tg, 'A2_corr')
        comb.append(row)
    comb = pd.DataFrame(comb).set_index('estimator')
    for col in [c for c in comb.columns if c.startswith('A')]:
        comb[f'gain_vs_run3_{col[2:]}'] = comb[col] / comb.loc['run3', col] - 1
        comb[f'gain_vs_map_{col[2:]}'] = comb[col] / comb.loc['map', col] - 1
    # time per event of N samples, scaled linearly from the saved run's own timing
    for g, tm in timings.items():
        n = comb.index.str.extract(r'circmean_(\d+)', expand=False).astype(float).values
        per_sample = (tm['sampling'] + tm['conversion'] + tm['phiCP']) / tm['n_flow_samples']
        comb[f'ms_per_event_{g}'] = n * per_sample * 1e3
        comb[f'ms_per_event_with_spin_{g}'] = n * (per_sample + tm['spin'] / tm['n_flow_samples']) * 1e3
    comb.to_csv(f'{a.outdir}/phiCP_asymmetry_combined.csv')
    with open(f'{a.outdir}/timing.json', 'w') as f:
        json.dump(timings, f, indent=1)
    show = [c for c in comb.columns if c.startswith(('gain_vs_run3_corr', 'gain_vs_map_corr', 'ms_per_event_had', 'ms_per_event_sl'))]
    print(comb[show].to_string(float_format=lambda v: f'{v:.4f}'))

    if spin_rows:
        spin_t = pd.DataFrame(spin_rows)
        spin_t.to_csv(f'{a.outdir}/spin_estimators.csv', index=False)
        last = spin_t[(spin_t['N'] == spin_t['N'].max()) | (spin_t['estimator'] == 'map')]
        print(last.pivot_table(index=['group', 'pair', 'variable'], columns='estimator', values='S').to_string(float_format=lambda v: f'{v:+.4f}'))

    # figures: gain vs N (top) and time vs N (bottom), separate axes rather than a second y-scale
    ns = comb.index.str.extract(r'circmean_(\d+)', expand=False).astype(float).values
    sel = ~np.isnan(ns)
    fig, (ax, axt) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    for g, color, label in (('total', '#2a6fdb', 'combined'), ('had', '#e8833a', 'hadronic'), ('sl', '#3f9d5a', 'semileptonic')):
        col = f'gain_vs_run3_corr_{g}'
        if col not in comb:
            continue
        ax.plot(ns[sel], 100 * comb[col].values[sel], marker='o', ms=4, lw=2, color=color, label=f'circular mean, {label}')
        ax.axhline(100 * comb.loc['map', col], color=color, lw=1.5, ls='--', label=f'MAP, {label}')
    ax.set_ylabel('gain in asymmetry over Run-3 [%]')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    for g, color in (('had', '#e8833a'), ('sl', '#3f9d5a')):
        if f'ms_per_event_{g}' in comb:
            axt.plot(ns[sel], comb[f'ms_per_event_{g}'].values[sel], lw=2, color=color, label=g)
    axt.set_xlabel('flow samples per event N')
    axt.set_ylabel('ms / event')
    axt.set_yscale('log')
    axt.grid(alpha=0.3)
    axt.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{a.outdir}/phiCP_gain_vs_nsamples.pdf')
    print(f'wrote {a.outdir}')


if __name__ == '__main__':
    main()
