"""Paired bootstrap and calibration comparison of several evaluate_polvec models on the SAME events.

(1) Paired Poisson bootstrap. Every replica draws one Poisson(1) weight per event, shared by all
    models and estimators, and recomputes each channel's asymmetry (make_asymmetry_yaml's metric,
    the bootstrap weight multiplying the TauSpinner weights) and the combined had+lep sensitivity
    S = sqrt((sum_i N_i,pass a_i^2 + S2_lep) / (N_had + N_lep)), as in uncertainty_cut_scan.py.
    Because the events are shared, the spread of the DIFFERENCE between two models is much smaller
    than either model's own spread -- that is the number that says whether a gap is real.
    Leptonic channels are fixed (nu-method yaml); the old-method reference is the published +18%.

    Variants: each model x each --column, and optionally with a common cut on the per-event phiCP
    uncertainty (--cut LABEL COLUMN PASS_FRACTION; threshold = that quantile of the model's own
    uncertainty over its hadronic events, as in uncertainty_cut_scan.py). The pass fraction is
    chosen on the full sample, so cut results carry a small optimism bias common to all models.

(2) Calibration of the per-event uncertainty (--err_column, the circular std of the flow samples)
    against the residual to the true phiCP: 1-sigma/2-sigma coverage and a reliability curve
    (circular std of the residuals vs the mean predicted uncertainty, in quantile bins of the
    prediction). Perfect calibration sits on the diagonal.

    python weighted_significance/compare_models_bootstrap.py \\
        --model unfrozen <parquet> --model frozen <parquet> ... [--ref unfrozen] \\
        [--cut frozen pred_phiCP_circmean 0.70 ...] --outdir weighted_significance/<dir>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_asymmetry_yaml import HADRONIC
from total_asymmetry import load_asymmetry_yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NU_METHOD_GAIN = 1.18
NBINS = 20
SHORT = {'pred_phiCP': 'MAP', 'pred_phiCP_circmean': 'circ. mean'}


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def circ_std(x):
    R = np.clip(np.hypot(np.mean(np.cos(x)), np.mean(np.sin(x))), 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(R))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', nargs=2, action='append', metavar=('LABEL', 'PARQUET'), required=True)
    ap.add_argument('--ref', default=None, help='label of the reference model for the differences (default: first)')
    ap.add_argument('--columns', default='pred_phiCP,pred_phiCP_circmean')
    ap.add_argument('--err_column', default='pred_phiCP_err')
    ap.add_argument('--cut', nargs=3, action='append', default=[], metavar=('LABEL', 'COLUMN', 'PASS_FRACTION'))
    ap.add_argument('--n_boot', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lep', default=os.path.join(HERE, 'asymmetry_lep_Polaris.yaml'))
    ap.add_argument('--ref_had', default=os.path.join(HERE, 'asymmetry_had_Polaris.yaml'))
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cols = args.columns.split(',')
    labels = [l for l, _ in args.model]
    ref = args.ref or labels[0]

    # ---- inputs: the event-level columns must agree across models (same events, same order)
    common = ['reco_taup_DM', 'reco_taun_DM', 'tauspinner_wt_alpha0', 'tauspinner_wt_alpha90', 'true_phiCP']
    data, base = {}, None
    for lab, path in args.model:
        df = pd.read_parquet(path, columns=common + cols + [args.err_column])
        if base is None:
            base = df[common].copy()
        else:
            assert len(df) == len(base), f'{lab}: {len(df)} rows vs {len(base)}'
            for c in common:
                assert np.array_equal(df[c].to_numpy(), base[c].to_numpy(), equal_nan=True), f'{lab}: column {c} differs -- not the same events'
        data[lab] = {c: df[c].to_numpy(float) for c in cols + [args.err_column]}
        del df
    n_ev = len(base)
    dm_p, dm_n = base['reco_taup_DM'].to_numpy(), base['reco_taun_DM'].to_numpy()
    w_e, w_o = base['tauspinner_wt_alpha0'].to_numpy(float), base['tauspinner_wt_alpha90'].to_numpy(float)
    true_phi = base['true_phiCP'].to_numpy(float)
    chan = np.full(n_ev, -1)
    for i, (p, n) in enumerate(HADRONIC):
        chan[(dm_p == p) & (dm_n == n)] = i
    had = chan >= 0
    n_ch = len(HADRONIC)
    print(f'>> {n_ev} events, {had.sum()} in the {n_ch} hadronic channels; models: {", ".join(labels)}')

    lep = load_asymmetry_yaml(args.lep)
    N_lep = sum(v['N'] for v in lep.values()); S2_lep = sum(v['N'] * v['asymmetry'] ** 2 for v in lep.values())
    ref_had = load_asymmetry_yaml(args.ref_had)
    S_nu = np.sqrt((sum(v['N'] * v['asymmetry'] ** 2 for v in ref_had.values()) + S2_lep)
                   / (sum(v['N'] for v in ref_had.values()) + N_lep))
    S_old = S_nu / NU_METHOD_GAIN

    # ---- variants: (kind, model) -> event indices, bin keys
    variants = {}
    for lab in labels:
        for c in cols:
            variants[(SHORT.get(c, c), lab)] = (c, 1.0)
    for lab, c, f in args.cut:
        variants[(f'{SHORT.get(c, c)}, cut', lab)] = (c, float(f))
    prepared = {}
    for (kind, lab), (c, f) in variants.items():
        phi = data[lab][c]; err = data[lab][args.err_column]
        ok = had & np.isfinite(phi)
        sel = ok.copy()
        X = np.inf
        if f < 1:
            e = err[ok]; X = float(np.quantile(e[np.isfinite(e)], f))
            sel &= np.isfinite(err) & (err <= X)
        b = np.clip((phi[sel] / (2 * np.pi) * NBINS).astype(int), 0, NBINS - 1)
        prepared[(kind, lab)] = dict(idx_ok=np.flatnonzero(ok), idx=np.flatnonzero(sel),
                                     key=chan[sel] * NBINS + b, ch=chan[sel], ch_ok=chan[ok], X=X, f=f)

    def sensitivity(v, pw):
        """combined had+lep S for variant v with per-event bootstrap weights pw (None = nominal)"""
        idx, key = v['idx'], v['key']
        we, wo = (w_e[idx], w_o[idx]) if pw is None else (w_e[idx] * pw[idx], w_o[idx] * pw[idx])
        E = np.bincount(key, we, minlength=n_ch * NBINS).reshape(n_ch, NBINS)
        O = np.bincount(key, wo, minlength=n_ch * NBINS).reshape(n_ch, NBINS)
        with np.errstate(invalid='ignore', divide='ignore'):
            a = np.sqrt(np.nansum((O / O.sum(1, keepdims=True) - E / E.sum(1, keepdims=True)) ** 2, axis=1))
        n_pass = np.bincount(v['ch'], None if pw is None else pw[idx], minlength=n_ch)
        n_all = np.bincount(v['ch_ok'], None if pw is None else pw[v['idx_ok']], minlength=n_ch)
        return np.sqrt((np.sum(n_pass * a ** 2) + S2_lep) / (n_all.sum() + N_lep))

    keys = list(prepared)
    nominal = {k: sensitivity(prepared[k], None) for k in keys}
    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.n_boot) for k in keys}
    for r in range(args.n_boot):
        pw = rng.poisson(1.0, n_ev).astype(float)
        for k in keys:
            boot[k][r] = sensitivity(prepared[k], pw)
        if (r + 1) % 25 == 0:
            print(f'   bootstrap {r + 1}/{args.n_boot}', flush=True)

    # ---- table: gain vs old method, and the paired difference to the reference model
    rows = []
    for kind, lab in keys:
        g = 100 * (boot[(kind, lab)] / S_old - 1)
        row = {'estimator': kind, 'model': lab, 'pass frac': prepared[(kind, lab)]['f'],
               'vs old [%]': 100 * (nominal[(kind, lab)] / S_old - 1), 'boot. std [pts]': g.std(ddof=1)}
        if (kind, ref) in boot and lab != ref:
            d = g - 100 * (boot[(kind, ref)] / S_old - 1)
            row[f'minus {ref} [pts]'] = 100 * (nominal[(kind, lab)] - nominal[(kind, ref)]) / S_old
            row['diff std [pts]'] = d.std(ddof=1)
            row['P(diff>0)'] = (d > 0).mean()
        rows.append(row)
    tab = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(f'\n=== combined had+lep sensitivity, {args.n_boot} paired Poisson-bootstrap replicas '
          f'(gains in % vs old method; differences in percentage points vs "{ref}") ===')
    print(tab.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    tab.to_csv(os.path.join(args.outdir, 'bootstrap_totals.csv'), index=False)

    # all pairwise differences within each estimator
    pair_rows = []
    for kind in dict.fromkeys(k for k, _ in keys):
        labs = [l for k, l in keys if k == kind]
        for i, a in enumerate(labs):
            for b in labs[i + 1:]:
                d = 100 * (boot[(kind, a)] - boot[(kind, b)]) / S_old
                pair_rows.append({'estimator': kind, 'A': a, 'B': b,
                                  'A - B [pts]': 100 * (nominal[(kind, a)] - nominal[(kind, b)]) / S_old,
                                  'std [pts]': d.std(ddof=1), 'P(A>B)': (d > 0).mean()})
    ptab = pd.DataFrame(pair_rows)
    print('\n=== all pairwise differences ===')
    print(ptab.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    ptab.to_csv(os.path.join(args.outdir, 'bootstrap_pairwise.csv'), index=False)

    # ---- (2) calibration of the per-event uncertainty
    crow, curves = [], {}
    q = np.linspace(0, 1, 21)
    for lab in labels:
        err = data[lab][args.err_column]
        for c in cols:
            phi = data[lab][c]
            m = had & np.isfinite(phi) & np.isfinite(err) & np.isfinite(true_phi)
            r = wrap(phi[m] - true_phi[m]); s = err[m]
            edges = np.quantile(s, q); bi = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, len(q) - 2)
            pred = np.array([s[bi == i].mean() for i in range(len(q) - 1)])
            obs = np.array([circ_std(r[bi == i]) for i in range(len(q) - 1)])
            curves[(lab, c)] = (pred, obs)
            crow.append({'model': lab, 'estimator': SHORT.get(c, c), 'median unc. [rad]': np.median(s),
                         'circ. std of residual [rad]': circ_std(r), '|res| < 1 unc.': np.mean(np.abs(r) < s),
                         '|res| < 2 unc.': np.mean(np.abs(r) < 2 * s),
                         'median obs/pred (reliability)': np.median(obs / pred),
                         'Spearman(unc, |res|)': pd.Series(s).rank().corr(pd.Series(np.abs(r)).rank())})
    ctab = pd.DataFrame(crow)
    print('\n=== calibration of the per-event phiCP uncertainty against the true phiCP '
          '(Gaussian reference: 68.3% / 95.4%) ===')
    print(ctab.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    ctab.to_csv(os.path.join(args.outdir, 'calibration.csv'), index=False)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(cols), figsize=(5.6 * len(cols), 5), squeeze=False)
    for ax, c in zip(axs[0], cols):
        for lab in labels:
            pred, obs = curves[(lab, c)]
            ax.plot(pred, obs, 'o-', ms=3, label=lab)
        lim = max(max(v[0].max(), v[1].max()) for (l, cc), v in curves.items() if cc == c) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', lw=1, label='perfect calibration')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f'predicted uncertainty ({args.err_column}) [rad], mean in bin')
        ax.set_ylabel(f'circular std of ({SHORT.get(c, c)} - true phiCP) [rad]')
        ax.set_title(f'{SHORT.get(c, c)}: 20 quantile bins of the uncertainty', fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, 'reliability.pdf')); plt.close(fig)
    print(f'\n>> wrote {args.outdir}/bootstrap_totals.csv, bootstrap_pairwise.csv, calibration.csv, reliability.pdf')


if __name__ == '__main__':
    main()
