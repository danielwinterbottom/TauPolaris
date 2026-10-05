"""Scan a cut on the per-event phiCP uncertainty and estimate the resulting sensitivity.

Events with uncertainty > X are dropped; the same X is used in every channel. Per channel
the CP-even/odd asymmetry a_i(X) is recomputed on the surviving events (same metric as
plot_phiCP.py / make_asymmetry_yaml.py). The statistical power of a channel scales like
a * sqrt(N), so the combined sensitivity keeps the full-sample normalisation:

    S(X) = sqrt( sum_i N_i,pass * a_i(X)^2  /  sum_i N_i,all )

i.e. a channel keeping a fraction f_i of its events contributes f_i * a_i^2 -- the sqrt(f)
loss is built in. With no cut this reduces exactly to total_asymmetry.py's number.

Leptonic channels (not in the results file) are added uncut from a yaml, and the result is
compared with the uncut model, with the run-3 method (per channel, from the stored recoRun3
histograms) and with the old method via the published nu-method +18%.

    python weighted_significance/uncertainty_cut_scan.py \\
        --results <evaluate_polvec results parquet> [--label Sep01] \\
        [--phi_column pred_phiCP --err_column pred_phiCP_err] [--lep asymmetry_lep_Polaris.yaml]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_asymmetry_yaml import HADRONIC, asymmetry
from total_asymmetry import load_asymmetry_yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RUN3_LOGS = os.path.join(HERE, '..', 'outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/phiCP/logs')
NU_METHOD_GAIN = 1.18      # published: nu-method total = 1.18 x old-method total


def run3_asymmetries():
    out = {}
    for f in glob.glob(os.path.join(RUN3_LOGS, '*_RecoRun3.npz')):
        z = np.load(f)
        e, o = z['even_counts'], z['odd_counts']          # density-normalised; divide by the sum
        out[os.path.basename(f).split('_')[0]] = float(np.sqrt(np.sum((e - o) ** 2)) / e.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--label', default=None)
    ap.add_argument('--phi_column', default='pred_phiCP')
    ap.add_argument('--err_column', default='pred_phiCP_err')
    ap.add_argument('--lep', default=os.path.join(HERE, 'asymmetry_lep_Polaris.yaml'))
    ap.add_argument('--ref_had', default=os.path.join(HERE, 'asymmetry_had_Polaris.yaml'),
                    help='nu-method hadronic yaml; with --lep it defines the +18%% reference')
    ap.add_argument('--fractions', default='1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    label = args.label or os.path.basename(os.path.dirname(os.path.abspath(args.results)))
    outdir = args.outdir or os.path.join(HERE, f'uncertainty_cut_{label}')
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_parquet(args.results, columns=[args.phi_column, args.err_column, 'reco_taup_DM', 'reco_taun_DM',
                                                'tauspinner_wt_alpha0', 'tauspinner_wt_alpha90'])
    phi, err = df[args.phi_column].to_numpy(float), df[args.err_column].to_numpy(float)
    dm_p, dm_n = df['reco_taup_DM'].to_numpy(), df['reco_taun_DM'].to_numpy()
    w_e, w_o = df['tauspinner_wt_alpha0'].to_numpy(float), df['tauspinner_wt_alpha90'].to_numpy(float)
    ok = np.isfinite(phi)
    print(f'>> {label}: {len(df)} events, uncertainty available for {np.isfinite(err).mean():.4f}')
    err = np.where(np.isfinite(err), err, np.inf)       # no error estimate -> treated as failing any finite cut

    chans = {f'DM{p}DM{n}': ok & (dm_p == p) & (dm_n == n) for p, n in HADRONIC}
    chans = {k: m for k, m in chans.items() if m.sum() >= 20}
    had_mask = np.zeros(len(df), bool)
    for m in chans.values():
        had_mask |= m
    lep = load_asymmetry_yaml(args.lep)
    N_lep = sum(v['N'] for v in lep.values()); S2_lep = sum(v['N'] * v['asymmetry'] ** 2 for v in lep.values())
    N_had = sum(int(m.sum()) for m in chans.values())

    # thresholds from the requested global pass fractions of the hadronic events
    fracs = [float(f) for f in args.fractions.split(',')]
    errs_h = err[had_mask]
    cuts = [np.inf if f >= 1 else float(np.quantile(errs_h[np.isfinite(errs_h)], f)) for f in fracs]

    rows, per_chan = [], {}
    for f, X in zip(fracs, cuts):
        S2 = 0.0
        for k, m in chans.items():
            sel = m & (err <= X)
            n = int(sel.sum())
            a = asymmetry(phi[sel], w_e[sel], w_o[sel]) if n >= 20 else 0.0
            S2 += n * a ** 2
            per_chan.setdefault(k, {})[f] = (n, a)
        rows.append({'pass fraction': f, 'cut X [rad]': X, 'S had': np.sqrt(S2 / N_had),
                     'S had+lep': np.sqrt((S2 + S2_lep) / (N_had + N_lep))})
    scan = pd.DataFrame(rows)

    ref_had = load_asymmetry_yaml(args.ref_had)
    S_ref = np.sqrt((sum(v['N'] * v['asymmetry'] ** 2 for v in ref_had.values()) + S2_lep)
                    / (sum(v['N'] for v in ref_had.values()) + N_lep))
    S_old = S_ref / NU_METHOD_GAIN
    S_nocut = scan.loc[scan['pass fraction'] == 1.0, 'S had+lep'].iloc[0]
    scan['vs no cut'] = scan['S had+lep'] / S_nocut - 1
    scan['vs old method'] = scan['S had+lep'] / S_old - 1
    best = scan.loc[scan['S had+lep'].idxmax()]
    pd.set_option('display.width', 200)
    print('\n=== scan (same cut in every hadronic channel; leptonic channels uncut, nu-method) ===')
    print(scan.to_string(index=False, formatters={'cut X [rad]': '{:.3f}'.format, 'S had': '{:.4f}'.format,
                                                  'S had+lep': '{:.4f}'.format, 'vs no cut': '{:+.1%}'.format,
                                                  'vs old method': '{:+.1%}'.format, 'pass fraction': '{:.2f}'.format}))
    print(f"\n>> optimum: pass fraction {best['pass fraction']:.2f} (cut X = {best['cut X [rad]']:.3f} rad): "
          f"S = {best['S had+lep']:.4f}, {best['vs no cut']:+.1%} vs no cut, {best['vs old method']:+.1%} vs old method "
          f"(nu-method reference {S_ref:.4f} = +18.0%)")
    scan.to_csv(os.path.join(outdir, 'scan.csv'), index=False)

    # per-channel view at the optimum vs run-3 and vs no cut
    fb = best['pass fraction']; run3 = run3_asymmetries()
    crow = []
    for k in chans:
        n0, a0 = per_chan[k][1.0]; nb, ab = per_chan[k][fb]
        eff0, effb = a0, ab * np.sqrt(nb / n0)          # per-channel sensitivity, normalised to the uncut N
        r = {'channel': k, 'N': n0, 'pass frac': nb / n0, 'asym no cut': a0, 'asym cut': ab,
             'sens. cut / no cut': effb / a0 - 1 if a0 > 0 else np.nan}
        if k in run3:
            r['no cut vs run-3'] = eff0 / run3[k] - 1
            r['cut vs run-3'] = effb / run3[k] - 1
        crow.append(r)
    ctab = pd.DataFrame(crow)
    print(f'\n=== per channel at pass fraction {fb:.2f}; "sens." = asym x sqrt(pass frac) ===')
    print(ctab.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    ctab.to_csv(os.path.join(outdir, f'per_channel_f{fb:.2f}.csv'), index=False)

    # ---- separate cut per channel: in each channel pick the within-channel pass fraction
    # maximising f * a(f)^2 (its contribution to S^2). Also a half-sample check, choosing the
    # cut on even-indexed events and measuring on odd-indexed ones, since 15 tuned cuts can
    # fit fluctuations in the small channels.
    fine = np.round(np.arange(1.0, 0.049, -0.025), 3)

    def best_cut(m_choose, m_eval):
        e = err[m_choose]; e = e[np.isfinite(e)]
        best_f, best_s2, best_X = 1.0, -1.0, np.inf
        for f in fine:
            X = np.inf if f >= 1 else float(np.quantile(e, f))
            sel = m_choose & (err <= X)
            if sel.sum() < 20:
                continue
            s2 = sel.sum() / m_choose.sum() * asymmetry(phi[sel], w_e[sel], w_o[sel]) ** 2
            if s2 > best_s2:
                best_f, best_s2, best_X = f, s2, X
        sel = m_eval & (err <= best_X)
        a = asymmetry(phi[sel], w_e[sel], w_o[sel]) if sel.sum() >= 20 else 0.0
        return best_f, best_X, sel.sum() / m_eval.sum(), a

    even_idx = (np.arange(len(df)) % 2) == 0
    prow = []
    S2_pc = S2_half = S2_half_ref = 0.0
    for k, m in chans.items():
        f, X, fpass, a = best_cut(m, m)
        _, _, fpass_h, a_h = best_cut(m & even_idx, m & ~even_idx)
        n0, a0 = per_chan[k][1.0]
        m_odd = m & ~even_idx
        a0_odd = asymmetry(phi[m_odd], w_e[m_odd], w_o[m_odd])
        S2_pc += n0 * fpass * a ** 2
        S2_half += n0 * fpass_h * a_h ** 2
        S2_half_ref += n0 * a0_odd ** 2
        r = {'channel': k, 'N': n0, 'cut X [rad]': X, 'pass frac': fpass, 'asym no cut': a0, 'asym cut': a,
             'gain from cut': a * np.sqrt(fpass) / a0 - 1,
             'gain, half-sample check': a_h * np.sqrt(fpass_h) / a0_odd - 1}
        if k in run3:
            r['cut vs run-3'] = a * np.sqrt(fpass) / run3[k] - 1
        prow.append(r)
    ptab = pd.DataFrame(prow)
    print('\n=== separate cut per channel (maximising f * a^2 in each channel) ===')
    print(ptab.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    S_pc = np.sqrt((S2_pc + S2_lep) / (N_had + N_lep))
    common = scan.loc[scan['S had+lep'].idxmax()]
    print(f"\n>> combined, had+lep:  no cut {S_nocut:.4f} ({S_nocut / S_old - 1:+.1%} vs old)   "
          f"common cut {common['S had+lep']:.4f} ({common['vs old method']:+.1%})   "
          f"per-channel cuts {S_pc:.4f} ({S_pc / S_old - 1:+.1%}, {S_pc / S_nocut - 1:+.1%} vs no cut)")
    print(f">> half-sample check of the per-channel cuts (hadronic only, gain over no cut on the same half): "
          f"{np.sqrt(S2_half / S2_half_ref) - 1:+.1%}  vs  {np.sqrt(S2_pc / sum(per_chan[k][1.0][0] * per_chan[k][1.0][1] ** 2 for k in chans)) - 1:+.1%} when tuned and measured on all events")
    r3k = [k for k in chans if k in run3]
    Nr = sum(per_chan[k][1.0][0] for k in r3k)
    t3 = np.sqrt(sum(per_chan[k][1.0][0] * run3[k] ** 2 for k in r3k) / Nr)
    tpc = np.sqrt(sum(float(ptab.loc[ptab.channel == k, 'pass frac'].iloc[0]) * per_chan[k][1.0][0]
                      * float(ptab.loc[ptab.channel == k, 'asym cut'].iloc[0]) ** 2 for k in r3k) / Nr)
    print(f'>> {len(r3k)} run-3 channels: per-channel cuts {tpc / t3 - 1:+.1%} vs run-3')
    ptab.to_csv(os.path.join(outdir, 'per_channel_optimised_cuts.csv'), index=False)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.plot(scan['pass fraction'], 100 * scan['vs old method'], 'o-', label=f'{label}, cut on {args.err_column}')
        ax.axhline(18.0, color='k', ls='--', lw=1, label='nu-method, no cut (+18%)')
        ax.set_xlabel('fraction of hadronic events passing the uncertainty cut'); ax.set_ylabel('sensitivity vs old method [%]')
        ax.invert_xaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=9); fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'scan.pdf')); plt.close(fig)
    except Exception as exc:
        print(f'(plot skipped: {exc})')
    print(f'>> outputs in {outdir}')


if __name__ == '__main__':
    main()
