"""CP sign-symmetry check of a phiCP estimate: does the method create a fake CP-odd asymmetry,
and how much of the real CP-odd (sign of the mixing angle) information does it keep?

TauSpinner weights are quadratic in the mixing angle alpha:
    w(alpha) = cos^2(alpha) w0 + sin^2(alpha) w90 + 2 sin(alpha) cos(alpha) w_int,
    w45 = (w0 + w90)/2 + w_int   =>   w(-45) = w0 + w90 - w45.
Under CP, alpha -> -alpha and phiCP -> 2pi - phiCP. So for an unbiased method:
  * alpha = 0 and alpha = 90 (CP-conserving): <sin phiCP> = 0,
  * alpha = +45 vs -45: <sin phiCP> flips sign, <cos phiCP> is the same, i.e. the mirror
    sum <sin>_{+45} + <sin>_{-45} = 0 (it equals 2 <sin> under the w0+w90 mixture).
Each <.> is a weighted mean with its standard error sqrt(sum w^2 (x - mean)^2) / sum w; "z" is the
value over its error. Also reported: the usual 20-bin CP-even vs CP-odd quadrature asymmetry
(alpha 0 vs 90) and the same between alpha +45 and -45 -- the CP-odd separation, which needs the sign.
The truth column (true_phiCP) shows what a perfect estimate would give on the same events.

    python weighted_significance/sign_symmetry_check.py --results <evaluate_polvec parquet> \\
        [--columns true_phiCP,pred_phiCP,pred_phiCP_circmean] [--label NAME]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_asymmetry_yaml import HADRONIC, asymmetry


def wmean(x, w):
    s = w.sum(); m = (w * x).sum() / s
    return m, np.sqrt((w ** 2 * (x - m) ** 2).sum()) / s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--columns', default='true_phiCP,pred_phiCP,pred_phiCP_circmean')
    ap.add_argument('--label', default=None)
    args = ap.parse_args()
    cols = args.columns.split(',')
    import pyarrow.parquet as pq
    have = pq.read_schema(args.results).names
    cols = [c for c in cols if c in have]
    df = pd.read_parquet(args.results, columns=cols + ['reco_taup_DM', 'reco_taun_DM', 'tauspinner_wt_alpha0',
                                                       'tauspinner_wt_alpha45', 'tauspinner_wt_alpha90'])
    w0, w45, w90 = (df[f'tauspinner_wt_alpha{a}'].to_numpy(float) for a in (0, 45, 90))
    W = {'0': w0, '90': w90, '+45': w45, '-45': w0 + w90 - w45}
    print(f">> {args.label or args.results}: {len(df)} events; min w(-45) = {W['-45'].min():.3g}")
    dp, dn = df['reco_taup_DM'].to_numpy(), df['reco_taun_DM'].to_numpy()
    had = np.zeros(len(df), bool)
    for p, n in HADRONIC:
        had |= (dp == p) & (dn == n)

    rows = []
    for c in cols:
        phi = df[c].to_numpy(float); m = had & np.isfinite(phi)
        s, co = np.sin(phi[m]), np.cos(phi[m])
        r = {'estimate': c}
        for a in ('0', '90', '+45', '-45'):
            ms, es = wmean(s, W[a][m]); mc, ec = wmean(co, W[a][m])
            r[f'<sin> a={a}'] = f'{ms:+.5f} ({ms / es:+.1f} sd)' if a in ('0', '90') else f'{ms:+.5f}'
            r[f'<cos> a={a}'] = f'{mc:+.5f}'
        mix = W['+45'][m] + W['-45'][m]                     # = w0 + w90: CP-symmetric mixture
        ms, es = wmean(s, mix)
        r['mirror: <sin>(+45) + <sin>(-45)'] = f'{2 * ms:+.5f} ({ms / es:+.1f} sd)'
        r['asym 0 vs 90'] = f"{asymmetry(phi[m], W['0'][m], W['90'][m]):.4f}"
        r['asym +45 vs -45'] = f"{asymmetry(phi[m], W['+45'][m], W['-45'][m]):.4f}"
        rows.append(r)
    t = pd.DataFrame(rows).set_index('estimate').T
    pd.set_option('display.width', 200)
    print(f'\n=== hadronic channels ({had.sum()} events); "sd" = value / standard error, |sd| <~ 2 is consistent with zero ===')
    print(t.to_string())

    # per channel: the largest fake <sin> at alpha 0 / 90 and in the mirror test, in standard errors
    print('\n=== per channel, largest |<sin>| / error over the 15 hadronic channels (alpha 0, alpha 90, mirror) ===')
    for c in cols:
        phi = df[c].to_numpy(float); worst = []
        for p, n in HADRONIC:
            m = (dp == p) & (dn == n) & np.isfinite(phi)
            if m.sum() < 100:
                continue
            s = np.sin(phi[m])
            for lab, w in (('a=0', W['0'][m]), ('a=90', W['90'][m]), ('mirror', W['+45'][m] + W['-45'][m])):
                ms, es = wmean(s, w); worst.append((abs(ms / es), f'DM{p}DM{n} {lab}'))
        worst.sort(reverse=True)
        n_tests = len(worst)
        print(f'  {c:22s} max {worst[0][0]:.1f} sd ({worst[0][1]}); {sum(z > 2 for z, _ in worst)} of {n_tests} tests above 2 sd '
              f'(~{0.046 * n_tests:.1f} expected by chance)')


if __name__ == '__main__':
    main()
