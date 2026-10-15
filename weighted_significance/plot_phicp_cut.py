"""phiCP per decay-mode pair with and without a cut on the per-event phiCP uncertainty.

One common threshold over the hadronic channels, chosen to keep --keep of the events. Per channel: the CP-even
(TauSpinner alpha = 0) and CP-odd (alpha = 90) phiCP distributions without the cut and with it, each normalised to
unit area, and the same for the events FAILING the cut (anti-cut), annotated with the quadrature asymmetry
(make_asymmetry_yaml's metric) without the cut, passing and failing it, the
fraction kept in that channel and the sensitivity change a * sqrt(N) (the sqrt(f) assumption).

    python weighted_significance/plot_phicp_cut.py --results <parquet> [--phi_column pred_phiCP]
        [--err_column pred_phiCP_err] [--keep 0.5] --label "<model, uncertainty>" --out <prefix>
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_asymmetry_yaml import HADRONIC, asymmetry

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'   # categorical slots 1-3: no cut / passing the cut / failing it
INK, MUTED, GRID = '#212121', '#6b7280', '#e5e7eb'
LAB = {0: '1π', 1: '1π1π⁰', 2: '1π2π⁰', 10: '3π', 11: '3π1π⁰'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--phi_column', default='pred_phiCP'); ap.add_argument('--err_column', default='pred_phiCP_err')
    ap.add_argument('--keep', type=float, default=0.5); ap.add_argument('--nbins', type=int, default=20)
    ap.add_argument('--label', default=''); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = pd.read_parquet(a.results, columns=[a.phi_column, a.err_column, 'reco_taup_DM', 'reco_taun_DM',
                                            'tauspinner_wt_alpha0', 'tauspinner_wt_alpha90'])
    phi, err = d[a.phi_column].to_numpy(float), d[a.err_column].to_numpy(float)
    dp, dn = d.reco_taup_DM.to_numpy(), d.reco_taun_DM.to_numpy()
    we, wo = d.tauspinner_wt_alpha0.to_numpy(float), d.tauspinner_wt_alpha90.to_numpy(float)
    chans = [(p, n, (dp == p) & (dn == n) & np.isfinite(phi)) for p, n in HADRONIC]
    had = np.zeros(len(d), bool)
    for _, _, m in chans:
        had |= m
    e_ok = np.where(np.isfinite(err), err, np.inf)
    X = float(np.quantile(e_ok[had & np.isfinite(e_ok)], a.keep))
    passed = e_ok <= X
    edges = np.linspace(0, 2 * np.pi, a.nbins + 1)
    fig, axs = plt.subplots(4, 4, figsize=(17, 14.5))
    S2_0 = S2_1 = Ntot = 0.0
    for ax, (p, n, m) in zip(axs.flat, chans):
        mc = m & passed; mf = m & ~passed
        def h(sel, w):
            c, _ = np.histogram(phi[sel], edges, weights=w[sel]); return c / (c.sum() * np.diff(edges))
        for sel, col, tag in ((m, BLUE, 'no cut'), (mc, ORANGE, f'pass (best {a.keep:.0%})'), (mf, AQUA, 'fail (anti-cut)')):
            ax.stairs(h(sel, we), edges, color=col, lw=1.8, baseline=None, label=f'{tag}, CP-even')
            ax.stairs(h(sel, wo), edges, color=col, lw=1.8, ls='--', baseline=None, label=f'{tag}, CP-odd')
        a0, a1, a2 = asymmetry(phi[m], we[m], wo[m]), asymmetry(phi[mc], we[mc], wo[mc]), asymmetry(phi[mf], we[mf], wo[mf])
        f = mc.sum() / m.sum()
        S2_0 += m.sum() * a0 ** 2; S2_1 += mc.sum() * a1 ** 2; Ntot += m.sum()
        ax.set_title(f'{LAB[p]} – {LAB[n]}', loc='left', color=INK, fontsize=11)
        ax.text(0.98, 0.04, f'asym: all {a0:.3f}, pass {a1:.3f}, fail {a2:.3f}\nkept {f:.0%}, a√N {100 * (a1 * np.sqrt(f) / a0 - 1):+.0f}%',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color=INK)
        ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(['0', 'π', '2π'])
        ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        lo, hi = ax.get_ylim(); ax.set_ylim(0, hi * 1.08)
    leg = axs.flat[-1]; leg.axis('off')
    hnd, lbl = axs.flat[0].get_legend_handles_labels()
    leg.legend(hnd, lbl, loc='upper left', frameon=False, fontsize=11, labelcolor=INK)
    tot = 100 * (np.sqrt(S2_1 / Ntot) / np.sqrt(S2_0 / Ntot) - 1)
    leg.text(0.0, 0.32, f'{a.label}\ncut: {a.err_column} ≤ {X:.3f} rad\n(keeps {a.keep:.0%} of hadronic events)\n'
             f'hadronic total a√N: {tot:+.1f}%\n(√f assumption)', fontsize=11, color=INK, va='top')
    fig.supxlabel(f'φCP ({a.phi_column}) [rad]', color=INK); fig.supylabel('normalised to unit area', color=INK)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', dpi=110)
    print(f'>> {a.out}.pdf/.png: threshold {X:.3f} rad, hadronic total sensitivity {tot:+.1f}% (sqrt(f))')


if __name__ == '__main__':
    main()
