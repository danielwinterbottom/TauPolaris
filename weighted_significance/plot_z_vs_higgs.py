"""Z->tautau background vs H->tautau signal for one flow model (hadronic channels): the per-event phiCP
uncertainty, and the phiCP distribution (circular mean of the flow samples, and the truth).

The Z sample is also shown reweighted to the Higgs visible-mass distribution (computed from the reco visible
decay products, which the model does not modify), so the part of the Z/H difference that is just mass can be
told apart from the rest.

    python weighted_significance/plot_z_vs_higgs.py --sig <signal results parquet> --bkg <Z results parquet> \\
        --sig_test <signal test parquet> --bkg_test <Z test parquet> --out <prefix>
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'     # categorical slots 1-3 (fixed order)
INK, MUTED, GRID = '#212121', '#6b7280', '#e5e7eb'


def visible_mass(path):
    cols = [f'reco_{t}_{o}_{c}' for t in ('taup', 'taun') for o in ('charged', 'pizero1') for c in ('e', 'px', 'py', 'pz')]
    d = pq.read_table(path, columns=cols).to_pandas()
    E = sum(d[f'reco_{t}_{o}_e'] for t in ('taup', 'taun') for o in ('charged', 'pizero1'))
    p = [sum(d[f'reco_{t}_{o}_p{c}'] for t in ('taup', 'taun') for o in ('charged', 'pizero1')) for c in 'xyz']
    return np.sqrt(np.clip(E ** 2 - p[0] ** 2 - p[1] ** 2 - p[2] ** 2, 0, None)).to_numpy()


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK); ax.set_ylabel(ylabel, color=INK)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def hist(ax, x, w, edges, color, label, ls='-'):
    h, _ = np.histogram(x, edges, weights=w)
    h = h / (h.sum() * np.diff(edges))
    ax.stairs(h, edges, color=color, lw=2, ls=ls, label=label, baseline=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sig', required=True); ap.add_argument('--bkg', required=True)
    ap.add_argument('--sig_test', required=True); ap.add_argument('--bkg_test', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--label', default='')
    a = ap.parse_args()
    cols = ['pred_phiCP_circmean', 'pred_phiCP_err', 'true_phiCP', 'reco_taup_DM', 'reco_taun_DM',
            'tauspinner_wt_alpha0', 'tauspinner_wt_alpha90']
    S, B = pd.read_parquet(a.sig, columns=cols), pd.read_parquet(a.bkg, columns=cols)
    ms, mb = visible_mass(a.sig_test), visible_mass(a.bkg_test)
    assert len(ms) == len(S) and len(mb) == len(B), 'results and test files are not row-aligned'
    had = lambda d: (d.reco_taup_DM.between(0, 11) & d.reco_taun_DM.between(0, 11)).to_numpy()
    hs_, hb_ = had(S), had(B)
    edges_m = np.arange(0, 302, 2.0)
    ps, _ = np.histogram(ms[hs_], edges_m, density=True); pb, _ = np.histogram(mb[hb_], edges_m, density=True)
    ratio = np.where(pb > 0, ps / np.where(pb > 0, pb, 1), 0)
    wm = ratio[np.clip(np.digitize(mb, edges_m) - 1, 0, len(ratio) - 1)]       # Z weight to the Higgs visible mass

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
    e = np.linspace(0, 2.6, 53)
    ax = axs[0]
    hist(ax, S.pred_phiCP_err[hs_], None, e, BLUE, f'H→ττ (median {np.nanmedian(S.pred_phiCP_err[hs_]):.2f} rad)')
    hist(ax, B.pred_phiCP_err[hb_], None, e, ORANGE, f'Z→ττ (median {np.nanmedian(B.pred_phiCP_err[hb_]):.2f} rad)')
    ok = hb_ & np.isfinite(B.pred_phiCP_err.to_numpy())
    zw = B.pred_phiCP_err.to_numpy()[ok]; ww = wm[ok]; o = np.argsort(zw); cw = np.cumsum(ww[o])
    hist(ax, zw, ww, e, ORANGE, f'Z→ττ, reweighted to the H visible mass (median {zw[o][np.searchsorted(cw, cw[-1] / 2)]:.2f} rad)', ls='--')
    style(ax, 'per-event φCP uncertainty (circular std of 100 flow samples) [rad]', 'normalised to unit area')
    ax.set_title('φCP uncertainty', loc='left', color=INK, fontsize=11)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', labelcolor=INK)

    ax = axs[1]
    e2 = np.linspace(0, 2 * np.pi, 41)
    hist(ax, S.pred_phiCP_circmean[hs_], S.tauspinner_wt_alpha0[hs_], e2, BLUE, 'H→ττ, CP-even (α = 0)')
    hist(ax, S.pred_phiCP_circmean[hs_], S.tauspinner_wt_alpha90[hs_], e2, BLUE, 'H→ττ, CP-odd (α = 90°)', ls='--')
    hist(ax, B.pred_phiCP_circmean[hb_], None, e2, ORANGE, 'Z→ττ')
    hist(ax, B.true_phiCP[hb_], None, e2, MUTED, 'Z→ττ, true φCP', ls=':')
    style(ax, 'φCP (circular mean of the flow samples) [rad]', 'normalised to unit area')
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]); ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.set_title('φCP', loc='left', color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc='lower center', labelcolor=INK, ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    fig.suptitle(f'Z→ττ vs H→ττ, hadronic channels{(" — " + a.label) if a.label else ""}', x=0.01, ha='left', color=INK, fontsize=12)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', dpi=130)
    print(f'>> wrote {a.out}.pdf/.png')


if __name__ == '__main__':
    main()
