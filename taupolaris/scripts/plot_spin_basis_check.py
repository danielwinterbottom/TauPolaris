"""How much does the reconstructed (n, r, k) basis matter? Spin-correlation observables with the polarimetric vectors
and the basis axes each taken from the reconstruction or from the generator level (referee: rotation of the k axis
leaks C_kk into C_rk at first order and dilutes the diagonal at second order).

Variants (pol-vecs / basis):  A reco/reco (nominal)   B reco/gen   C gen/gen (ideal)   D gen/reco (basis effect alone)

Events are weighted to the Standard Model spin correlations exactly as nll_spincorr_fits.py does (w = 1 + sum C_ij
wt_hp_i_hm_j with C = diag(1, 1, -1)). Plots: 1D distributions of cos_r(+)cos_k(-) and cos_k(+)cos_r(-) for the four
variants (legend: raw -9 <...>, before any unfolding), 2D event-by-event comparisons reco basis vs gen basis, and a
table of the raw -9 <cos_i(+) cos_j(-)> for all nine elements.

    python taupolaris/scripts/plot_spin_basis_check.py [--results <nu-method results parquet>] [--test_file <row-aligned
        prepared test parquet>] [--n_events N] --outdir <dir>
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from taupolaris.python.spin_basis import spin_cosines, taus_from, polvecs_from, COS_KEYS

AX = 'nrk'
VARIANTS = {'A': ('reco', 'reco'), 'B': ('reco', 'gen'), 'C': ('gen', 'gen'), 'D': ('gen', 'reco')}
LABEL = {'A': 'A: reco pol-vecs, reco basis', 'B': 'B: reco pol-vecs, gen basis',
         'C': 'C: gen pol-vecs, gen basis', 'D': 'D: gen pol-vecs, reco basis'}
COL = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a', 'D': '#e87ba4'}         # categorical slots 1-3, 5
INK, MUTED, GRID = '#212121', '#6b7280', '#e5e7eb'


def load(results, test_file, n_events, prefix):
    from taupolaris.python.Evaluation_Tools import compute_spin_vars
    raw = ([f'{prefix}_tau_{s}_{c}' for s in ('plus', 'minus') for c in ('E', 'px', 'py', 'pz')] +
           [f'true_tau_{s}_{c}' for s in ('plus', 'minus') for c in ('E', 'px', 'py', 'pz')] +
           [f'reco_{t}_{p}_{c}' for t in ('taup', 'taun') for p in ('pi1', 'pi2', 'pi3', 'pizero1', 'charged') for c in ('E', 'px', 'py', 'pz')] +
           [f'reco_{t}_{f}' for t in ('taup', 'taun') for f in ('haspizero', 'ishadronic', 'is3prong')] +
           [f'wt_hp_{a}_hm_{b}' for a in AX for b in AX])
    tcols = [f'ts_hh_{t}_{c}' for t in ('taup', 'taun') for c in 'xyz'] + ['undecayed_taup_px']
    out, n = [], 0
    for rb, tb in zip(pq.ParquetFile(results).iter_batches(batch_size=500_000, columns=raw),
                      pq.ParquetFile(test_file).iter_batches(batch_size=500_000, columns=tcols)):
        r, t = rb.to_pandas(), tb.to_pandas()
        if np.abs(r['true_tau_plus_px'].to_numpy() - t['undecayed_taup_px'].to_numpy()).max() > 1e-2:
            raise SystemExit(f'{test_file} is not row-aligned with {results}')
        r = compute_spin_vars(r, tau_pred_prefix=f'{prefix}_', tau_vis_prefix='reco_', store_polvecs=True)
        h = {'reco': polvecs_from(r, f'{prefix}_'),
             'gen': (t[tcols[0:3]].to_numpy(float), t[tcols[3:6]].to_numpy(float))}
        taus = {'reco': taus_from(r, f'{prefix}_'), 'gen': taus_from(r, 'true_')}
        blk = {'w': 1 + sum(c * r[f'wt_hp_{a}_hm_{a}'].to_numpy() for a, c in zip(AX, (1, 1, -1)))}
        for v, (hs, bs) in VARIANTS.items():
            cs = spin_cosines(*h[hs], *taus[bs])
            for k in COS_KEYS:
                blk[f'{v}_{k}'] = cs[k]
        out.append(pd.DataFrame(blk)); n += len(r)
        print(f'  {n} events', flush=True)
        if n_events and n >= n_events:
            break
    d = pd.concat(out, ignore_index=True)
    d['w'] = d['w'].clip(lower=0)
    return d.iloc[:n_events] if n_events else d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/output_results.parquet')
    ap.add_argument('--test_file', default='prepared_LHC_data_sep01/ppToHToTauTau_AllDM_UnCorr_Jul20/test_dataframe__leptonic_mode_0_transformer.parquet')
    ap.add_argument('--var_prefix', default='map_pred')
    ap.add_argument('--n_events', type=int, default=0)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    d = load(a.results, a.test_file, a.n_events, a.var_prefix)
    w = d['w'].to_numpy()
    prod = lambda v, i, j: d[f'{v}_cos{i}_plus'].to_numpy() * d[f'{v}_cos{j}_minus'].to_numpy()
    rawC = lambda v, i, j: -9 * np.average(prod(v, i, j), weights=w)

    # table of the raw (reconstruction-level, not unfolded) -9 <cos_i(+) cos_j(-)>
    tab = pd.DataFrame({f'C_{i}{j}': {LABEL[v]: rawC(v, i, j) for v in VARIANTS} for i in AX for j in AX})
    pd.set_option('display.width', 250)
    print(f'\n{len(d)} events, SM weights; raw -9<cos_i(+)cos_j(-)> (true SM: C_nn = C_rr = 1, C_kk = -1, off-diagonal 0):')
    print(tab.to_string(float_format=lambda x: f'{x:+.3f}'))
    tab.to_csv(os.path.join(a.outdir, 'raw_C_by_variant.csv'))

    # 1D
    edges = np.linspace(-1, 1, 41)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (i, j) in zip(axs, (('r', 'k'), ('k', 'r'))):
        for v in VARIANTS:
            hh, _ = np.histogram(prod(v, i, j), edges, weights=w)
            ax.stairs(hh / (hh.sum() * np.diff(edges)), edges, color=COL[v], lw=1.8, baseline=None,
                      label=f'{LABEL[v]}   (raw C$_{{{i}{j}}}$ = {rawC(v, i, j):+.3f})')
        ax.set_yscale('log'); ax.set_xlabel(f'cos θ$_{i}^+$ · cos θ$_{j}^-$', color=INK); ax.set_ylabel('normalised to unit area', color=INK)
        ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc='lower center', labelcolor=INK)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo / 20, hi * 1.5)
    fig.suptitle('SM-weighted H→ττ (hadronic); true C$_{rk}$ = C$_{kr}$ = 0', x=0.01, ha='left', color=INK, fontsize=11)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.outdir, f'cosr_cosk_1d.{ext}'), dpi=120)
    plt.close(fig)

    # 2D: reco basis vs gen basis, event by event
    pairs = [('D', 'C', 'gen pol-vecs'), ('A', 'B', 'reco pol-vecs')]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.9))
    e2 = np.linspace(-1, 1, 81)
    for ax, (vr, vg, hl) in zip(axs[:2], pairs):
        x, y = prod(vg, 'r', 'k'), prod(vr, 'r', 'k')
        h2 = ax.hist2d(x, y, bins=[e2, e2], weights=w, norm=LogNorm(), cmap='Blues')
        ax.plot([-1, 1], [-1, 1], color=MUTED, lw=0.8, ls='--')
        ax.set_xlabel(f'cos θ$_r^+$·cos θ$_k^-$, gen basis ({hl})', color=INK); ax.set_ylabel('same, reco basis', color=INK)
        ax.set_title(f'{hl}: corr. {np.corrcoef(x, y)[0, 1]:.3f}', loc='left', fontsize=10, color=INK)
        fig.colorbar(h2[3], ax=ax, fraction=0.046)
    x, y = d['C_cosk_plus'].to_numpy(), d['D_cosk_plus'].to_numpy()
    h2 = axs[2].hist2d(x, y, bins=[e2, e2], weights=w, norm=LogNorm(), cmap='Blues')
    axs[2].plot([-1, 1], [-1, 1], color=MUTED, lw=0.8, ls='--')
    axs[2].set_xlabel('cos θ$_k^+$ (gen pol-vec), gen basis', color=INK); axs[2].set_ylabel('same, reco basis', color=INK)
    axs[2].set_title(f'k axis alone: corr. {np.corrcoef(x, y)[0, 1]:.3f}', loc='left', fontsize=10, color=INK)
    fig.colorbar(h2[3], ax=axs[2], fraction=0.046)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.outdir, f'basis_reco_vs_gen_2d.{ext}'), dpi=120)
    plt.close(fig)
    print(f'>> wrote {a.outdir}/raw_C_by_variant.csv, cosr_cosk_1d.pdf/png, basis_reco_vs_gen_2d.pdf/png')


if __name__ == '__main__':
    main()
