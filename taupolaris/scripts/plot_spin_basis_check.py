"""cos_r(+) cos_k(-) with the helicity basis built from the reconstructed taus vs from the generator taus (referee:
resolution of the k axis). The polarimetric vectors are the same in both (reconstructed by default, --polvecs gen for
the generator ones), so the difference is the basis alone.

Events are weighted to the Standard Model spin correlations as in nll_spincorr_fits.py (w = 1 + sum C_ij wt_hp_i_hm_j,
C = diag(1, 1, -1)). Writes a 1D comparison (weighted mean and RMS in the legend), the 1D event-by-event difference
reco basis - gen basis, and a 2D reco-basis vs gen-basis plot.

    python taupolaris/scripts/plot_spin_basis_check.py [--results <nu-method results parquet>] [--test_file <row-aligned
        prepared test parquet>] [--var rk|kr] [--polvecs reco|gen] [--n_events N] --outdir <dir>
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.getcwd())
from taupolaris.python.spin_basis import spin_cosines, taus_from, polvecs_from

AX = 'nrk'
RECO, GEN = '#2a78d6', '#eb6834'
INK, MUTED, GRID = '#212121', '#6b7280', '#e5e7eb'


def load(results, test_file, n_events, prefix, polvecs, var):
    from taupolaris.python.Evaluation_Tools import compute_spin_vars
    i, j = var
    raw = ([f'{prefix}_tau_{s}_{c}' for s in ('plus', 'minus') for c in ('E', 'px', 'py', 'pz')] +
           [f'true_tau_{s}_{c}' for s in ('plus', 'minus') for c in ('E', 'px', 'py', 'pz')] +
           [f'reco_{t}_{p}_{c}' for t in ('taup', 'taun') for p in ('pi1', 'pi2', 'pi3', 'pizero1', 'charged') for c in ('E', 'px', 'py', 'pz')] +
           [f'reco_{t}_{f}' for t in ('taup', 'taun') for f in ('haspizero', 'ishadronic', 'is3prong')] +
           [f'wt_hp_{a}_hm_{a}' for a in AX])
    tcols = [f'ts_hh_{t}_{c}' for t in ('taup', 'taun') for c in 'xyz'] + ['undecayed_taup_px']
    out, n = [], 0
    for rb, tb in zip(pq.ParquetFile(results).iter_batches(batch_size=500_000, columns=raw),
                      pq.ParquetFile(test_file).iter_batches(batch_size=500_000, columns=tcols)):
        r, t = rb.to_pandas(), tb.to_pandas()
        if np.abs(r['true_tau_plus_px'].to_numpy() - t['undecayed_taup_px'].to_numpy()).max() > 1e-2:
            raise SystemExit(f'{test_file} is not row-aligned with {results}')
        if polvecs == 'reco':
            r = compute_spin_vars(r, tau_pred_prefix=f'{prefix}_', tau_vis_prefix='reco_', store_polvecs=True)
            h = polvecs_from(r, f'{prefix}_')
        else:
            h = (t[tcols[0:3]].to_numpy(float), t[tcols[3:6]].to_numpy(float))
        blk = {'w': 1 + sum(c * r[f'wt_hp_{a}_hm_{a}'].to_numpy() for a, c in zip(AX, (1, 1, -1)))}
        for name, pre in (('reco', f'{prefix}_'), ('gen', 'true_')):
            cs = spin_cosines(*h, *taus_from(r, pre))
            blk[name] = cs[f'cos{i}_plus'] * cs[f'cos{j}_minus']
        out.append(pd.DataFrame(blk)); n += len(r)
        print(f'  {n} events', flush=True)
        if n_events and n >= n_events:
            break
    d = pd.concat(out, ignore_index=True)
    d['w'] = d['w'].clip(lower=0)
    return d.iloc[:n_events] if n_events else d


def style(ax):
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/output_results.parquet')
    ap.add_argument('--test_file', default='prepared_LHC_data_sep01/ppToHToTauTau_AllDM_UnCorr_Jul20/test_dataframe__leptonic_mode_0_transformer.parquet')
    ap.add_argument('--var_prefix', default='map_pred')
    ap.add_argument('--var', default='rk', choices=['rk', 'kr'], help='rk: cos_r(+) cos_k(-), kr: cos_k(+) cos_r(-)')
    ap.add_argument('--polvecs', default='reco', choices=['reco', 'gen'])
    ap.add_argument('--n_events', type=int, default=0)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    d = load(a.results, a.test_file, a.n_events, a.var_prefix, a.polvecs, a.var)
    w = d['w'].to_numpy()
    i, j = a.var
    xlab = f'cos θ$_{i}^+$ · cos θ$_{j}^-$'
    tag = f'{a.var}_{a.polvecs}polvecs'

    # 1D
    edges = np.linspace(-1, 1, 41)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, col in (('reco', RECO), ('gen', GEN)):
        x = d[name].to_numpy()
        mean = np.average(x, weights=w)
        rms = np.sqrt(np.average((x - mean) ** 2, weights=w))
        hh, _ = np.histogram(x, edges, weights=w)
        ax.stairs(hh / (hh.sum() * np.diff(edges)), edges, color=col, lw=1.8, baseline=None,
                  label=f'{name} τ basis:  mean = {mean:+.4f},  RMS = {rms:.4f}')
        print(f'{name} basis: mean {mean:+.5f}  RMS {rms:.5f}')
    ax.set_xlabel(xlab, color=INK); ax.set_ylabel('normalised to unit area', color=INK)
    ax.set_title(f'SM-weighted H→ττ (hadronic), {a.polvecs} polarimetric vectors', loc='left', fontsize=10, color=INK)
    style(ax)
    lo, hi = ax.get_ylim(); ax.set_ylim(0, hi * 1.2)
    ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.outdir, f'cos{tag}_1d.{ext}'), dpi=120)
    plt.close(fig)

    # 1D difference, event by event (log y: the core is narrow, the interest is in the tails)
    dx = d['reco'].to_numpy() - d['gen'].to_numpy()
    mean = np.average(dx, weights=w)
    rms = np.sqrt(np.average((dx - mean) ** 2, weights=w))
    out = np.average(np.abs(dx) > 0.5, weights=w)
    ed = np.linspace(-0.5, 0.5, 101)
    hh, _ = np.histogram(np.clip(dx, ed[0] + 1e-9, ed[-1] - 1e-9), ed, weights=w)   # overflow into the edge bins
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.stairs(hh / (hh.sum() * np.diff(ed)), ed, color=RECO, lw=1.8, baseline=None,
              label=f'mean = {mean:+.5f},  RMS = {rms:.4f}\n|Δ| > 0.5: {out:.2e} (in edge bins)')
    ax.set_yscale('log')
    ax.set_xlabel(f'{xlab} (reco τ basis) − (gen τ basis)', color=INK); ax.set_ylabel('normalised to unit area', color=INK)
    ax.set_title(f'SM-weighted H→ττ (hadronic), {a.polvecs} polarimetric vectors', loc='left', fontsize=10, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK, handlelength=0)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.outdir, f'cos{tag}_diff.{ext}'), dpi=120)
    plt.close(fig)
    print(f'difference: mean {mean:+.6f}  RMS {rms:.5f}  |diff|>0.5 fraction {out:.2e}')

    # 2D, linear z
    e2 = np.linspace(-1, 1, 81)
    x, y = d['gen'].to_numpy(), d['reco'].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    h2 = ax.hist2d(x, y, bins=[e2, e2], weights=w, cmap='Blues')
    ax.plot([-1, 1], [-1, 1], color=MUTED, lw=0.8, ls='--')
    ax.set_xlabel(f'{xlab}, gen τ basis', color=INK); ax.set_ylabel(f'{xlab}, reco τ basis', color=INK)
    ax.set_title(f'{a.polvecs} polarimetric vectors; correlation {np.corrcoef(x, y)[0, 1]:.3f}', loc='left', fontsize=10, color=INK)
    fig.colorbar(h2[3], ax=ax, fraction=0.046, label='weighted events')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.outdir, f'cos{tag}_2d.{ext}'), dpi=120)
    plt.close(fig)
    print(f'>> wrote {a.outdir}/cos{tag}_{{1d,diff,2d}}.pdf/png')


if __name__ == '__main__':
    main()
