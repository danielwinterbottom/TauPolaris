"""
phiCP performance of the VISIBLE-ONLY polarimetric vector -- the eps -> 0
("ghost neutrino") limit of the exact per-decay-mode formulas, with the ghost
direction taken along the visible tau momentum. See
acoplanarity_tools.polarimetric_vec_visible_only.

No neutrino, no tau reconstruction, no secondary vertex: this is what can be
computed from the reco decay products alone. The point is to measure how much
CP-separating power that carries on its own, per decay mode, before deciding
whether to hand it to the flow as an extra input feature.

DM0 is excluded throughout (its polarimetric vector is just the pion direction,
so the limit is trivial and there is nothing to learn).

Variants compared:
  truth            ts_hh_* + true tau directions                      (ceiling)
  visible-only     h_vis + visible directions, visible-visible frame  (no tau at all)
  vis-h + true dir h_vis + TRUE tau directions                        (isolates h_vis)
  nu-method        the neutrino-regression flow                       (context)
  direct-method    the direct polvec flow                             (context)

Everything is accumulated in fixed-size histograms chunk by chunk rather than
held in memory, so this runs over the full multi-million-event results files
without needing them to fit in RAM. Metrics that look like they need the raw
values do not: the CP asymmetries are functions of the weighted phiCP
histograms, <cos dphi> is a running mean, and the median h_vis-to-truth angle
comes from a fine (0.25 deg) angle histogram.

Run (needs ROOT on the path, so use the venv rather than the torch env):
    source /Users/dw515/venvs/myenv/bin/activate
    PYTHONPATH="$PWD:$PYTHONPATH" python taupolaris/scripts/visible_polvec_study.py \
        --nu     outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/output_results.parquet \
        --direct outputs_model_LHC_TransformerFlow_PolVecDirect_Hadronic_Weighted_onorm_July29/polvec_eval_results_output_results_UnCorr.parquet \
        --outdir visible_polvec_study
"""
import argparse
import os
from collections import defaultdict

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:  # tqdm isn't in every environment this runs in
    def tqdm(it, **kwargs):
        return it

from taupolaris.utils.acoplanarity_tools import (
    polarimetric_vec_visible_only, spatial, boost_vec, boost4,
)
from taupolaris.utils.coordinate_conversions import _build_nrk_basis_from_visible_tau
from taupolaris.scripts.compare_polvec_methods import (
    nu_columns, direct_columns, dm_codes, vec3, unit, AXES, PHICP_BINS,
    phiCP_from_vectors, tau_dirs_in_ditau_frame,
    nu_method_observables, direct_method_observables, cos_between,
)

DMS = (1, 2, 10, 11)
DM_PAIRS = [(1, 1), (1, 2), (2, 2), (1, 10), (2, 10), (10, 10),
            (1, 11), (2, 11), (10, 11), (11, 11)]

TRUTH = 'truth (ts_hh + true taus)'
VIS = 'visible-only (h_vis + visible dirs)'
VIS_TRUEDIR = 'h_vis + true tau dirs'
NU = 'nu-method'
DIRECT = 'direct-method'

ANG_BINS = 720  # 0.25 deg resolution, for the median h_vis-to-truth angle
PHI_EDGES = np.linspace(0, 2 * np.pi, PHICP_BINS + 1)

# true-vs-estimated 2D correlation plots, in the (n,r,k) basis built from the
# visible tau momentum -- i.e. exactly the basis the direct model is trained in
# (coordinate_conversions._build_nrk_basis_from_visible_tau), so the panels show
# the correlation in the coordinates the network actually predicts.
NRK = ('n', 'r', 'k')
H2_BINS = 60
H2_EDGES = np.linspace(-1, 1, H2_BINS + 1)


# --------------------------------------------------------------------------
# chunked, row-aligned reading
# --------------------------------------------------------------------------
def iter_chunks(path, columns, chunk_rows, max_rows=None):
    """Yield DataFrames of exactly chunk_rows (last one may be shorter).
    parquet row groups don't line up between the two results files, so the
    batches have to be re-cut to a fixed size before they can be zipped."""
    pf = pq.ParquetFile(path)
    buf, have, emitted = [], 0, 0
    for batch in pf.iter_batches(batch_size=65536, columns=list(columns)):
        buf.append(batch.to_pandas())
        have += len(buf[-1])
        while have >= chunk_rows:
            df = pd.concat(buf, ignore_index=True) if len(buf) > 1 else buf[0]
            out = df.iloc[:chunk_rows].reset_index(drop=True)
            rest = df.iloc[chunk_rows:]
            buf = [rest.reset_index(drop=True)] if len(rest) else []
            have = len(rest)
            yield out
            emitted += len(out)
            if max_rows is not None and emitted >= max_rows:
                return
    if have:
        df = pd.concat(buf, ignore_index=True) if len(buf) > 1 else buf[0]
        if max_rows is not None:
            df = df.iloc[:max(0, max_rows - emitted)]
        if len(df):
            yield df.reset_index(drop=True)


def n_rows(path):
    return pq.ParquetFile(path).metadata.num_rows


# --------------------------------------------------------------------------
# accumulators
# --------------------------------------------------------------------------
class Accumulator:
    """Running histograms/sums. Everything reported is a function of these, so
    no chunk's raw values are ever kept."""

    def __init__(self):
        # (category, variant) -> [n, sum cos(dphi), even counts, odd counts]
        self.cat = defaultdict(lambda: [0, 0.0, np.zeros(PHICP_BINS), np.zeros(PHICP_BINS)])
        # leg DM -> [n legs, sum cos, angle histogram]
        self.leg = defaultdict(lambda: [0, 0.0, np.zeros(ANG_BINS)])
        # (leg DM, variant, component) -> 2D true-vs-estimated histogram
        self.h2 = defaultdict(lambda: np.zeros((H2_BINS, H2_BINS)))
        # (leg DM, variant, component) -> [n, sx, sy, sxx, syy, sxy] for Pearson r
        self.corr = defaultdict(lambda: np.zeros(6))

    def add_phicp(self, category, variant, phicp, truth_phicp, w_even, w_odd, mask):
        m = mask & np.isfinite(phicp) & np.isfinite(truth_phicp)
        if not m.any():
            return
        s = self.cat[(category, variant)]
        dphi = (phicp[m] - truth_phicp[m] + np.pi) % (2 * np.pi) - np.pi
        s[0] += int(m.sum())
        s[1] += float(np.cos(dphi).sum())
        s[2] += np.histogram(phicp[m], bins=PHI_EDGES, weights=w_even[m])[0]
        s[3] += np.histogram(phicp[m], bins=PHI_EDGES, weights=w_odd[m])[0]

    def add_legs(self, dm, cos_vals):
        s = self.leg[dm]
        s[0] += len(cos_vals)
        s[1] += float(cos_vals.sum())
        ang = np.degrees(np.arccos(np.clip(cos_vals, -1, 1)))
        s[2] += np.histogram(ang, bins=ANG_BINS, range=(0, 180))[0]


    def add_2d(self, dm, variant, comp, true_v, pred_v):
        good = np.isfinite(true_v) & np.isfinite(pred_v)
        true_v, pred_v = true_v[good], pred_v[good]
        if not len(true_v):
            return
        self.h2[(dm, variant, comp)] += np.histogram2d(
            true_v, pred_v, bins=(H2_EDGES, H2_EDGES))[0]
        self.corr[(dm, variant, comp)] += np.array([
            len(true_v), true_v.sum(), pred_v.sum(),
            (true_v ** 2).sum(), (pred_v ** 2).sum(), (true_v * pred_v).sum()])


def pearson(c):
    """Pearson r from accumulated [n, sx, sy, sxx, syy, sxy]."""
    n, sx, sy, sxx, syy, sxy = c
    if n < 2:
        return np.nan
    num = n * sxy - sx * sy
    den = np.sqrt(max(n * sxx - sx ** 2, 0.0) * max(n * syy - sy ** 2, 0.0))
    return float(num / den) if den > 0 else np.nan


def spread_ratio(c):
    """std(estimated)/std(true). Together with the Pearson r this separates two
    very different failure modes that r alone cannot: a NOISY estimator has low r
    but a spread near 1, while a COLLAPSED one (regressing towards the sample
    mean, which is what a likelihood-trained model does when it cannot resolve an
    event) has a spread well below 1. For a polarimetric vector the second is
    much worse -- a collapsed prediction carries almost no per-event
    information even where r looks non-zero."""
    n, sx, sy, sxx, syy, sxy = c
    if n < 2:
        return np.nan
    vx = max(n * sxx - sx ** 2, 0.0)
    vy = max(n * syy - sy ** 2, 0.0)
    return float(np.sqrt(vy / vx)) if vx > 0 else np.nan


def hist_median(counts, lo, hi):
    """Median from a histogram, linearly interpolated inside the crossing bin."""
    total = counts.sum()
    if total == 0:
        return np.nan
    edges = np.linspace(lo, hi, len(counts) + 1)
    cum = np.cumsum(counts)
    i = int(np.searchsorted(cum, total / 2.0))
    i = min(i, len(counts) - 1)
    before = cum[i - 1] if i > 0 else 0.0
    frac = (total / 2.0 - before) / counts[i] if counts[i] > 0 else 0.0
    return edges[i] + frac * (edges[i + 1] - edges[i])


def cp_metrics(even_counts, odd_counts):
    """(quadrature asymmetry, chi2 separation) from pre-binned weighted counts.
    Same definitions as compare_polvec_methods.cp_separation, which bins
    internally -- kept consistent so numbers are comparable across the two
    scripts."""
    pe = even_counts / max(even_counts.sum(), 1e-12)
    po = odd_counts / max(odd_counts.sum(), 1e-12)
    width = PHI_EDGES[1] - PHI_EDGES[0]
    asym = float(np.sqrt(np.sum((pe / width - po / width) ** 2)))
    denom = np.maximum((pe + po) / 2.0, 1e-12)
    return asym, float(np.sqrt(np.sum((pe - po) ** 2 / denom)))


# --------------------------------------------------------------------------
# physics
# --------------------------------------------------------------------------
def M4(df, tau, name, prefix='reco_'):
    return ak.zip({c: df[f'{prefix}{tau}_{name}_{c}'].to_numpy(dtype=float)
                   for c in ('px', 'py', 'pz', 'E')}, with_name="Momentum4D")


def visible_4vec(df, tau, dm):
    """Reco visible system: pi1 (+pi2+pi3 for 3-prong) (+pi0 where the DM has one)."""
    is3 = (dm == 10) | (dm == 11)
    has_pi0 = (dm == 1) | (dm == 2) | (dm == 11)
    piz = M4(df, tau, 'pizero1')
    piz = ak.with_name(ak.where(has_pi0, piz, piz * 0.0), "Momentum4D")
    v1 = M4(df, tau, 'pi1') + piz
    v3 = M4(df, tau, 'pi1') + M4(df, tau, 'pi2') + M4(df, tau, 'pi3') + piz
    return ak.with_name(ak.where(is3, v3, v1), "Momentum4D")


def to_np3(v):
    return np.stack([ak.to_numpy(v.x), ak.to_numpy(v.y), ak.to_numpy(v.z)], axis=1)


def visible_only_in_frame(df, dm_p, dm_n, boost_to, eps):
    """h_vis for both legs, inputs first boosted into `boost_to`'s rest frame --
    matching the two-step lab -> ditau-frame -> decay-frame convention that
    ts_hh_* is defined in. Returns (h_p, h_m, vis_p, vis_m) in that frame."""
    bv = boost_vec(boost_to)
    out = {}
    for tau, dm in (('taup', dm_p), ('taun', dm_n)):
        parts = {n: boost4(M4(df, tau, n), bv) for n in ('pi1', 'pi2', 'pi3', 'pizero1')}
        q = 1.0 if tau == 'taup' else -1.0
        out[tau] = (
            polarimetric_vec_visible_only(parts['pi1'], parts['pi2'], parts['pi3'],
                                          parts['pizero1'], dm, q, eps=eps),
            boost4(visible_4vec(df, tau, dm), bv),
        )
    return out['taup'][0], out['taun'][0], out['taup'][1], out['taun'][1]


def process_chunk(nu, dr, eps, acc, categories_fn):
    dm_p, dm_n = dm_codes(nu, 'reco')
    w_even = nu['tauspinner_wt_alpha0'].to_numpy()
    w_odd = nu['tauspinner_wt_alpha90'].to_numpy()

    tp = vec3(nu, ['true_tau_plus_px', 'true_tau_plus_py', 'true_tau_plus_pz'])
    tm = vec3(nu, ['true_tau_minus_px', 'true_tau_minus_py', 'true_tau_minus_pz'])
    Ep, Em = nu['true_tau_plus_E'].to_numpy(), nu['true_tau_minus_E'].to_numpy()
    tdp, tdm = tau_dirs_in_ditau_frame(tp, Ep, tm, Em)
    th_p = unit(vec3(nu, [f'ts_hh_taup_{a}' for a in AXES]))
    th_m = unit(vec3(nu, [f'ts_hh_taun_{a}' for a in AXES]))
    truth_phicp = phiCP_from_vectors(th_p, tdp, th_m, tdm)

    visvis = ak.with_name(visible_4vec(nu, 'taup', dm_p) + visible_4vec(nu, 'taun', dm_n),
                          "Momentum4D")
    hv_p, hv_m, vp, vm = visible_only_in_frame(nu, dm_p, dm_n, visvis, eps)
    vis_phicp = phiCP_from_vectors(hv_p, unit(to_np3(spatial(vp))),
                                   hv_m, unit(to_np3(spatial(vm))))

    true_ditau = ak.zip({'px': tp[:, 0] + tm[:, 0], 'py': tp[:, 1] + tm[:, 1],
                         'pz': tp[:, 2] + tm[:, 2], 'E': Ep + Em}, with_name="Momentum4D")
    hv2_p, hv2_m, _, _ = visible_only_in_frame(nu, dm_p, dm_n, true_ditau, eps)
    vis_truedir_phicp = phiCP_from_vectors(hv2_p, tdp, hv2_m, tdm)

    variants = {TRUTH: truth_phicp, VIS: vis_phicp, VIS_TRUEDIR: vis_truedir_phicp}
    # per-leg polarimetric vectors, for the true-vs-estimated 2D correlations
    legs = {VIS: (hv2_p, hv2_m)}
    if dr is not None:
        a, b, c, d = nu_method_observables(nu, 'map_pred', dm_p, dm_n)
        variants[NU] = phiCP_from_vectors(a, b, c, d)
        legs[NU] = (a, c)
        a, b, c, d = direct_method_observables(dr)
        variants[DIRECT] = phiCP_from_vectors(a, b, c, d)
        legs[DIRECT] = (a, c)

    # per-leg h_vis quality (referenced to the true ditau frame, as above)
    for d in DMS:
        for h, t, dv in ((hv2_p, th_p, dm_p), (hv2_m, th_m, dm_n)):
            s = (dv == d) & ~np.isnan(h).any(axis=1)
            if s.any():
                acc.add_legs(d, cos_between(h[s], t[s]))

    # --- true vs estimated, component by component, in the (n,r,k) basis the
    # direct model is trained in. Basis built from the LAB-frame visible momenta
    # exactly as at data-prep time, even though the vectors themselves live in
    # the ditau frame -- that mismatch is the codebase's own convention (see
    # ConvertToOrthonormalNRK), so reproducing it keeps these panels directly
    # comparable to the model's actual target.
    for tau, dv, th in (('taup', dm_p, th_p), ('taun', dm_n, th_m)):
        basis = _build_nrk_basis_from_visible_tau(
            nu, f'reco_{tau}_charged_', f'reco_{tau}_pizero1_')
        true_proj = {c: np.sum(th * b, axis=1) for c, b in zip(NRK, basis)}
        for name, (hp, hm) in legs.items():
            h = hp if tau == 'taup' else hm
            ok = np.isfinite(h).all(axis=1) & (np.abs(h).max(axis=1) < 100)
            for c, b in zip(NRK, basis):
                pred_proj = np.sum(h * b, axis=1)
                for d in DMS:
                    s = ok & (dv == d)
                    if s.any():
                        acc.add_2d(d, name, c, true_proj[c][s], pred_proj[s])

    for cat, mask in categories_fn(dm_p, dm_n).items():
        for name, ph in variants.items():
            acc.add_phicp(cat, name, ph, truth_phicp, w_even, w_odd, mask)


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_cp_curves(acc, category, variants, title, outpath):
    """One panel per variant, CP-even vs CP-odd phiCP, shared y-axis so the
    panels compare by eye. Drawn from the accumulated counts, density-normalised
    the same way ax.hist(density=True) would."""
    fig, axes = plt.subplots(1, len(variants), figsize=(3.5 * len(variants), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    width = PHI_EDGES[1] - PHI_EDGES[0]
    for ax, name in zip(axes, variants):
        s = acc.cat.get((category, name))
        if s is None or s[0] == 0:
            ax.set_axis_off()
            continue
        de = s[2] / max(s[2].sum(), 1e-12) / width
        do = s[3] / max(s[3].sum(), 1e-12) / width
        ax.stairs(de, PHI_EDGES, color='steelblue', linewidth=1.7, label='CP-even')
        ax.stairs(do, PHI_EDGES, color='tomato', linewidth=1.7, label='CP-odd')
        asym, _ = cp_metrics(s[2], s[3])
        ax.set_title(f'{name}\nasym = {asym:.4f}', fontsize=8.5)
        ax.set_xlabel(r'$\phi_{CP}$ [rad]')
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel('a.u.')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_2d_correlations(acc, dm, variants, outpath):
    """Rows = estimator, columns = (n,r,k) component: true vs estimated, with the
    Pearson r for that panel. Empty bins are left white (same convention as
    evaluate_polvec.py's cmin=1) and the colour scale is logarithmic, since the
    density spans several orders of magnitude between the diagonal and the tails."""
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(len(variants), len(NRK),
                             figsize=(3.7 * len(NRK), 3.4 * len(variants)),
                             squeeze=False)
    for i, v in enumerate(variants):
        for j, comp in enumerate(NRK):
            ax = axes[i][j]
            H = acc.h2.get((dm, v, comp))
            if H is None or H.sum() == 0:
                ax.set_axis_off()
                continue
            ax.pcolormesh(H2_EDGES, H2_EDGES, np.ma.masked_equal(H.T, 0),
                          cmap='viridis', norm=LogNorm())
            ax.plot([-1, 1], [-1, 1], 'r--', linewidth=1)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            ax.set_aspect('equal')
            cc = acc.corr[(dm, v, comp)]
            ax.set_title(f'$h_{comp}$   r = {pearson(cc):.3f}   '
                         r'$\sigma_{est}/\sigma_{true}$ = ' f'{spread_ratio(cc):.2f}',
                         fontsize=9)
            ax.set_xlabel(f'true $h_{comp}$', fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f'{v}\nestimated $h_{comp}$', fontsize=8.5)
            else:
                ax.set_ylabel(f'estimated $h_{comp}$', fontsize=8.5)
    fig.suptitle(f'Polarimetric vector, true vs estimated -- leg DM{dm} '
                 f'(visible-momentum n,r,k basis)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_correlation_summary(acc, dms, variants, outpath):
    """Pearson r per component, grouped by decay mode -- the one-page read of
    which estimator tracks the truth in which coordinate."""
    fig, axes = plt.subplots(1, len(NRK), figsize=(4.4 * len(NRK), 4.2), sharey=True)
    x = np.arange(len(dms))
    width = 0.8 / max(len(variants), 1)
    for j, comp in enumerate(NRK):
        ax = np.atleast_1d(axes)[j]
        for i, v in enumerate(variants):
            vals = [pearson(acc.corr[(d, v, comp)]) for d in dms]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=v)
        ax.set_xticks(x); ax.set_xticklabels([f'DM{d}' for d in dms])
        ax.set_title(f'$h_{comp}$', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.8)
    np.atleast_1d(axes)[0].set_ylabel('Pearson r vs true polarimetric vector')
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_retained_summary(table, order, outpath):
    variants = [v for v in table['variant'].unique() if v != TRUTH]
    piv = table.pivot(index='category', columns='variant', values='retained').reindex(order)
    x = np.arange(len(piv))
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(max(9, 1.0 * len(piv) + 3), 5))
    for i, v in enumerate(variants):
        ax.bar(x + i * width - 0.4 + width / 2, piv[v].to_numpy(), width, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('retained CP asymmetry (variant / truth)')
    ax.set_title('phiCP CP-even vs CP-odd separation retained, by decay-mode pair', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nu', required=True)
    ap.add_argument('--direct', default=None, help='optional, for context in the table')
    ap.add_argument('--outdir', default='visible_polvec_study')
    ap.add_argument('--max_events', type=int, default=None,
                    help='default: every row in the (shorter) results file')
    ap.add_argument('--chunk_rows', type=int, default=200000,
                    help='events held in memory at once')
    ap.add_argument('--min_events', type=int, default=200,
                    help='skip decay-mode combinations with fewer events than this')
    ap.add_argument('--eps', type=float, default=1e-6,
                    help='ghost-neutrino scale. A numerical stand-in for the eps -> 0 '
                         'limit, not a physical parameter; results are stable over 1e-2..1e-6.')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    total = n_rows(args.nu)
    if args.direct:
        total = min(total, n_rows(args.direct))
    if args.max_events is not None:
        total = min(total, args.max_events)
    print(f">> Processing {total} events in chunks of {args.chunk_rows}")

    def categories_fn(dm_p, dm_n):
        both = np.isin(dm_p, DMS) & np.isin(dm_n, DMS)
        cats = {'inclusive': both}
        for a, b in DM_PAIRS:
            cats[f'DM{a}-DM{b}'] = both & (((dm_p == a) & (dm_n == b)) |
                                           ((dm_p == b) & (dm_n == a)))
        return cats

    acc = Accumulator()
    nu_it = iter_chunks(args.nu, nu_columns(), args.chunk_rows, total)
    dr_it = iter_chunks(args.direct, direct_columns(), args.chunk_rows, total) if args.direct \
        else iter([None] * (total // args.chunk_rows + 2))
    checked = False
    for nu_df, dr_df in tqdm(zip(nu_it, dr_it), total=int(np.ceil(total / args.chunk_rows)),
                             desc="chunks"):
        if dr_df is not None:
            n = min(len(nu_df), len(dr_df))
            nu_df, dr_df = nu_df.iloc[:n].reset_index(drop=True), dr_df.iloc[:n].reset_index(drop=True)
            if not checked:
                # relative comparison with a floor, NOT np.allclose: the two files
                # are written at different float precisions (nu float64, direct
                # float32), so a fixed atol trips on rows where px happens to sit
                # near zero even though the events line up perfectly.
                a = nu_df['true_tau_plus_px'].to_numpy()
                b = dr_df['true_tau_plus_px'].to_numpy()
                frac_ok = float((np.abs(a - b) / np.maximum(np.abs(a), 1.0) < 1e-4).mean())
                if frac_ok < 0.999:
                    raise SystemExit("ERROR: nu and direct results files are not row-aligned "
                                     f"({100 * frac_ok:.3f}% of rows match)")
                checked = True
        process_chunk(nu_df, dr_df, args.eps, acc, categories_fn)

    variants = [TRUTH, VIS, VIS_TRUEDIR] + ([NU, DIRECT] if args.direct else [])

    print("\n=== visible-only polarimetric vector vs true ts_hh, per leg DM ===")
    print(f"{'DM':>4s} {'n legs':>10s} {'median angle':>13s} {'<cos>':>9s}")
    leg_rows = []
    for d in DMS:
        n, sc, hist = acc.leg[d]
        if n == 0:
            continue
        med = hist_median(hist, 0, 180)
        print(f"{d:4d} {n:10d} {med:12.2f}d {sc / n:9.4f}")
        leg_rows.append({'leg_DM': d, 'n_legs': n, 'median_angle_deg': med, 'meancos': sc / n})
    pd.DataFrame(leg_rows).to_csv(os.path.join(args.outdir, 'h_vis_quality_by_dm.csv'), index=False)

    def rows_for(category):
        out = []
        tr = acc.cat.get((category, TRUTH))
        if tr is None or tr[0] == 0:
            return out
        a_t, chi_t = cp_metrics(tr[2], tr[3])
        for name in variants:
            s = acc.cat.get((category, name))
            if s is None or s[0] == 0:
                continue
            a_p, chi_p = cp_metrics(s[2], s[3])
            out.append({'category': category, 'variant': name, 'N': s[0],
                        'phiCP_meancos': s[1] / s[0], 'CP_asym': a_p, 'CP_asym_truth': a_t,
                        'retained': a_p / a_t if a_t > 0 else np.nan,
                        'chi2_sep': chi_p,
                        'chi2_retained': chi_p / chi_t if chi_t > 0 else np.nan})
        return out

    inc = pd.DataFrame(rows_for('inclusive'))
    print(f"\n=== phiCP, both legs in DM{{{','.join(map(str, DMS))}}} "
          f"({int(inc['N'].iloc[0])} events) ===")
    print(inc.drop(columns='category').to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    inc.to_csv(os.path.join(args.outdir, 'phiCP_inclusive.csv'), index=False)

    plot_cp_curves(acc, 'inclusive', variants,
                   f"All events, both legs in DM{{{','.join(map(str, DMS))}}} "
                   f"({int(inc['N'].iloc[0])} events)",
                   os.path.join(args.outdir, 'phiCP_cp_curves.pdf'))

    dm_outdir = os.path.join(args.outdir, 'phiCP_by_dm')
    os.makedirs(dm_outdir, exist_ok=True)
    per_dm, order = [], []
    for a, b in DM_PAIRS:
        cat = f'DM{a}-DM{b}'
        rows = rows_for(cat)
        if not rows or rows[0]['N'] < args.min_events:
            continue
        order.append(cat)
        per_dm += rows
        plot_cp_curves(acc, cat, variants, f"{cat}  ({rows[0]['N']} events)",
                       os.path.join(dm_outdir, f'phiCP_cp_{cat}.pdf'))

    per_dm = pd.DataFrame(per_dm)
    if not per_dm.empty:
        print("\n=== retained CP asymmetry (variant/truth) by decay-mode pair ===")
        piv = per_dm.pivot(index='category', columns='variant', values='retained').reindex(order)
        piv.insert(0, 'N', per_dm[per_dm.variant == TRUTH].set_index('category')['N'].reindex(order))
        print(piv.to_string(float_format=lambda v: f'{v:.4f}'))
        per_dm.to_csv(os.path.join(args.outdir, 'phiCP_by_dm.csv'), index=False)
        plot_retained_summary(per_dm, order,
                              os.path.join(args.outdir, 'retained_asymmetry_by_dm.pdf'))

    # ---- true vs estimated 2D correlations ----
    corr_variants = [v for v in (VIS, NU, DIRECT) if any((d, v, 'n') in acc.h2 for d in DMS)]
    corr_dir = os.path.join(args.outdir, 'polvec_correlation')
    os.makedirs(corr_dir, exist_ok=True)
    corr_rows = []
    for d in DMS:
        if not any((d, v, 'n') in acc.h2 for v in corr_variants):
            continue
        plot_2d_correlations(acc, d, corr_variants,
                             os.path.join(corr_dir, f'polvec_true_vs_est_DM{d}.pdf'))
        for v in corr_variants:
            row = {'leg_DM': d, 'variant': v,
                   'n_legs': int(acc.corr[(d, v, 'n')][0])}
            row.update({f'r_{c}': pearson(acc.corr[(d, v, c)]) for c in NRK})
            row.update({f'spread_{c}': spread_ratio(acc.corr[(d, v, c)]) for c in NRK})
            corr_rows.append(row)
    if corr_rows:
        corr = pd.DataFrame(corr_rows)
        print("\n=== Pearson r vs the true polarimetric vector, by leg DM "
              "(visible-momentum n,r,k basis) ===")
        print(corr.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        corr.to_csv(os.path.join(args.outdir, 'polvec_correlation_by_dm.csv'), index=False)
        plot_correlation_summary(acc, DMS, corr_variants,
                                 os.path.join(args.outdir, 'polvec_correlation_summary.pdf'))

    print(f"\n>> plots + csv written to {args.outdir}/ (per-DM curves in {dm_outdir}/, "
          f"correlations in {corr_dir}/)")


if __name__ == '__main__':
    main()
