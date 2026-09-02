"""
Head-to-head comparison of the two ways this project reconstructs the tau
polarimetric vectors and tau momenta:

  * "nu-method"     -- a flow regresses the two neutrino momenta
                       (taupolaris/scripts/evaluate.py output parquet). The
                       polarimetric vectors are then *derived* analytically
                       from (predicted tau, reco decay products) with
                       acoplanarity_tools.get_ditau_polarimetric.
  * "direct-method" -- a flow regresses the polarimetric vectors and the
                       undecayed tau momenta directly
                       (taupolaris/scripts/evaluate_polvec.py output parquet).

WHY THIS SCRIPT EXISTS RATHER THAN JUST DIFFING THE TWO PARQUETS: the
`pred_cosn/r/k_plus/minus` columns the two evaluation scripts store are NOT
the same variable.

  - evaluate.py         -> Evaluation_Tools.compute_spin_vars ->
                           kinematic_helpers.compute_spin_angles, i.e. the
                           canonical spin basis with k_hat = the *tau+*
                           direction in the ditau rest frame.
  - evaluate_polvec.py  -> the model's own training-target basis, i.e. the
                           visible-tau (charged+pi0) referenced (n,r,k) of
                           coordinate_conversions._build_nrk_basis_from_visible_tau,
                           built in the *lab* frame.

Comparing those two columns directly would be meaningless. Everything here is
therefore recomputed from the Cartesian quantities both files do share, in one
common convention (the ditau/Higgs rest frame, matching the frame the truth
`ts_hh_taup/taun_{x,y,z}` branches are stored in -- see
run_delphes.py::_hh_to_higgs_rf).

The two truth definitions are also NOT interchangeable for every decay mode:
`ts_hh_*` is TauSpinner's exact HH vector for whatever the tau actually did,
while the nu-method's analytic reconstruction uses a per-DM formula. They agree
to ~1e-5 for DM 0/1 and to ~2e-3 for DM 10, but the DM=2 and DM=11 formulas
differ by construction. `--truth` selects which reference to score against;
the default (`ts_hh`) is the common one and is what the direct method trains on.

Usage
-----
    python taupolaris/scripts/compare_polvec_methods.py \
        --nu    outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/output_results.parquet \
        --direct outputs_model_LHC_TransformerFlow_PolVecDirect_Hadronic_Weighted_onorm_July29/polvec_eval_results_output_results_UnCorr.parquet \
        --max_events 400000 --outdir compare_nu_vs_direct

`--direct` is repeatable, so several direct-method trainings can be scored against
the same nu-method baseline on the same events (they must all have been evaluated
on the same test dataframe -- the script asserts row alignment):

    python taupolaris/scripts/compare_polvec_methods.py \
        --nu     outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/output_results.parquet \
        --direct July29-postfix=outputs_..._onorm_July29/polvec_eval_results_output_results_UnCorr.parquet \
        --direct July16-prefix=outputs_..._onorm_July16/polvec_eval_results_output_results_UnCorr.parquet \
        --max_events 100000 --outdir compare_three_way

Note this needs ROOT on the path (via PolarimetricA1), so run it in the venv that
has ROOT rather than the torch conda env:

    source /Users/dw515/venvs/myenv/bin/activate
    PYTHONPATH="$PWD:$PYTHONPATH" python taupolaris/scripts/compare_polvec_methods.py ...
"""
import argparse
import os
import re
import warnings

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from taupolaris.utils.acoplanarity_tools import get_ditau_polarimetric, compute_aco_polarimetric
from taupolaris.utils.kinematic_helpers import boost, boost_vector

M_TAU = 1.77686
PHICP_BINS = 20
AXES = ('x', 'y', 'z')

# decay-mode combinations reported in the per-DM tables, in the order they are
# printed. Unordered pairs -- (a,b) also matches (b,a), same as
# evaluate_polvec.py's phiCP-by-DM plots.
DM_PAIRS = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2),
            (0, 10), (1, 10), (2, 10), (10, 10),
            (0, 11), (1, 11), (2, 11), (10, 11), (11, 11)]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, eps)


def vec3(df, cols):
    return df[list(cols)].to_numpy(dtype=float)


def to_ak3(arr):
    return ak.zip({"x": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2]}, with_name="Vector3D")


def dm_codes(df, prefix):
    """Per-tau DM code (0/1/2/10/11 hadronic, 100 leptonic, -1 unmatched).
    Identical convention to evaluate.py::_phiCP_dm_codes, evaluate_polvec.py::add_DM
    and plot_phiCP.py::add_DM -- kept in lockstep so the categories in this
    comparison mean the same thing as in every existing plot."""
    out = {}
    for tau in ('taup', 'taun'):
        is_lep = df[f'{prefix}_{tau}_ishadronic'].to_numpy() == 0
        npizero = df[f'{prefix}_{tau}_npizero'].to_numpy()
        is3prong = df[f'{prefix}_{tau}_is3prong'].to_numpy()
        is_dm0 = (npizero == 0) & (is3prong == 0) & (~is_lep)
        is_dm1 = (npizero == 1) & (is3prong == 0) & (~is_lep)
        is_dm2 = ((npizero == 1) | (npizero == 2)) & (is3prong == 0) & (~is_lep)
        is_dm10 = (npizero == 0) & (is3prong == 1) & (~is_lep)
        is_dm11 = (npizero == 1) & (is3prong == 1) & (~is_lep)
        out[tau] = np.where(is_dm0, 0,
                     np.where(is_dm1, 1,
                       np.where(is_dm2, 2,
                         np.where(is_dm10, 10,
                           np.where(is_dm11, 11,
                             np.where(is_lep, 100, -1))))))
    return out['taup'], out['taun']


def phiCP_from_vectors(h_p, dir_p, h_m, dir_m):
    """phiCP from the two polarimetric vectors and the two tau directions, all
    already in the ditau rest frame. Same formula (and therefore same
    normalisation/orientation convention) as evaluate.py and evaluate_polvec.py,
    both of which route through acoplanarity_tools.compute_aco_polarimetric.

    NOTE the (R1,P1,R2,P2) ordering is irrelevant here: swapping the two legs
    flips both the cross-product order and the sign of the reference direction
    (the taus are exactly back-to-back in this frame), so the result is
    unchanged. evaluate.py passes tau+ first, evaluate_polvec.py passes tau-
    first, and they agree."""
    return ak.to_numpy(compute_aco_polarimetric(to_ak3(h_p), to_ak3(dir_p),
                                                to_ak3(h_m), to_ak3(dir_m)))


def tau_dirs_in_ditau_frame(p_p, E_p, p_m, E_m):
    """Unit tau+/tau- directions in the ditau rest frame. Matches
    evaluate_polvec.py::tau_directions_com (and get_ditau_polarimetric's
    spatial(tau_rf).unit(), which is the same thing up to the exactly
    back-to-back relation)."""
    t_p = np.column_stack([E_p, p_p])
    t_m = np.column_stack([E_m, p_m])
    bv = boost_vector(t_p + t_m)
    return unit(boost(t_p, -bv)[:, 1:]), unit(boost(t_m, -bv)[:, 1:])


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def read_head(path, columns, n_rows):
    """Read the first n_rows of the named columns. Streams row groups rather
    than pq.read_table(...).head(): these results files are multi-GB and reading
    them whole just to slice the front is what makes the naive version unusable."""
    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    missing = [c for c in columns if c not in have]
    if missing:
        raise KeyError(f"{os.path.basename(path)} is missing required columns: {missing}")
    chunks, got = [], 0
    for batch in pf.iter_batches(batch_size=65536, columns=list(columns)):
        chunks.append(batch.to_pandas())
        got += len(chunks[-1])
        if got >= n_rows:
            break
    df = pd.concat(chunks, ignore_index=True)
    return df.iloc[:n_rows].reset_index(drop=True)


def nu_columns():
    cols = []
    for pref in ('true_', 'pred_', 'map_pred_'):
        for t in ('plus', 'minus'):
            cols += [f'{pref}tau_{t}_{c}' for c in ('E', 'px', 'py', 'pz')]
    for tau in ('taup', 'taun'):
        for obj in ('pi1', 'pi2', 'pi3', 'pizero1', 'charged'):
            cols += [f'reco_{tau}_{obj}_{c}' for c in ('E', 'px', 'py', 'pz')]
        cols += [f'reco_{tau}_ishadronic', f'reco_{tau}_npizero', f'reco_{tau}_is3prong']
        cols += [f'true_{tau}_ishadronic', f'true_{tau}_npizero', f'true_{tau}_is3prong']
        cols += [f'ts_hh_{tau}_{a}' for a in AXES]
    cols += ['tauspinner_wt_alpha0', 'tauspinner_wt_alpha90']
    return cols


def direct_columns():
    cols = []
    for t in ('plus', 'minus'):
        for pref in ('true_', 'pred_'):
            cols += [f'{pref}tau_{t}_{c}' for c in ('E', 'px', 'py', 'pz')]
    for tau in ('taup', 'taun'):
        cols += [f'true_ts_hh_{tau}_{a}' for a in AXES]
        cols += [f'pred_ts_hh_{tau}_{a}' for a in AXES]
        cols += [f'reco_{tau}_DM', f'gen_{tau}_DM']
    cols += ['true_phiCP', 'pred_phiCP', 'tauspinner_wt_alpha0', 'tauspinner_wt_alpha90']
    return cols


# --------------------------------------------------------------------------
# per-method observable extraction
# --------------------------------------------------------------------------
def nu_method_observables(nu, tau_prefix, dm_p, dm_n):
    """Polarimetric vectors + tau directions in the ditau rest frame for the
    nu-method, via exactly the path evaluate.py/plot_phiCP.py use for its phiCP
    (get_ditau_polarimetric), so this reproduces that method's real physics
    performance rather than a re-derivation of it.

    tau_prefix: 'true', 'pred' (single flow sample) or 'map_pred' (MAP estimate).
    """
    df = nu.copy()
    df['taup_DM'] = dm_p
    df['taun_DM'] = dm_n
    h_p, dir_p, h_m, dir_m = get_ditau_polarimetric(df, tau_prefix=tau_prefix,
                                                    reco_pions=True, add_ghosts=True)
    to_np = lambda v: np.stack([ak.to_numpy(v.x), ak.to_numpy(v.y), ak.to_numpy(v.z)], axis=1)
    return to_np(h_p), to_np(dir_p), to_np(h_m), to_np(dir_m)


def direct_method_observables(dr):
    """Polarimetric vectors + tau directions in the ditau rest frame for the
    direct method. pred_ts_hh_* is already the model's own (unit-normalised)
    output in this frame; the tau directions come from its regressed tau momenta."""
    h_p = unit(vec3(dr, [f'pred_ts_hh_taup_{a}' for a in AXES]))
    h_m = unit(vec3(dr, [f'pred_ts_hh_taun_{a}' for a in AXES]))
    p_p = vec3(dr, ['pred_tau_plus_px', 'pred_tau_plus_py', 'pred_tau_plus_pz'])
    p_m = vec3(dr, ['pred_tau_minus_px', 'pred_tau_minus_py', 'pred_tau_minus_pz'])
    E_p = np.sqrt((p_p ** 2).sum(1) + M_TAU ** 2)
    E_m = np.sqrt((p_m ** 2).sum(1) + M_TAU ** 2)
    dir_p, dir_m = tau_dirs_in_ditau_frame(p_p, E_p, p_m, E_m)
    return h_p, dir_p, h_m, dir_m


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def cos_between(a, b):
    return np.clip(np.sum(unit(a) * unit(b), axis=1), -1.0, 1.0)


def asymmetry_quadrature(counts_a, counts_b):
    """Same metric as plot_phiCP.py / evaluate_polvec.py: sqrt(sum((a-b)^2)) over
    per-bin counts that each sum to 1. Note PROBABILITY-normalised, not density:
    plot_phiCP.py normalises the weights (w / w.sum()) and histograms with
    density=False, so dividing by the bin width here would inflate the result by
    1/binwidth = 20/2pi = 3.183 and break comparability with its plots."""
    return float(np.sqrt(np.sum((counts_a - counts_b) ** 2)))


def cp_separation(phicp, w_even, w_odd, bins=PHICP_BINS):
    """CP-even vs CP-odd separation power of a phiCP distribution.

    Returns (asym_quadrature, chi2_sep). asym_quadrature is the existing
    codebase metric. chi2_sep is a bin-count-independent symmetric chi2 distance
    sqrt(sum (p_e-p_o)^2 / ((p_e+p_o)/2)) over probability-normalised bins,
    which is the quantity that actually scales like the statistical
    separation between the two hypotheses -- useful because the quadrature sum
    changes if anyone edits PHICP_BINS."""
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    # Normalise the WEIGHTS before histogramming, not the counts after -- that is
    # what plot_phiCP.py::plot_phicp_histogram does, and the two differ whenever
    # any event's phiCP is NaN or outside [0, 2pi]: such events contribute to the
    # weight sum but to no bin, so normalising afterwards silently rescales.
    def _probs(w):
        w = np.asarray(w, dtype=float)
        total = w.sum()
        if total != 0:
            w = w / total
        return np.histogram(phicp, bins=edges, weights=w)[0]

    pe, po = _probs(w_even), _probs(w_odd)
    denom = np.maximum((pe + po) / 2.0, 1e-12)
    return asymmetry_quadrature(pe, po), float(np.sqrt(np.sum((pe - po) ** 2 / denom)))


def metrics_for_mask(obs, truth, mask, w_even, w_odd):
    """All comparison metrics for one method on one event subset.

    obs   : dict from *_method_observables (h_p, dir_p, h_m, dir_m) + predicted tau 3-momenta
    truth : dict with the same keys built from truth
    """
    n = int(mask.sum())
    if n == 0:
        return None
    m = mask

    res = {'N': n}

    # --- polarimetric vector: angle to truth. <cos> is the physically
    # meaningful one -- it is the dilution factor multiplying every
    # spin-correlation observable built from h.
    for leg, key in (('taup', 'h_p'), ('taun', 'h_m')):
        c = cos_between(obs[key][m], truth[key][m])
        res[f'{leg}_h_meancos'] = float(c.mean())
        res[f'{leg}_h_medang'] = float(np.degrees(np.arccos(np.median(c))))
    res['h_meancos'] = 0.5 * (res['taup_h_meancos'] + res['taun_h_meancos'])

    # --- tau direction in the ditau rest frame (what phiCP's projection axis is)
    c_dir = cos_between(obs['dir_p'][m], truth['dir_p'][m])
    res['tau_dir_meancos'] = float(c_dir.mean())
    res['tau_dir_medang'] = float(np.degrees(np.arccos(np.median(c_dir))))

    # --- tau momentum in the lab
    for leg in ('p', 'm'):
        pt = truth[f'tau{leg}_p3'][m]
        pp = obs[f'tau{leg}_p3'][m]
        mag_t = np.linalg.norm(pt, axis=1)
        mag_p = np.linalg.norm(pp, axis=1)
        rel = (mag_p - mag_t) / np.maximum(mag_t, 1e-9)
        res[f'tau{leg}_pmag_bias'] = float(np.median(rel))
        res[f'tau{leg}_pmag_iqr'] = float(np.subtract(*np.percentile(rel, [75, 25])))
    res['tau_pmag_iqr'] = 0.5 * (res['taup_pmag_iqr'] + res['taum_pmag_iqr'])

    # --- ditau invariant mass
    res['ditau_mass_bias'] = float(np.median(
        (obs['ditau_mass'][m] - truth['ditau_mass'][m]) / np.maximum(truth['ditau_mass'][m], 1e-9)))

    # --- phiCP
    dphi = (obs['phiCP'][m] - truth['phiCP'][m] + np.pi) % (2 * np.pi) - np.pi
    res['phiCP_meancos'] = float(np.mean(np.cos(dphi)))
    res['phiCP_medabs'] = float(np.median(np.abs(dphi)))

    a_pred, chi_pred = cp_separation(obs['phiCP'][m], w_even[m], w_odd[m])
    a_true, chi_true = cp_separation(truth['phiCP'][m], w_even[m], w_odd[m])
    res['phiCP_asym_pred'] = a_pred
    res['phiCP_asym_true'] = a_true
    res['phiCP_asym_ratio'] = a_pred / a_true if a_true > 0 else np.nan
    res['phiCP_chi2sep_pred'] = chi_pred
    res['phiCP_chi2sep_true'] = chi_true
    res['phiCP_chi2sep_ratio'] = chi_pred / chi_true if chi_true > 0 else np.nan
    return res


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_overlay_hist(series, bins, xlabel, title, outpath, density=True, weights=None):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, vals in series.items():
        ax.hist(vals, bins=bins, histtype='step', linewidth=1.6, density=density,
                weights=weights, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('a.u.' if density else 'events')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_metric_by_dm(table, metric, ylabel, title, outpath, higher_is_better=True):
    """One grouped bar chart per metric: category on x, one bar per method."""
    cats = [c for c in table['category'].unique()]
    methods = [m for m in table['method'].unique()]
    x = np.arange(len(cats))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(cats) + 3), 5))
    for i, meth in enumerate(methods):
        sub = table[table['method'] == meth].set_index('category')
        vals = [sub[metric].get(c, np.nan) for c in cats]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=meth)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  ({'higher' if higher_is_better else 'lower'} is better)", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_phiCP_cp_curves(truth_phicp, method_phicp, w_even, w_odd, title, outpath):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bins = np.linspace(0, 2 * np.pi, PHICP_BINS + 1)
    ax.hist(truth_phicp, bins=bins, weights=w_even, density=True, histtype='step',
            linewidth=1.8, linestyle='--', color='k', label='truth, CP-even')
    ax.hist(truth_phicp, bins=bins, weights=w_odd, density=True, histtype='step',
            linewidth=1.8, linestyle=':', color='k', label='truth, CP-odd')
    colours = {'nu-method': 'steelblue'}
    for label, vals in method_phicp.items():
        col = colours.get(label, None)
        ax.hist(vals, bins=bins, weights=w_even, density=True, histtype='step',
                linewidth=1.6, color=col, label=f'{label}, CP-even')
        ax.hist(vals, bins=bins, weights=w_odd, density=True, histtype='step',
                linewidth=1.6, linestyle='-.', color=col, label=f'{label}, CP-odd')
    ax.set_xlabel(r'$\phi_{CP}$ [rad]')
    ax.set_ylabel('a.u.')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nu', required=True, help='evaluate.py results parquet (neutrino-regression model)')
    ap.add_argument('--direct', required=True, action='append', metavar='[LABEL=]PATH',
                    help='evaluate_polvec.py results parquet (direct polvec model). Repeatable, so '
                         'several direct-method trainings (e.g. before/after a fix) can be compared '
                         'against the same nu-method baseline on the same events. Prefix with '
                         '"LABEL=" to name the column, otherwise the containing directory is used.')
    ap.add_argument('--outdir', default='compare_polvec_methods')
    ap.add_argument('--max_events', type=int, default=400000,
                    help='number of (row-aligned) events to compare')
    ap.add_argument('--nu_tau_prefix', default='map_pred', choices=['map_pred', 'pred'],
                    help="which nu-method estimate to use. 'map_pred' (default) is the MAP "
                         "estimate, i.e. the like-for-like counterpart of the direct method's "
                         "own MAP prediction; 'pred' is a single posterior sample.")
    ap.add_argument('--truth', default='ts_hh', choices=['ts_hh', 'analytic'],
                    help="reference polarimetric vector. 'ts_hh' (default) = TauSpinner truth, "
                         "the common reference and the direct model's training target. "
                         "'analytic' = get_ditau_polarimetric evaluated with the TRUE taus and "
                         "reco pions, i.e. the nu-method's own best-case ceiling -- useful to "
                         "separate 'the flow is worse' from 'the analytic formula for this DM "
                         "is not the same quantity as ts_hh'.")
    ap.add_argument('--dm_level', default='reco', choices=['reco', 'true'],
                    help='classify decay modes at reco or gen level')
    ap.add_argument('--min_events', type=int, default=200,
                    help='skip DM categories with fewer events than this')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f">> Reading {args.max_events} events from each results file...")
    nu = read_head(args.nu, nu_columns(), args.max_events)

    directs = {}
    for spec in args.direct:
        if '=' in spec and not os.path.exists(spec):
            label, path = spec.split('=', 1)
        else:
            label, path = None, spec
        if label is None:
            # name it after the model output directory, trimmed of the shared
            # 'outputs_model_LHC_TransformerFlow_PolVecDirect_' prefix so the
            # tables stay readable when several are compared at once
            parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
            label = 'direct-' + re.sub(r'^outputs_(model_)?(LHC_TransformerFlow_)?(PolVecDirect_)?', '', parent)
        directs[label] = read_head(path, direct_columns(), args.max_events)

    n = min([len(nu)] + [len(d) for d in directs.values()])
    nu = nu.iloc[:n].reset_index(drop=True)
    directs = {k: v.iloc[:n].reset_index(drop=True) for k, v in directs.items()}
    print(f">> Comparing {n} events across: nu-method, {', '.join(directs)}")

    # --- row alignment check. The two evaluation scripts write their rows in the
    # same order (same prepared test dataframe), but nothing enforces it, and a
    # silent misalignment would look exactly like "one model is worse".
    scale = np.maximum(np.abs(nu['true_tau_plus_px'].to_numpy()), 1.0)
    for label, dr in directs.items():
        align = np.abs(nu['true_tau_plus_px'].to_numpy() - dr['true_tau_plus_px'].to_numpy())
        frac_ok = float((align / scale < 1e-4).mean())
        print(f">> Row-alignment check (nu vs {label}) on true_tau_plus_px: "
              f"{100 * frac_ok:.4f}% of rows match")
        if frac_ok < 0.999:
            raise SystemExit(f"ERROR: {label} is not row-aligned with the nu-method file -- the "
                             "events in row i are not the same event. Comparison aborted.")

        # --- truth polarimetric vectors must also agree between the files
        for tau in ('taup', 'taun'):
            d = np.abs(vec3(nu, [f'ts_hh_{tau}_{a}' for a in AXES])
                       - vec3(dr, [f'true_ts_hh_{tau}_{a}' for a in AXES])).max()
            print(f">>   max |ts_hh_{tau} (nu) - true_ts_hh_{tau} ({label})| = {d:.2e}")

    dm_p, dm_n = dm_codes(nu, args.dm_level)
    if args.dm_level == 'reco':
        for label, dr in directs.items():
            agree = float((dr['reco_taup_DM'].to_numpy() == dm_p).mean())
            print(f">> Recomputed reco taup_DM agrees with {label}'s stored value for "
                  f"{100 * agree:.3f}% of events")

    w_even = nu['tauspinner_wt_alpha0'].to_numpy()
    w_odd = nu['tauspinner_wt_alpha90'].to_numpy()

    # ---------------- truth reference ----------------
    true_p3_p = vec3(nu, ['true_tau_plus_px', 'true_tau_plus_py', 'true_tau_plus_pz'])
    true_p3_m = vec3(nu, ['true_tau_minus_px', 'true_tau_minus_py', 'true_tau_minus_pz'])
    true_E_p = nu['true_tau_plus_E'].to_numpy()
    true_E_m = nu['true_tau_minus_E'].to_numpy()
    true_dir_p, true_dir_m = tau_dirs_in_ditau_frame(true_p3_p, true_E_p, true_p3_m, true_E_m)

    if args.truth == 'ts_hh':
        true_h_p = unit(vec3(nu, [f'ts_hh_taup_{a}' for a in AXES]))
        true_h_m = unit(vec3(nu, [f'ts_hh_taun_{a}' for a in AXES]))
    else:
        true_h_p, _, true_h_m, _ = nu_method_observables(nu, 'true', dm_p, dm_n)

    truth = {
        'h_p': true_h_p, 'h_m': true_h_m,
        'dir_p': true_dir_p, 'dir_m': true_dir_m,
        'taup_p3': true_p3_p, 'taum_p3': true_p3_m,
        'phiCP': phiCP_from_vectors(true_h_p, true_dir_p, true_h_m, true_dir_m),
        'ditau_mass': np.sqrt(np.clip((true_E_p + true_E_m) ** 2
                                      - ((true_p3_p + true_p3_m) ** 2).sum(1), 0, None)),
    }

    # ---------------- methods ----------------
    methods = {}

    print(f">> Building nu-method observables (tau_prefix={args.nu_tau_prefix})...")
    h_p, d_p, h_m, d_m = nu_method_observables(nu, args.nu_tau_prefix, dm_p, dm_n)
    p3_p = vec3(nu, [f'{args.nu_tau_prefix}_tau_plus_{c}' for c in ('px', 'py', 'pz')])
    p3_m = vec3(nu, [f'{args.nu_tau_prefix}_tau_minus_{c}' for c in ('px', 'py', 'pz')])
    E_p = nu[f'{args.nu_tau_prefix}_tau_plus_E'].to_numpy()
    E_m = nu[f'{args.nu_tau_prefix}_tau_minus_E'].to_numpy()
    methods['nu-method'] = {
        'h_p': h_p, 'h_m': h_m, 'dir_p': d_p, 'dir_m': d_m,
        'taup_p3': p3_p, 'taum_p3': p3_m,
        'phiCP': phiCP_from_vectors(h_p, d_p, h_m, d_m),
        'ditau_mass': np.sqrt(np.clip((E_p + E_m) ** 2 - ((p3_p + p3_m) ** 2).sum(1), 0, None)),
    }

    for label, dr in directs.items():
        print(f">> Building {label} observables...")
        h_p, d_p, h_m, d_m = direct_method_observables(dr)
        p3_p = vec3(dr, ['pred_tau_plus_px', 'pred_tau_plus_py', 'pred_tau_plus_pz'])
        p3_m = vec3(dr, ['pred_tau_minus_px', 'pred_tau_minus_py', 'pred_tau_minus_pz'])
        E_p = np.sqrt((p3_p ** 2).sum(1) + M_TAU ** 2)
        E_m = np.sqrt((p3_m ** 2).sum(1) + M_TAU ** 2)
        methods[label] = {
            'h_p': h_p, 'h_m': h_m, 'dir_p': d_p, 'dir_m': d_m,
            'taup_p3': p3_p, 'taum_p3': p3_m,
            'phiCP': phiCP_from_vectors(h_p, d_p, h_m, d_m),
            'ditau_mass': np.sqrt(np.clip((E_p + E_m) ** 2 - ((p3_p + p3_m) ** 2).sum(1), 0, None)),
        }

    # sanity: each direct file already stores its own true/pred phiCP -- recomputing
    # them here must reproduce those, otherwise this script's conventions are wrong.
    checks = [('truth', truth['phiCP'], next(iter(directs.values()))['true_phiCP'].to_numpy())]
    checks += [(f'{label} pred', methods[label]['phiCP'], dr['pred_phiCP'].to_numpy())
               for label, dr in directs.items()]
    for label, mine, stored in checks:
        d = np.abs((mine - stored + np.pi) % (2 * np.pi) - np.pi)
        print(f">> phiCP cross-check vs stored {label}: median |diff| = {np.median(d):.2e} rad, "
              f"frac < 1e-3 = {(d < 1e-3).mean():.4f}")

    # ---------------- categories ----------------
    valid = (dm_p >= 0) & (dm_n >= 0) & (dm_p != 100) & (dm_n != 100)
    for meth in methods.values():
        # get_ditau_polarimetric returns -9999 sentinels for unhandled DMs
        valid &= np.abs(meth['h_p']).max(axis=1) < 100
        valid &= np.abs(meth['h_m']).max(axis=1) < 100
    valid &= np.isfinite(truth['phiCP'])
    for meth in methods.values():
        valid &= np.isfinite(meth['phiCP'])
    print(f">> {valid.sum()} / {n} events usable in all methods")

    categories = {'inclusive': valid}
    for a, b in DM_PAIRS:
        mask = valid & (((dm_p == a) & (dm_n == b)) | ((dm_p == b) & (dm_n == a)))
        if mask.sum() >= args.min_events:
            categories[f'DM{a}-DM{b}'] = mask
    # per-leg categories: every event where AT LEAST ONE tau has this DM,
    # scored on that leg only -- this is where a per-DM regression like the
    # DM=10 one shows up most directly, without being diluted by the partner leg.
    leg_categories = {}
    for d in (0, 1, 2, 10, 11):
        leg_categories[f'leg DM{d}'] = d

    rows = []
    for cat, mask in categories.items():
        for name, meth in methods.items():
            r = metrics_for_mask(meth, truth, mask, w_even, w_odd)
            if r is None:
                continue
            r['category'] = cat
            r['method'] = name
            rows.append(r)
    table = pd.DataFrame(rows)
    cols = ['category', 'method'] + [c for c in table.columns if c not in ('category', 'method')]
    table = table[cols]

    # --- per-leg polarimetric-vector table (both legs pooled per DM)
    leg_rows = []
    for label, d in leg_categories.items():
        for name, meth in methods.items():
            cs, ns = [], 0
            for leg_dm, key in ((dm_p, 'h_p'), (dm_n, 'h_m')):
                m = valid & (leg_dm == d)
                if m.sum() == 0:
                    continue
                cs.append(cos_between(meth[key][m], truth[key][m]))
                ns += int(m.sum())
            if ns < args.min_events:
                continue
            c = np.concatenate(cs)
            leg_rows.append({'leg_category': label, 'method': name, 'N_legs': ns,
                             'h_meancos': float(c.mean()),
                             'h_medang_deg': float(np.degrees(np.arccos(np.median(c)))),
                             'h_frac_within_10deg': float((c > np.cos(np.radians(10))).mean())})
    leg_table = pd.DataFrame(leg_rows)

    # ---------------- output ----------------
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 100)

    headline = ['category', 'method', 'N', 'h_meancos', 'tau_dir_meancos', 'tau_pmag_iqr',
                'phiCP_meancos', 'phiCP_asym_pred', 'phiCP_asym_true', 'phiCP_asym_ratio']
    print("\n=== HEADLINE METRICS ===")
    print("h_meancos       : <cos(angle(pred h, true h))>, averaged over both legs -- the spin dilution factor (1 = perfect)")
    print("tau_dir_meancos : <cos(angle)> between predicted and true tau+ direction in the ditau rest frame")
    print("tau_pmag_iqr    : IQR of (|p_pred|-|p_true|)/|p_true|, averaged over both taus (lower is better)")
    print("phiCP_meancos   : <cos(phiCP_pred - phiCP_true)> (1 = perfect)")
    print("phiCP_asym_*    : CP-even vs CP-odd quadrature asymmetry of the phiCP distribution;")
    print("                  _true uses truth phiCP on the same events, so _ratio is the retained separation power\n")
    print(table[headline].to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    print("\n=== POLARIMETRIC VECTOR BY SINGLE-LEG DECAY MODE ===")
    if not leg_table.empty:
        print(leg_table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    table.to_csv(os.path.join(args.outdir, 'metrics_by_category.csv'), index=False)
    leg_table.to_csv(os.path.join(args.outdir, 'metrics_by_leg_dm.csv'), index=False)

    # --- head-to-head deltas, the thing actually being asked
    piv = table.pivot(index='category', columns='method')
    order = [c for c in categories if c in piv.index]
    deltas = []
    for label in directs:
        print(f"\n=== {label} minus nu-method (positive = direct is better, except *_iqr) ===")
        delta = pd.DataFrame({
            'N': piv[('N', 'nu-method')],
            'd_h_meancos': piv[('h_meancos', label)] - piv[('h_meancos', 'nu-method')],
            'd_tau_dir_meancos': piv[('tau_dir_meancos', label)] - piv[('tau_dir_meancos', 'nu-method')],
            'd_tau_pmag_iqr': piv[('tau_pmag_iqr', label)] - piv[('tau_pmag_iqr', 'nu-method')],
            'd_phiCP_meancos': piv[('phiCP_meancos', label)] - piv[('phiCP_meancos', 'nu-method')],
            'd_phiCP_asym_ratio': piv[('phiCP_asym_ratio', label)] - piv[('phiCP_asym_ratio', 'nu-method')],
        }).reindex(order)
        print(delta.to_string(float_format=lambda v: f'{v:+.4f}'))
        delta.insert(0, 'direct_model', label)
        deltas.append(delta)
    pd.concat(deltas).to_csv(os.path.join(args.outdir, 'delta_direct_minus_nu.csv'))

    # ---------------- plots ----------------
    print(f"\n>> Writing plots to {args.outdir}/")
    plotdir = args.outdir
    ang_bins = np.linspace(0, 180, 91)
    for cat, mask in categories.items():
        safe = cat.replace(' ', '_')
        series = {}
        for name, meth in methods.items():
            c = np.concatenate([cos_between(meth['h_p'][mask], truth['h_p'][mask]),
                                cos_between(meth['h_m'][mask], truth['h_m'][mask])])
            series[name] = np.degrees(np.arccos(c))
        plot_overlay_hist(series, ang_bins, 'angle(pred h, true h) [deg]',
                          f'Polarimetric vector resolution -- {cat}',
                          os.path.join(plotdir, f'polvec_angle_{safe}.pdf'))

        series = {name: ((meth['phiCP'][mask] - truth['phiCP'][mask] + np.pi) % (2 * np.pi) - np.pi)
                  for name, meth in methods.items()}
        plot_overlay_hist(series, np.linspace(-np.pi, np.pi, 81),
                          r'$\phi_{CP}^{pred} - \phi_{CP}^{true}$ [rad]',
                          f'phiCP resolution -- {cat}',
                          os.path.join(plotdir, f'phiCP_resolution_{safe}.pdf'))

        plot_phiCP_cp_curves(truth['phiCP'][mask],
                             {k: v['phiCP'][mask] for k, v in methods.items()},
                             w_even[mask], w_odd[mask],
                             f'phiCP, CP-even vs CP-odd -- {cat}',
                             os.path.join(plotdir, f'phiCP_cp_{safe}.pdf'))

        series = {}
        for name, meth in methods.items():
            rel = np.concatenate([
                (np.linalg.norm(meth['taup_p3'][mask], axis=1) - np.linalg.norm(truth['taup_p3'][mask], axis=1))
                / np.linalg.norm(truth['taup_p3'][mask], axis=1),
                (np.linalg.norm(meth['taum_p3'][mask], axis=1) - np.linalg.norm(truth['taum_p3'][mask], axis=1))
                / np.linalg.norm(truth['taum_p3'][mask], axis=1)])
            series[name] = rel
        plot_overlay_hist(series, np.linspace(-1, 1, 101),
                          r'$(|p|_{pred} - |p|_{true}) / |p|_{true}$',
                          f'Tau momentum resolution -- {cat}',
                          os.path.join(plotdir, f'tau_pmag_{safe}.pdf'))

    for metric, ylabel, better in (
        ('h_meancos', r'$\langle\cos\Delta\theta_h\rangle$', True),
        ('phiCP_meancos', r'$\langle\cos\Delta\phi_{CP}\rangle$', True),
        ('phiCP_asym_ratio', 'retained CP asymmetry (pred/true)', True),
        ('tau_pmag_iqr', 'IQR of relative tau |p| error', False),
    ):
        plot_metric_by_dm(table, metric, ylabel, metric,
                          os.path.join(plotdir, f'summary_{metric}.pdf'), higher_is_better=better)

    print(f">> Done. Tables: {args.outdir}/metrics_by_category.csv, metrics_by_leg_dm.csv, "
          f"delta_direct_minus_nu.csv")


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()
