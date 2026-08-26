import ROOT
ROOT.gROOT.SetBatch(True)
from numpy import arange
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from array import array
from taupolaris.utils.kinematic_helpers import EntanglementVariables
import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import mplhep as hep

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18,
                      "ytick.labelsize": 18, "legend.fontsize": 18})
# main.tex loads no special math-font package, so the paper's math is default
# LaTeX Computer Modern; 'cm' reproduces that here (the CMS style otherwise
# maps \mathcal etc. to its sans-serif text font, e.g. C wouldn't get the
# calligraphic C used in the paper for concurrence).
plt.rcParams['mathtext.fontset'] = 'cm'

# Collaborator's shared palette (models.tex figures).
bkg_fit_color = '#0072B2'   # Background (Z/gamma* -> tautau)   (== sampled_color)
sig_fit_color = '#e42536'   # Best-fit signal                    (== pred_color)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, help="Determines which model to assume for the signal. 1 = CP-even Higgs, 2 = spin0 no entanglement, 3 = uncorrelated", default=1)
parser.add_argument("--sig-file", default="outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/output_results.parquet", help="Sig file for the fully-hadronic (tt) channel.")
parser.add_argument("--bkg-file", default="outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/output_Z.parquet", help="Bkg file for the fully-hadronic (tt) channel.")
parser.add_argument("--sig-file-sl", default='outputs_Flow_Uncorr_Masked_Semileptonic_100e_July28/output_results.parquet', help="Sig file for the semileptonic (mt+et) channels -- mu+tauh and e+tauh both come from the same file, distinguished by reco_taup/taun_ismuon/iselectron. If not provided (together with --bkg-file-sl), mt/et categories are skipped.")
parser.add_argument("--bkg-file-sl", default='outputs_Flow_Uncorr_Masked_Semileptonic_100e_July28/output_Z.parquet', help="Bkg file for the semileptonic (mt+et) channels.")
parser.add_argument("--no-replace", action="store_true", help="Sample toys without replacement (non-overlapping chunks). Limits N_toys to the smallest category's pool_size/N_events.")
parser.add_argument("--n-toys", type=int, default=1000, help="Number of toys to generate. If --no-replace is set, this will be limited by the smallest category's pool.")
parser.add_argument("--outdir", default="nll_fits_aug7", help="Directory to write output files to.")
parser.add_argument("--measure-B", action="store_true", help="Also measure B vector (tau polarization) elements.")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)


_ij_map = {'nn':(0,0),'rr':(1,1),'nr':(0,1),'rn':(1,0),'kn':(2,0),'kr':(2,1),'nk':(0,2),'rk':(1,2),'kk':(2,2)}
_i_map = {'n': 0, 'r': 1, 'k': 2}
_axes = ['n', 'r', 'k']
_wt_cols = [f'wt_hp_{a}_hm_{b}' for a in _axes for b in _axes]
_B_wt_cols = [f'wt_hp_{a}' for a in _axes] + [f'wt_hm_{b}' for b in _axes]
Cij_elements = ["nn", "rr", "nr", "rn", "kn", "kr", "nk", "rk", "kk"]
B_elements = [('p','n'), ('p','r'), ('p','k'), ('m','n'), ('m','r'), ('m','k')]


def precompute_model_hists(df_sig, prod_vals, C_base, ii, jj, rho_vals, nbins, B_p_base=None, B_m_base=None):
    """
    For each value in rho_vals, set C_base[ii,jj]=rho, reweight df_sig, histogram prod_vals.
    B_p_base and B_m_base are held fixed during the Cij scan.
    Returns array of shape (len(rho_vals), nbins), normalised to sum=1.
    """
    model_hists = np.zeros((len(rho_vals), nbins), dtype=np.float64)
    C = C_base.copy()
    for k, rho in enumerate(rho_vals):
        C[ii, jj] = rho
        wt = compute_event_weights(df_sig, C, B_p_base, B_m_base)
        h, _ = np.histogram(prod_vals, bins=nbins, range=(-1, 1), weights=wt)
        h = h.astype(np.float64)
        if h.sum() > 0:
            h /= h.sum()
        model_hists[k] = h
    return model_hists


def precompute_model_hists_B(df_sig, var_vals, which, idx, rho_vals, nbins, C_base, B_p_base, B_m_base):
    """
    For B vector scanning: which='p' (tau+) or 'm' (tau-), idx=0/1/2 (n/r/k).
    Uses 1D histogram of cos_i+ or cos_j- weighted by the full spin weight.
    Returns array of shape (len(rho_vals), nbins), normalised to sum=1.
    """
    model_hists = np.zeros((len(rho_vals), nbins), dtype=np.float64)
    B_p = B_p_base.copy()
    B_m = B_m_base.copy()
    for k, rho in enumerate(rho_vals):
        if which == 'p':
            B_p[idx] = rho
        else:
            B_m[idx] = rho
        wt = compute_event_weights(df_sig, C_base, B_p, B_m)
        h, _ = np.histogram(var_vals, bins=nbins, range=(-1, 1), weights=wt)
        h = h.astype(np.float64)
        if h.sum() > 0:
            h /= h.sum()
        model_hists[k] = h
    return model_hists


def _nll_curve(data_counts, model_hists, rho_vals, N_sig_exp=None, N_bkg_exp=None, bkg_counts=None):
    """Per-rho -2*log(L) (up to an additive constant) for a single dataset/category."""
    if bkg_counts is not None and (N_sig_exp is None or N_bkg_exp is None):
        raise ValueError("If bkg_counts is provided, N_sig_exp and N_bkg_exp must be provided.")

    Nexp = (N_sig_exp + N_bkg_exp) if bkg_counts is not None else (N_sig_exp if N_sig_exp is not None else data_counts.sum())

    nll_values = np.full(len(rho_vals), np.inf)
    for k, (rho, pred) in enumerate(zip(rho_vals, model_hists)):
        if np.any(pred < 0):
            continue

        if bkg_counts is not None:
            combined = pred * N_sig_exp
            bkg_norm = bkg_counts * (N_bkg_exp / bkg_counts.sum() if bkg_counts.sum() > 0 else 0)
            combined = combined + bkg_norm
            total = combined.sum()
            if total > 0:
                combined /= total
            pred = combined

        mu = Nexp * pred
        n = data_counts
        mask = mu > 0
        nll_values[k] = 2.0 * np.sum(mu[mask] - n[mask] * np.log(mu[mask]))
    return nll_values


def _find_best_and_errors(nll_values, rho_vals):
    best_nll = nll_values.min()
    # When the NLL is flat over a range of rho values (degenerate model),
    # np.argmin returns the first occurrence (left edge). Instead take the midpoint.
    flat_mask = np.isclose(nll_values, best_nll, rtol=0, atol=1e-6)
    flat_indices = np.where(flat_mask)[0]
    i_best = flat_indices[len(flat_indices) // 2]
    best_rho = rho_vals[i_best]
    delta_nll = nll_values - best_nll

    left = rho_vals < best_rho
    right = rho_vals > best_rho

    rho_low = np.interp(1.0, delta_nll[left][::-1], rho_vals[left][::-1]) if left.any() else best_rho
    rho_high = np.interp(1.0, delta_nll[right], rho_vals[right]) if right.any() else best_rho

    if rho_low > rho_high:
        rho_low, rho_high = rho_high, rho_low

    return best_rho, rho_low, rho_high


def nll_scan_np(data_counts, model_hists, rho_vals, N_sig_exp=None, N_bkg_exp=None, bkg_counts=None):
    """NLL scan for a single dataset/category."""
    nll_values = _nll_curve(data_counts, model_hists, rho_vals, N_sig_exp, N_bkg_exp, bkg_counts)
    return _find_best_and_errors(nll_values, rho_vals)


def nll_scan_combined_np(category_terms, rho_vals):
    """
    Combined NLL scan across several categories (channels): each category
    contributes an independent Poisson likelihood term for the same physics
    parameter (a given Cij or B component, common across categories via the
    signal model), and the per-category -2*log(L) curves are summed at each
    scan point before finding the minimum -- a standard multi-channel
    combination.

    category_terms: list of dicts with keys 'data_counts', 'model_hists',
    'N_sig_exp', 'N_bkg_exp', 'bkg_counts' (same meaning as nll_scan_np's args,
    one entry per category).
    """
    total_nll = np.zeros(len(rho_vals))
    for term in category_terms:
        total_nll += _nll_curve(
            term['data_counts'], term['model_hists'], rho_vals,
            N_sig_exp=term.get('N_sig_exp'), N_bkg_exp=term.get('N_bkg_exp'),
            bkg_counts=term.get('bkg_counts'),
        )
    return _find_best_and_errors(total_nll, rho_vals)


def GetSpinCorrelationMatrix(phiCP):
    # return the expected spin correlation matrix for Higgs bosons production with different CP-mixing angles
    delta = np.radians(phiCP)

    C = np.array(
        [[np.cos(2*delta),np.sin(2*delta),0],
        [-np.sin(2*delta),np.cos(2*delta),0],
        [0,0,-1]], dtype=float
    )

    return C

def GetMatrixCoefficient(C, element):
    # element "nn" or "rr" or "nr" or "rn", "kn" etc...
    if element == "nn":
        return C[0,0]
    elif element == "rr":
        return C[1,1]
    elif element == "nr":
        return C[0,1]
    elif element == "rn":
        return C[1,0]
    elif element == "kn":
        return C[2,0]
    elif element == "kr":
        return C[2,1]
    elif element == "nk":
        return C[0,2]
    elif element == "rk":
        return C[1,2]
    elif element == "kk":
        return C[2,2]
    else:
        raise ValueError(f"Invalid element: {element}. Allowed elements are: nn, rr, nr, rn, kn, kr, nk, rk, kk")


def compute_event_weights(df, C, B_p=None, B_m=None):
    """Compute per-event spin weights from C matrix and optional B vectors."""
    wt = np.ones(len(df), dtype=np.float64)
    for i, a in enumerate(_axes):
        for j, b in enumerate(_axes):
            if C[i, j] != 0:
                wt += C[i, j] * df[f'wt_hp_{a}_hm_{b}'].values
    if B_p is not None:
        for i, a in enumerate(_axes):
            if B_p[i] != 0:
                wt += B_p[i] * df[f'wt_hp_{a}'].values
    if B_m is not None:
        for j, b in enumerate(_axes):
            if B_m[j] != 0:
                wt += B_m[j] * df[f'wt_hm_{b}'].values
    return np.clip(wt, 0, None)


def _reco_dm_mask(df, tau, dm):
    """Boolean mask for a single tau leg ('taup' or 'taun') matching a
    reco-level decay-mode label used in yields_run3: pi = 1-prong no pi0,
    rho = 1-prong 1 pi0, a11pr = 1-prong 2 pi0, a1 = 3-prong no pi0,
    mu/e = leptonic (muon/electron)."""
    if dm == 'pi':
        return (df[f'reco_{tau}_is3prong'] == 0) & (df[f'reco_{tau}_npizero'] == 0)
    elif dm == 'rho':
        return (df[f'reco_{tau}_is3prong'] == 0) & (df[f'reco_{tau}_npizero'] == 1)
    elif dm == 'a11pr':
        return (df[f'reco_{tau}_is3prong'] == 0) & (df[f'reco_{tau}_npizero'] == 2)
    elif dm == 'a1':
        return (df[f'reco_{tau}_is3prong'] == 1) & (df[f'reco_{tau}_npizero'] == 0)
    elif dm == 'mu':
        return df[f'reco_{tau}_ismuon'] == 1
    elif dm == 'e':
        return df[f'reco_{tau}_iselectron'] == 1
    else:
        raise ValueError(f"Unknown decay-mode label: {dm}")


def _dm_pair_mask(dm1, dm2):
    """Selection for a (dm1, dm2) decay-mode combination, symmetric in
    taup/taun since physical charge doesn't determine which leg has which
    decay mode."""
    return lambda df: (
        (_reco_dm_mask(df, 'taup', dm1) & _reco_dm_mask(df, 'taun', dm2)) |
        (_reco_dm_mask(df, 'taup', dm2) & _reco_dm_mask(df, 'taun', dm1))
    )


def _dm_pair_mask_any(*pairs):
    """Union of several (dm1, dm2) pair selections (see _dm_pair_mask) -- for
    yields_run3 bins that lump more than one decay-mode combination together,
    e.g. 'rhoa11pr' = (rho, a11pr) events plus (a11pr, a11pr) events."""
    def _mask(df):
        result = None
        for dm1, dm2 in pairs:
            m = _dm_pair_mask(dm1, dm2)(df)
            result = m if result is None else (result | m)
        return result
    return _mask


_no_gen_cut = lambda df: np.ones(len(df), dtype=bool)


# ============================================================================
# Category definitions
# ============================================================================
# Each category is an independent event selection (e.g. a decay-mode
# combination) with its own expected signal/background yield. The combined
# fit sums the per-category -2*log(L) at each scan point (see
# nll_scan_combined_np), so categories with more/better-constrained events
# pull the combined result harder -- a standard multi-channel combination.
#
# To add a category, append another dict below with its own gen_mask/reco_mask
# (boolean masks on the raw sig/bkg dataframes, combined with & internally) and
# its own N_sig/N_bkg (expected event yields for that category, e.g. from a
# luminosity scale-up of a data/MC yield table).
#
# Categories below are the fully-hadronic (tt) decay-mode combinations from
# `yields_run3`, using its lumi-scaled sig/bkg numbers (lum_scale=48.076923)
# directly as N_sig/N_bkg. No gen-level cut is needed/applied -- selection is
# reco-only, matching how the analysis categorizes data.
#
# Semileptonic (mt/et) categories from yields_run3 (murho, mupi, mua1,
# mua11pr, erho, epi, ea1, ea11pr) are not included yet: the current input
# files (--sig-file/--bkg-file) only contain fully-hadronic (hadhad) events.
# Adding them requires switching to semileptonic input files and reworking the
# cos-variable/weight-column handling for the leptonic leg -- left for later.
CATEGORIES = [
    # tt (fully hadronic) -- from yields_run3
    {'name': 'rhorho',   'channel': 'tt', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('rho', 'rho'),   'N_sig': 635, 'N_bkg': 2069},
    {'name': 'rhoa11pr', 'channel': 'tt', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask_any(('rho', 'a11pr'), ('a11pr', 'a11pr')), 'N_sig': 322, 'N_bkg': 1229},
    {'name': 'rhoa1',    'channel': 'tt', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('rho', 'a1'),    'N_sig': 631, 'N_bkg': 1632},
    {'name': 'a1a1',     'channel': 'tt', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('a1', 'a1'),     'N_sig': 164, 'N_bkg': 481},
    {'name': 'pirho',    'channel': 'tt', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('pi', 'rho'),    'N_sig': 400, 'N_bkg': 1263},
    {'name': 'pipi',     'channel': 'tt', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('pi', 'pi'),     'N_sig': 180, 'N_bkg': 1069},
    {'name': 'pia1',     'channel': 'tt', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('pi', 'a1'),     'N_sig': 233, 'N_bkg': 535},
    {'name': 'pia11pr',  'channel': 'tt', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('pi', 'a11pr'),  'N_sig': 93,  'N_bkg': 299},
    {'name': 'a11pra1',  'channel': 'tt', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('a11pr', 'a1'),  'N_sig': 137, 'N_bkg': 460},

    # mt (muon + hadronic tau) and et (electron + hadronic tau) -- from
    # yields_run3. Both come from the same semileptonic file (distinguished by
    # reco_taup/taun_ismuon/iselectron), so both use the 'sl' channel. Only
    # used if --sig-file-sl/--bkg-file-sl are both provided; otherwise these
    # are skipped automatically.
    {'name': 'murho',    'channel': 'sl', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('mu', 'rho'),   'N_sig': 2193, 'N_bkg': 11729},
    {'name': 'mupi',     'channel': 'sl', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('mu', 'pi'),    'N_sig': 859,  'N_bkg': 3894},
    {'name': 'mua1',     'channel': 'sl', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('mu', 'a1'),    'N_sig': 1144, 'N_bkg': 6940},
    {'name': 'mua11pr',  'channel': 'sl', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('mu', 'a11pr'), 'N_sig': 453,  'N_bkg': 3076},

    {'name': 'erho',     'channel': 'sl', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('e', 'rho'),    'N_sig': 1880, 'N_bkg': 18706},
    {'name': 'epi',      'channel': 'sl', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('e', 'pi'),     'N_sig': 684,  'N_bkg': 5419},
    {'name': 'ea1',      'channel': 'sl', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('e', 'a1'),     'N_sig': 1910, 'N_bkg': 28698},
    {'name': 'ea11pr',   'channel': 'sl', 'gen_mask': _no_gen_cut, 'reco_mask': _dm_pair_mask('e', 'a11pr'),  'N_sig': 428,  'N_bkg': 5082},
]

var_prefix = 'map_pred'

_cut_cols = [
    'reco_taup_npizero', 'reco_taun_npizero', 'reco_taup_is3prong', 'reco_taun_is3prong',
    'reco_taup_ismuon', 'reco_taun_ismuon', 'reco_taup_iselectron', 'reco_taun_iselectron',
]


def _get_file_columns(path):
    return set(pq.ParquetFile(path).schema.names)


def _swap_to_physical_charge(df, var_prefix):
    """For semileptonic eval outputs, the 'taup'/'tau_plus' labels mean the
    LEPTONIC tau (tau1), not the physical tau+ (see convert_semileptonic_df).
    The wt_hp_*/wt_hm_* spin weights ARE defined by physical charge though, so
    without correction the labels and weights are mismatched for the ~half of
    events where the lepton is physically the tau-. If the file carries the
    'taup_charge' column (added to the pipeline for exactly this purpose),
    swap all taup<->taun and tau_plus<->tau_minus columns for events where the
    labeled taup is physically negative, so that after this call the labels
    are true physical charge everywhere and match the weights."""
    if 'taup_charge' not in df.columns:
        return df
    flip = df['taup_charge'].values < 0
    if not flip.any():
        return df
    print(f"  restoring physical-charge labels using 'taup_charge': swapping legs for "
          f"{flip.sum()}/{len(df)} events where the labeled taup is physically tau-")
    swapped = set()
    for col in df.columns:
        if col in swapped:
            continue
        if 'taup' in col:
            partner = col.replace('taup', 'taun')
        elif 'tau_plus' in col:
            partner = col.replace('tau_plus', 'tau_minus')
        else:
            continue
        if partner not in df.columns:
            continue
        a = df[col].values.copy()
        b = df[partner].values.copy()
        a[flip], b[flip] = b[flip], a[flip].copy()
        df[col] = a
        df[partner] = b
        swapped.add(col)
        swapped.add(partner)
    return df


def _load_one(path, var_prefix, extra_cols):
    """Load a single parquet file, returning the cos{n,r,k}_{plus,minus}
    columns (for var_prefix) plus extra_cols.

    Always (re)derives the cos columns from raw kinematics via
    compute_spin_vars when the raw inputs are available, rather than trusting
    any cos{n,r,k}_{plus,minus} columns already cached in the file. This
    matters for a1 (3-prong) legs: files produced before the a1 polarimetric
    fix have cos values for those legs computed with the old, incorrect
    1-prong-only approximation, and simply reading the cached columns would
    silently keep using those stale values. reco_taup/taun_ishadronic +
    reco_taup/taun_charged also correctly handle a leptonically-decaying leg
    (see Evaluation_Tools.compute_spin_vars). Only falls back to the cached
    cos columns if the raw inputs needed to derive them aren't present.
    """
    cos_cols = [f'{var_prefix}_cos{a}_{s}' for a in _axes for s in ('plus', 'minus')]
    avail = _get_file_columns(path)

    raw_needed = (
        [f'{var_prefix}_tau_plus_{c}' for c in ('E', 'px', 'py', 'pz')] +
        [f'{var_prefix}_tau_minus_{c}' for c in ('E', 'px', 'py', 'pz')] +
        [f'reco_{tau}_{part}_{c}' for tau in ('taup', 'taun') for part in ('pi1', 'pi2', 'pi3', 'pizero1', 'charged') for c in ('E', 'px', 'py', 'pz')] +
        [f'reco_{tau}_haspizero' for tau in ('taup', 'taun')] +
        [f'reco_{tau}_ishadronic' for tau in ('taup', 'taun')] +
        [f'reco_{tau}_is3prong' for tau in ('taup', 'taun')]
    )
    if all(c in avail for c in raw_needed):
        needed = list(dict.fromkeys(raw_needed + extra_cols))
        if 'taup_charge' in avail:
            needed.append('taup_charge')
        df = pd.read_parquet(path, columns=needed)
        # A relabeled semileptonic file WITHOUT the charge column can't be
        # corrected -- warn loudly so stale files aren't used silently.
        if 'taup_charge' not in df.columns and (df['reco_taup_ishadronic'].values == 0).mean() > 0.9:
            print(f"  WARNING: {path} looks like a relabeled semileptonic file (taup is "
                  f"the leptonic tau in >90% of events) but has no 'taup_charge' column. "
                  f"Labels cannot be mapped back to physical charge, so the spin weights "
                  f"are mismatched for ~half the events (diluted sensitivity, B+/B- and "
                  f"CP-odd Cij mixing). Regenerate with the updated prepare+eval pipeline.")
        df = _swap_to_physical_charge(df, var_prefix)
        # imported lazily -- Evaluation_Tools.py pulls in torch at module scope
        from taupolaris.python.Evaluation_Tools import compute_spin_vars
        df = compute_spin_vars(df, tau_pred_prefix=f'{var_prefix}_', tau_vis_prefix='reco_')
        return df.reset_index(drop=True)

    if all(c in avail for c in cos_cols):
        print(f"  WARNING: raw kinematics not found in {path}, falling back to "
              f"cached cos columns for '{var_prefix}' -- these may be stale for a1 "
              f"(3-prong) legs if the file predates the a1 polarimetric fix.")
        needed = list(dict.fromkeys(cos_cols + extra_cols))
        return pd.read_parquet(path, columns=needed).reset_index(drop=True)

    raise ValueError(f"{path}: neither cos{{n,r,k}}_{{plus,minus}} columns nor the raw "
                      f"kinematics needed to derive them are present.")


def load_channel_files(sig_path, bkg_path, var_prefix, measure_B):
    sig_extra = _wt_cols + ((_B_wt_cols) if measure_B else []) + _cut_cols
    bkg_extra = _cut_cols
    df_sig = _load_one(sig_path, var_prefix, sig_extra)
    df_bkg = _load_one(bkg_path, var_prefix, bkg_extra)
    return df_sig, df_bkg


print("Loading parquet files...")
channel_files = {'tt': (args.sig_file, args.bkg_file)}
if args.sig_file_sl and args.bkg_file_sl:
    channel_files['sl'] = (args.sig_file_sl, args.bkg_file_sl)
else:
    print("--sig-file-sl/--bkg-file-sl not both provided: mt/et (semileptonic) categories will be skipped.")

channel_raw = {}
for ch, (sig_path, bkg_path) in channel_files.items():
    print(f"Loading channel '{ch}': sig={sig_path}, bkg={bkg_path}")
    df_sig_raw, df_bkg_raw = load_channel_files(sig_path, bkg_path, var_prefix, args.measure_B)
    channel_raw[ch] = (df_sig_raw, df_bkg_raw)
    print(f"  loaded {len(df_sig_raw)} sig events, {len(df_bkg_raw)} bkg events")


n_bins=20
step=0.01
vals = np.linspace(-1, 1, int(2/step) + 1)


if args.model == 1:
    print("Using CP-even Higgs model for signal")
    phiCP = 0
    C_rwt = GetSpinCorrelationMatrix(phiCP)
elif args.model == 2:
    print("Using spin-0 no entanglement model for signal")
    C_rwt = np.array(
            [[0,0,0],
            [0,0,0],
            [0,0,-1]], dtype=float
        )
elif args.model == 3:
    print("Using uncorrelated model for signal")
    C_rwt = np.array(
            [[0,0,0],
            [0,0,0],
            [0,0,0]], dtype=float
        )
else:
    raise ValueError(f"Invalid model: {args.model}. Allowed models are: 1 = CP-even Higgs, 2 = spin0 no entanglement, 3 = uncorrelated")


# B vectors (tau polarization) — zero for all current models
B_rwt_p = np.zeros(3, dtype=float)
B_rwt_m = np.zeros(3, dtype=float)


def build_category(cat_spec, df_sig_raw, df_bkg_raw, C_rwt, B_rwt_p, B_rwt_m, n_bins, vals, measure_B):
    """Apply a category's selection to the raw sig/bkg dataframes and precompute
    everything needed for the NLL scans (event weights, product/single-cos
    caches, per-rho model histograms, fixed background templates)."""
    name = cat_spec['name']
    sig_mask = cat_spec['gen_mask'](df_sig_raw) & cat_spec['reco_mask'](df_sig_raw)
    bkg_mask = cat_spec['gen_mask'](df_bkg_raw) & cat_spec['reco_mask'](df_bkg_raw)
    df_sig_cat = df_sig_raw[sig_mask].reset_index(drop=True)
    df_bkg_cat = df_bkg_raw[bkg_mask].reset_index(drop=True)
    print(f"Category '{name}': N_sig(expected)={cat_spec['N_sig']}, N_bkg(expected)={cat_spec['N_bkg']}, "
          f"pool: {len(df_sig_cat)} sig, {len(df_bkg_cat)} bkg events")

    base_C = C_rwt.copy()
    sig_model_wt = compute_event_weights(df_sig_cat, base_C, B_rwt_p, B_rwt_m)

    sig_prod_vals, bkg_prod_vals = {}, {}
    model_hists_per_Cij, fixed_bkg_Cij = {}, {}
    for Cij in Cij_elements:
        ii, jj = _ij_map[Cij]
        x_var = f'{var_prefix}_cos{Cij[0]}_plus'
        y_var = f'{var_prefix}_cos{Cij[1]}_minus'
        prod_sig = df_sig_cat[x_var].values * df_sig_cat[y_var].values
        prod_bkg = df_bkg_cat[x_var].values * df_bkg_cat[y_var].values
        sig_prod_vals[Cij] = prod_sig
        bkg_prod_vals[Cij] = prod_bkg
        model_hists_per_Cij[Cij] = precompute_model_hists(df_sig_cat, prod_sig, base_C, ii, jj, vals, n_bins, B_rwt_p, B_rwt_m)
        tmpl, _ = np.histogram(prod_bkg, bins=n_bins, range=(-1, 1))
        tmpl = tmpl.astype(np.float64)
        if tmpl.sum() > 0:
            tmpl *= cat_spec['N_bkg'] / tmpl.sum()
        fixed_bkg_Cij[Cij] = tmpl

    cat = {
        'name': name, 'N_sig': cat_spec['N_sig'], 'N_bkg': cat_spec['N_bkg'],
        'sig_model_wt': sig_model_wt,
        'sig_prod_vals': sig_prod_vals, 'bkg_prod_vals': bkg_prod_vals,
        'model_hists_per_Cij': model_hists_per_Cij, 'fixed_bkg_Cij': fixed_bkg_Cij,
        'n_sig_total': len(df_sig_cat), 'n_bkg_total': len(df_bkg_cat),
    }

    if measure_B:
        sig_B_vals, bkg_B_vals = {}, {}
        model_hists_per_B, fixed_bkg_B = {}, {}
        for which, ax in B_elements:
            key = (which, ax)
            var = f'{var_prefix}_cos{ax}_{"plus" if which == "p" else "minus"}'
            sig_B_vals[key] = df_sig_cat[var].values
            bkg_B_vals[key] = df_bkg_cat[var].values
            model_hists_per_B[key] = precompute_model_hists_B(
                df_sig_cat, sig_B_vals[key], which, _i_map[ax], vals, n_bins, base_C, B_rwt_p, B_rwt_m
            )
            tmpl, _ = np.histogram(bkg_B_vals[key], bins=n_bins, range=(-1, 1))
            tmpl = tmpl.astype(np.float64)
            if tmpl.sum() > 0:
                tmpl *= cat_spec['N_bkg'] / tmpl.sum()
            fixed_bkg_B[key] = tmpl
        cat.update({
            'sig_B_vals': sig_B_vals, 'bkg_B_vals': bkg_B_vals,
            'model_hists_per_B': model_hists_per_B, 'fixed_bkg_B': fixed_bkg_B,
        })

    return cat


print(f"\nBuilding categories (channels loaded: {list(channel_raw.keys())})...")
categories = []
for spec in CATEGORIES:
    ch = spec.get('channel', 'tt')
    if ch not in channel_raw:
        print(f"  Skipping category '{spec['name']}' (channel '{ch}' not loaded -- "
              f"provide --sig-file-{ch}/--bkg-file-{ch} to include it)")
        continue
    df_sig_raw, df_bkg_raw = channel_raw[ch]
    categories.append(build_category(spec, df_sig_raw, df_bkg_raw, C_rwt, B_rwt_p, B_rwt_m, n_bins, vals, args.measure_B))
print(f"Built {len(categories)} categories: {[c['name'] for c in categories]}")
del channel_raw


def _asimov_counts(sig_vals, bkg_vals, N_sig, N_bkg, weights_sig=None):
    h_sig, _ = np.histogram(sig_vals, bins=n_bins, range=(-1, 1), weights=weights_sig)
    h_sig = h_sig.astype(np.float64)
    if h_sig.sum() > 0:
        h_sig *= N_sig / h_sig.sum()
    h_bkg, _ = np.histogram(bkg_vals, bins=n_bins, range=(-1, 1))
    h_bkg = h_bkg.astype(np.float64)
    if h_bkg.sum() > 0:
        h_bkg *= N_bkg / h_bkg.sum()
    return h_sig + h_bkg, h_bkg


def _plot_model_shapes(model_hists, vals, xlabel, component_label, outpath, headroom=1.3):
    """Overlay the model-predicted (signal-only, background-free) shape at
    rho = -1, 0, +1 -- the extremes and midpoint of the reweighting -- to show
    how much the observable's shape actually changes with the fit parameter,
    i.e. the raw discriminating power of the reweighting for this
    variable/category, independent of the Asimov data or background.
    component_label is the bare mathtext Cij/B name (e.g. 'C_{nn}',
    'B^{+}_{n}', no surrounding $), used in the legend in place of the
    generic rho. headroom scales the y-axis max so the histogram peak stays
    clear of the legend (B components need more, since their legend text is
    longer)."""
    idx_m1 = 0
    idx_0 = int(np.argmin(np.abs(vals)))
    idx_p1 = len(vals) - 1

    edges = np.linspace(-1, 1, n_bins + 1)
    curves = [(model_hists[idx_m1], vals[idx_m1], bkg_fit_color),
              (model_hists[idx_0],  vals[idx_0],  'black'),
              (model_hists[idx_p1], vals[idx_p1], sig_fit_color)]

    fig, ax = plt.subplots(figsize=(8, 6))
    for hist, val, color in curves:
        ax.step(edges, np.append(hist, hist[-1]), where='post', color=color,
                linewidth=2, label=f'${component_label} = {val:.2f}$')

    ax.set_ylim(0, max(h.max() for h, _, _ in curves) * headroom)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Normalised')
    ax.legend(loc='upper right')

    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def _plot_asimov_overlay(data_counts, bkg_counts, pred_corr_np, xlabel, legend_lines, outpath, headroom=1.35):
    """Asimov data vs best-fit signal + background overlay for one category/element.
    headroom scales the y-axis max so the histogram content stays clear of the
    legend. The legend uses two columns -- data/background/signal on the
    left, the fit-result text on the right -- so it needs only as many rows
    as the taller column (3), not 3+len(legend_lines) (5) as a single column
    would; figsize is widened so the legend also has clear margin either side
    at 18pt font, rather than nearly spanning the full axes width."""
    edges = np.linspace(-1, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    data_err = np.sqrt(np.abs(data_counts))

    fig, ax = plt.subplots(figsize=(9.5, 6))
    # stairs() draws a single connected step outline -- a vertical segment
    # appears only where the bin content actually changes, not a full
    # rectangle border per bin (which is what bar() draws, and what produced
    # the "picket fence" of internal vertical lines between similar-height
    # bins in the previous version).
    ax.stairs(bkg_counts, edges, baseline=0, fill=True,
              facecolor=bkg_fit_color, edgecolor='black', linewidth=1.0, zorder=1)
    ax.stairs(bkg_counts + pred_corr_np, edges, baseline=bkg_counts, fill=True,
              facecolor=sig_fit_color, edgecolor='black', linewidth=1.0, zorder=1)
    # Using the returned ErrorbarContainer (rather than a bare marker) as the
    # legend handle so the legend swatch itself shows the error bar, not just
    # the point.
    data_handle = ax.errorbar(centers, data_counts, yerr=data_err, fmt='o', color='black',
                               markersize=6, linewidth=1.5, capsize=0, zorder=3,
                               label='Asimov data')

    stack_max = (bkg_counts + pred_corr_np).max()
    data_top = (data_counts + data_err).max()
    ax.set_ylim(0, max(stack_max, data_top) * headroom)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Counts')

    handles = [
        data_handle,
        Patch(facecolor=bkg_fit_color, edgecolor='black', linewidth=0.7, label='Background'),
        Patch(facecolor=sig_fit_color, edgecolor='black', linewidth=0.7, label='Best-fit signal'),
    ] + [Patch(color='none', label=line) for line in legend_lines]
    ax.legend(handles=handles, ncol=2, loc='upper center', borderaxespad=1.0)

    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


C_exp = C_rwt
measured_Cij_values = {}

print("\nRunning combined Asimov fit for each Cij (summed over categories)...")
for Cij in Cij_elements:
    cat_terms = []
    per_cat_hists = []  # (cat, data_counts, bkg_counts), kept for plotting below
    for cat in categories:
        data_counts, bkg_counts = _asimov_counts(
            cat['sig_prod_vals'][Cij], cat['bkg_prod_vals'][Cij], cat['N_sig'], cat['N_bkg'],
            weights_sig=cat['sig_model_wt'],
        )
        cat_terms.append({
            'data_counts': data_counts, 'model_hists': cat['model_hists_per_Cij'][Cij],
            'N_sig_exp': cat['N_sig'], 'N_bkg_exp': cat['N_bkg'], 'bkg_counts': bkg_counts,
        })
        per_cat_hists.append((cat, data_counts, bkg_counts))

    best_rho, rho_low, rho_high = nll_scan_combined_np(cat_terms, vals)
    print(f"C{Cij}: rho = {best_rho:.3f} (+{rho_high-best_rho:.3f}/-{best_rho-rho_low:.3f})  [combined over {len(categories)} categories]")
    measured_Cij_values[Cij] = (best_rho, rho_low, rho_high)

    # plotting: one plot per category (Asimov data vs best-fit signal + background for that category)
    i_best = np.argmin(np.abs(vals - best_rho))
    if args.model == 1:
        model_tag = f"fitted_phiCP{phiCP:.0f}"
    elif args.model == 2:
        model_tag = "fitted_spin0_noentanglement"
    else:
        model_tag = "fitted_uncorrelated"
    for cat, data_counts, bkg_counts in per_cat_hists:
        pred_corr_np = cat['model_hists_per_Cij'][Cij][i_best] * cat['N_sig']
        _plot_asimov_overlay(
            data_counts, bkg_counts, pred_corr_np,
            f"$\\cos\\theta_{{{Cij[0]}}}^{{+}}\\cos\\theta_{{{Cij[1]}}}^{{-}}$",
            [f"Combined-fit $C_{{{Cij}}} = {best_rho:.2f}^{{+{rho_high-best_rho:.2f}}}_{{-{best_rho-rho_low:.2f}}}$",
             f"True $C_{{{Cij}}} = {GetMatrixCoefficient(C_exp, Cij):.2f}$"],
            f"{args.outdir}/{cat['name']}_C{Cij}_{model_tag}.pdf",
        )

        _plot_model_shapes(
            cat['model_hists_per_Cij'][Cij], vals,
            f"$\\cos\\theta_{{{Cij[0]}}}^{{+}}\\cos\\theta_{{{Cij[1]}}}^{{-}}$",
            f"C_{{{Cij}}}",
            f"{args.outdir}/{cat['name']}_C{Cij}_shapes.pdf",
        )

print("\nMeasured Cij values (combined reweighting fit):")
for Cij, (Cij_val, Cij_low, Cij_high) in measured_Cij_values.items():
    print(f"C{Cij}: C{Cij} = {Cij_val:.3f} (+{Cij_high-Cij_val:.3f}/-{Cij_val-Cij_low:.3f})")
print("\nTrue Cij values:")
for Cij in Cij_elements:
    print(f"C{Cij}: C{Cij} = {GetMatrixCoefficient(C_exp, Cij):.3f}")

# --- B vector measurement ---
measured_B_values = {}
if args.measure_B:
    print("\nRunning combined Asimov fit for each B element (summed over categories)...")
    for which, ax in B_elements:
        key = (which, ax)
        cat_terms = []
        per_cat_hists_B = []  # (cat, data_counts, bkg_counts), kept for plotting below
        for cat in categories:
            data_counts, bkg_counts = _asimov_counts(
                cat['sig_B_vals'][key], cat['bkg_B_vals'][key], cat['N_sig'], cat['N_bkg'],
                weights_sig=cat['sig_model_wt'],
            )
            cat_terms.append({
                'data_counts': data_counts, 'model_hists': cat['model_hists_per_B'][key],
                'N_sig_exp': cat['N_sig'], 'N_bkg_exp': cat['N_bkg'], 'bkg_counts': bkg_counts,
            })
            per_cat_hists_B.append((cat, data_counts, bkg_counts))
        best_rho, rho_low, rho_high = nll_scan_combined_np(cat_terms, vals)
        measured_B_values[key] = (best_rho, rho_low, rho_high)
        label = f"B{'tau+' if which=='p' else 'tau-'}_{ax}"
        label_tex = f"B^{{{'+' if which=='p' else '-'}}}_{{{ax}}}"
        print(f"{label}: {best_rho:.3f} (+{rho_high-best_rho:.3f}/-{best_rho-rho_low:.3f})  [true=0, combined over {len(categories)} categories]")

        xlabel = f"$\\cos\\theta_{{{ax}}}^{{{'+' if which=='p' else '-'}}}$"
        i_best = np.argmin(np.abs(vals - best_rho))
        for cat, data_counts, bkg_counts in per_cat_hists_B:
            pred_corr_np = cat['model_hists_per_B'][key][i_best] * cat['N_sig']
            _plot_asimov_overlay(
                data_counts, bkg_counts, pred_corr_np, xlabel,
                [f"Combined-fit ${label_tex} = {best_rho:.2f}^{{+{rho_high-best_rho:.2f}}}_{{-{best_rho-rho_low:.2f}}}$",
                 f"True ${label_tex} = 0.00$"],
                f"{args.outdir}/{cat['name']}_B{which}{ax}_{model_tag}.pdf",
            )
            _plot_model_shapes(
                cat['model_hists_per_B'][key], vals, xlabel, label_tex,
                f"{args.outdir}/{cat['name']}_B{which}{ax}_shapes.pdf",
                headroom=1.6,
            )

    print("\nMeasured B values (combined reweighting fit):")
    for (which, ax), (B_val, B_low, B_high) in measured_B_values.items():
        label = f"B{'tau+' if which=='p' else 'tau-'}_{ax}"
        print(f"{label}: {label} = {B_val:.3f} (+{B_high-B_val:.3f}/-{B_val-B_low:.3f})")
    print("\nTrue B values: all 0.000")


C = np.array([[measured_Cij_values["nn"][0], measured_Cij_values["nr"][0], measured_Cij_values["nk"][0]],
              [measured_Cij_values["rn"][0], measured_Cij_values["rr"][0], measured_Cij_values["rk"][0]],
              [measured_Cij_values["kn"][0], measured_Cij_values["kr"][0], measured_Cij_values["kk"][0]]])
if args.measure_B:
    _Bp = np.array([[measured_B_values[('p',a)][0]] for a in _axes])
    _Bm = np.array([[measured_B_values[('m',a)][0]] for a in _axes])
    con, m12 = EntanglementVariables(C, Bplus=_Bp, Bminus=_Bm)
else:
    con, m12 = EntanglementVariables(C)

print("C:", C)
print("Con:", con)
print("m12:", m12)

#now to get the errors:
# we will loop over toys and, for each category, take a chunk of events (sampling a
# poisson with mean = N_sig/N_bkg for that category), fit each toy with a combined
# (summed-over-categories) NLL scan for each Cij/B element, compute concurrence, m12
# etc from the combined C matrix, and use these toys to get the errors on the
# measured quantities.
N_toys = args.n_toys
con_toys = []
m12_toys = []
Cnn_toys = []
B_toys = {(w, a): [] for w, a in B_elements} if args.measure_B else {}

print_every = 100

if args.no_replace:
    max_toys = N_toys
    for cat in categories:
        nonzero_mask = cat['sig_model_wt'] > 0
        max_toys_sig = int(nonzero_mask.sum()) // cat['N_sig'] if cat['N_sig'] > 0 else N_toys
        max_toys_bkg = cat['n_bkg_total'] // cat['N_bkg'] if cat['N_bkg'] > 0 else N_toys
        max_toys = min(max_toys, max_toys_sig, max_toys_bkg)
    if max_toys < N_toys:
        print(f"Warning: --no-replace limits toys to {max_toys} (smallest category's pool)")
        N_toys = max_toys
    for cat in categories:
        # pre-shuffle pool indices weighted by sig_model_wt for signal, only sampling
        # from events with non-zero weight
        nonzero_mask = cat['sig_model_wt'] > 0
        nonzero_idx = np.where(nonzero_mask)[0]
        nonzero_wt = cat['sig_model_wt'][nonzero_idx]
        n_needed = N_toys * cat['N_sig']
        cat['sig_pool'] = nonzero_idx[np.random.choice(len(nonzero_idx), size=min(n_needed, len(nonzero_idx)), replace=False, p=nonzero_wt/nonzero_wt.sum())]
        cat['bkg_pool'] = np.random.permutation(cat['n_bkg_total'])

# create output file with only toy_tree
fout = ROOT.TFile.Open(f"{args.outdir}/toys_tree.root", "RECREATE")
fout.cd()
toy_tree_out = ROOT.TTree("toy_tree", "toy_tree")
branches = [
  'iToy', 'con', 'm12',
  'Cnn', 'Cnr', 'Cnk',
  'Crn', 'Crr', 'Crk',
  'Ckn', 'Ckr', 'Ckk'
]
if args.measure_B:
    branches += ['Bpn', 'Bpr', 'Bpk', 'Bmn', 'Bmr', 'Bmk']
branch_vals = {}
for b in branches:
    branch_vals[b] = array('f', [0])
    toy_tree_out.Branch(b, branch_vals[b], '%s/F' % b)

for toy in range(N_toys):
    N_sig_toy_total = 0
    N_bkg_toy_total = 0
    per_cat_toy = []  # per-category sampled indices/weights/yields for this toy
    for cat in categories:
        N_sig_toy = np.random.poisson(cat['N_sig'])
        N_bkg_toy = np.random.poisson(cat['N_bkg'])
        N_sig_toy_total += N_sig_toy
        N_bkg_toy_total += N_bkg_toy

        if args.no_replace:
            sig_idx = cat['sig_pool'][toy * cat['N_sig'] : toy * cat['N_sig'] + N_sig_toy]
            bkg_idx = cat['bkg_pool'][toy * cat['N_bkg'] : toy * cat['N_bkg'] + N_bkg_toy]
            toy_sig_wt = cat['sig_model_wt'][sig_idx]
        else:
            sig_idx = np.random.choice(cat['n_sig_total'], size=N_sig_toy, replace=True, p=cat['sig_model_wt']/cat['sig_model_wt'].sum())
            bkg_idx = np.random.choice(cat['n_bkg_total'], size=N_bkg_toy, replace=True)
            toy_sig_wt = None

        per_cat_toy.append({
            'cat': cat, 'sig_idx': sig_idx, 'bkg_idx': bkg_idx, 'toy_sig_wt': toy_sig_wt,
            'N_sig_toy': N_sig_toy, 'N_bkg_toy': N_bkg_toy,
        })

    measured_B_values_toy = {}
    if args.measure_B:
        for which, ax in B_elements:
            key = (which, ax)
            cat_terms = []
            for pc in per_cat_toy:
                cat = pc['cat']
                var_vals = cat['sig_B_vals'][key]
                toy_sig_B, _ = np.histogram(var_vals[pc['sig_idx']], bins=n_bins, range=(-1, 1), weights=pc['toy_sig_wt'])
                toy_sig_B = toy_sig_B.astype(np.float64)
                if toy_sig_B.sum() > 0:
                    toy_sig_B *= pc['N_sig_toy'] / toy_sig_B.sum()

                bkg_var_vals = cat['bkg_B_vals'][key]
                if pc['N_bkg_toy'] > 0:
                    toy_bkg_B, _ = np.histogram(bkg_var_vals[pc['bkg_idx']], bins=n_bins, range=(-1, 1))
                    toy_bkg_B = toy_bkg_B.astype(np.float64)
                    if toy_bkg_B.sum() > 0:
                        toy_bkg_B_scaled = toy_bkg_B * (pc['N_bkg_toy'] / toy_bkg_B.sum())
                    else:
                        toy_bkg_B_scaled = np.zeros(n_bins, dtype=np.float64)
                else:
                    toy_bkg_B_scaled = np.zeros(n_bins, dtype=np.float64)

                toy_counts_B = toy_sig_B + toy_bkg_B_scaled
                cat_terms.append({
                    'data_counts': toy_counts_B, 'model_hists': cat['model_hists_per_B'][key],
                    'N_sig_exp': pc['N_sig_toy'], 'N_bkg_exp': pc['N_bkg_toy'],
                    'bkg_counts': cat['fixed_bkg_B'][key],
                })
            best_rho, _, _ = nll_scan_combined_np(cat_terms, vals)
            measured_B_values_toy[key] = best_rho

    measured_Cij_values_toy = {}
    for Cij in Cij_elements:
        cat_terms = []
        for pc in per_cat_toy:
            cat = pc['cat']
            sig_prod_all = cat['sig_prod_vals'][Cij]
            toy_sig_counts, _ = np.histogram(sig_prod_all[pc['sig_idx']], bins=n_bins, range=(-1, 1), weights=pc['toy_sig_wt'])
            toy_sig_counts = toy_sig_counts.astype(np.float64)
            if toy_sig_counts.sum() > 0:
                toy_sig_counts *= pc['N_sig_toy'] / toy_sig_counts.sum()

            if pc['N_bkg_toy'] > 0:
                bkg_prod_all = cat['bkg_prod_vals'][Cij]
                toy_bkg_counts, _ = np.histogram(bkg_prod_all[pc['bkg_idx']], bins=n_bins, range=(-1, 1))
                toy_bkg_counts = toy_bkg_counts.astype(np.float64)
                if toy_bkg_counts.sum() > 0:
                    toy_bkg_counts *= pc['N_bkg_toy'] / toy_bkg_counts.sum()
            else:
                toy_bkg_counts = np.zeros(n_bins, dtype=np.float64)

            toy_counts = toy_sig_counts + toy_bkg_counts
            cat_terms.append({
                'data_counts': toy_counts, 'model_hists': cat['model_hists_per_Cij'][Cij],
                'N_sig_exp': pc['N_sig_toy'], 'N_bkg_exp': pc['N_bkg_toy'],
                'bkg_counts': cat['fixed_bkg_Cij'][Cij],
            })
        best_rho, rho_low, rho_high = nll_scan_combined_np(cat_terms, vals)
        measured_Cij_values_toy[Cij] = best_rho  # rho = Cij directly

    C_toy = np.array([[measured_Cij_values_toy["nn"], measured_Cij_values_toy["nr"], measured_Cij_values_toy["nk"]],
                      [measured_Cij_values_toy["rn"], measured_Cij_values_toy["rr"], measured_Cij_values_toy["rk"]],
                      [measured_Cij_values_toy["kn"], measured_Cij_values_toy["kr"], measured_Cij_values_toy["kk"]]])
    if args.measure_B:
        _Bp_toy = np.array([[measured_B_values_toy.get(('p',a), 0.)] for a in _axes])
        _Bm_toy = np.array([[measured_B_values_toy.get(('m',a), 0.)] for a in _axes])
        con_toy, m12_toy = EntanglementVariables(C_toy, Bplus=_Bp_toy, Bminus=_Bm_toy)
    else:
        con_toy, m12_toy = EntanglementVariables(C_toy)
    con_toys.append(con_toy)
    m12_toys.append(m12_toy)
    Cnn_toys.append(C_toy[0,0])
    if args.measure_B:
        for w, a in B_elements:
            B_toys[(w, a)].append(measured_B_values_toy.get((w, a), 0.))

    branch_vals['iToy'][0] = toy
    branch_vals['con'][0] = con_toy
    branch_vals['m12'][0] = m12_toy
    branch_vals['Cnn'][0] = C_toy[0,0]
    branch_vals['Cnr'][0] = C_toy[0,1]
    branch_vals['Cnk'][0] = C_toy[0,2]
    branch_vals['Crn'][0] = C_toy[1,0]
    branch_vals['Crr'][0] = C_toy[1,1]
    branch_vals['Crk'][0] = C_toy[1,2]
    branch_vals['Ckn'][0] = C_toy[2,0]
    branch_vals['Ckr'][0] = C_toy[2,1]
    branch_vals['Ckk'][0] = C_toy[2,2]
    if args.measure_B:
        branch_vals['Bpn'][0] = measured_B_values_toy.get(('p','n'), 0.)
        branch_vals['Bpr'][0] = measured_B_values_toy.get(('p','r'), 0.)
        branch_vals['Bpk'][0] = measured_B_values_toy.get(('p','k'), 0.)
        branch_vals['Bmn'][0] = measured_B_values_toy.get(('m','n'), 0.)
        branch_vals['Bmr'][0] = measured_B_values_toy.get(('m','r'), 0.)
        branch_vals['Bmk'][0] = measured_B_values_toy.get(('m','k'), 0.)
    toy_tree_out.Fill()

    if toy % print_every == 0 or toy == N_toys - 1:
        print(f"------------------------------------------------")
        print(f"Toy {toy+1}/{N_toys}")
        print(f"N_sig_toy_events (all categories) = {N_sig_toy_total}, N_bkg_toy_events = {N_bkg_toy_total}")
        print(f"Con = {con_toy:.3f}, M12 = {m12_toy:.3f}")
        print(f"C = {C_toy}")

        # get uncertainties on con and m12 using toys (asymetric errors)

        con_toys_array = np.array(con_toys)
        m12_toys_array = np.array(m12_toys)

        con_err_low = np.percentile(con_toys_array, 16)
        con_err_high = np.percentile(con_toys_array, 84)

        m12_err_low = np.percentile(m12_toys_array, 16)
        m12_err_high = np.percentile(m12_toys_array, 84)

        con_mean = np.mean(con_toys_array)
        m12_mean = np.mean(m12_toys_array)

        con_median = np.median(con_toys_array)
        m12_median = np.median(m12_toys_array)

        Cnn_toys_array = np.array(Cnn_toys)

        Cnn_err_low = np.percentile(Cnn_toys_array, 16)
        Cnn_err_high = np.percentile(Cnn_toys_array, 84)
        Cnn_median = np.median(Cnn_toys_array)

        print(f"Mean Con from toys: {con_mean:.3f}")
        print(f"Mean M12 from toys: {m12_mean:.3f}")
        print(f"Median Con from toys: {con_median:.3f}")
        print(f"Median M12 from toys: {m12_median:.3f}")
        print(f"Concurrence 68% interval: {con_err_low:.3f} - {con_err_high:.3f}")
        print(f"M12 68% interval: {m12_err_low:.3f} - {m12_err_high:.3f}\n")

        azimov_Cnn = measured_Cij_values["nn"]
        print(f"Cnn  | Asimov: {azimov_Cnn[0]:.3f} (+{azimov_Cnn[2]-azimov_Cnn[0]:.3f}/-{azimov_Cnn[0]-azimov_Cnn[1]:.3f})  |  Toy median: {Cnn_median:.3f} ({Cnn_err_low-Cnn_median:.3f}/+{Cnn_err_high-Cnn_median:.3f})\n")

        if args.measure_B:
            print("B components (Asimov vs toy median):")
            B_labels = {'p': 'tau+', 'm': 'tau-'}
            for w, a in B_elements:
                az = measured_B_values[(w, a)]
                b_arr = np.array(B_toys[(w, a)])
                b_med = np.median(b_arr)
                b_lo  = np.percentile(b_arr, 16)
                b_hi  = np.percentile(b_arr, 84)
                print(f"  B{B_labels[w]}_{a} | Asimov: {az[0]:.3f} (+{az[2]-az[0]:.3f}/-{az[0]-az[1]:.3f})  |  Toy median: {b_med:.3f} ({b_lo-b_med:.3f}/+{b_hi-b_med:.3f})")
            print()

        fout.cd()
        toy_tree_out.Write("", ROOT.TObject.kOverwrite)
        fout.Flush()

fout.Close()
