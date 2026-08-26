import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import pandas as pd
import mplhep as hep
import numpy as np
import awkward as ak
import argparse
import os

import math
from taupolaris.utils.calculate_hh import Particle, _prepare_kinematic_for_hh, calculateHH, getHHVectors

from taupolaris.utils.acoplanarity_tools import (
    compute_aco_polarimetric,
    get_R_P_vectors_all,
    compute_aco_classic,
    compute_aco_classic_a1a1,
    get_ditau_polarimetric,
)

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18,
                      "ytick.labelsize": 18, "legend.fontsize": 18})

options = {
    'files':{  # set files here (ones from eval have all info we need)
#'even': 'outputs_TransformerFlows_originalHP_June/outputs_model_LHC_TransformerFlow_Hadronic_AllDMs_25e_June7/output_results_CPEven.parquet',
#'odd': 'outputs_TransformerFlows_originalHP_June/outputs_model_LHC_TransformerFlow_Hadronic_AllDMs_25e_June7/output_results_CPOdd.parquet',
#'sl_even': '/vols/cms/lcr119/offline/HiggsCP/DiTauEntanglement/outputs_model_LHC_TransformerFlow_Semileptonic_AllDMs_25e_June8/output_results_CPEven.parquet',
#'sl_odd': '/vols/cms/lcr119/offline/HiggsCP/DiTauEntanglement/outputs_model_LHC_TransformerFlow_Semileptonic_AllDMs_25e_June8/output_results_CPOdd.parquet',
'mix': None,
# single TauSpinner-reweightable ("uncorrelated") sample, used with --uncorrelated instead
# of separate even/odd/mix files -- e.g. an evaluate_polvec.py output parquet, which carries
# tauspinner_wt_alpha0/45/90 weight columns
'uncorr': 'outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/output_inc_error_large.parquet',
# 'uncorr': 'outputs_PlainTransformer_Uncorr_Masked_Hadronic_25e_July28/output_results.parquet',
#'sl_uncorr': 'outputs_Flow_Uncorr_Masked_Semileptonic_100e_July28/output_results.parquet',
},
    'gen': {
        'label': 'Generator Neutrino',
        'tag':   'POL_GEN',
    },
    'gen_ts': {
        'label': 'Generator (TauSpinner)',
        'tag':   'POL_GEN_TS',
    },
    'recoRun3': {
        'label': 'Approximate Methods',
        'tag':   'RecoRun3',
    },
    'recoNu': {
        'label': 'TauPolaris',
        'tag':   'RecoNu_Smeared',
    },
    'recoNu_ts': {
        'label': 'TauPolaris',
        'tag':   'RecoNu_Smeared_TS',
    },
    'recoNu_hybrid': {
        'label': 'TauPolaris',
        'tag':   'RecoNu_Smeared_Hybrid',
    },
    'recoPolvec': {
        'label': 'Polarimetric Vector (Flow, direct)',
        'tag':   'PolVecDirect',
    },
}

def _build_daughters_for_hh(row, side, tau_prefix='true', pion_prefix='true'):
    """Build Particle list for calculateHH from a dataframe row."""
    prefix = f'{pion_prefix}_{side}'
    decay_prefix = f'true_{side}'  # decay mode flags always from true
    is_had     = row[f'{decay_prefix}_ishadronic']
    is3prong   = row[f'{decay_prefix}_is3prong']
    npizero    = int(row[f'{decay_prefix}_npizero'])
    ismuon     = row[f'{decay_prefix}_ismuon']
    iselectron = row[f'{decay_prefix}_iselectron']

    nu_key    = f'{tau_prefix}_nubar' if side == 'taup' else f'{tau_prefix}_nu'
    nu_pdgid  = -16 if side == 'taup' else  16
    # For 1-prong: pi_pdgid is the single charged pion (same sign as tau)
    pi_pdgid  =  211 if side == 'taup' else -211
    pip_pdgid = -211 if side == 'taup' else  211
    # For 3-prong: parquet stores pi1=OS (opposite sign to tau), pi2=SS1, pi3=SS2.
    # TAUOLA damppk expects [nu, PIM1(SS), PIM2(SS), PIM3(OS)] — OS pion must be last.
    piOS_pdgid = -211 if side == 'taup' else  211   # pi1 in parquet
    piSS_pdgid =  211 if side == 'taup' else -211   # pi2, pi3 in parquet

    nu = Particle(row[f'{nu_key}_px'], row[f'{nu_key}_py'],
                  row[f'{nu_key}_pz'], row[f'{nu_key}_E'], nu_pdgid)

    def pi(n):
        return Particle(row[f'{prefix}_pi{n}_px'], row[f'{prefix}_pi{n}_py'],
                        row[f'{prefix}_pi{n}_pz'], row[f'{prefix}_pi{n}_E'], pi_pdgid)
    def pip():
        return Particle(row[f'{prefix}_pi2_px'], row[f'{prefix}_pi2_py'],
                        row[f'{prefix}_pi2_pz'], row[f'{prefix}_pi2_E'], pip_pdgid)
    def pi0(n=1):
        return Particle(row[f'{prefix}_pizero{n}_px'], row[f'{prefix}_pizero{n}_py'],
                        row[f'{prefix}_pizero{n}_pz'], row[f'{prefix}_pizero{n}_E'], 111)

    if is_had:
        if is3prong:
            p_os  = Particle(row[f'{prefix}_pi1_px'], row[f'{prefix}_pi1_py'],
                             row[f'{prefix}_pi1_pz'], row[f'{prefix}_pi1_E'], piOS_pdgid)
            p_ss1 = Particle(row[f'{prefix}_pi2_px'], row[f'{prefix}_pi2_py'],
                             row[f'{prefix}_pi2_pz'], row[f'{prefix}_pi2_E'], piSS_pdgid)
            p_ss2 = Particle(row[f'{prefix}_pi3_px'], row[f'{prefix}_pi3_py'],
                             row[f'{prefix}_pi3_pz'], row[f'{prefix}_pi3_E'], piSS_pdgid)
            daughters = [nu, p_ss1, p_ss2, p_os]
            if npizero >= 1:
                daughters.append(pi0())
            return daughters
        else:
            if npizero == 0: return [nu, pi(1)]
            if npizero == 1 or npizero == 2: return [nu, pi(1), pi0()]
            return None
    elif ismuon:
        mu_pdgid  = -13 if side == 'taup' else 13
        nul_pdgid =  14 if side == 'taup' else -14
        return [nu,
                Particle(row[f'{prefix}_pi1_px'], row[f'{prefix}_pi1_py'],
                         row[f'{prefix}_pi1_pz'], row[f'{prefix}_pi1_E'], mu_pdgid),
                Particle(row[f'{prefix}_pizero1_px'], row[f'{prefix}_pizero1_py'],
                         row[f'{prefix}_pizero1_pz'], row[f'{prefix}_pizero1_E'], nul_pdgid)]
    elif iselectron:
        e_pdgid   = -11 if side == 'taup' else 11
        nul_pdgid =  12 if side == 'taup' else -12
        return [nu,
                Particle(row[f'{prefix}_pi1_px'], row[f'{prefix}_pi1_py'],
                         row[f'{prefix}_pi1_pz'], row[f'{prefix}_pi1_E'], e_pdgid),
                Particle(row[f'{prefix}_pizero1_px'], row[f'{prefix}_pizero1_py'],
                         row[f'{prefix}_pizero1_pz'], row[f'{prefix}_pizero1_E'], nul_pdgid)]
    return None


def _hh_to_higgs_rf(hh3, tau_part, boson_part):
    """
    Convert a TauSpinner HH polarimetric vector to Higgs-rest-frame Cartesian
    coordinates, matching the frame used by get_ditau_polarimetric.

    TauSpinner computes HH in a rotated frame where the tau points to -Z.
    This function undoes that rotation by recovering the tau's polar angles
    (phi, theta) in the Higgs RF and applying the inverse rotations.
    """
    tau_c = tau_part.copy()
    bos_c = boson_part.copy()
    tau_c.boostToRestFrame(bos_c)          # boost tau to Higgs RF
    phi   = tau_c.getAnglePhi()
    tau_c.rotateXY(-phi)
    theta = tau_c.getAngleTheta()          # polar angle of tau in Higgs RF
    # Undo the rotation: inverse of [rotateXY(-phi), rotateXZ(pi-theta)]
    h = Particle(hh3[0], hh3[1], hh3[2], 0.0, 0)
    h.rotateXZ(theta - math.pi)
    h.rotateXY(phi)
    return np.array([h.px(), h.py(), h.pz()])


def _run_hh_loop(df, n_events=None, tau_prefix='true', pion_prefix='true', fix_tau_mass=False, event_mask=None):
    """
    Run calculateHH for each event and return Higgs-RF polarimetric vectors.

    tau_prefix:  prefix for tau/neutrino 4-momenta ('true', 'map_pred', 'pred')
    pion_prefix: prefix for pion 4-momenta ('true' or 'reco')
    event_mask:  optional boolean array of length len(df); if given, only events where
                 event_mask is True are processed. hh_p/hh_m are still returned with
                 length len(df), with NaN for unprocessed events.

    Returns (hh_p, hh_m, dm_p_arr, dm_m_arr) where hh_p/hh_m are (n, 3) float arrays
    with NaN for events where calculateHH failed or the decay mode is unsupported.
    """
    df_full = df.iloc[:n_events].reset_index(drop=True) if n_events is not None else df.reset_index(drop=True)
    n_full = len(df_full)
    dm_p_arr = np.array(df_full['taup_DM'])
    dm_m_arr = np.array(df_full['taun_DM'])

    hh_p_full = np.full((n_full, 3), np.nan)
    hh_m_full = np.full((n_full, 3), np.nan)

    if event_mask is not None:
        event_mask = np.asarray(event_mask, dtype=bool)
        df_sub = df_full[event_mask].reset_index(drop=True)
        subset_indices = np.where(event_mask)[0]
    else:
        df_sub = df_full
        subset_indices = np.arange(n_full)

    n = len(df_sub)

    tau_p_pdg = -15
    tau_m_pdg =  15

    _MTAU = 1.77682  # physical tau mass in GeV

    def _enforce_tau_mass(tau_part, pdgid):
        """Return tau_part with energy set to sqrt(|p|^2 + m_tau^2), keeping 3-momentum fixed."""
        pmag2 = tau_part.px()**2 + tau_part.py()**2 + tau_part.pz()**2
        E_phys = math.sqrt(pmag2 + _MTAU**2)
        return Particle(tau_part.px(), tau_part.py(), tau_part.pz(), E_phys, pdgid)

    print(f"  Running calculateHH loop over {n} events (tau_prefix={tau_prefix}, pion_prefix={pion_prefix})...", flush=True)
    for i, (_, row) in enumerate(df_sub.iterrows()):
        if n > 5000 and i % 10000 == 0:
            print(f"  {i}/{n}", flush=True)
        dau_p = _build_daughters_for_hh(row, 'taup', tau_prefix=tau_prefix, pion_prefix=pion_prefix)
        dau_m = _build_daughters_for_hh(row, 'taun', tau_prefix=tau_prefix, pion_prefix=pion_prefix)
        if dau_p is None or dau_m is None:
            continue
        tau_p_part = Particle(row[f'{tau_prefix}_tau_plus_px'],  row[f'{tau_prefix}_tau_plus_py'],
                              row[f'{tau_prefix}_tau_plus_pz'],  row[f'{tau_prefix}_tau_plus_E'],  tau_p_pdg)
        tau_m_part = Particle(row[f'{tau_prefix}_tau_minus_px'], row[f'{tau_prefix}_tau_minus_py'],
                              row[f'{tau_prefix}_tau_minus_pz'], row[f'{tau_prefix}_tau_minus_E'], tau_m_pdg)
        if fix_tau_mass:
            tau_p_part = _enforce_tau_mass(tau_p_part, tau_p_pdg)
            tau_m_part = _enforce_tau_mass(tau_m_part, tau_m_pdg)
        boson = Particle(row[f'{tau_prefix}_tau_plus_px']  + row[f'{tau_prefix}_tau_minus_px'],
                         row[f'{tau_prefix}_tau_plus_py']  + row[f'{tau_prefix}_tau_minus_py'],
                         row[f'{tau_prefix}_tau_plus_pz']  + row[f'{tau_prefix}_tau_minus_pz'],
                         row[f'{tau_prefix}_tau_plus_E']   + row[f'{tau_prefix}_tau_minus_E'], 25)
        try:
            HHp_vec, _, HHm_vec, _ = getHHVectors(boson, tau_p_part, tau_m_part, dau_p, dau_m)
            full_i = subset_indices[i]
            hh_p_full[full_i] = _hh_to_higgs_rf(HHp_vec[:3], tau_p_part, boson)
            hh_m_full[full_i] = _hh_to_higgs_rf(HHm_vec[:3], tau_m_part, boson)
        except Exception:
            pass

    return hh_p_full, hh_m_full, dm_p_arr, dm_m_arr


def compute_phicp_all(df, option, use_map=True, output_dir='.'):
    # Compute phiCP for all events in the df (splitting of methods by DM done automatically, vectorised)
    df = df.copy()
    if option == 'gen':
        R1, P1, R2, P2 = get_ditau_polarimetric(df, tau_prefix='true', reco_pions=True)
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'gen_ts':
        # Use true tau directions for P1/P2 (same as gen), polarimetric vectors from TauSpinner
        _, P1, _, P2 = get_ditau_polarimetric(df, tau_prefix='true', reco_pions=False)
        hh_p, hh_m, _, _ = _run_hh_loop(df)
        R1 = ak.zip({"x": hh_p[:, 0], "y": hh_p[:, 1], "z": hh_p[:, 2]}, with_name="Vector3D")
        R2 = ak.zip({"x": hh_m[:, 0], "y": hh_m[:, 1], "z": hh_m[:, 2]}, with_name="Vector3D")
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'recoNu':
        tau_prefix = 'map_pred' if use_map else 'pred'
        R1, P1, R2, P2 = get_ditau_polarimetric(df, tau_prefix=tau_prefix, reco_pions=True)
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'recoNu_ts':
        tau_prefix = 'map_pred' if use_map else 'pred'
        R1_old, P1, R2_old, P2 = get_ditau_polarimetric(df, tau_prefix=tau_prefix, reco_pions=True)
        hh_p, hh_m, _, _ = _run_hh_loop(df, tau_prefix=tau_prefix, pion_prefix='reco', fix_tau_mass=False) # fix_tau_mass=True - needs to be studies more 
        nan_p = np.isnan(hh_p[:, 0])
        nan_m = np.isnan(hh_m[:, 0])
        if nan_p.any() or nan_m.any():
            print(f"  recoNu_ts: falling back to recoNu for {nan_p.sum()} tau+ and {nan_m.sum()} tau- events where calculateHH failed")
        hh_p[nan_p] = np.stack([np.array(R1_old.x)[nan_p], np.array(R1_old.y)[nan_p], np.array(R1_old.z)[nan_p]], axis=1)
        hh_m[nan_m] = np.stack([np.array(R2_old.x)[nan_m], np.array(R2_old.y)[nan_m], np.array(R2_old.z)[nan_m]], axis=1)
        R1 = ak.zip({"x": hh_p[:, 0], "y": hh_p[:, 1], "z": hh_p[:, 2]}, with_name="Vector3D")
        R2 = ak.zip({"x": hh_m[:, 0], "y": hh_m[:, 1], "z": hh_m[:, 2]}, with_name="Vector3D")
        #TODO need to study how to deal with the nans properly, for now we just use the old R1/R2 for those events
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'recoNu_hybrid':
        tau_prefix = 'map_pred' if use_map else 'pred'
        R1, P1, R2, P2 = get_ditau_polarimetric(df, tau_prefix=tau_prefix, reco_pions=True)
        dm_p_arr = np.array(df['taup_DM'])
        dm_m_arr = np.array(df['taun_DM'])
        needs_ts = (dm_p_arr == 11) | (dm_m_arr == 11)
        hh_p, hh_m, _, _ = _run_hh_loop(df, tau_prefix=tau_prefix, pion_prefix='reco', fix_tau_mass=False, event_mask=needs_ts)
        r1_arr = np.stack([np.array(R1.x), np.array(R1.y), np.array(R1.z)], axis=1)
        r2_arr = np.stack([np.array(R2.x), np.array(R2.y), np.array(R2.z)], axis=1)
        nan_p = np.isnan(hh_p[:, 0])
        nan_m = np.isnan(hh_m[:, 0])
        # Replace each leg independently: only swap if that leg is DM=11
        use_ts_p = (dm_p_arr == 11) & ~nan_p
        use_ts_m = (dm_m_arr == 11) & ~nan_m
        r1_arr[use_ts_p] = hh_p[use_ts_p]
        r2_arr[use_ts_m] = hh_m[use_ts_m]
        R1 = ak.zip({"x": r1_arr[:, 0], "y": r1_arr[:, 1], "z": r1_arr[:, 2]}, with_name="Vector3D")
        R2 = ak.zip({"x": r2_arr[:, 0], "y": r2_arr[:, 1], "z": r2_arr[:, 2]}, with_name="Vector3D")
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'recoRun3':
        R1, P1, leg1_is_dp = get_R_P_vectors_all(df, tau_prefix='taup', use_map=use_map)
        R2, P2, leg2_is_dp = get_R_P_vectors_all(df, tau_prefix='taun', use_map=use_map)
        phiCPmain = compute_aco_classic(R1, P1, R2, P2, leg1_is_dp, leg2_is_dp)
        phiCPa1a1 = compute_aco_classic_a1a1(df)
        phiCP = np.where((df['taup_DM'] == 10) & (df['taun_DM'] == 10), phiCPa1a1, phiCPmain)
    elif option == 'recoPolvec':
        # phiCP already computed by evaluate_polvec.py -- just read it, no calculateHH needed
        phiCP = df['pred_phiCP'].values
    df['phiCP'] = np.array(phiCP)
    return df

def add_DM(df, dm_prefix='reco'):
    for tau in ['taup', 'taun']:
        tau_is_lep = df[f'{dm_prefix}_{tau}_ishadronic'].values == 0
        tau_is_dm0 = (df[f"{dm_prefix}_{tau}_npizero"].values == 0) & (df[f'{dm_prefix}_{tau}_is3prong'] == 0) & (~tau_is_lep)
        tau_is_dm1 = (df[f"{dm_prefix}_{tau}_npizero"].values == 1) & (df[f'{dm_prefix}_{tau}_is3prong'] == 0) & (~tau_is_lep)
        tau_is_dm2 = ((df[f"{dm_prefix}_{tau}_npizero"].values == 1) | (df[f"{dm_prefix}_{tau}_npizero"].values == 2)) & (df[f'{dm_prefix}_{tau}_is3prong'] == 0) & (~tau_is_lep)
        tau_is_dm10 = (df[f"{dm_prefix}_{tau}_npizero"].values == 0) & (df[f'{dm_prefix}_{tau}_is3prong'] == 1) & (~tau_is_lep)
        tau_is_dm11 = (df[f"{dm_prefix}_{tau}_npizero"].values == 1) & (df[f'{dm_prefix}_{tau}_is3prong'] == 1) & (~tau_is_lep)
        df[f'{tau}_DM'] = np.where(tau_is_dm0, 0,
                             np.where(tau_is_dm1, 1,
                                      np.where(tau_is_dm2, 2,
                                               np.where(tau_is_dm10, 10, 
                                                    np.where(tau_is_dm11, 11,
                                                        np.where(tau_is_lep, 100, -1))))))
    return df

def add_or_get_DM(df, dm_prefix='reco'):
    """Sets df['taup_DM']/df['taun_DM']. evaluate_polvec.py's output parquets (used with
    --option recoPolvec and/or --uncorrelated) already carry precomputed DM codes as
    '{gen,reco}_taup_DM'/'{gen,reco}_taun_DM' (note: 'gen' not 'true' for the truth-level
    prefix, unlike this script's own raw-flag convention) -- use those directly if present,
    otherwise fall back to computing from raw ishadronic/npizero/is3prong flags as before."""
    alt_prefix = {'true': 'gen', 'gen': 'true'}.get(dm_prefix, dm_prefix)
    for pfx in (dm_prefix, alt_prefix):
        taup_col, taun_col = f'{pfx}_taup_DM', f'{pfx}_taun_DM'
        if taup_col in df.columns and taun_col in df.columns:
            df['taup_DM'] = df[taup_col]
            df['taun_DM'] = df[taun_col]
            return df
    return add_DM(df, dm_prefix=dm_prefix)

class HandlerStepLine(HandlerPatch):
    """Draws step-histogram legend entries as a line with a shaded band, matching the
    line+fill_between style used on the axes instead of matplotlib's default box."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        color = orig_handle.get_edgecolor()
        band = Rectangle((xdescent, ydescent), width, height, facecolor=color, edgecolor='none',
                          alpha=0.25, transform=trans)
        line = Line2D([xdescent, xdescent + width], [ydescent + height / 2, ydescent + height / 2],
                      color=color, linewidth=2, transform=trans)
        return [band, line]


def plot_phicp_histogram(ax, data, bin_edges, variable, label, color, hide_errors=False, weights=None):

    step_x = np.repeat(bin_edges, 2)[1:-1]
    N = len(data[variable])

    # Use 1/N weights instead of densitiy
    if weights is None:
        w = np.ones(N) / N
    else:
        w = np.asarray(weights)
        w = w / w.sum() # check tauspinner weights are normalised to 1


    counts, _ = np.histogram(data[variable], bins=bin_edges, weights=w)
    ax.hist(data[variable], bins=bin_edges, histtype='step', label=label,
            density=False, linewidth=2, color=color, weights=w)

    if not hide_errors:
        w2_counts, _ = np.histogram(data[variable], bins=bin_edges, weights=w**2)
        err = np.sqrt(w2_counts)
        ax.fill_between(step_x, np.repeat(counts - err, 2),
                        np.repeat(counts + err, 2), alpha=0.25, color=color)
    return counts


def compute_vis_pt(df, prefix):
    taup_px = df[f'reco_taup_charged_px'] + df[f'reco_taup_pizero1_px']
    taun_px = df[f'reco_taun_charged_px'] + df[f'reco_taun_pizero1_px']
    taup_py = df[f'reco_taup_charged_py'] + df[f'reco_taup_pizero1_py']
    taun_py = df[f'reco_taun_charged_py'] + df[f'reco_taun_pizero1_py']
    return np.sqrt((taup_px + taun_px)**2 + (taup_py + taun_py)**2)


def load_data(prefix='', extra_pt_cut=-1, uncorrelated=False):
    cfg = options['files']
    read = pd.read_parquet

    if uncorrelated:
        uncorr_df = read(cfg[f'{prefix}uncorr'])
        print(f'UNCORRELATED File: {cfg[f"{prefix}uncorr"]}')
        if extra_pt_cut > 0:
            uncorr_df['vis_pt'] = compute_vis_pt(uncorr_df, prefix)
            uncorr_df = uncorr_df[uncorr_df['vis_pt'] > extra_pt_cut]
        return uncorr_df

    mix_df = read(cfg[f'{prefix}mix']) if cfg.get(f'{prefix}mix') is not None else None
    zprime_df = read(cfg[f'{prefix}Zprime']) if cfg.get(f'{prefix}Zprime') is not None else None
    even_df = read(cfg[f'{prefix}even'])
    print(f'EVEN File: {cfg[f"{prefix}even"]}')
    odd_df = read(cfg[f'{prefix}odd'])
    print(f'ODD File: {cfg[f"{prefix}odd"]}')
    if mix_df is not None:
        print(f'MIX File: {cfg[f"{prefix}mix"]}')
    if zprime_df is not None:
        print(f'ZPRIME File: {cfg[f"{prefix}Zprime"]}')
    # estimate visible pT from sum of true_taun_charged_px true_taun_pizero1_px, etc and apply cut if extra_pt_cut>0
    if extra_pt_cut > 0:
        even_df['vis_pt'] = compute_vis_pt(even_df, prefix)
        odd_df['vis_pt'] = compute_vis_pt(odd_df, prefix)
        even_df = even_df[even_df['vis_pt'] > extra_pt_cut]
        odd_df = odd_df[odd_df['vis_pt'] > extra_pt_cut]
        if mix_df is not None:
            mix_df['vis_pt'] = compute_vis_pt(mix_df, prefix)
            mix_df = mix_df[mix_df['vis_pt'] > extra_pt_cut]
        if zprime_df is not None:
            zprime_df['vis_pt'] = compute_vis_pt(zprime_df, prefix)
            zprime_df = zprime_df[zprime_df['vis_pt'] > extra_pt_cut]
    return even_df, odd_df, mix_df, zprime_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--option', choices=['gen', 'gen_ts', 'recoRun3', 'recoNu', 'recoNu_ts', 'recoNu_hybrid', 'recoPolvec'],
                        default='gen', help="Reconstruction method to use.")
    parser.add_argument('--output-dir', default='.', help="Directory for output PDFs.")
    parser.add_argument('--useMLP', action='store_true')
    parser.add_argument('--GENfilter', action='store_true',
                        help="Use true_ prefix for DM/prong masks instead of reco_.")
    parser.add_argument('--hide-errors', action='store_true',
                        help="Hide Poisson error bands on the bins (shown by default).")
    parser.add_argument('--leptonic_mode', default=0, type=int, choices=[0,1,2],
                        help="If 0 use hadronic decay modes, for 1 use semileptonic, for 2 use fully leptonic (not currently supported).")
    parser.add_argument('--uncorrelated', action='store_true',
                        help="Load a single TauSpinner-reweightable ('uncorr') sample instead of "
                             "separate even/odd/mix files, and use tauspinner_wt_alpha0/90/45 as "
                             "histogram weights for the CP-even/CP-odd/CP-mix curves.")

    args = parser.parse_args()

    if args.leptonic_mode == 2:
        raise NotImplementedError("Fully leptonic mode not currently supported.")

    do_DM10= False

    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.option}", exist_ok=True)

    prefix = 'sl_' if args.leptonic_mode == 1 else ''
    use_map = not args.useMLP
    dm_pfx = 'true' if args.GENfilter else 'reco'

    if args.uncorrelated:
        phi_error_cut = 1.4 # ~80 degree error
        uncorr_df = load_data(prefix=prefix, uncorrelated=True)
        #uncorr_df = uncorr_df[(uncorr_df['pred_phiCP_err'] > phi_error_cut)]
        uncorr_df = add_or_get_DM(uncorr_df, dm_prefix=dm_pfx)
        uncorr_df = compute_phicp_all(uncorr_df, args.option, use_map=use_map, output_dir=args.output_dir)
        even_df = odd_df = mix_df = uncorr_df
        zprime_df = None
    else:
        even_df, odd_df, mix_df, zprime_df = load_data(prefix=prefix)
        even_df = add_or_get_DM(even_df, dm_prefix=dm_pfx)
        odd_df = add_or_get_DM(odd_df, dm_prefix=dm_pfx)
        even_df = compute_phicp_all(even_df, args.option, use_map=use_map, output_dir=args.output_dir)
        odd_df  = compute_phicp_all(odd_df,  args.option, use_map=use_map, output_dir=args.output_dir)
        if mix_df is not None:
            mix_df = add_or_get_DM(mix_df, dm_prefix=dm_pfx)
            mix_df = compute_phicp_all(mix_df, args.option, use_map=use_map, output_dir=args.output_dir)
        if zprime_df is not None:
            zprime_df = add_or_get_DM(zprime_df, dm_prefix=dm_pfx)
            zprime_df = compute_phicp_all(zprime_df, args.option, use_map=use_map)

    if args.leptonic_mode == 1:
        dm_combs = [[100, 0], [100,1], [100,2], [100,10]]
        if args.option in ('gen_ts', 'recoNu_ts', 'recoNu_hybrid'):
            dm_combs += [[100, 11]]
    else: 
        dm_combs = [[0, 0]] #, [0,1], [1,1], [2,2], [1,2], [0,2], [10,10], [0,10], [1,10], [2,10]]
        #if args.option in ('gen_ts', 'recoNu_ts', 'recoNu_hybrid'):
        #    dm_combs += [[0, 11], [1,11], [2,11], [10,11], [11,11]]

    for dm_taup, dm_taun in dm_combs:

        dm_mask = lambda df, p=dm_taup, n=dm_taun: ((df['taup_DM'] == p) & (df['taun_DM'] == n)) | ((df['taun_DM'] == n) & (df['taup_DM'] == p))
        even = even_df[dm_mask(even_df)]
        odd  = odd_df[dm_mask(odd_df)]

        print(f"DM{dm_taup}-DM{dm_taun}: {len(even)} CP even, {len(odd)} CP odd events")

        fig, ax = plt.subplots(figsize=(8, 6))
        bin_edges = np.linspace(0, 2 * np.pi, 21)
        hide = args.hide_errors
        if args.uncorrelated:
            # even/odd/mix are all the same reweightable sample here -- apply the TauSpinner
            # weight for each CP hypothesis instead of using separately-generated samples
            even_counts = plot_phicp_histogram(ax, even, bin_edges, 'phiCP', r'CP-even ($\alpha=0^\circ$)', 'red',
                                                hide, weights=even['tauspinner_wt_alpha0'].values)
            odd_counts  = plot_phicp_histogram(ax, odd,  bin_edges, 'phiCP', r'CP-odd ($\alpha=90^\circ$)',  'blue',
                                                hide, weights=odd['tauspinner_wt_alpha90'].values)
            plot_phicp_histogram(ax, even, bin_edges, 'phiCP', r'CP-mix ($\alpha=45^\circ$)', 'green',
                                 hide, weights=even['tauspinner_wt_alpha45'].values)
        else:
            even_counts = plot_phicp_histogram(ax, even, bin_edges, 'phiCP', r'CP-even ($\alpha=0^\circ$)', 'red',   hide)
            odd_counts  = plot_phicp_histogram(ax, odd,  bin_edges, 'phiCP', r'CP-odd ($\alpha=90^\circ$)',  'blue',  hide)
            if mix_df is not None:
                mix = mix_df[dm_mask(mix_df)]
                plot_phicp_histogram(ax, mix, bin_edges, 'phiCP', r'CP-mix ($\alpha=45^\circ$)', 'green', hide)
        if zprime_df is not None:
            zprime = zprime_df[dm_mask(zprime_df)]
            plot_phicp_histogram(ax, zprime, bin_edges, 'phiCP', r'Zprime ($\alpha=180^\circ$)', 'black', hide)
        avg = 0.5 * (even_counts + odd_counts)
        asymmetry = np.mean(np.abs(even_counts - odd_counts) / avg)

        significance = 0 
        for i in range(len(even_counts)):
            b_est = (odd_counts[i] + even_counts[i])*0.5*4
            #temp = odd_counts[i] - even_counts[i] + (even_counts[i]+b_est)*np.log((even_counts[i]+b_est)/(odd_counts[i]+b_est)) if even_counts[i] > 0 and odd_counts[i] > 0 else 0
            temp = (odd_counts[i] - even_counts[i])**2
            significance += temp
        #significance = np.sqrt(2 * significance)
        significance = np.sqrt(significance)

        

        dm_labels = {0: r'$1\pi^\pm0\pi^0$', 1: r'$1\pi^\pm1\pi^0$', 2: r'$1\pi^\pm2\pi^0$', 10: r'$3\pi^\pm0\pi^0$', 11: r'$3\pi^\pm1\pi^0$', 100: r'$\tau_\ell$'}
        dm_taup_label = dm_labels.get(dm_taup, f'DM{dm_taup}')
        dm_taun_label = dm_labels.get(dm_taun, f'DM{dm_taun}')

        ax.set_xlabel(r'$\phi_{CP}$')
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 0.11)
        ax.set_ylabel('Normalized counts')
        ax.legend(loc='upper right', handler_map={matplotlib.patches.Polygon: HandlerStepLine()})
        ax.text(0.05, 0.95, f'{dm_taup_label} - {dm_taun_label}', transform=ax.transAxes,
                verticalalignment='top', fontweight='bold')
        ax.text(0.05, 0.88, f'Asymmetry: {significance:.3f}', transform=ax.transAxes,
                verticalalignment='top', fontweight='bold')
        #ax.text(0.05, 0.82, r'$\sigma_{\phi_{CP}}>1.4$', transform=ax.transAxes,
        ax.text(0.05, 0.82, r'$\forall\,\sigma_{\phi_{CP}}$', transform=ax.transAxes,
                        verticalalignment='top', fontweight='bold')
        option_label = options[args.option]["label"] if not args.useMLP else "Transformer"
        ax.text(0.05, 0.05, option_label, transform=ax.transAxes,
                verticalalignment='bottom', fontweight='bold')
        # ax.text(0.05, 0.85, f'Asymmetry (quadrature): {significance:.4f}', transform=ax.transAxes,
                # verticalalignment='top', fontweight='bold')
        out = f"{args.output_dir}/{args.option}/DM{dm_taup}DM{dm_taun}_{options[args.option]['tag']}.pdf"
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print(f"Saved {out}")

        # save numpy arrays to remake plots in future
        np.savez(
            f"{args.output_dir}/logs/DM{dm_taup}DM{dm_taun}_{options[args.option]['tag']}.npz",
            even_counts=even_counts,
            odd_counts=odd_counts,
            bin_edges=bin_edges,
        )

if __name__ == '__main__':
    main()
