"""
Evaluation script for the direct polarimetric-vector / tau-momentum flow
(config_polvec.yaml style models: 12 outputs = ts_hh_taup/taun (polarimetric
vectors) + undecayed_taup/taun (tau momenta)).

Compares true vs. predicted (MAP estimate):
  - tau 4-vectors (px, py, pz, E)
  - polarimetric vectors (x, y, z + angular separation)
  - phiCP
  - entanglement / spin-correlation "cos" variables (the B+/B-/C matrix)

Handles both the standard (leptonic_mode != 1: physical-charge based
taup/taun naming) and semileptonic (leptonic_mode == 1: tau1=leptonic,
tau2=hadronic, see DataProcessing.convert_semileptonic_df) trainings -- the
tau labels used throughout this script are derived once from leptonic_mode
near the top of main() and threaded through every physical-tau-specific
column name, instead of hardcoding 'taup'/'taun'.

Deliberately kept separate from evaluate.py (which handles the older
neutrino-regression models and already has a lot of leptonic-mode/coordinate
branching) -- this one only has to handle one fixed 12-value output layout.

Test data is read from data_config['test_dataset']/['test_output_name'] (single
string or parallel lists), the same convention as evaluate.py, via
DataProcessing.get_test_dataset.

Run (from the DiTauEntanglement directory):
    python3 taupolaris/scripts/evaluate_polvec.py --config config_polvec.yaml --max_events 2000
"""
import argparse
import os
import numpy as np
import pandas as pd
import awkward as ak
import torch
import yaml
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from taupolaris.python.DataProcessing import RegressionDataset, get_test_dataset, _resolve_polvec_feature_level
from taupolaris.python.NN_Tools import load_model, get_device, is_legacy_pizero_proj_checkpoint
from taupolaris.python.Evaluation_Tools import flow_map_predict, plot_spin_density_matrix
from taupolaris.utils.coordinate_conversions import (
    ConvertFromOrthonormalNRK_Predictions_PolVec,
    ConvertFromOrthonormalNRK_Predictions_PolVec_Angular,
)
from taupolaris.utils.kinematic_helpers import compute_spin_density_vars, boost, boost_vector
from taupolaris.utils.acoplanarity_tools import compute_aco_polarimetric

M_TAU = 1.77686
PHICP_BINS = 20  # phiCP always uses 20 bins, independent of --num_bins


def cartesian_output_order(tau_labels):
    t1, t2 = tau_labels
    return [
        f'ts_hh_{t1}_x', f'ts_hh_{t1}_y', f'ts_hh_{t1}_z',
        f'ts_hh_{t2}_x', f'ts_hh_{t2}_y', f'ts_hh_{t2}_z',
        f'undecayed_{t1}_px', f'undecayed_{t1}_py', f'undecayed_{t1}_pz',
        f'undecayed_{t2}_px', f'undecayed_{t2}_py', f'undecayed_{t2}_pz',
    ]


def onorm_output_order(tau_labels):
    t1, t2 = tau_labels
    return [
        f'ts_hh_{t1}_n', f'ts_hh_{t1}_r', f'ts_hh_{t1}_k',
        f'ts_hh_{t2}_n', f'ts_hh_{t2}_r', f'ts_hh_{t2}_k',
        f'undecayed_{t1}_n', f'undecayed_{t1}_r', f'undecayed_{t1}_k',
        f'undecayed_{t2}_n', f'undecayed_{t2}_r', f'undecayed_{t2}_k',
    ]


def angular_output_order(tau_labels):
    t1, t2 = tau_labels
    return [
        f'ts_hh_{t1}_costheta', f'ts_hh_{t1}_phi',
        f'ts_hh_{t2}_costheta', f'ts_hh_{t2}_phi',
        f'undecayed_{t1}_n', f'undecayed_{t1}_r', f'undecayed_{t1}_k',
        f'undecayed_{t2}_n', f'undecayed_{t2}_r', f'undecayed_{t2}_k',
    ]


def hh_native_prefix(output_features, tau_label):
    """Return the ts_hh_* column prefix ('ts_hh_' or 'ts_hh_approx_') that
    output_features actually uses for this tau's native (onorm/onorm_angular)
    polarimetric-vector output. 'Approx' models substitute ts_hh_approx_ for the
    leptonically-decaying tau's leg (see DataProcessing._approx_leptonic_polvec),
    while the other leg still uses the full ts_hh_ truth -- so t1 and t2 can
    resolve to different prefixes and both must be looked up per-tau rather than
    assumed. The prepared dataframe carries both ts_hh_{tau}_* and
    ts_hh_approx_{tau}_* columns, so this same prefix is valid for indexing the
    true df as well as pred_native_df/samples_native_df."""
    for suffix in (f'{tau_label}_n', f'{tau_label}_costheta'):
        for col in output_features:
            if col.startswith('ts_hh') and col.endswith(suffix):
                return col[:-len(suffix)]
    raise ValueError(f"No ts_hh_* output feature found for tau '{tau_label}' in {output_features}")


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _to_vec3(arr):
    """(N,3) numpy array -> awkward Vector3D, for acoplanarity_tools.compute_aco_polarimetric."""
    return ak.zip({"x": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2]}, with_name="Vector3D")


def compute_phiCP(h1, n1, h2, n2):
    """
    phiCP via acoplanarity_tools.compute_aco_polarimetric (same arXiv:1811.03969
    eq. 3-4 formula validated against this codebase's stored true_cosn/r/k and
    against TauSpinner CP-mixing weights earlier in this project). R1=h1/P1=n1
    and R2=h2/P2=n2 follow that function's own convention (R1=tau+ polarimetric,
    P1=tau+ direction in compute_aco_classic_a1a1) -- doesn't matter which
    physical tau is "1" vs "2" as long as the same labelling is used for both
    the true and predicted vectors being compared.

    h1, n1, h2, n2 : (N,3) numpy arrays
    """
    phicp = compute_aco_polarimetric(_to_vec3(h1), _to_vec3(n1), _to_vec3(h2), _to_vec3(n2))
    return ak.to_numpy(phicp)


def tau_directions_com(tau1_p3, tau2_p3, tau1_E, tau2_E):
    """Boost the two taus to their common (ditau) rest frame; return unit direction vectors."""
    tau1_4 = np.column_stack([tau1_E, tau1_p3])
    tau2_4 = np.column_stack([tau2_E, tau2_p3])
    com_boost = boost_vector(tau1_4 + tau2_4)
    tau1_4_com = boost(tau1_4, -com_boost)
    kx = unit(tau1_4_com[:, 1:])
    return kx, -kx  # tau1 direction, tau2 direction


def leptonic_polvec_from_tau_and_lepton(tau4, other_tau4, lep4):
    """Approximate leptonic-tau polarimetric vector derived analytically from an
    arbitrary tau 4-momentum (e.g. the model's own *predicted* undecayed-tau
    output) and the charged-lepton 4-momentum, as an alternative to reading the
    model's directly-predicted native h -- lets us compare "predict h directly"
    vs. "predict the tau momentum, then derive h analytically from tau+lepton"
    from the same trained model. Same closed-form definition as
    DataProcessing._approx_leptonic_polvec / acoplanarity_tools.polarimetric_vec_leptonic
    (times_by_bl=False): the charged-lepton direction in the tau rest frame,
    sign-flipped (boosted lab -> ditau rest frame -> tau rest frame, matching the
    two-step boost used everywhere else ts_hh is defined in this codebase).

    tau4, other_tau4, lep4: (N,4) arrays [E, px, py, pz] (kinematic_helpers
    convention -- matches tau_directions_com above, NOT DataProcessing's
    [px,py,pz,E]).
    """
    higgs4 = tau4 + other_tau4
    higgs_bv = boost_vector(higgs4)
    lep_hf = boost(lep4, -higgs_bv)
    tau_hf = boost(tau4, -higgs_bv)
    tau_hf_bv = boost_vector(tau_hf)
    lep_trf = boost(lep_hf, -tau_hf_bv)
    return unit(-lep_trf[:, 1:])


def circular_std(angles, axis=-1):
    """Circular standard deviation (Mardia & Jupp), used instead of a plain np.std
    to estimate the flow-sampling error on phiCP. phiCP is periodic on [0, 2*pi),
    so a naive std would report a spuriously huge error for any event whose MAP
    estimate sits near the 0/2*pi seam and whose flow samples land just across it
    (e.g. MAP~0.02 rad with samples at ~0.01 and ~6.27 -- actually tightly
    clustered, ~0.01 rad apart, not ~6 rad apart)."""
    mean_cos = np.mean(np.cos(angles), axis=axis)
    mean_sin = np.mean(np.sin(angles), axis=axis)
    R = np.clip(np.sqrt(mean_cos ** 2 + mean_sin ** 2), 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(R))


def convert_native_to_cartesian(coordinates, native_df, onorm_cols, angular_cols, cartesian_cols,
                                 reco_t1_charged, reco_t1_pizero, reco_t2_charged, reco_t2_pizero):
    """Convert flow native-space (onorm/onorm_angular/standard) predictions to a
    Cartesian DataFrame with cartesian_cols. native_df must be a DataFrame indexable
    by onorm_cols/angular_cols/cartesian_cols (e.g. columns=output_features).
    reco_t1/t2_charged/pizero must have the same row count as native_df -- use
    np.repeat to tile them when converting multiple flow samples per event."""
    if coordinates == 'onorm':
        cart = ConvertFromOrthonormalNRK_Predictions_PolVec(
            native_df[onorm_cols].values,
            reco_taup_charged=reco_t1_charged, reco_taup_pizero=reco_t1_pizero,
            reco_taun_charged=reco_t2_charged, reco_taun_pizero=reco_t2_pizero,
        )
    elif coordinates == 'onorm_angular':
        cart = ConvertFromOrthonormalNRK_Predictions_PolVec_Angular(
            native_df[angular_cols].values,
            reco_taup_charged=reco_t1_charged, reco_taup_pizero=reco_t1_pizero,
            reco_taun_charged=reco_t2_charged, reco_taun_pizero=reco_t2_pizero,
        )
    elif coordinates == 'standard':
        cart = native_df[cartesian_cols].values
    else:
        raise ValueError(f"coordinates='{coordinates}' not supported by this script (only 'standard'/'onorm'/'onorm_angular')")
    return pd.DataFrame(cart, columns=cartesian_cols)


def plot_true_vs_pred_2d(true_vals, pred_vals, labels, suptitle, outpath, num_bins, sym_range=None):
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
    if len(labels) == 1:
        axes = [axes]
    for ax, label, tv, pv in zip(axes, labels, true_vals, pred_vals):
        if sym_range == 'auto':
            lim = np.percentile(np.abs(tv), 99)
            rng = [[-lim, lim], [-lim, lim]]
        elif sym_range is not None:
            rng = [list(sym_range), list(sym_range)]
        else:
            rng = None
        ax.hist2d(tv, pv, bins=num_bins, range=rng, cmin=1)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
        ax.set_xlabel(f'true {label}')
        ax.set_ylabel(f'pred {label}')
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def add_DM(df, tau_labels, dm_prefix='reco'):
    """Per-tau decay-mode code, same categorisation/convention as plot_phiCP.py's
    add_DM: 0=1-prong no pi0, 1=1-prong+1pi0, 2=1-prong+2pi0, 10=3-prong, 11=3-prong+pi0,
    100=leptonic, -1=unmatched. Returns (dm_tau1, dm_tau2) arrays, following
    whatever tau_labels are in use ((taup,taun) or (tau1,tau2)).

    dm_prefix='reco' reads reco_{tau}_ishadronic etc. (reco-level); dm_prefix=''
    reads the bare {tau}_ishadronic etc. (gen-level -- gen columns have no prefix
    in this dataframe convention, unlike reco which is explicitly 'reco_'-prefixed)."""
    col_prefix = f'{dm_prefix}_' if dm_prefix else ''
    dm = {}
    for tau in tau_labels:
        is_lep = df[f'{col_prefix}{tau}_ishadronic'].values == 0
        is_dm0 = (df[f'{col_prefix}{tau}_npizero'].values == 0) & (df[f'{col_prefix}{tau}_is3prong'].values == 0) & (~is_lep)
        is_dm1 = (df[f'{col_prefix}{tau}_npizero'].values == 1) & (df[f'{col_prefix}{tau}_is3prong'].values == 0) & (~is_lep)
        is_dm2 = ((df[f'{col_prefix}{tau}_npizero'].values == 1) | (df[f'{col_prefix}{tau}_npizero'].values == 2)) & (df[f'{col_prefix}{tau}_is3prong'].values == 0) & (~is_lep)
        is_dm10 = (df[f'{col_prefix}{tau}_npizero'].values == 0) & (df[f'{col_prefix}{tau}_is3prong'].values == 1) & (~is_lep)
        is_dm11 = (df[f'{col_prefix}{tau}_npizero'].values == 1) & (df[f'{col_prefix}{tau}_is3prong'].values == 1) & (~is_lep)
        dm[tau] = np.where(is_dm0, 0,
                    np.where(is_dm1, 1,
                        np.where(is_dm2, 2,
                            np.where(is_dm10, 10,
                                np.where(is_dm11, 11,
                                    np.where(is_lep, 100, -1))))))
    return dm[tau_labels[0]], dm[tau_labels[1]]


def normalised_phicp_counts(phicp, weights, bin_edges):
    """Per-bin phiCP counts normalised so they sum to 1, matching
    plot_phiCP.py::plot_phicp_histogram.

    That function normalises the WEIGHTS (w / w.sum()) and histograms with
    density=False, so each bin holds the fraction of total weight it contains.
    Using matplotlib's density=True instead would divide by the bin width and
    inflate every downstream asymmetry by 1/binwidth (= 20/2pi = 3.183 for the
    20 bins used here) -- which is exactly what this script used to do, making
    its numbers incomparable with plot_phiCP.py's.
    """
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    w = w / total if total != 0 else w
    counts, _ = np.histogram(phicp, bins=bin_edges, weights=w)
    return counts


def asymmetry_quadrature(counts_a, counts_b):
    """The 'Asymmetry' metric plot_phiCP.py reports: sqrt(sum((a-b)^2)) over two
    phiCP histograms whose bins each sum to 1 (see normalised_phicp_counts).
    Feed it probability-normalised counts, NOT densities."""
    return np.sqrt(np.sum((counts_a - counts_b) ** 2))


def plot_phiCP_cp_comparison(true_phiCP, pred_phiCP, w_cpeven, w_cpodd, title, outpath, num_bins):
    """The true/predicted x CP-even/CP-odd 4-curve phiCP comparison, factored out
    so it can be reused both for the full sample and per decay-mode combination."""
    fig, ax = plt.subplots(figsize=(7, 6))
    bins = np.linspace(0, 2 * np.pi, num_bins + 1)
    # 1/N-style weight normalisation with density=False, exactly as
    # plot_phiCP.py does -- so both the curves and the asymmetry printed on them
    # are directly comparable between the two scripts.
    series = (
        ('true, CP-even',      true_phiCP, w_cpeven, 'steelblue', '--'),
        ('predicted, CP-even', pred_phiCP, w_cpeven, 'steelblue', '-'),
        ('true, CP-odd',       true_phiCP, w_cpodd,  'tomato',    '--'),
        ('predicted, CP-odd',  pred_phiCP, w_cpodd,  'tomato',    '-'),
    )
    counts = {}
    for label, vals, w, colour, style in series:
        counts[label] = normalised_phicp_counts(vals, w, bins)
        ax.stairs(counts[label], bins, color=colour, linestyle=style,
                  linewidth=1.5, label=label)

    true_asym = asymmetry_quadrature(counts['true, CP-even'], counts['true, CP-odd'])
    pred_asym = asymmetry_quadrature(counts['predicted, CP-even'], counts['predicted, CP-odd'])
    ax.text(0.05, 0.95, f'Asymmetry, true: {true_asym:.4f}', transform=ax.transAxes,
            verticalalignment='top', fontweight='bold', fontsize=9)
    ax.text(0.05, 0.89, f'Asymmetry, pred: {pred_asym:.4f}', transform=ax.transAxes,
            verticalalignment='top', fontweight='bold', fontsize=9)

    ax.set_xlabel(r'$\phi_{CP}$ [rad]')
    ax.set_ylabel('Normalized counts')
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', required=True, help='path to the configuration file')
    argparser.add_argument('--useCPU', action='store_true', help='force CPU evaluation')
    argparser.add_argument('--max_events', type=int, default=5000, help='number of test events to evaluate on')
    argparser.add_argument('--num_bins', type=int, default=50)
    argparser.add_argument('--oneprong', action='store_true', help='whether to only evaluate on 1-prong taus only')
    argparser.add_argument('--threeprong', action='store_true', help='whether to only evaluate on events with at least 1 3-prong tau')
    argparser.add_argument('--n_flow_samples', type=int, default=50,
                            help='number of flow samples per event used to estimate an error on pred_phiCP (circular std)')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']

    device = torch.device('cpu') if args.useCPU else get_device()
    print(f">> Using device: {device}")

    output_dir = f"outputs_{nn_config['model_name']}"

    input_features = data_config['Features']['input_features']
    coordinates = data_config['coordinates']
    output_features = data_config['Features']['output_features'][coordinates]
    leptonic_mode = data_config.get('leptonic_mode', -1)
    # tau1=leptonic/tau2=hadronic for semileptonic training (see
    # DataProcessing.convert_semileptonic_df); otherwise taup/taun (physical
    # charge, symmetric). Threaded through every column name built below
    # instead of hardcoding 'taup'/'taun'.
    tau_labels = ('tau1', 'tau2') if leptonic_mode == 1 else ('taup', 'taun')
    t1, t2 = tau_labels
    norm_data = np.load(f'{output_dir}/normalization_params.npz')

    # --- load model (same for every test_dataset below) ---
    # prefer the final saved model (written once training completes); fall back to
    # the best/partial checkpoints under plots/ (same convention as evaluate.py) so
    # this also works against a still-training or interrupted run.
    output_plots_dir = os.path.join(output_dir, 'plots')
    model_path = os.path.join(output_dir, f"{nn_config['model_name']}.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(output_plots_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(output_plots_dir, 'partial_model.pth')
    print(f">> Loading model weights from {model_path}")
    # map_location='cpu' always works regardless of what device the checkpoint was
    # saved on (and regardless of whether a GPU is available here); the model is
    # moved to the target device separately below.
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    legacy_pizero_proj = is_legacy_pizero_proj_checkpoint(state_dict)
    if legacy_pizero_proj:
        print(">> Checkpoint predates the pizero_proj/final-LayerNorm architecture change; "
              "building the matching (older) model architecture.")
    model = load_model(nn_config['hyperparams'], input_features, output_features,
                        useTransformer=nn_config.get('use_transformer', True), leptonic_mode=leptonic_mode,
                        legacy_pizero_proj=legacy_pizero_proj,
                        polvec_feature_level=_resolve_polvec_feature_level(data_config))
    model.load_state_dict(state_dict)
    model.float().to(device)
    model.eval()

    # --- resolve test_dataset / test_output_name (single string or parallel lists),
    # same convention as evaluate.py ---
    if isinstance(data_config['test_dataset'], list):
        test_datasets = data_config['test_dataset']
        if not isinstance(data_config['test_output_name'], list):
            raise ValueError("If test_dataset is a list, test_output_name must also be a list")
        if len(data_config['test_output_name']) != len(test_datasets):
            raise ValueError("test_output_name must have the same length as test_dataset")
        test_output_names = data_config['test_output_name']
    else:
        test_datasets = [data_config['test_dataset']]
        if isinstance(data_config['test_output_name'], list):
            raise ValueError("If test_dataset is not a list, test_output_name must not be a list")
        test_output_names = [data_config['test_output_name']]

    for test_dataset_path, test_output_name in zip(test_datasets, test_output_names):
        print(f">> Evaluating on test dataset {test_dataset_path} (output name: {test_output_name})")
        data_config['test_dataset'] = test_dataset_path
        dataset, df, _, _ = get_test_dataset(data_config, norm_data, oneprong=args.oneprong, threeprong=args.threeprong)

        if args.max_events is not None and len(df) > args.max_events:
            df = df.iloc[:args.max_events].reset_index(drop=True)
            dataset = RegressionDataset(
                df, input_features, output_features,
                normalize_inputs=True, normalize_outputs=True,
                input_mean=torch.from_numpy(norm_data['input_mean']),
                input_std=torch.from_numpy(norm_data['input_std']),
                output_mean=torch.from_numpy(norm_data['output_mean']),
                output_std=torch.from_numpy(norm_data['output_std']),
            )
        print(f">> Evaluating on {len(df)} events")

        outdir = os.path.join(output_dir, 'plots_eval_polvec', test_output_name)
        os.makedirs(outdir, exist_ok=True)

        # --- MAP prediction, in the model's native (training) coordinate space ---
        X, _ = dataset[:]
        X = X.to(device)
        print(f">> Running MAP prediction (method={nn_config.get('map_method', 'gradient')})...")
        #chunk_size = 25000 
        chunk_size = nn_config.get('chunk_size', 25000 if device.type == 'cpu' else 25000)
        _, predictions_native = flow_map_predict(
            model, X, test_dataset=dataset,
            method=nn_config.get('map_method', 'gradient'),
            num_draws=nn_config.get('map_num_draws', 100),
            chunk_size=chunk_size
        )
        pred_native_df = pd.DataFrame(predictions_native, columns=output_features)

        cartesian_cols = cartesian_output_order(tau_labels)
        # onorm_cols/angular_cols must match the actual column names of
        # pred_native_df/samples_native_df (built with columns=output_features
        # above), not a generic hardcoded naming -- for "Approx" models the
        # config substitutes ts_hh_approx_{t1}_* for the leptonic tau's leg
        # (see DataProcessing._approx_leptonic_polvec), which the generic
        # onorm_output_order()/angular_output_order() helpers don't know about.
        # output_features is already in the right order for both the native
        # predictions and (after convert_semileptonic_df's substring-based
        # taup_/taun_ -> tau1_/tau2_ rename) the true df's matching columns.
        onorm_cols = output_features if coordinates == 'onorm' else onorm_output_order(tau_labels)
        angular_cols = output_features if coordinates == 'onorm_angular' else angular_output_order(tau_labels)

        # --- convert predictions to Cartesian (true values are already Cartesian in df) ---
        # kept as named arrays (not just inlined into the conversion call) so they can be
        # reused/tiled for the flow-sampling phiCP error estimate below
        reco_t1_charged = df[[f'reco_{t1}_charged_px', f'reco_{t1}_charged_py', f'reco_{t1}_charged_pz']].values
        reco_t1_pizero  = df[[f'reco_{t1}_pizero1_px', f'reco_{t1}_pizero1_py', f'reco_{t1}_pizero1_pz']].values
        reco_t2_charged = df[[f'reco_{t2}_charged_px', f'reco_{t2}_charged_py', f'reco_{t2}_charged_pz']].values
        reco_t2_pizero  = df[[f'reco_{t2}_pizero1_px', f'reco_{t2}_pizero1_py', f'reco_{t2}_pizero1_pz']].values
        pred_cart_df = convert_native_to_cartesian(
            coordinates, pred_native_df, onorm_cols, angular_cols, cartesian_cols,
            reco_t1_charged, reco_t1_pizero, reco_t2_charged, reco_t2_pizero,
        )

        true_cart_df = df[cartesian_cols].reset_index(drop=True)

        # decay mode (gen and reco), for the per-DM phiCP plots below and saved to results_df
        gen_dm_t1, gen_dm_t2 = add_DM(df, tau_labels, dm_prefix='')
        reco_dm_t1, reco_dm_t2 = add_DM(df, tau_labels, dm_prefix='reco')

        # === 1. tau 4-vector comparison ===
        print(">> Plotting tau 4-vector comparisons...")
        for tau in tau_labels:
            true_p = true_cart_df[[f'undecayed_{tau}_px', f'undecayed_{tau}_py', f'undecayed_{tau}_pz']].values
            pred_p = pred_cart_df[[f'undecayed_{tau}_px', f'undecayed_{tau}_py', f'undecayed_{tau}_pz']].values
            true_E = df[f'undecayed_{tau}_e'].values
            pred_E = np.sqrt(np.sum(pred_p ** 2, axis=1) + M_TAU ** 2)

            true_vals = [true_p[:, 0], true_p[:, 1], true_p[:, 2], true_E]
            pred_vals = [pred_p[:, 0], pred_p[:, 1], pred_p[:, 2], pred_E]
            plot_true_vs_pred_2d(
                true_vals, pred_vals, [f'{tau}_px', f'{tau}_py', f'{tau}_pz', f'{tau}_E'],
                f'Tau 4-vector: {tau}', os.path.join(outdir, f'tau4vec_{tau}.pdf'),
                args.num_bins, sym_range='auto',
            )

        # === 2. polarimetric vector comparison ===
        print(">> Plotting polarimetric vector comparisons...")
        for tau in tau_labels:
            true_h = true_cart_df[[f'ts_hh_{tau}_x', f'ts_hh_{tau}_y', f'ts_hh_{tau}_z']].values
            pred_h_unit = unit(pred_cart_df[[f'ts_hh_{tau}_x', f'ts_hh_{tau}_y', f'ts_hh_{tau}_z']].values)

            plot_true_vs_pred_2d(
                [true_h[:, 0], true_h[:, 1], true_h[:, 2]],
                [pred_h_unit[:, 0], pred_h_unit[:, 1], pred_h_unit[:, 2]],
                [f'ts_hh_{tau}_x', f'ts_hh_{tau}_y', f'ts_hh_{tau}_z'],
                f'Polarimetric vector: {tau}', os.path.join(outdir, f'polarimetric_{tau}.pdf'),
                args.num_bins, sym_range=(-1, 1),
            )

            # true_h isn't always exactly unit-length (see calculate_hh.py's rare
            # ~0.1-0.5% numerical-singularity tail), so it must be normalized here
            # too -- otherwise the dot product isn't cos(angle) and the clip(-1,1)
            # silently masks the distortion instead of computing the true angle.
            ang_sep = np.degrees(np.arccos(np.clip(np.sum(unit(true_h) * pred_h_unit, axis=1), -1, 1)))
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(ang_sep, bins=args.num_bins, histtype='step', linewidth=1.5)
            ax.set_xlabel(f'angle(true, pred) [deg] -- {tau}')
            ax.set_ylabel('events')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f'polarimetric_angle_{tau}.pdf'), dpi=130)
            plt.close(fig)

        # === 2b. phi periodicity/seam diagnostics (onorm_angular only) ===
        # phi is trained as a plain unconstrained regression target (see the
        # ConvertNRKToAngular/periodicity discussion) -- there's no built-in
        # knowledge that phi=0 and phi=2*pi are the same point. If that hurts
        # the model, it should show up as elevated prediction error specifically
        # for events whose true phi lands near the 0/2*pi seam, not spread
        # uniformly across phi. These plots/columns are the direct test of that.
        phi_diag_cols = None
        if coordinates == 'onorm_angular':
            print(">> Computing phi periodicity/seam diagnostics...")
            phi_diag_outdir = os.path.join(outdir, 'phi_periodicity')
            os.makedirs(phi_diag_outdir, exist_ok=True)
            phi_diag_cols = {}
            n_phi_bins = args.num_bins
            for tau in tau_labels:
                true_phi = df[f'ts_hh_{tau}_phi'].values
                pred_phi = pred_native_df[f'ts_hh_{tau}_phi'].values
                raw_dphi = pred_phi - true_phi
                # correct minimal angular difference, wrapped into [-pi, pi)
                wrapped_dphi = (raw_dphi + np.pi) % (2 * np.pi) - np.pi
                needed_wrap = np.abs(raw_dphi) > np.pi

                phi_diag_cols[f'{tau}_phi_dphi_raw'] = raw_dphi
                phi_diag_cols[f'{tau}_phi_dphi_wrapped'] = wrapped_dphi
                phi_diag_cols[f'{tau}_phi_needed_wrap'] = needed_wrap

                frac_wrap = needed_wrap.mean()
                print(f"  {tau}: fraction of events where the naive (unwrapped) "
                      f"delta_phi was off by a full wrap: {100 * frac_wrap:.3f}%")

                # 1) raw vs wrapped delta_phi overlay -- if these differ meaningfully,
                # naive (non-periodic-aware) error metrics would be misleading.
                fig, ax = plt.subplots(figsize=(6, 5))
                bins = np.linspace(-2 * np.pi, 2 * np.pi, args.num_bins + 1)
                ax.hist(raw_dphi, bins=bins, histtype='step', linewidth=1.5, label='raw (unwrapped)')
                ax.hist(wrapped_dphi, bins=bins, histtype='step', linewidth=1.5, label='wrapped to [-pi,pi)')
                ax.set_xlabel(f'pred_phi - true_phi [rad] -- {tau}')
                ax.set_ylabel('events')
                ax.legend()
                ax.text(0.05, 0.95, f'needed wrap: {100 * frac_wrap:.2f}%', transform=ax.transAxes,
                        verticalalignment='top', fontweight='bold', fontsize=9)
                fig.tight_layout()
                fig.savefig(os.path.join(phi_diag_outdir, f'dphi_raw_vs_wrapped_{tau}.pdf'), dpi=130)
                plt.close(fig)

                # 2) profile: mean |wrapped dphi| vs true_phi -- the direct test of
                # whether error is elevated specifically near the seam (true_phi near
                # 0 or 2*pi) compared to the bulk (e.g. near pi).
                phi_bins = np.linspace(0, 2 * np.pi, n_phi_bins + 1)
                bin_idx = np.clip(np.digitize(np.mod(true_phi, 2 * np.pi), phi_bins) - 1, 0, n_phi_bins - 1)
                bin_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
                mean_abs_err = np.full(n_phi_bins, np.nan)
                for b in range(n_phi_bins):
                    mask = bin_idx == b
                    if mask.sum() > 0:
                        mean_abs_err[b] = np.mean(np.abs(wrapped_dphi[mask]))

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(bin_centers, mean_abs_err, marker='o', linewidth=1.5)
                ax.axvspan(0, phi_bins[1], color='red', alpha=0.1, label='seam (phi~0 / phi~2*pi)')
                ax.axvspan(phi_bins[-2], 2 * np.pi, color='red', alpha=0.1)
                ax.set_xlabel(f'true_phi [rad] -- {tau}')
                ax.set_ylabel('mean |wrapped delta_phi| [rad]')
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(phi_diag_outdir, f'dphi_vs_truephi_profile_{tau}.pdf'), dpi=130)
                plt.close(fig)

                # 3) true_phi distribution, using the same bin edges as the profile above --
                # checks whether a non-flat error profile is just tracking non-uniform
                # training/test statistics in phi (fewer/more events in some bins) rather
                # than a genuine periodicity or hard-to-predict-region effect.
                counts, _ = np.histogram(np.mod(true_phi, 2 * np.pi), bins=phi_bins)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.bar(bin_centers, counts, width=(phi_bins[1] - phi_bins[0]), align='center',
                       color='steelblue', alpha=0.7)
                ax.axvspan(0, phi_bins[1], color='red', alpha=0.1, label='seam (phi~0 / phi~2*pi)')
                ax.axvspan(phi_bins[-2], 2 * np.pi, color='red', alpha=0.1)
                ax.set_xlabel(f'true_phi [rad] -- {tau}')
                ax.set_ylabel('events')
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(phi_diag_outdir, f'truephi_distribution_{tau}.pdf'), dpi=130)
                plt.close(fig)

                # single-number summary: error in the seam bins (first + last) vs the
                # bulk (all other bins) -- a ratio >> 1 indicates a real periodicity
                # problem; a ratio ~1 means the model handles the seam no worse than
                # anywhere else in phi.
                seam_err = np.nanmean([mean_abs_err[0], mean_abs_err[-1]])
                bulk_err = np.nanmean(mean_abs_err[1:-1])
                if bulk_err > 0:
                    print(f"  {tau}: mean |wrapped dphi| in seam bins = {seam_err:.4f} rad, "
                          f"bulk bins = {bulk_err:.4f} rad, ratio = {seam_err / bulk_err:.2f}")
            print(f">> Saved phi periodicity diagnostic plots to {phi_diag_outdir}")

        # === 3. phiCP ===
        print(">> Computing phiCP (true vs predicted)...")
        def get_phiCP(cart_df, E_t1, E_t2):
            h1 = cart_df[[f'ts_hh_{t2}_x', f'ts_hh_{t2}_y', f'ts_hh_{t2}_z']].values
            h2 = cart_df[[f'ts_hh_{t1}_x', f'ts_hh_{t1}_y', f'ts_hh_{t1}_z']].values
            p_t1 = cart_df[[f'undecayed_{t1}_px', f'undecayed_{t1}_py', f'undecayed_{t1}_pz']].values
            p_t2 = cart_df[[f'undecayed_{t2}_px', f'undecayed_{t2}_py', f'undecayed_{t2}_pz']].values
            n2, n1 = tau_directions_com(p_t1, p_t2, E_t1, E_t2)
            return compute_phiCP(h1, n1, h2, n2)

        true_E_t1 = df[f'undecayed_{t1}_e'].values
        true_E_t2 = df[f'undecayed_{t2}_e'].values
        pred_p_t1 = pred_cart_df[[f'undecayed_{t1}_px', f'undecayed_{t1}_py', f'undecayed_{t1}_pz']].values
        pred_p_t2 = pred_cart_df[[f'undecayed_{t2}_px', f'undecayed_{t2}_py', f'undecayed_{t2}_pz']].values
        pred_E_t1 = np.sqrt(np.sum(pred_p_t1 ** 2, axis=1) + M_TAU ** 2)
        pred_E_t2 = np.sqrt(np.sum(pred_p_t2 ** 2, axis=1) + M_TAU ** 2)

        true_phiCP = get_phiCP(true_cart_df, true_E_t1, true_E_t2)
        pred_phiCP = get_phiCP(pred_cart_df, pred_E_t1, pred_E_t2)

        # === 3'. leptonic-leg alternatives to the model's native h_t1 prediction --
        # only meaningful for the leptonically-decaying leg (t1 in semileptonic
        # training), and only stored/plotted alongside (not instead of) the model's
        # direct native prediction above. The hadronic leg (h_t2, direction n_t2)
        # is shared and unchanged (still the existing pol-vec method) across both.
        pred_h_t1_derived, pred_phiCP_derived = None, None
        pred_R_t1_run3lep, pred_phiCP_run3lep = None, None
        if leptonic_mode == 1:
            lep_cols = [f'reco_{t1}_lep_e', f'reco_{t1}_lep_px', f'reco_{t1}_lep_py', f'reco_{t1}_lep_pz']
            if all(c in df.columns for c in lep_cols):
                reco_lep4 = df[lep_cols].values.astype(float)
                pred_tau1_4 = np.column_stack([pred_E_t1, pred_p_t1])
                pred_tau2_4 = np.column_stack([pred_E_t2, pred_p_t2])
                pred_h_t2_native = pred_cart_df[[f'ts_hh_{t2}_x', f'ts_hh_{t2}_y', f'ts_hh_{t2}_z']].values
                n2_dir, n1_dir = tau_directions_com(pred_p_t1, pred_p_t2, pred_E_t1, pred_E_t2)

                # 3'a. regressed tau + lepton, via the same closed-form leptonic
                # polvec approximation as ts_hh_approx (see
                # leptonic_polvec_from_tau_and_lepton's docstring).
                pred_h_t1_derived = leptonic_polvec_from_tau_and_lepton(pred_tau1_4, pred_tau2_4, reco_lep4)
                pred_phiCP_derived = compute_phiCP(pred_h_t2_native, n1_dir, pred_h_t1_derived, n2_dir)

                # 3'b. "Run3 classic" lepton-side (R, P) definition, i.e. what
                # acoplanarity_tools.get_R_P_vectors_all/plot_phiCP.py's 'recoRun3'
                # option uses for a leptonic leg: R = the lepton track's impact
                # parameter, P = the lepton's own momentum (NOT the regressed tau
                # momentum). Unlike 'recoRun3' (which boosts into the visible-momenta
                # sum's rest frame before projecting, since it never reconstructs the
                # full tau), here R/P are boosted into the same ditau (tau1+tau2)
                # rest frame the hadronic leg's (h_t2, n_t2) already live in, so both
                # legs can be combined self-consistently in one compute_phiCP call --
                # this only changes how the lepton leg's observable is defined, the
                # hadronic leg still uses the existing pol-vec method unchanged.
                ip_cols = [f'reco_{t1}_lep_ipx', f'reco_{t1}_lep_ipy', f'reco_{t1}_lep_ipz']
                if all(c in df.columns for c in ip_cols):
                    lep_ip3 = df[ip_cols].values.astype(float)
                    com_boost = boost_vector(pred_tau1_4 + pred_tau2_4)
                    lep_ip4 = np.concatenate([np.zeros((len(df), 1)), lep_ip3], axis=1)
                    R_lep_com = boost(lep_ip4, -com_boost)[:, 1:]
                    P_lep_com = boost(reco_lep4, -com_boost)[:, 1:]
                    pred_phiCP_run3lep = compute_phiCP(pred_h_t2_native, n1_dir, R_lep_com, P_lep_com)
                    pred_R_t1_run3lep = R_lep_com
                else:
                    print(f">> WARNING: {ip_cols} not all found in test dataframe -- "
                          "skipping Run3-classic lepton-side (R, P) polarimetric variable.")
            else:
                print(f">> WARNING: {lep_cols} not all found in test dataframe -- "
                      "skipping leptonic-leg (regressed-tau/Run3) polarimetric variables.")

        # === 3a. phiCP uncertainty, from repeated flow sampling ===
        # MAP gives a single point estimate; sampling the flow n_flow_samples times per
        # event and taking the (circular) spread of the resulting phiCP gives a per-event
        # error estimate on pred_phiCP.
        n_fs = args.n_flow_samples
        # each call below processes (events_in_chunk * n_fs) rows in one shot -- reusing
        # MAP's own chunk_size here would mean e.g. 25000*50 = 1.25M rows per call (MAP
        # itself never multiplies chunk_size by anything), which was empirically much
        # slower than linear scaling predicted. Divide by n_fs so each call stays close
        # to MAP's own per-call scale.
        err_chunk_size = max(1, chunk_size // n_fs)
        print(f">> Estimating pred_phiCP uncertainty from {n_fs} flow samples/event "
              f"(chunk size {err_chunk_size} events, {err_chunk_size * n_fs} rows/call)...")
        pred_phiCP_err = np.empty(len(df))
        n_sample_failures = 0
        for start in tqdm(range(0, len(df), err_chunk_size), desc="Processing chunks (phiCP uncertainty)"):
            end = min(start + err_chunk_size, len(df))
            C = end - start
            # model.sample() occasionally hits a rare, non-reproducible numerical
            # instability in nflows's rational-quadratic spline inverse (an assertion
            # on a negative discriminant, from FP32 precision near bin-boundary
            # derivatives for an unlucky random draw) -- not tied to any particular
            # input event, confirmed by retrying the exact same chunk with fresh
            # random noise. Retry a few times before giving up on the chunk.
            samples_norm = None
            for attempt in range(5):
                try:
                    with torch.no_grad():
                        samples_norm = model.sample(num_samples=n_fs, context=X[start:end])  # [C, n_fs, F]
                    break
                except AssertionError:
                    continue
            if samples_norm is None:
                n_sample_failures += 1
                pred_phiCP_err[start:end] = np.nan
                continue
            samples_native = dataset.destandardize_outputs(samples_norm).cpu().numpy().reshape(C * n_fs, -1)
            samples_native_df = pd.DataFrame(samples_native, columns=output_features)

            rep = lambda arr: np.repeat(arr[start:end], n_fs, axis=0)
            sample_cart_df = convert_native_to_cartesian(
                coordinates, samples_native_df, onorm_cols, angular_cols, cartesian_cols,
                rep(reco_t1_charged), rep(reco_t1_pizero), rep(reco_t2_charged), rep(reco_t2_pizero),
            )
            p_t1_s = sample_cart_df[[f'undecayed_{t1}_px', f'undecayed_{t1}_py', f'undecayed_{t1}_pz']].values
            p_t2_s = sample_cart_df[[f'undecayed_{t2}_px', f'undecayed_{t2}_py', f'undecayed_{t2}_pz']].values
            E_t1_s = np.sqrt(np.sum(p_t1_s ** 2, axis=1) + M_TAU ** 2)
            E_t2_s = np.sqrt(np.sum(p_t2_s ** 2, axis=1) + M_TAU ** 2)
            phiCP_samples = get_phiCP(sample_cart_df, E_t1_s, E_t2_s).reshape(C, n_fs)
            pred_phiCP_err[start:end] = circular_std(phiCP_samples, axis=1)

        if n_sample_failures > 0:
            print(f"  WARNING: flow sampling failed on 5 retries for {n_sample_failures} chunk(s) "
                  f"({n_sample_failures * err_chunk_size} events, upper bound); pred_phiCP_err set to NaN for those events.")

        # Reweight the (single, UnCorr) sample to CP-even/CP-odd hypotheses using
        # TauSpinner's own per-event weights, and compare true vs. predicted phiCP
        # under each hypothesis -- this tests whether the model's *predicted*
        # polarimetric vectors preserve enough information to still separate
        # CP-even from CP-odd once reweighted, not just whether they match truth
        # event-by-event.
        have_weights = 'tauspinner_wt_alpha0' in df.columns and 'tauspinner_wt_alpha90' in df.columns
        if have_weights:
            w_cpeven = df['tauspinner_wt_alpha0'].values
            w_cpodd = df['tauspinner_wt_alpha90'].values

        def _plot_phiCP_suite(pred_phiCP_variant, filename_stem, dm_subdir):
            """Factored out so the same 'All events' + per-decay-mode phiCP plot suite
            can be produced both for the model's native prediction and (if computed)
            the derived regressed-tau+lepton alternative, without duplicating the
            plotting logic or overwriting either set of files."""
            if have_weights:
                plot_phiCP_cp_comparison(true_phiCP, pred_phiCP_variant, w_cpeven, w_cpodd,
                                          'All events', os.path.join(outdir, f'{filename_stem}.pdf'), PHICP_BINS)
            else:
                fig, ax = plt.subplots(figsize=(6, 5))
                bins = np.linspace(0, 2 * np.pi, PHICP_BINS + 1)
                ones = np.ones(len(true_phiCP))
                ax.stairs(normalised_phicp_counts(true_phiCP, ones, bins), bins,
                          linewidth=1.5, label='true')
                ax.stairs(normalised_phicp_counts(pred_phiCP_variant, ones, bins), bins,
                          linewidth=1.5, label='predicted')
                ax.set_xlabel(r'$\phi_{CP}$ [rad]')
                ax.set_ylabel('Normalized counts')
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(outdir, f'{filename_stem}.pdf'), dpi=130)
                plt.close(fig)

            if not have_weights:
                return
            dm_t1, dm_t2 = reco_dm_t1, reco_dm_t2
            dm_combs = [
                [0, 0], [0, 1], [1, 1], [2, 2], [1, 2], [0, 2], [10, 10], [0, 10], [1, 10], [2, 10],
                [0, 11], [1, 11], [2, 11], [10, 11], [11, 11],
                [100, 0], [100, 1], [100, 2], [100, 10], [100, 11], [100, 100],
            ]
            dm_outdir = os.path.join(outdir, dm_subdir)
            os.makedirs(dm_outdir, exist_ok=True)
            min_events = 20
            for dm_p, dm_n in dm_combs:
                mask = ((dm_t1 == dm_p) & (dm_t2 == dm_n)) | ((dm_t1 == dm_n) & (dm_t2 == dm_p))
                n_events = mask.sum()
                if n_events < min_events:
                    continue
                plot_phiCP_cp_comparison(
                    true_phiCP[mask], pred_phiCP_variant[mask], w_cpeven[mask], w_cpodd[mask],
                    f'DM{dm_p} - DM{dm_n} ({n_events} events)',
                    os.path.join(dm_outdir, f'{filename_stem}_DM{dm_p}_DM{dm_n}.pdf'), PHICP_BINS,
                )

        if not have_weights:
            print(">> WARNING: tauspinner_wt_alpha0/90 not found in test dataframe -- "
                  "falling back to unweighted true-vs-predicted phiCP.")
        print(">> Plotting phiCP (all events + per decay-mode combination)...")
        _plot_phiCP_suite(pred_phiCP, 'phiCP', 'phiCP_by_dm')
        print(f">> Saved phiCP plots to {outdir}")

        if pred_h_t1_derived is not None:
            print(">> Plotting derived (regressed-tau + lepton) phiCP (all events + per decay-mode combination)...")
            _plot_phiCP_suite(pred_phiCP_derived, 'phiCP_derivedApproxLep', 'phiCP_by_dm_derivedApproxLep')
            print(f">> Saved derived phiCP plots to {outdir}")

        if pred_R_t1_run3lep is not None:
            print(">> Plotting Run3-classic lepton-side (R, P) phiCP (all events + per decay-mode combination)...")
            _plot_phiCP_suite(pred_phiCP_run3lep, 'phiCP_run3lep', 'phiCP_by_dm_run3lep')
            print(f">> Saved Run3-classic lepton-side phiCP plots to {outdir}")

        # === 3c. ditau (boson candidate) invariant mass ===
        print(">> Plotting ditau invariant mass...")
        true_p_t1 = true_cart_df[[f'undecayed_{t1}_px', f'undecayed_{t1}_py', f'undecayed_{t1}_pz']].values
        true_p_t2 = true_cart_df[[f'undecayed_{t2}_px', f'undecayed_{t2}_py', f'undecayed_{t2}_pz']].values

        def ditau_mass(E_t1, p_t1, E_t2, p_t2):
            Etot = E_t1 + E_t2
            ptot = p_t1 + p_t2
            m2 = Etot ** 2 - np.sum(ptot ** 2, axis=1)
            return np.sqrt(np.clip(m2, 0, None))

        true_mass = ditau_mass(true_E_t1, true_p_t1, true_E_t2, true_p_t2)
        pred_mass = ditau_mass(pred_E_t1, pred_p_t1, pred_E_t2, pred_p_t2)

        plot_true_vs_pred_2d(
            [true_mass], [pred_mass], ['ditau_mass'],
            'Ditau invariant mass', os.path.join(outdir, 'ditau_mass_2d.pdf'),
            args.num_bins, sym_range=None,
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        lo, hi = np.percentile(true_mass, [0.5, 99.5])
        bins = np.linspace(lo, hi, args.num_bins + 1)
        ax.hist(true_mass, bins=bins, density=True, histtype='step', linewidth=1.5, label='true')
        ax.hist(pred_mass, bins=bins, density=True, histtype='step', linewidth=1.5, label='predicted')
        ax.set_xlabel('ditau mass [GeV]')
        ax.set_ylabel('a.u.')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'ditau_mass_1d.pdf'), dpi=130)
        plt.close(fig)

        # === 4. entanglement / spin-correlation cos variables ===
        # NOTE: these n/r/k projections use the *visible-tau-referenced* basis the
        # model was trained against (see ConvertToOrthonormalNRK), not the
        # true-tau-direction basis of kinematic_helpers.compute_spin_angles -- this
        # compares the model's predictions against its own training-target
        # definition, not the fully "canonical" spin-density-matrix basis.
        ent_df = None
        if coordinates in ('onorm', 'onorm_angular'):
            print(">> Computing entanglement/spin-correlation variables...")
            # .values throughout: df may have a non-contiguous index (e.g. after
            # get_test_dataset's oneprong/threeprong filtering, which doesn't reset
            # it), while pred_native_df always has a fresh 0..N-1 index -- mixing
            # Series with different indices in one dict would silently misalign rows.
            #
            # Truth always comes from the raw (n,r,k) projections (kept as a
            # passthrough column even in onorm_angular mode -- see
            # ConvertNRKToAngular's drop_nrk=False in DataProcessing.py), NOT from
            # decoding costheta/phi, since that would silently force truth onto
            # the unit sphere and hide the real (if rare) |h|!=1 deviations in the
            # data -- see calculate_hh.py. The prediction, by contrast, only ever
            # exists as (costheta, phi) in onorm_angular mode (that's the model's
            # actual output), so it has to be decoded.
            # resolved per-tau since 'Approx' models substitute ts_hh_approx_ for
            # only the leptonically-decaying tau's leg (see hh_native_prefix)
            hh_p1, hh_p2 = hh_native_prefix(output_features, t1), hh_native_prefix(output_features, t2)
            true_n_p, true_r_p, true_k_p = df[f'{hh_p1}{t1}_n'].values, df[f'{hh_p1}{t1}_r'].values, df[f'{hh_p1}{t1}_k'].values
            true_n_m, true_r_m, true_k_m = df[f'{hh_p2}{t2}_n'].values, df[f'{hh_p2}{t2}_r'].values, df[f'{hh_p2}{t2}_k'].values
            if coordinates == 'onorm':
                pred_n_p, pred_r_p, pred_k_p = pred_native_df[f'{hh_p1}{t1}_n'].values, pred_native_df[f'{hh_p1}{t1}_r'].values, pred_native_df[f'{hh_p1}{t1}_k'].values
                pred_n_m, pred_r_m, pred_k_m = pred_native_df[f'{hh_p2}{t2}_n'].values, pred_native_df[f'{hh_p2}{t2}_r'].values, pred_native_df[f'{hh_p2}{t2}_k'].values
            else:  # onorm_angular: model only outputs (costheta, phi) -> decode to (n, r, k)
                def _angular_to_nrk(costheta, phi):
                    sintheta = np.sqrt(np.clip(1.0 - costheta ** 2, 0.0, None))
                    return sintheta * np.cos(phi), sintheta * np.sin(phi), costheta
                pred_n_p, pred_r_p, pred_k_p = _angular_to_nrk(pred_native_df[f'{hh_p1}{t1}_costheta'].values, pred_native_df[f'{hh_p1}{t1}_phi'].values)
                pred_n_m, pred_r_m, pred_k_m = _angular_to_nrk(pred_native_df[f'{hh_p2}{t2}_costheta'].values, pred_native_df[f'{hh_p2}{t2}_phi'].values)

            ent_df = pd.DataFrame({
                'true_cosn_plus': true_n_p, 'true_cosr_plus': true_r_p, 'true_cosk_plus': true_k_p,
                'true_cosn_minus': true_n_m, 'true_cosr_minus': true_r_m, 'true_cosk_minus': true_k_m,
                'pred_cosn_plus': pred_n_p, 'pred_cosr_plus': pred_r_p, 'pred_cosk_plus': pred_k_p,
                'pred_cosn_minus': pred_n_m, 'pred_cosr_minus': pred_r_m, 'pred_cosk_minus': pred_k_m,
                # same values again under the raw ts_hh_{t1}/{t2}_{n,r,k} naming
                # (the cosX_plus/minus naming above is just what
                # compute_spin_density_vars expects as input)
                f'true_ts_hh_{t1}_n': true_n_p, f'true_ts_hh_{t1}_r': true_r_p, f'true_ts_hh_{t1}_k': true_k_p,
                f'true_ts_hh_{t2}_n': true_n_m, f'true_ts_hh_{t2}_r': true_r_m, f'true_ts_hh_{t2}_k': true_k_m,
                f'pred_ts_hh_{t1}_n': pred_n_p, f'pred_ts_hh_{t1}_r': pred_r_p, f'pred_ts_hh_{t1}_k': pred_k_p,
                f'pred_ts_hh_{t2}_n': pred_n_m, f'pred_ts_hh_{t2}_r': pred_r_m, f'pred_ts_hh_{t2}_k': pred_k_m,
            })
            true_vars = compute_spin_density_vars(ent_df, prefix='true_')
            pred_vars = compute_spin_density_vars(ent_df, prefix='pred_')
            print('True  (B+, B-, C, concurrence, m12):', true_vars)
            print('Pred  (B+, B-, C, concurrence, m12):', pred_vars)
            plot_spin_density_matrix({'true': true_vars, 'pred': pred_vars}, dm_category='polvec_quicktest', outdir=outdir)
        else:
            print(">> Skipping entanglement-matrix plot (only implemented for coordinates='onorm'/'onorm_angular').")

        # === 5. save results (incl. TauSpinner weights) to parquet ===
        print(">> Saving results dataframe (with TauSpinner weights passed through)...")
        pred_h_t1_unit = unit(pred_cart_df[[f'ts_hh_{t1}_x', f'ts_hh_{t1}_y', f'ts_hh_{t1}_z']].values)
        pred_h_t2_unit = unit(pred_cart_df[[f'ts_hh_{t2}_x', f'ts_hh_{t2}_y', f'ts_hh_{t2}_z']].values)

        # friendly 4-vector output labels: keep the original 'tau_plus'/'tau_minus'
        # naming (backward compatible with existing results parquet schema) when
        # tau_labels are physical-charge based (taup/taun); for semileptonic
        # (tau1/tau2) that naming would be actively misleading (tau1 is "whichever
        # tau was leptonic", not tied to charge), so use tau1/tau2 directly there.
        d1, d2 = ('tau_plus', 'tau_minus') if tau_labels == ('taup', 'taun') else tau_labels

        results_df = pd.DataFrame({
            f'true_{d1}_E':  true_E_t1,
            f'true_{d1}_px': true_cart_df[f'undecayed_{t1}_px'].values,
            f'true_{d1}_py': true_cart_df[f'undecayed_{t1}_py'].values,
            f'true_{d1}_pz': true_cart_df[f'undecayed_{t1}_pz'].values,
            f'true_{d2}_E':  true_E_t2,
            f'true_{d2}_px': true_cart_df[f'undecayed_{t2}_px'].values,
            f'true_{d2}_py': true_cart_df[f'undecayed_{t2}_py'].values,
            f'true_{d2}_pz': true_cart_df[f'undecayed_{t2}_pz'].values,
            f'pred_{d1}_E':  pred_E_t1,
            f'pred_{d1}_px': pred_cart_df[f'undecayed_{t1}_px'].values,
            f'pred_{d1}_py': pred_cart_df[f'undecayed_{t1}_py'].values,
            f'pred_{d1}_pz': pred_cart_df[f'undecayed_{t1}_pz'].values,
            f'pred_{d2}_E':  pred_E_t2,
            f'pred_{d2}_px': pred_cart_df[f'undecayed_{t2}_px'].values,
            f'pred_{d2}_py': pred_cart_df[f'undecayed_{t2}_py'].values,
            f'pred_{d2}_pz': pred_cart_df[f'undecayed_{t2}_pz'].values,
            f'true_ts_hh_{t1}_x': true_cart_df[f'ts_hh_{t1}_x'].values,
            f'true_ts_hh_{t1}_y': true_cart_df[f'ts_hh_{t1}_y'].values,
            f'true_ts_hh_{t1}_z': true_cart_df[f'ts_hh_{t1}_z'].values,
            f'true_ts_hh_{t2}_x': true_cart_df[f'ts_hh_{t2}_x'].values,
            f'true_ts_hh_{t2}_y': true_cart_df[f'ts_hh_{t2}_y'].values,
            f'true_ts_hh_{t2}_z': true_cart_df[f'ts_hh_{t2}_z'].values,
            f'pred_ts_hh_{t1}_x': pred_h_t1_unit[:, 0],
            f'pred_ts_hh_{t1}_y': pred_h_t1_unit[:, 1],
            f'pred_ts_hh_{t1}_z': pred_h_t1_unit[:, 2],
            f'pred_ts_hh_{t2}_x': pred_h_t2_unit[:, 0],
            f'pred_ts_hh_{t2}_y': pred_h_t2_unit[:, 1],
            f'pred_ts_hh_{t2}_z': pred_h_t2_unit[:, 2],
            'true_phiCP': true_phiCP,
            'pred_phiCP': pred_phiCP,
            'pred_phiCP_err': pred_phiCP_err,
            f'gen_{t1}_DM': gen_dm_t1,
            f'gen_{t2}_DM': gen_dm_t2,
            f'reco_{t1}_DM': reco_dm_t1,
            f'reco_{t2}_DM': reco_dm_t2,
        })

        # derived (regressed tau + lepton) alternative to the model's native h_t1
        # prediction above -- stored under its own column names so it doesn't
        # overwrite pred_ts_hh_{t1}_*/pred_phiCP.
        if pred_h_t1_derived is not None:
            results_df[f'pred_ts_hh_derivedApproxLep_{t1}_x'] = pred_h_t1_derived[:, 0]
            results_df[f'pred_ts_hh_derivedApproxLep_{t1}_y'] = pred_h_t1_derived[:, 1]
            results_df[f'pred_ts_hh_derivedApproxLep_{t1}_z'] = pred_h_t1_derived[:, 2]
            results_df['pred_phiCP_derivedApproxLep'] = pred_phiCP_derived

        # Run3-classic lepton-side (R, P) alternative -- also its own column names,
        # doesn't overwrite pred_ts_hh_{t1}_*/pred_ts_hh_derivedApproxLep_{t1}_*.
        # Note R is the (ditau-frame-boosted) lepton impact parameter, not a unit
        # polarimetric vector -- kept unnormalised to match get_R_P_vectors_all's
        # own R_lep_ip convention.
        if pred_R_t1_run3lep is not None:
            results_df[f'pred_R_run3lep_{t1}_x'] = pred_R_t1_run3lep[:, 0]
            results_df[f'pred_R_run3lep_{t1}_y'] = pred_R_t1_run3lep[:, 1]
            results_df[f'pred_R_run3lep_{t1}_z'] = pred_R_t1_run3lep[:, 2]
            results_df['pred_phiCP_run3lep'] = pred_phiCP_run3lep

        # cos_n/r/k projections (true + predicted) that the entanglement/spin-density
        # matrix (compute_spin_density_vars) is built from -- saved so the C_ij matrix
        # can be recomputed/inspected downstream from the output tree directly.
        if ent_df is not None:
            results_df = pd.concat([results_df, ent_df.reset_index(drop=True)], axis=1)

        if phi_diag_cols is not None:
            results_df = pd.concat([results_df, pd.DataFrame(phi_diag_cols).reset_index(drop=True)], axis=1)

        # native onorm/onorm_angular-space values (true + predicted), i.e. before
        # ConvertFromOrthonormalNRK_Predictions_PolVec[_Angular] undoes the per-tau
        # visible-momentum (n,r,k) basis back to Cartesian.
        if coordinates == 'onorm':
            for col in onorm_cols:
                results_df[f'true_{col}'] = df[col].values
                results_df[f'pred_{col}'] = pred_native_df[col].values
        elif coordinates == 'onorm_angular':
            for col in angular_cols:
                results_df[f'true_{col}'] = df[col].values
                results_df[f'pred_{col}'] = pred_native_df[col].values
            # |h| before it was forced to a unit vector during data prep (see
            # ConvertNRKToAngular) -- a rare (~0.1-0.5%) heavy-tailed deviation
            # from 1 traced to numerical singularities in calculate_hh.py, not
            # a physics effect. Passed through so bad events (|h| far from 1)
            # can be identified/cut in downstream analysis of this results file.
            for norm_col in [f'ts_hh_{t1}_norm', f'ts_hh_{t2}_norm']:
                if norm_col in df.columns:
                    results_df[f'true_{norm_col}'] = df[norm_col].values

        # pass through TauSpinner weights (and any spin-correlation weight pieces) if present
        _AXES = ('n', 'r', 'k')
        _passthrough_cols = (
            [f'tauspinner_wt_alpha{a}' for a in [0, 45, 90]] +
            [f'wt_hp_{a}' for a in _AXES] +
            [f'wt_hm_{a}' for a in _AXES] +
            [f'wt_hp_{a}_hm_{b}' for a in _AXES for b in _AXES]
        )
        cols_to_pass = [c for c in _passthrough_cols if c in df.columns]
        missing_cols = [c for c in _passthrough_cols if c not in df.columns]
        if missing_cols:
            print(f">> WARNING: the following TauSpinner weight columns were not found and won't be saved: {missing_cols}")
        if cols_to_pass:
            results_df = pd.concat([results_df, df[cols_to_pass].reset_index(drop=True)], axis=1)

        results_path = os.path.join(output_dir, f'polvec_eval_results_{test_output_name}.parquet')
        results_df.to_parquet(results_path)
        print(f">> Saved results dataframe ({len(results_df)} events, {len(results_df.columns)} columns) to {results_path}")

        print(f">> Done with {test_output_name}. Plots saved to {outdir}")


if __name__ == "__main__":
    main()
