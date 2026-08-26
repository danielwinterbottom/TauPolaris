"""
Plot the per-event sampled phiCP distribution for a handful of individual
DM=0/DM=0 events, so a "good" (small spread) and "poor" (large spread) event
can be picked out by eye for the paper -- as opposed to evaluate.py's
--inc_phiCP_error, which draws the same kind of samples per event but only
ever keeps the collapsed circular_std summary, not the individual phiCP
values, so there's nothing there to make a per-event histogram from.

Reuses the same model-loading, dataset-loading, DM=0/DM=0 filter, and flow
sampling (with its AssertionError-retry) as evaluate.py, restricted to a
small number of selected events with more samples per event, so this stays
consistent with the actual method described in the paper.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from taupolaris.python.NN_Tools import load_model, get_device, is_legacy_pizero_proj_checkpoint
from taupolaris.python.DataProcessing import get_test_dataset, RegressionDataset
from taupolaris.utils.coordinate_conversions import convert_coordinates_pred
from taupolaris.utils.kinematic_helpers import add_energies_pair
from taupolaris.utils.acoplanarity_tools import get_ditau_polarimetric, compute_aco_polarimetric


def circular_std(angles, axis=-1):
    """Same definition as evaluate.py's circular_std (Mardia & Jupp)."""
    mean_cos = np.mean(np.cos(angles), axis=axis)
    mean_sin = np.mean(np.sin(angles), axis=axis)
    R = np.clip(np.sqrt(mean_cos ** 2 + mean_sin ** 2), 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(R))


def circular_mean(angles, axis=-1):
    """Circular mean, wrapped to [0, 2*pi) to match phiCP's usual convention
    (arctan2's own principal range is (-pi, pi], which would otherwise silently
    switch branch relative to compute_aco_polarimetric's [0, 2*pi) output)."""
    mean = np.arctan2(np.mean(np.sin(angles), axis=axis), np.mean(np.cos(angles), axis=axis))
    return np.mod(mean, 2 * np.pi)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', required=True, help='path to the configuration file')
    argparser.add_argument('--useCPU', action='store_true')
    argparser.add_argument('--n_events', type=int, default=10, help='number of individual DM=0/DM=0 events to plot')
    argparser.add_argument('--n_samples', type=int, default=200, help='number of flow samples per event')
    argparser.add_argument('--seed', type=int, default=42, help='random seed used to pick which events to plot')
    argparser.add_argument('--n_bins', type=int, default=40)
    argparser.add_argument('--outdir', type=str, default='phiCP_event_samples')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']
    coordinates = data_config['coordinates']
    use_reco = data_config['use_reco']
    leptonic_mode = data_config.get('leptonic_mode', 0)

    device = torch.device('cpu') if args.useCPU else get_device()

    output_dir = f"outputs_{nn_config['model_name']}"
    output_plots_dir = f"{output_dir}/plots"

    input_features = data_config['Features']['input_features']
    output_features = data_config['Features']['output_features'][coordinates]
    hp = nn_config['hyperparams']
    is_transformer = nn_config.get('use_transformer', False)

    model_path = f'{output_plots_dir}/best_model.pth'
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        model_path = f'{output_plots_dir}/partial_model.pth'
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    legacy_pizero_proj = is_legacy_pizero_proj_checkpoint(state_dict)
    model = load_model(hp, input_features, output_features, batch_norm=False,
                        useMLP=False, useTransformer=is_transformer, useTransformerMLP=False,
                        leptonic_mode=leptonic_mode, legacy_pizero_proj=legacy_pizero_proj)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.float().to(device)

    norm_data = np.load(f'{output_dir}/normalization_params.npz')

    # test_dataset/test_output_name are always stored as lists in the config (even for a
    # single entry, per evaluate.py's convention) -- this script only ever needs the first.
    if isinstance(data_config["test_dataset"], list):
        if len(data_config["test_dataset"]) > 1:
            print(f">> Multiple test_dataset entries in config; using the first: "
                  f"{data_config['test_dataset'][0]}")
        data_config["test_dataset"] = data_config["test_dataset"][0]
        data_config["test_output_name"] = data_config["test_output_name"][0]

    test_dataset, test_df, _, _ = get_test_dataset(data_config, norm_data)
    del test_dataset

    # Same DM=0/DM=0 filter as evaluate.py's temporary filter.
    dm0_mask = ((test_df['reco_taup_npizero'] == 0) & (test_df['reco_taup_is3prong'] == 0) &
                (test_df['reco_taun_npizero'] == 0) & (test_df['reco_taun_is3prong'] == 0))
    test_df = test_df[dm0_mask].reset_index(drop=True)
    print(f">> DM=0/DM=0 events available: {len(test_df)}")

    rng = np.random.RandomState(args.seed)
    n_events = min(args.n_events, len(test_df))
    event_idx = rng.choice(len(test_df), size=n_events, replace=False)
    sel_df = test_df.iloc[event_idx].reset_index(drop=True)

    sel_dataset = RegressionDataset(
        sel_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True,
        input_mean=torch.from_numpy(norm_data['input_mean']), input_std=torch.from_numpy(norm_data['input_std']),
        output_mean=torch.from_numpy(norm_data['output_mean']), output_std=torch.from_numpy(norm_data['output_std']),
    )
    X_sel, _ = sel_dataset[:]
    X_sel = X_sel.to(device)

    pion_prefix = 'reco'
    # get_ditau_polarimetric expects '..._E' (uppercase) on its inputs, but the raw
    # test dataframe stores energy as '..._e' (lowercase) -- same relabeling evaluate.py
    # does when building its own results_df from the raw test_df.
    def _raw_col(tau, part, comp):
        raw_comp = 'e' if comp == 'E' else comp
        return f'{pion_prefix}_{tau}_{part}_{raw_comp}'

    pion_out_cols, pion_raw_cols = [], []
    for tau in ('taup', 'taun'):
        for part in ('pi1', 'pi2', 'pi3', 'pizero1', 'charged'):
            for comp in ('E', 'px', 'py', 'pz'):
                raw_col = _raw_col(tau, part, comp)
                if raw_col in sel_df.columns:
                    pion_out_cols.append(f'{pion_prefix}_{tau}_{part}_{comp}')
                    pion_raw_cols.append(raw_col)
    pion_vals = sel_df[pion_raw_cols].values  # values, ordered to match pion_out_cols

    def _charged_pi0(tau):
        charged = sel_df[[_raw_col(tau, 'charged', c) for c in ('E', 'px', 'py', 'pz')]].values
        pi0 = sel_df[[_raw_col(tau, 'pizero1', c) for c in ('E', 'px', 'py', 'pz')]].values
        return charged, pi0
    taup_charged, taup_pi0 = _charged_pi0('taup')
    taun_charged, taun_pi0 = _charged_pi0('taun')

    n_fs = args.n_samples
    print(f">> Drawing {n_fs} flow samples for each of {n_events} events...")

    samples_norm = None
    for attempt in range(5):
        try:
            with torch.no_grad():
                samples_norm = model.sample(num_samples=n_fs, context=X_sel)  # [n_events, n_fs, F]
            break
        except AssertionError:
            continue
    if samples_norm is None:
        raise RuntimeError("flow sampling failed on 5 retries -- try a different --seed or fewer --n_samples")

    samples = sel_dataset.destandardize_outputs(samples_norm).cpu().numpy().reshape(n_events * n_fs, -1)

    rep = lambda arr: np.repeat(arr, n_fs, axis=0)
    samples = convert_coordinates_pred(
        samples, coordinates=coordinates, output_features=output_features,
        tau1_charged=rep(taup_charged), tau1_pi0=rep(taup_pi0),
        tau2_charged=rep(taun_charged), tau2_pi0=rep(taun_pi0),
        leptonic_mode=leptonic_mode,
    )
    samples = add_energies_pair(samples)

    tau_plus_samp = samples[:, 0:4] + rep(taup_charged) + rep(taup_pi0)
    tau_minus_samp = samples[:, 4:8] + rep(taun_charged) + rep(taun_pi0)

    tmp_df = pd.DataFrame(rep(pion_vals), columns=pion_out_cols)
    tmp_df['taup_DM'] = 0
    tmp_df['taun_DM'] = 0
    tmp_df['samp_tau_plus_E'], tmp_df['samp_tau_plus_px'] = tau_plus_samp[:, 0], tau_plus_samp[:, 1]
    tmp_df['samp_tau_plus_py'], tmp_df['samp_tau_plus_pz'] = tau_plus_samp[:, 2], tau_plus_samp[:, 3]
    tmp_df['samp_tau_minus_E'], tmp_df['samp_tau_minus_px'] = tau_minus_samp[:, 0], tau_minus_samp[:, 1]
    tmp_df['samp_tau_minus_py'], tmp_df['samp_tau_minus_pz'] = tau_minus_samp[:, 2], tau_minus_samp[:, 3]

    R1, P1, R2, P2 = get_ditau_polarimetric(tmp_df, tau_prefix='samp', reco_pions=True)
    phiCP_flat = np.asarray(compute_aco_polarimetric(R1, P1, R2, P2))
    phiCP_samples = phiCP_flat.reshape(n_events, n_fs)  # [n_events, n_fs]

    sigmas = circular_std(phiCP_samples, axis=1)
    means = circular_mean(phiCP_samples, axis=1)

    os.makedirs(args.outdir, exist_ok=True)

    summary = pd.DataFrame({
        'event_idx_in_dm0dm0_sample': event_idx,
        'mean_phiCP_rad': means,
        'sigma_phiCP_rad': sigmas,
    }).sort_values('sigma_phiCP_rad').reset_index(drop=True)
    summary_path = os.path.join(args.outdir, 'summary.csv')
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f">> Wrote {summary_path}")
    print(f">> Smallest sigma: event {summary.iloc[0]['event_idx_in_dm0dm0_sample']:.0f} "
          f"(sigma={summary.iloc[0]['sigma_phiCP_rad']:.3f} rad)")
    print(f">> Largest sigma:  event {summary.iloc[-1]['event_idx_in_dm0dm0_sample']:.0f} "
          f"(sigma={summary.iloc[-1]['sigma_phiCP_rad']:.3f} rad)")

    # Raw samples aren't otherwise persisted anywhere -- save them so replotting
    # (different range/binning/style) doesn't require re-running the flow model.
    samples_path = os.path.join(args.outdir, 'phiCP_samples_rad.npz')
    np.savez(samples_path, event_idx=event_idx, phiCP_samples=phiCP_samples)
    print(f">> Wrote {samples_path}")

    pi_ticks = np.array([0, 0.5, 1, 1.5, 2]) * np.pi
    pi_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']

    for i in range(n_events):
        idx = event_idx[i]
        # Fixed [0, 2*pi) axis for easy by-eye comparison across events, at the cost of a
        # spurious-looking split for any event whose distribution straddles the seam (mean
        # near 0/2*pi with a large enough spread) -- check summary.csv's mean_phiCP_rad
        # before reading too much into an odd-looking bimodal plot.
        vals = np.mod(phiCP_samples[i], 2 * np.pi)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(vals, bins=args.n_bins, range=(0, 2 * np.pi), histtype='step', linewidth=1.8, color='#e42536')
        ax.axvline(means[i], color='grey', linestyle='--', linewidth=1.0)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(pi_ticks)
        ax.set_xticklabels(pi_labels)
        ax.set_xlabel(r'$\phi_{CP}$ [rad]')
        ax.set_ylabel('Samples / bin')
        ax.text(0.05, 0.95, rf'$\sigma_{{\phi_{{CP}}}} = {sigmas[i]:.3f}$ rad',
                transform=ax.transAxes, va='top', ha='left')
        fig.tight_layout()
        fname = os.path.join(args.outdir, f'event{idx}_phiCP_samples.pdf')
        fig.savefig(fname)
        plt.close(fig)
        print(f">> Saved {fname} (sigma={sigmas[i]:.3f} rad)")


if __name__ == "__main__":
    main()
