import torch
import argparse
import yaml
import os
import numpy as np
import pandas as pd
from taupolaris.python.NN_Tools import load_model
from taupolaris.python.DataProcessing import get_test_dataset
from taupolaris.python.Evaluation_Tools import save_sampled_pdfs_LHC, flow_map_predict
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 16})

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', required=True, type=str,
                           help='path to the config file used for training/evaluation')
    argparser.add_argument('--output_name', '-o', required=True, type=str,
                           help='test_output_name used in evaluate.py (sets output subdirectory name)')
    argparser.add_argument('--events', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                           help='event indices to plot')
    argparser.add_argument('--num_samples', type=int, default=50000,
                           help='number of flow samples per event')
    argparser.add_argument('--bins', type=int, default=100)
    argparser.add_argument('--no_map', action='store_true',
                           help='skip MAP estimate overlay')
    argparser.add_argument('--stochastic_map', action='store_true',
                           help='also compute and overlay the stochastic MAP estimate, in addition to the gradient MAP')
    argparser.add_argument('--transformer_input', type=str, default=None,
                           help='path to a parquet file with transformer predictions (same pred_nu_*/pred_nubar_* '
                                'column naming as evaluate.py output, row-aligned with the test dataset); '
                                'overlaid as a green line on the regressed-neutrino plots if given')
    argparser.add_argument('--useCPU', action='store_true')
    argparser.add_argument('--scan_pdf', action='store_true')
    args = argparser.parse_args()

    torch.manual_seed(42)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.useCPU else 'cpu')

    output_dir = f"outputs_{nn_config['model_name']}"
    output_plots_dir = f"{output_dir}/plots"

    input_features = data_config['Features']['input_features']
    output_features = data_config['Features']['output_features'][data_config['coordinates']]
    is_transformer = nn_config.get('use_transformer', False)
    leptonic_mode = data_config.get('leptonic_mode', 0)
    hp = nn_config['hyperparams']

    model = load_model(hp, input_features, output_features, batch_norm=False,
                       useMLP=False, useTransformer=is_transformer,
                       useTransformerMLP=False, leptonic_mode=leptonic_mode)

    model_path = f'{output_plots_dir}/best_model.pth'
    if not os.path.exists(model_path):
        model_path = f'{output_plots_dir}/partial_model.pth'
    print(f"Loading model from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model = model.to(device)
    print(">> Successfully loaded model")

    norm_data = np.load(f'{output_dir}/normalization_params.npz')

    data_config['test_dataset'] = data_config['test_dataset'] if not isinstance(data_config['test_dataset'], list) else data_config['test_dataset'][0]
    data_config['test_output_name'] = args.output_name
    test_dataset, test_df, _, _ = get_test_dataset(data_config, norm_data, oneprong=False)

    outdir = f"{output_dir}/pdf_slices_sampled_{args.output_name}"
    os.makedirs(outdir, exist_ok=True)
    print(f"Saving plots to {outdir}/")

    scan_outdir = os.path.join(outdir, "likelihood_scans")
    if args.scan_pdf:
        os.makedirs(scan_outdir, exist_ok=True)

    map_method  = nn_config.get('map_method', 'gradient')
    map_num_draws = nn_config.get('map_num_draws', 100)

    transformer_df = None
    if args.transformer_input:
        transformer_df = pd.read_parquet(args.transformer_input)
        print(f">> Loaded transformer predictions from {args.transformer_input}")

    # Compute MAP estimates for all requested events in a single batched call
    # instead of looping event-by-event: the gradient optimisation (200 Adam
    # steps through the flow) is the dominant cost, and batching lets every
    # event share those steps rather than repeating them serially per event.
    map_pred_all = None
    map_pred_all_stochastic = None
    if not args.no_map:
        print(f"Computing MAP estimates for {len(args.events)} event(s) (batched, method='{map_method}')...")
        X_events = test_dataset.X[args.events].to(device)
        map_norm_all, map_pred_all, log_prob_z_all = flow_map_predict(
            model, X_events,
            test_dataset=test_dataset,
            num_draws=map_num_draws,
            chunk_size=len(args.events),
            method=map_method,
            return_log_prob=True
        )
        for event_number, log_prob_z in zip(args.events, log_prob_z_all):
            print(f"  event {event_number}: found likelihood {log_prob_z:.4f}")

        if args.stochastic_map:
            print(f"Computing alternative MAP estimates for {len(args.events)} event(s) (batched, method='stochastic')...")
            _, map_pred_all_stochastic = flow_map_predict(
                model, X_events,
                test_dataset=test_dataset,
                num_draws=map_num_draws,
                chunk_size=len(args.events),
                method='stochastic',
            )

    for i, event_number in enumerate(args.events):
        print(f"  Sampling event {event_number}...")

        map_value = None
        map_value2 = None
        if not args.no_map:
            map_value = map_pred_all[i]
            if map_pred_all_stochastic is not None:
                map_value2 = map_pred_all_stochastic[i]
            X_event = X_events[i:i + 1]
            map_norm = map_norm_all[i:i + 1]
            log_prob_z = log_prob_z_all[i]

            if args.scan_pdf:
                _plot_likelihood_scan(
                    model, X_event, map_norm[0], log_prob_z,
                    event_number, test_dataset, scan_outdir, device
                )

        transformer_value = None
        if transformer_df is not None:
            transformer_value = _transformer_nu_momenta(transformer_df, event_number)

        save_sampled_pdfs_LHC(
            model=model,
            device=device,
            dataset=test_dataset,
            output_features=output_features,
            event_number=event_number,
            num_samples=args.num_samples,
            bins=args.bins,
            outdir=outdir,
            map_value=map_value,
            map_value2=map_value2,
            transformer_value=transformer_value,
            df=test_df,
            coordinates=data_config['coordinates'],
            leptonic_mode=leptonic_mode
        )

    print("Done.")


def _transformer_nu_momenta(transformer_df, event_number):
    """Extract [taup_nu_px, taup_nu_py, taup_nu_pz, taun_nu_px, taun_nu_py, taun_nu_pz]
    for one event from a transformer predictions parquet using evaluate.py's
    pred_nubar_*/pred_nu_* column naming (taup_nu <-> nubar, taun_nu <-> nu)."""
    row = transformer_df.iloc[event_number]
    return np.array([
        row['pred_nubar_px'], row['pred_nubar_py'], row['pred_nubar_pz'],
        row['pred_nu_px'], row['pred_nu_py'], row['pred_nu_pz'],
    ])


def _plot_likelihood_scan(model, X_event, map_norm, log_prob_z, event_number, test_dataset, outdir, device):
    """Profile log p against a single shifted component (others frozen), for both
    taus, using one batched forward pass per tau instead of a per-shift Python loop."""
    scan_indices = {'taun': 5, 'taup': 2}  # 6th / 3rd element respectively
    shifts = np.arange(-1, 1, 0.01)
    shifts_t = torch.from_numpy(shifts).float().to(device)
    n_shifts = len(shifts)

    x_map_norm = map_norm.to(device)
    X_batch = X_event.repeat(n_shifts, 1)

    for tau, scan_idx in scan_indices.items():
        print(f">> Profiling likelihood for k in {tau} (others frozen)")
        x_shifted = x_map_norm.unsqueeze(0).repeat(n_shifts, 1)
        x_shifted[:, scan_idx] += shifts_t

        with torch.no_grad():
            z, logabsdet_enc = model.encode(x_shifted, context=X_batch)
            scan_log_p = (-0.5 * (z ** 2).sum(dim=-1) + logabsdet_enc).cpu().numpy()

        x_shifted_phys = test_dataset.destandardize_outputs(x_shifted.cpu())[:, scan_idx].numpy()
        x_range = x_map_norm[scan_idx].item() + shifts

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x_range, scan_log_p)
        ax.axhline(y=float(log_prob_z), color='red', linestyle='--', label='MAP log p')
        ax.legend()
        ax.set_xlabel(f"k component {tau} (entry {scan_idx+1}) - standardised")
        ax.set_ylabel('log(p)')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"profile_event{event_number}_{tau}_standardised.pdf"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x_shifted_phys, scan_log_p)
        ax.axhline(y=float(log_prob_z), color='red', linestyle='--', label='MAP log p')
        ax.legend()
        ax.set_xlabel(f"k component {tau} (entry {scan_idx+1}) - physical")
        ax.set_ylabel('log(p)')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"profile_event{event_number}_{tau}_physical.pdf"))
        plt.close(fig)


if __name__ == '__main__':
    main()