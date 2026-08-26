import torch
import pandas as pd
import argparse
import yaml
import os
import uproot
import numpy as np
from tqdm import tqdm
from taupolaris.python.NN_Tools import load_model, get_device, is_legacy_pizero_proj_checkpoint
from taupolaris.python.DataProcessing import get_test_dataset
from taupolaris.utils.coordinate_conversions import convert_coordinates_pred
from taupolaris.python.Evaluation_Tools import flow_map_predict, compute_spin_vars, save_sampled_pdfs, plot_spin_density_matrix
from taupolaris.utils.kinematic_helpers import compute_spin_density_vars, add_energies_pair, add_energy, inv_mass
from taupolaris.utils.acoplanarity_tools import get_ditau_polarimetric, compute_aco_polarimetric


def circular_std(angles, axis=-1):
    """Circular standard deviation (Mardia & Jupp). phiCP is periodic on
    [0, 2*pi), so a naive std would report a spuriously large error for any
    event whose central estimate sits near the 0/2*pi seam with samples
    landing just across it (e.g. central~0.02 rad with samples at ~0.01 and
    ~6.27 -- actually tightly clustered, not ~6 rad apart). Same definition
    as evaluate_polvec.py's circular_std."""
    mean_cos = np.mean(np.cos(angles), axis=axis)
    mean_sin = np.mean(np.sin(angles), axis=axis)
    R = np.clip(np.sqrt(mean_cos ** 2 + mean_sin ** 2), 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(R))


def _phiCP_dm_codes(df, prefix):
    """Per-tau decay-mode code (0/1/2/10/11 hadronic, 100 leptonic, -1
    unmatched), needed by get_ditau_polarimetric. Same convention as
    plot_phiCP.py's add_DM / evaluate_polvec.py's add_DM. `prefix` is the
    reconstruction level to classify at ('reco' or 'true'), matching
    whichever level the pion four-momenta are taken from."""
    dm = {}
    for tau in ('taup', 'taun'):
        is_lep = df[f'{prefix}_{tau}_ishadronic'].values == 0
        npizero = df[f'{prefix}_{tau}_npizero'].values
        is3prong = df[f'{prefix}_{tau}_is3prong'].values
        is_dm0 = (npizero == 0) & (is3prong == 0) & (~is_lep)
        is_dm1 = (npizero == 1) & (is3prong == 0) & (~is_lep)
        is_dm2 = ((npizero == 1) | (npizero == 2)) & (is3prong == 0) & (~is_lep)
        is_dm10 = (npizero == 0) & (is3prong == 1) & (~is_lep)
        is_dm11 = (npizero == 1) & (is3prong == 1) & (~is_lep)
        dm[tau] = np.where(is_dm0, 0,
                    np.where(is_dm1, 1,
                        np.where(is_dm2, 2,
                            np.where(is_dm10, 10,
                                np.where(is_dm11, 11,
                                    np.where(is_lep, 100, -1))))))
    return dm['taup'], dm['taun']


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', help='path to the configuration file', type=str, default='taupolaris/config/LEP.yaml', required=True)
    argparser.add_argument('--useMLP', help='whether to use a simple MLP instead of a normalizing flow', action='store_true')
    argparser.add_argument('--useTransformerBaseline', help='whether to use a transformer encoder + regression head instead of a normalizing flow', action='store_true')
    argparser.add_argument('--useCPU', help='whether to use CPU only for evaluation', action='store_true')
    argparser.add_argument('--oneprong', help='whether to only evaluate on 1-prong taus only', action='store_true')
    argparser.add_argument('--threeprong', help='whether to only evaluate on events with at least 1 3-prong tau', action='store_true')
    argparser.add_argument('--make_root_output', help='whether to save the results in a root file as well as a pandas dataframe', action='store_true')
    argparser.add_argument('--inc_significance', help='whether to include the estimate of the neutrino \"significance\"', action='store_true')
    argparser.add_argument('--inc_phiCP_error', help='whether to compute phiCP from the predicted neutrino (pred_phiCP/map_pred_phiCP) and its per-event uncertainty (pred_phiCP_err) from repeated flow sampling -- the error from the neutrino regression alone, with all visible momenta held fixed. Needs the flow\'s posterior, so has no effect with --useMLP/--useTransformerBaseline.', action='store_true')
    argparser.add_argument('--n_flow_samples_phiCP', type=int, default=50, help='number of flow samples per event used to estimate the phiCP uncertainty (only used with --inc_phiCP_error)')
    args = argparser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']

    nn_config   = config['SetupNN']
    coordinates = data_config['coordinates']
    use_reco = data_config['use_reco']

    if use_reco: prefix = 'reco_'
    else: prefix = ''

    # set gpu or cpu
    device = torch.device('cpu') if args.useCPU else get_device()

    output_dir = f"outputs_{nn_config['model_name']}"
    output_plots_dir = f"{output_dir}/plots"

    # load model
    input_features = data_config['Features']['input_features']
    output_features = data_config['Features']['output_features'][data_config['coordinates']]

    if args.useMLP:
        hp = nn_config['MLP_hyperparams']
    elif args.useTransformerBaseline:
        hp = nn_config['TransformerBaseline_hyperparams']
    else:
        hp = nn_config['hyperparams']
    is_transformer = nn_config.get('use_transformer', False)
    leptonic_mode = data_config.get('leptonic_mode', 0)
    model_path = f'{output_plots_dir}/best_model.pth'
    print(f"Using model {nn_config['model_name']}")
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):  # check if model exists, if not take partial model
        model_path = f'{output_plots_dir}/partial_model.pth'
    # map_location='cpu' always works regardless of what device the checkpoint was
    # saved on (and regardless of whether a GPU is available here)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    legacy_pizero_proj = is_legacy_pizero_proj_checkpoint(state_dict)
    if legacy_pizero_proj:
        print(">> Checkpoint predates the pizero_proj/final-LayerNorm architecture change; "
              "building the matching (older) model architecture.")
    model = load_model(hp, input_features, output_features, batch_norm=False, useMLP=args.useMLP, useTransformer=is_transformer, useTransformerMLP=args.useTransformerBaseline, leptonic_mode=leptonic_mode, legacy_pizero_proj=legacy_pizero_proj)
    model.load_state_dict(state_dict)
    print(">> Successfully loaded model")
    model.eval()

    norm_data = np.load(f'{output_dir}/normalization_params.npz') # we get the means and stds used in training the model so that we can apply the same normalization to the test dataset

    #check if data_config["test_dataset"] is a list, if so we will loop over it and replace test_dataset with each list element in turn
    if isinstance(data_config["test_dataset"], list):
        test_datasets = data_config["test_dataset"]
        # check if test_output_name is also a list of the same length, if not raise error
        if not isinstance(data_config["test_output_name"], list):
            raise ValueError("If test_dataset is a list, test_output_name must also be a list")
        if isinstance(data_config["test_output_name"], list):
            if len(data_config["test_output_name"]) != len(data_config["test_dataset"]):
                raise ValueError("test_output_name must have the same length as test_dataset")
        test_output_names = data_config["test_output_name"]
    else:
        test_datasets = [data_config["test_dataset"]]
        # raise exception if test_output_name is a list
        if isinstance(data_config["test_output_name"], list):
            raise ValueError("If test_dataset is not a list, test_output_name must not be a list")
        test_output_names = [data_config["test_output_name"]]

    for test_dataset_name, test_output_name in zip(test_datasets, test_output_names):

        data_config["test_dataset"] = test_dataset_name
        data_config["test_output_name"] = test_output_name
        test_dataset, test_df, _, _ = get_test_dataset(data_config, norm_data, oneprong=args.oneprong)

        print(f'Evaluating on test dataset {test_dataset_name}')
        print(f'Number of events in test dataset: {len(test_dataset)}')
    
        if 'taup_nu_px' in test_df.columns:
            tau1_prefix = 'taup'
            tau2_prefix = 'taun'
        else:
            tau1_prefix = 'tau1'
            tau2_prefix = 'tau2'
    
        # get tau pi and pizero four vectors from test_df
        true_taun_pi = test_df[[f'{tau2_prefix}_pi1_e', f'{tau2_prefix}_pi1_px', f'{tau2_prefix}_pi1_py', f'{tau2_prefix}_pi1_pz']].values
        true_taup_pi  = test_df[[f'{tau1_prefix}_pi1_e', f'{tau1_prefix}_pi1_px', f'{tau1_prefix}_pi1_py', f'{tau1_prefix}_pi1_pz']].values
        true_taun_pizero  = test_df[[f'{tau2_prefix}_pizero1_e', f'{tau2_prefix}_pizero1_px', f'{tau2_prefix}_pizero1_py', f'{tau2_prefix}_pizero1_pz']].values
        true_taup_pizero = test_df[[f'{tau1_prefix}_pizero1_e', f'{tau1_prefix}_pizero1_px', f'{tau1_prefix}_pizero1_py', f'{tau1_prefix}_pizero1_pz']].values


        # gets ips as well
        true_taun_pi_ip = test_df[[f'{tau2_prefix}_pi1_ipx', f'{tau2_prefix}_pi1_ipy', f'{tau2_prefix}_pi1_ipz']].values
        true_taup_pi_ip = test_df[[f'{tau1_prefix}_pi1_ipx', f'{tau1_prefix}_pi1_ipy', f'{tau1_prefix}_pi1_ipz']].values
    
        # pi2/pi3 for 3-prong taus
        if 'taup_pi2_e' in test_df.columns or 'tau1_pi2_e' in test_df.columns:
            true_taup_pi2 = test_df[[f'{tau1_prefix}_pi2_e', f'{tau1_prefix}_pi2_px', f'{tau1_prefix}_pi2_py', f'{tau1_prefix}_pi2_pz']].values
            true_taun_pi2 = test_df[[f'{tau2_prefix}_pi2_e', f'{tau2_prefix}_pi2_px', f'{tau2_prefix}_pi2_py', f'{tau2_prefix}_pi2_pz']].values
            true_taup_pi3 = test_df[[f'{tau1_prefix}_pi3_e', f'{tau1_prefix}_pi3_px', f'{tau1_prefix}_pi3_py', f'{tau1_prefix}_pi3_pz']].values
            true_taun_pi3 = test_df[[f'{tau2_prefix}_pi3_e', f'{tau2_prefix}_pi3_px', f'{tau2_prefix}_pi3_py', f'{tau2_prefix}_pi3_pz']].values
        else:
            true_taup_pi2 = np.zeros((len(test_df), 4))
            true_taun_pi2 = np.zeros((len(test_df), 4))
            true_taup_pi3 = np.zeros((len(test_df), 4))
            true_taun_pi3 = np.zeros((len(test_df), 4))
    
        inc_new_vars = 'taup_charged_e' in test_df.columns or 'tau1_charged_e' in test_df.columns
    
        # check if charged component exists
        if inc_new_vars:
            true_taup_charged = test_df[[f'{tau1_prefix}_charged_e', f'{tau1_prefix}_charged_px', f'{tau1_prefix}_charged_py', f'{tau1_prefix}_charged_pz']].values
            true_taun_charged = test_df[[f'{tau2_prefix}_charged_e', f'{tau2_prefix}_charged_px', f'{tau2_prefix}_charged_py', f'{tau2_prefix}_charged_pz']].values
            true_taun_charged_ip = test_df[[f'{tau2_prefix}_charged_ipx', f'{tau2_prefix}_charged_ipy', f'{tau2_prefix}_charged_ipz']].values
            true_taup_charged_ip = test_df[[f'{tau1_prefix}_charged_ipx', f'{tau1_prefix}_charged_ipy', f'{tau1_prefix}_charged_ipz']].values
            true_taup_sv = test_df[[f'{tau1_prefix}_sv_x', f'{tau1_prefix}_sv_y', f'{tau1_prefix}_sv_z']].values
            true_taun_sv = test_df[[f'{tau2_prefix}_sv_x', f'{tau2_prefix}_sv_y', f'{tau2_prefix}_sv_z']].values
    
        else: # else use the pi four vectors as the charged component 
            true_taup_charged = true_taup_pi
            true_taun_charged = true_taun_pi
            true_taun_charged_ip = true_taun_pi_ip
            true_taup_charged_ip = true_taup_pi_ip
            # set sv's to 0
            true_taup_sv = np.zeros((len(test_df), 3))
            true_taun_sv = np.zeros((len(test_df), 3))
    
        if use_reco:
            reco_taun_pi = test_df[[f'reco_{tau2_prefix}_pi1_e', f'reco_{tau2_prefix}_pi1_px', f'reco_{tau2_prefix}_pi1_py', f'reco_{tau2_prefix}_pi1_pz']].values
            reco_taup_pi  = test_df[[f'reco_{tau1_prefix}_pi1_e', f'reco_{tau1_prefix}_pi1_px', f'reco_{tau1_prefix}_pi1_py', f'reco_{tau1_prefix}_pi1_pz']].values
            reco_taun_pizero  = test_df[[f'reco_{tau2_prefix}_pizero1_e', f'reco_{tau2_prefix}_pizero1_px', f'reco_{tau2_prefix}_pizero1_py', f'reco_{tau2_prefix}_pizero1_pz']].values
            reco_taup_pizero = test_df[[f'reco_{tau1_prefix}_pizero1_e', f'reco_{tau1_prefix}_pizero1_px', f'reco_{tau1_prefix}_pizero1_py', f'reco_{tau1_prefix}_pizero1_pz']].values
    
            # get ips for reco pis as well
            reco_taun_pi_ip = test_df[[f'reco_{tau2_prefix}_pi1_ipx', f'reco_{tau2_prefix}_pi1_ipy', f'reco_{tau2_prefix}_pi1_ipz']].values
            reco_taup_pi_ip = test_df[[f'reco_{tau1_prefix}_pi1_ipx', f'reco_{tau1_prefix}_pi1_ipy', f'reco_{tau1_prefix}_pi1_ipz']].values
    
            if 'reco_taup_pi2_e' in test_df.columns or 'reco_tau1_pi2_e' in test_df.columns:
                reco_taup_pi2 = test_df[[f'reco_{tau1_prefix}_pi2_e', f'reco_{tau1_prefix}_pi2_px', f'reco_{tau1_prefix}_pi2_py', f'reco_{tau1_prefix}_pi2_pz']].values
                reco_taun_pi2 = test_df[[f'reco_{tau2_prefix}_pi2_e', f'reco_{tau2_prefix}_pi2_px', f'reco_{tau2_prefix}_pi2_py', f'reco_{tau2_prefix}_pi2_pz']].values
                reco_taup_pi3 = test_df[[f'reco_{tau1_prefix}_pi3_e', f'reco_{tau1_prefix}_pi3_px', f'reco_{tau1_prefix}_pi3_py', f'reco_{tau1_prefix}_pi3_pz']].values
                reco_taun_pi3 = test_df[[f'reco_{tau2_prefix}_pi3_e', f'reco_{tau2_prefix}_pi3_px', f'reco_{tau2_prefix}_pi3_py', f'reco_{tau2_prefix}_pi3_pz']].values
            else:
                reco_taup_pi2 = np.zeros((len(test_df), 4))
                reco_taun_pi2 = np.zeros((len(test_df), 4))
                reco_taup_pi3 = np.zeros((len(test_df), 4))
                reco_taun_pi3 = np.zeros((len(test_df), 4))
    
            if inc_new_vars:
                reco_taup_charged = test_df[[f'reco_{tau1_prefix}_charged_e', f'reco_{tau1_prefix}_charged_px', f'reco_{tau1_prefix}_charged_py', f'reco_{tau1_prefix}_charged_pz']].values
                reco_taun_charged = test_df[[f'reco_{tau2_prefix}_charged_e', f'reco_{tau2_prefix}_charged_px', f'reco_{tau2_prefix}_charged_py', f'reco_{tau2_prefix}_charged_pz']].values
                reco_taun_charged_ip = test_df[[f'reco_{tau2_prefix}_charged_ipx', f'reco_{tau2_prefix}_charged_ipy', f'reco_{tau2_prefix}_charged_ipz']].values
                reco_taup_charged_ip = test_df[[f'reco_{tau1_prefix}_charged_ipx', f'reco_{tau1_prefix}_charged_ipy', f'reco_{tau1_prefix}_charged_ipz']].values
                reco_taup_sv = test_df[[f'reco_{tau1_prefix}_sv_x', f'reco_{tau1_prefix}_sv_y', f'reco_{tau1_prefix}_sv_z']].values if use_reco else None
                reco_taun_sv = test_df[[f'reco_{tau2_prefix}_sv_x', f'reco_{tau2_prefix}_sv_y', f'reco_{tau2_prefix}_sv_z']].values if use_reco else None
            else: # else use the pi four vectors as the charged component (this is not ideal but we just want to see how much difference it makes to the results)
                reco_taup_charged = reco_taup_pi
                reco_taun_charged = reco_taun_pi
                reco_taun_charged_ip = reco_taun_pi_ip
                reco_taup_charged_ip = reco_taup_pi_ip
                # set sv's to 0
                reco_taup_sv = np.zeros((len(test_df), 3))
                reco_taun_sv = np.zeros((len(test_df), 3))
    
        # move X_test and model to device
        X_test, _ = test_dataset[:]
        del _
        X_test = X_test.to(device)
        model = model.float().to(device)  # nflows' StandardNormal buffer is float64; MPS needs float32
    
        samples_map = None
        sample_chunk_size = nn_config.get('chunk_size', 50000 if device.type == 'cpu' else 100000)
        
        if args.useTransformerBaseline or args.useMLP:
            # chunk to avoid OOM from (N, 13, d_model) token tensors
            pred_chunks = []
            with torch.no_grad():
                for start in range(0, X_test.shape[0], sample_chunk_size):
                    pred_chunks.append(model(X_test[start:start + sample_chunk_size]).cpu())
            predictions_norm = torch.cat(pred_chunks, dim=0)
            del pred_chunks
        else:
            # sample from the normflow pdf in chunks to avoid memory issues
            pred_chunks = []
            with torch.no_grad():
                for start in range(0, X_test.shape[0], sample_chunk_size):
                    chunk = model.sample(num_samples=1, context=X_test[start:start + sample_chunk_size]).squeeze(1)
                    pred_chunks.append(chunk.cpu())
            predictions_norm = torch.cat(pred_chunks, dim=0)
            del pred_chunks
    
        # destandardize predictions so that they are in physical units
        predictions = test_dataset.destandardize_outputs(predictions_norm).cpu().numpy() 
    
        # identify is this is hadronic only, semileptonic, or leptonic based on number of columns in predictions
        leptonic_mode = 0
        if predictions.shape[1] == 6:
            leptonic_mode = 0
        elif predictions.shape[1] == 7:
            leptonic_mode = 1
        elif predictions.shape[1] == 8:
            leptonic_mode = 2
    
        print(f"Leptonic mode identified as {leptonic_mode} based on number of output features in predictions")   
    
        if use_reco:
          conv_kwargs = dict(coordinates=coordinates, output_features=output_features,
                        tau1_charged=reco_taup_charged, tau1_pi0=reco_taup_pizero, tau2_charged=reco_taun_charged, tau2_pi0=reco_taun_pizero, leptonic_mode=leptonic_mode)
        else:
          conv_kwargs = dict(coordinates=coordinates, output_features=output_features,
                        tau1_charged=true_taup_charged, tau1_pi0=true_taup_pizero, tau2_charged=true_taun_charged, tau2_pi0=true_taun_pizero, leptonic_mode=leptonic_mode)
    
        # get the gen values of the neutrinos in x,y,z coordinates
    
        true_values = convert_coordinates_pred(test_df[output_features].values, **conv_kwargs)
        true_values = add_energies_pair(true_values)
    
        predictions = convert_coordinates_pred(predictions, **conv_kwargs)
    
        if not args.useMLP and not args.useTransformerBaseline:
            chunk_size = nn_config.get('chunk_size', 1000 if device.type == 'cpu' else 10000)
            # define alternative prediction by taking most probable value from flow, do this by sampling to find MAP estimate
            print("Computing alternative predictions using flow_map_predict...")
            map_method = nn_config.get('map_method', 'latent_zero')
            print(f">> Method: {map_method} with chunk size {chunk_size}")
            _, samples_map = flow_map_predict(
                model, X_test,
                test_dataset=test_dataset,
                num_draws=nn_config.get('map_num_draws', 100),
                chunk_size= chunk_size,
                method=map_method,
            )
            samples_map = convert_coordinates_pred(samples_map, **conv_kwargs)
      
        # add energies (E=|p|) and build neutrino 4-vectors
        predictions     = add_energies_pair(predictions)
        predictions_map = add_energies_pair(samples_map) if samples_map is not None else None

        if args.inc_significance:
            # compute neutrino significance by taking N samples, computing E for each one, getting variance, then dividing by mean
            print("Computing neutrino significance...")
            N_significance_samples = 10
            nu_E_significance = []
            nubar_E_significance = []
            significance_chunks = []
            for i in range(0, X_test.shape[0], sample_chunk_size):
                chunk = model.sample(num_samples=N_significance_samples, context=X_test[i:i + sample_chunk_size])
                significance_chunks.append(chunk.cpu())
            significances = torch.cat(significance_chunks, dim=0)
            # destandardize
            significances = test_dataset.destandardize_outputs(significances)
            # convert coordinates, need to do this for each sample
            significance_converted = []
            for j in range(significances.shape[1]):
                pred_j = significances[:, j, :].cpu().numpy()
                pred_j = convert_coordinates_pred(pred_j, **conv_kwargs)
                # compute and add energies
                pred_j = add_energies_pair(pred_j)
                significance_converted.append(torch.tensor(pred_j))
            
            significances = torch.stack(significance_converted, dim=1)
            #get nubar_E and nu_E 
            nu_E = significances[:, :, 4]
            nubar_E = significances[:, :, 0]
            nuE_mean = torch.mean(nu_E, dim=1)
            nubarE_mean = torch.mean(nubar_E, dim=1)
            nu_E_std = torch.std(nu_E, dim=1)
            nubar_E_std = torch.std(nubar_E, dim=1)
            nu_significance = nuE_mean / nu_E_std
            nubar_significance = nubarE_mean / nubar_E_std


        #### temp set predictions to zeros
        ###N = predictions.shape[0]
        #### zero neutrino 4-vectors: [nu_p(4), nu_n(4)]
        ###predictions = np.zeros((N, 8), dtype=predictions.dtype)
        ###predictions_map = np.zeros_like(predictions)

    
        # get true taus by summing with pis and pizeros
        true_taus = np.concatenate([true_values[:, 0:4] + true_taup_charged + true_taup_pizero,
                                    true_values[:, 4:8] + true_taun_charged + true_taun_pizero], axis=1)
        
        # now use predicted neutrino but add to visible products to get predicted taus
        if use_reco:
            pred_taus = np.concatenate([predictions[:, 0:4] + reco_taup_charged + reco_taup_pizero,
                                    predictions[:, 4:8] + reco_taun_charged + reco_taun_pizero], axis=1)
        else:
            pred_taus = np.concatenate([predictions[:, 0:4] + true_taup_charged + true_taup_pizero,
                                    predictions[:, 4:8] + true_taun_charged + true_taun_pizero], axis=1)
        
        # same for the MAP predictions
        pred_taus_map = None
        if predictions_map is not None:
            if use_reco:
                pred_taus_map = np.concatenate([predictions_map[:, 0:4] + reco_taup_charged + reco_taup_pizero,
                                            predictions_map[:, 4:8] + reco_taun_charged + reco_taun_pizero], axis=1)
            else:
                pred_taus_map = np.concatenate([predictions_map[:, 0:4] + true_taup_charged + true_taup_pizero,
                                            predictions_map[:, 4:8] + true_taun_charged + true_taun_pizero], axis=1)
      
      
        # build dataframe for results
        true_taup_haspizero = test_df[f'{tau1_prefix}_haspizero'].values.reshape(-1,1)
        true_taun_haspizero = test_df[f'{tau2_prefix}_haspizero'].values.reshape(-1,1)
        if inc_new_vars:
            true_taup_ishadronic = test_df[f'{tau1_prefix}_ishadronic'].values.reshape(-1,1)
            true_taun_ishadronic = test_df[f'{tau2_prefix}_ishadronic'].values.reshape(-1,1)
            true_taup_npizero = test_df[f'{tau1_prefix}_npizero'].values.reshape(-1,1)
            true_taun_npizero = test_df[f'{tau2_prefix}_npizero'].values.reshape(-1,1)
            true_taup_is3prong = test_df[f'{tau1_prefix}_is3prong'].values.reshape(-1,1)
            true_taun_is3prong = test_df[f'{tau2_prefix}_is3prong'].values.reshape(-1,1)
            true_taup_ismuon = test_df[f'{tau1_prefix}_ismuon'].values.reshape(-1,1)
            true_taun_ismuon = test_df[f'{tau2_prefix}_ismuon'].values.reshape(-1,1)
            true_taup_iselectron = test_df[f'{tau1_prefix}_iselectron'].values.reshape(-1,1)
            true_taun_iselectron = test_df[f'{tau2_prefix}_iselectron'].values.reshape(-1,1)
        else:
            # set defaults such that this will still work with old setup based on dm 0 and 1 only
            true_taup_ishadronic = np.ones((len(test_df), 1))
            true_taun_ishadronic = np.ones((len(test_df), 1))
            true_taup_npizero = true_taup_haspizero
            true_taun_npizero = true_taun_haspizero
            true_taup_is3prong = np.zeros((len(test_df), 1))
            true_taun_is3prong = np.zeros((len(test_df), 1))
            true_taup_ismuon = np.zeros((len(test_df), 1))
            true_taun_ismuon = np.zeros((len(test_df), 1))
            true_taup_iselectron = np.zeros((len(test_df), 1))
            true_taun_iselectron = np.zeros((len(test_df), 1))
    
    
        if use_reco:
            reco_taup_haspizero = test_df[f'reco_{tau1_prefix}_haspizero'].values.reshape(-1,1) if use_reco else None
            reco_taun_haspizero = test_df[f'reco_{tau2_prefix}_haspizero'].values.reshape(-1,1) if use_reco else None
            if inc_new_vars:
                reco_taup_ishadronic = test_df[f'reco_{tau1_prefix}_ishadronic'].values.reshape(-1,1) if use_reco else None
                reco_taun_ishadronic = test_df[f'reco_{tau2_prefix}_ishadronic'].values.reshape(-1,1) if use_reco else None
                reco_taup_npizero = test_df[f'reco_{tau1_prefix}_npizero'].values.reshape(-1,1) if use_reco else None
                reco_taun_npizero = test_df[f'reco_{tau2_prefix}_npizero'].values.reshape(-1,1) if use_reco else None
                reco_taup_is3prong = test_df[f'reco_{tau1_prefix}_is3prong'].values.reshape(-1,1) if use_reco else None
                reco_taun_is3prong = test_df[f'reco_{tau2_prefix}_is3prong'].values.reshape(-1,1) if use_reco else None
                reco_taup_ismuon = test_df[f'reco_{tau1_prefix}_ismuon'].values.reshape(-1,1) if use_reco else None
                reco_taun_ismuon = test_df[f'reco_{tau2_prefix}_ismuon'].values.reshape(-1,1) if use_reco else None
                reco_taup_iselectron = test_df[f'reco_{tau1_prefix}_iselectron'].values.reshape(-1,1) if use_reco else None
                reco_taun_iselectron = test_df[f'reco_{tau2_prefix}_iselectron'].values.reshape(-1,1) if use_reco else None
            else:
                reco_taup_haspizero = test_df[f'reco_{tau1_prefix}_haspizero'].values.reshape(-1,1) if use_reco else None
                reco_taun_haspizero = test_df[f'reco_{tau2_prefix}_haspizero'].values.reshape(-1,1) if use_reco else None
                reco_taup_ishadronic = np.zeros((len(test_df), 1))
                reco_taun_ishadronic = np.zeros((len(test_df), 1))
                reco_taup_npizero = np.zeros((len(test_df), 1))
                reco_taun_npizero = np.zeros((len(test_df), 1))
                reco_taup_is3prong = np.zeros((len(test_df), 1))
                reco_taun_is3prong = np.zeros((len(test_df), 1))
                reco_taup_ismuon = np.zeros((len(test_df), 1))
                reco_taun_ismuon = np.zeros((len(test_df), 1))
                reco_taup_iselectron = np.zeros((len(test_df), 1))
                reco_taun_iselectron = np.zeros((len(test_df), 1))
    
        # Pass through optional columns from the input dataframe if present
        _AXES = ('n', 'r', 'k')
        _passthrough_cols = (
            [f'tauspinner_wt_alpha{a}' for a in [0, 45, 90]] +
            [f'wt_hp_{a}' for a in _AXES] +
            [f'wt_hm_{a}' for a in _AXES] +
            [f'wt_hp_{a}_hm_{b}' for a in _AXES for b in _AXES] +
            ['ts_hh_taup_x', 'ts_hh_taup_y', 'ts_hh_taup_z',
             'ts_hh_taun_x', 'ts_hh_taun_y', 'ts_hh_taun_z'] +
            [f'undecayed_{tau}_{comp}'
             for tau in ['taup', 'taun']
             for comp in ['px', 'py', 'pz', 'e']]
        )
        # tau1_charge/tau2_charge: physical charge of the labeled legs, stored by
        # convert_semileptonic_df for semileptonic (tau1=leptonic) dataframes.
        # Passed through (renamed to the taup/taun output convention, since this
        # script writes tau1 out as 'taup') so downstream consumers can map the
        # leptonic/hadronic labels back to physical charge -- the wt_hp_*/wt_hm_*
        # spin weights are defined by physical charge and are NOT relabeled.
        _passthrough_cols = _passthrough_cols + ['tau1_charge', 'tau2_charge']
        cols_to_pass = [c for c in _passthrough_cols if c in test_df.columns]
        if cols_to_pass:
            tauspinner_info_df = test_df[cols_to_pass].reset_index(drop=True)
            tauspinner_info_df = tauspinner_info_df.rename(
                columns={'tau1_charge': 'taup_charge', 'tau2_charge': 'taun_charge'})
        else:
            tauspinner_info_df = None

        # store the mass constraint if exists
        has_fastmtt_constraint = 'FastMTT_mass_constraint' in test_df.columns
        if has_fastmtt_constraint:
            fastmtt_mass_constraint = test_df['FastMTT_mass_constraint'].values
            fastmtt_pt_constraint = test_df['FastMTT_pt_constraint'].values
            fastmtt_pt_1_constraint = test_df['FastMTT_pt_1_constraint'].values
            fastmtt_pt_2_constraint = test_df['FastMTT_pt_2_constraint'].values

        del test_df

        # collect true and predicted nus, true and predicted taus, and pi's into pandas dataframe, label the columns
        results_df = pd.DataFrame(data=np.concatenate([true_values, predictions, true_taus, pred_taus, true_taun_haspizero,
                                true_taup_haspizero, true_taup_ishadronic, true_taun_ishadronic, true_taup_npizero, true_taun_npizero,
                                true_taup_is3prong, true_taun_is3prong, true_taup_ismuon, true_taun_ismuon, true_taup_iselectron, true_taun_iselectron,
                                true_taup_pi, true_taup_pizero, true_taun_pi, true_taun_pizero, true_taun_pi_ip, true_taup_pi_ip, true_taup_charged,
                                true_taun_charged, true_taup_charged_ip, true_taun_charged_ip, true_taup_sv, true_taun_sv,
                                true_taup_pi2, true_taun_pi2, true_taup_pi3, true_taun_pi3], axis=1),
                                  columns=[
                                           'true_nubar_E', 'true_nubar_px', 'true_nubar_py', 'true_nubar_pz',
                                           'true_nu_E', 'true_nu_px', 'true_nu_py', 'true_nu_pz',
                                           'pred_nubar_E', 'pred_nubar_px', 'pred_nubar_py', 'pred_nubar_pz',
                                           'pred_nu_E', 'pred_nu_px', 'pred_nu_py', 'pred_nu_pz',
                                           'true_tau_plus_E',  'true_tau_plus_px',  'true_tau_plus_py',  'true_tau_plus_pz',
                                           'true_tau_minus_E', 'true_tau_minus_px', 'true_tau_minus_py', 'true_tau_minus_pz',
                                           'pred_tau_plus_E',  'pred_tau_plus_px',  'pred_tau_plus_py',  'pred_tau_plus_pz',
                                           'pred_tau_minus_E', 'pred_tau_minus_px', 'pred_tau_minus_py', 'pred_tau_minus_pz',
                                           'true_taun_haspizero', 'true_taup_haspizero',
                                           'true_taup_ishadronic', 'true_taun_ishadronic',
                                           'true_taup_npizero', 'true_taun_npizero',
                                           'true_taup_is3prong', 'true_taun_is3prong',
                                           'true_taup_ismuon', 'true_taun_ismuon',
                                           'true_taup_iselectron', 'true_taun_iselectron',
                                           'true_taup_pi1_E', 'true_taup_pi1_px', 'true_taup_pi1_py', 'true_taup_pi1_pz',
                                           'true_taup_pizero1_E', 'true_taup_pizero1_px', 'true_taup_pizero1_py', 'true_taup_pizero1_pz',
                                           'true_taun_pi1_E', 'true_taun_pi1_px', 'true_taun_pi1_py', 'true_taun_pi1_pz',
                                           'true_taun_pizero1_E', 'true_taun_pizero1_px', 'true_taun_pizero1_py', 'true_taun_pizero1_pz',
                                           'true_taun_pi1_ipx', 'true_taun_pi1_ipy', 'true_taun_pi1_ipz',
                                           'true_taup_pi1_ipx', 'true_taup_pi1_ipy', 'true_taup_pi1_ipz',
                                           'true_taup_charged_E', 'true_taup_charged_px', 'true_taup_charged_py', 'true_taup_charged_pz',
                                           'true_taun_charged_E', 'true_taun_charged_px', 'true_taun_charged_py', 'true_taun_charged_pz',
                                           'true_taup_charged_ipx', 'true_taup_charged_ipy', 'true_taup_charged_ipz',
                                           'true_taun_charged_ipx', 'true_taun_charged_ipy', 'true_taun_charged_ipz',
                                           'true_taup_sv_x', 'true_taup_sv_y', 'true_taup_sv_z',
                                           'true_taun_sv_x', 'true_taun_sv_y', 'true_taun_sv_z',
                                           'true_taup_pi2_E', 'true_taup_pi2_px', 'true_taup_pi2_py', 'true_taup_pi2_pz',
                                           'true_taun_pi2_E', 'true_taun_pi2_px', 'true_taun_pi2_py', 'true_taun_pi2_pz',
                                           'true_taup_pi3_E', 'true_taup_pi3_px', 'true_taup_pi3_py', 'true_taup_pi3_pz',
                                           'true_taun_pi3_E', 'true_taun_pi3_px', 'true_taun_pi3_py', 'true_taun_pi3_pz',
                                           ])

        #now add tauspinner infor if present
        if tauspinner_info_df is not None:
            results_df = pd.concat([results_df, tauspinner_info_df], axis=1)
        del tauspinner_info_df

        # invariant masses (per tau and for the pair)
        results_df['true_tau_plus_mass']  = inv_mass(true_taus, 0)
        results_df['true_tau_minus_mass'] = inv_mass(true_taus, 4)
        results_df['pred_tau_plus_mass']  = inv_mass(pred_taus, 0)
        results_df['pred_tau_minus_mass'] = inv_mass(pred_taus, 4)
        results_df['pred_boson_mass'] = np.sqrt(np.maximum(
            (pred_taus[:,0]+pred_taus[:,4])**2 - (pred_taus[:,1]+pred_taus[:,5])**2
            - (pred_taus[:,2]+pred_taus[:,6])**2 - (pred_taus[:,3]+pred_taus[:,7])**2, 0))

        if has_fastmtt_constraint:
            results_df['FastMTT_mass_constraint'] = fastmtt_mass_constraint
            results_df['FastMTT_pt_constraint'] = fastmtt_pt_constraint
            results_df['FastMTT_pt_taup_constraint'] = fastmtt_pt_1_constraint
            results_df['FastMTT_pt_taun_constraint'] = fastmtt_pt_2_constraint
            del fastmtt_mass_constraint, fastmtt_pt_constraint, fastmtt_pt_1_constraint, fastmtt_pt_2_constraint

        # delete everything that has been concatinated into results_df to save memory
        del true_values, predictions, true_taus, pred_taus, true_taun_haspizero, true_taup_haspizero, true_taup_ishadronic, true_taun_ishadronic, true_taup_npizero, true_taun_npizero, true_taup_is3prong, true_taun_is3prong, true_taup_ismuon, true_taun_ismuon, true_taup_iselectron, true_taun_iselectron, true_taup_pi, true_taup_pizero, true_taun_pi, true_taun_pizero, true_taun_pi_ip, true_taup_pi_ip, true_taup_charged, true_taun_charged, true_taup_charged_ip, true_taun_charged_ip, true_taup_sv, true_taun_sv, true_taup_pi2, true_taun_pi2, true_taup_pi3, true_taun_pi3

        if use_reco:
            results_df_extra = pd.DataFrame(data=np.concatenate([reco_taup_haspizero, reco_taun_haspizero, reco_taup_ishadronic, reco_taun_ishadronic, reco_taup_npizero, reco_taun_npizero, reco_taup_is3prong, reco_taun_is3prong, reco_taup_ismuon, reco_taun_ismuon, reco_taup_iselectron, reco_taun_iselectron,
                                                                reco_taup_pi, reco_taup_pizero, reco_taun_pi, reco_taun_pizero, reco_taun_pi_ip, reco_taup_pi_ip, reco_taup_charged, reco_taun_charged, reco_taup_charged_ip, reco_taun_charged_ip, reco_taup_sv, reco_taun_sv,
                                                                reco_taup_pi2, reco_taun_pi2, reco_taup_pi3, reco_taun_pi3], axis=1),
                                  columns=[
                                            'reco_taup_haspizero', 'reco_taun_haspizero', 
                                            'reco_taup_ishadronic', 'reco_taun_ishadronic',
                                            'reco_taup_npizero', 'reco_taun_npizero',
                                            'reco_taup_is3prong', 'reco_taun_is3prong',
                                            'reco_taup_ismuon', 'reco_taun_ismuon',
                                            'reco_taup_iselectron', 'reco_taun_iselectron',
                                            'reco_taup_pi1_E', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz',
                                            'reco_taup_pizero1_E', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz',
                                            'reco_taun_pi1_E', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz',
                                            'reco_taun_pizero1_E', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz',
                                            'reco_taun_pi1_ipx', 'reco_taun_pi1_ipy', 'reco_taun_pi1_ipz',
                                            'reco_taup_pi1_ipx', 'reco_taup_pi1_ipy', 'reco_taup_pi1_ipz',
                                            'reco_taup_charged_E', 'reco_taup_charged_px', 'reco_taup_charged_py', 'reco_taup_charged_pz',
                                            'reco_taun_charged_E', 'reco_taun_charged_px', 'reco_taun_charged_py', 'reco_taun_charged_pz',
                                            'reco_taup_charged_ipx', 'reco_taup_charged_ipy', 'reco_taup_charged_ipz',
                                            'reco_taun_charged_ipx', 'reco_taun_charged_ipy', 'reco_taun_charged_ipz',
                                            'reco_taup_sv_x', 'reco_taup_sv_y', 'reco_taup_sv_z',
                                            'reco_taun_sv_x', 'reco_taun_sv_y', 'reco_taun_sv_z',
                                            'reco_taup_pi2_E', 'reco_taup_pi2_px', 'reco_taup_pi2_py', 'reco_taup_pi2_pz',
                                            'reco_taun_pi2_E', 'reco_taun_pi2_px', 'reco_taun_pi2_py', 'reco_taun_pi2_pz',
                                            'reco_taup_pi3_E', 'reco_taup_pi3_px', 'reco_taup_pi3_py', 'reco_taup_pi3_pz',
                                            'reco_taun_pi3_E', 'reco_taun_pi3_px', 'reco_taun_pi3_py', 'reco_taun_pi3_pz',
                                           ])

            # delete the reco variables that have been concatenated into results_df_extra to save memory
            del reco_taup_haspizero, reco_taun_haspizero, reco_taup_ishadronic, reco_taun_ishadronic, reco_taup_npizero, reco_taun_npizero, reco_taup_is3prong, reco_taun_is3prong, reco_taup_ismuon, reco_taun_ismuon, reco_taup_iselectron, reco_taun_iselectron, reco_taup_pi, reco_taup_pizero, reco_taun_pi, reco_taun_pizero, reco_taun_pi_ip, reco_taup_pi_ip, reco_taup_charged, reco_taun_charged, reco_taup_charged_ip, reco_taun_charged_ip, reco_taup_sv, reco_taun_sv, reco_taup_pi2, reco_taun_pi2, reco_taup_pi3, reco_taun_pi3

            results_df = pd.concat([results_df, results_df_extra], axis=1)
            del results_df_extra

    
    
        if predictions_map is not None:
            results_df_extra = pd.DataFrame(
                data=np.concatenate([predictions_map, pred_taus_map], axis=1),
                columns=[
                    'map_pred_nubar_E', 'map_pred_nubar_px', 'map_pred_nubar_py', 'map_pred_nubar_pz',
                    'map_pred_nu_E', 'map_pred_nu_px', 'map_pred_nu_py', 'map_pred_nu_pz',
                    'map_pred_tau_plus_E',  'map_pred_tau_plus_px',  'map_pred_tau_plus_py',  'map_pred_tau_plus_pz',
                    'map_pred_tau_minus_E', 'map_pred_tau_minus_px', 'map_pred_tau_minus_py', 'map_pred_tau_minus_pz',
                ])
            results_df = pd.concat([results_df, results_df_extra], axis=1)
            del results_df_extra
            
        if args.inc_significance:
            results_df['nu_significance'] = nu_significance
            results_df['nubar_significance'] = nubar_significance
            del nu_significance, nubar_significance
        
        # invariant masses (per tau and for the pair)
        if predictions_map is not None:
            results_df['map_pred_tau_plus_mass']  = inv_mass(pred_taus_map, 0)
            results_df['map_pred_tau_minus_mass'] = inv_mass(pred_taus_map, 4)
            results_df['map_pred_boson_mass'] = np.sqrt(np.maximum(
                (pred_taus_map[:,0]+pred_taus_map[:,4])**2 - (pred_taus_map[:,1]+pred_taus_map[:,5])**2
                - (pred_taus_map[:,2]+pred_taus_map[:,6])**2 - (pred_taus_map[:,3]+pred_taus_map[:,7])**2, 0))

            del pred_taus_map
    
        if leptonic_mode == 0:
            # spin variables - not implemented yet for leptonic modes 
            print("Computing spin variables...")
            results_df = compute_spin_vars(results_df, tau_pred_prefix='true_', tau_vis_prefix='true_') 
            results_df = compute_spin_vars(results_df, tau_pred_prefix='pred_',  tau_vis_prefix='reco_' if use_reco else 'true_')
            # get spin vars for MAP prediction if present
            if predictions_map is not None:
                results_df = compute_spin_vars(results_df, tau_pred_prefix='map_pred_', tau_vis_prefix='reco_' if use_reco else 'true_')
    
            # loop over dm categories and compute spin density matrix variables for each
            # TODO: could also do splitting based on reco dm category - both give us different but useful information
            dm_masks = {
                'all':    results_df,
                'dm_0_0': results_df[(results_df['true_taup_npizero'] == 0) & (results_df['true_taun_npizero'] == 0) & (results_df['true_taup_ishadronic'] == 1) & (results_df['true_taun_ishadronic'] == 1) & (results_df['true_taup_is3prong'] == 0) & (results_df['true_taun_is3prong'] == 0)],
                'dm_0_1': results_df[(((results_df['true_taup_npizero'] == 0) & (results_df['true_taun_npizero'] == 1)) |
                                      ((results_df['true_taup_npizero'] == 1) & (results_df['true_taun_npizero'] == 0))) & (results_df['true_taup_ishadronic'] == 1) & (results_df['true_taun_ishadronic'] == 1) & (results_df['true_taup_is3prong'] == 0) & (results_df['true_taun_is3prong'] == 0)],
                'dm_1_1': results_df[(results_df['true_taup_npizero'] == 1) & (results_df['true_taun_npizero'] == 1) & (results_df['true_taup_ishadronic'] == 1) & (results_df['true_taun_ishadronic'] == 1) & (results_df['true_taup_is3prong'] == 0) & (results_df['true_taun_is3prong'] == 0)],
            }
            spin_plot_dir = f"{output_plots_dir}/spin_density/{data_config['test_output_name']}"
            for dm_category, results_df_dm in dm_masks.items():
                true_Bplus, true_Bminus, true_C, true_con, true_m12 = compute_spin_density_vars(results_df_dm, prefix='true_')
                pred_Bplus, pred_Bminus, pred_C, pred_con, pred_m12 = compute_spin_density_vars(results_df_dm, prefix='pred_')
                if predictions_map is not None:
                    map_pred_Bplus, map_pred_Bminus, map_pred_C, map_pred_con, map_pred_m12 = compute_spin_density_vars(results_df_dm, prefix='map_pred_')
    
                print('\n===== DM CATEGORY:', dm_category, '=====')
                print(f'Number of events in this category: {len(results_df_dm)}')
                print('\n True spin density matrix variables:')
                print(true_Bplus)
                print(true_Bminus)
                print(true_C)
                print(true_con, true_m12)
                print()
    
                print('\n Sampled predicted spin density matrix variables:')
                print(pred_Bplus)
                print(pred_Bminus)
                print(pred_C)
                print(pred_con, pred_m12)
    
                if predictions_map is not None:
                    print('\n MAP estimate spin density matrix variables:')
                    print(map_pred_Bplus)
                    print(map_pred_Bminus)
                    print(map_pred_C)
                    print(map_pred_con, map_pred_m12)
    
                # collect results for plotting
                plot_results = {'True': (true_Bplus, true_Bminus, true_C, true_con, true_m12)}
                plot_results['Sampled'] = (pred_Bplus, pred_Bminus, pred_C, pred_con, pred_m12)
                if predictions_map is not None:
                    plot_results['MAP'] = (map_pred_Bplus, map_pred_Bminus, map_pred_C, map_pred_con, map_pred_m12)
                plot_spin_density_matrix(plot_results, dm_category, outdir=spin_plot_dir)

                del true_Bplus, true_Bminus, true_C, true_con, true_m12, pred_Bplus, pred_Bminus, pred_C, pred_con, pred_m12
                if predictions_map is not None:
                    del map_pred_Bplus, map_pred_Bminus, map_pred_C, map_pred_con, map_pred_m12
            del dm_masks

        # === phiCP from the predicted neutrino, and its uncertainty from repeated flow sampling ===
        # Central estimate: single-sample ('pred') and, if available, MAP ('map_pred') phiCP,
        # via the same get_ditau_polarimetric + compute_aco_polarimetric method plot_phiCP.py's
        # 'recoNu' option uses downstream on this script's own output -- computed here too since
        # an error estimate is meaningless without the corresponding central value alongside it.
        # Uncertainty: draw n_flow_samples_phiCP fresh posterior samples/event, recompute phiCP
        # for each with the visible momenta held fixed, and take the circular spread (phiCP is
        # periodic, so a plain std would misreport events whose phiCP sits near the 0/2*pi seam)
        # -- this isolates the error coming from the neutrino regression alone.
        if args.inc_phiCP_error:
            if args.useMLP or args.useTransformerBaseline:
                print("WARNING: --inc_phiCP_error needs the flow's posterior (model.sample()), "
                      "which --useMLP/--useTransformerBaseline don't have; skipping.")
            else:
                pion_prefix = 'reco' if use_reco else 'true'
                reco_pions = (pion_prefix == 'reco')
                print(f"Computing phiCP from the predicted neutrino (pion_prefix={pion_prefix}) and its "
                      f"uncertainty from {args.n_flow_samples_phiCP} flow samples/event...")

                results_df['taup_DM'], results_df['taun_DM'] = _phiCP_dm_codes(results_df, pion_prefix)

                def _phiCP_from_df(df, tau_prefix):
                    R1, P1, R2, P2 = get_ditau_polarimetric(df, tau_prefix=tau_prefix, reco_pions=reco_pions)
                    return np.asarray(compute_aco_polarimetric(R1, P1, R2, P2))

                # central (point) estimates
                results_df['pred_phiCP'] = _phiCP_from_df(results_df, 'pred')
                if predictions_map is not None:
                    results_df['map_pred_phiCP'] = _phiCP_from_df(results_df, 'map_pred')

                # fixed (sample-independent) inputs needed per event: visible pion/pizero/charged
                # four-vectors (for get_ditau_polarimetric) and the decay-mode codes, all read back
                # from results_df's own saved columns rather than the loose per-particle numpy
                # arrays used earlier in this function, which are already deleted by this point.
                pion_cols = [f'{pion_prefix}_{tau}_{part}_{comp}'
                             for tau in ('taup', 'taun')
                             for part in ('pi1', 'pi2', 'pi3', 'pizero1', 'charged')
                             for comp in ('E', 'px', 'py', 'pz')]
                pion_cols = [c for c in pion_cols if c in results_df.columns]
                pion_vals_full = results_df[pion_cols].values
                dm_vals_full = results_df[['taup_DM', 'taun_DM']].values

                def _charged_pi0(tau):
                    charged = results_df[[f'{pion_prefix}_{tau}_charged_{c}' for c in ('E', 'px', 'py', 'pz')]].values
                    pi0     = results_df[[f'{pion_prefix}_{tau}_pizero1_{c}' for c in ('E', 'px', 'py', 'pz')]].values
                    return charged, pi0
                taup_charged_full, taup_pi0_full = _charged_pi0('taup')
                taun_charged_full, taun_pi0_full = _charged_pi0('taun')

                n_fs = args.n_flow_samples_phiCP
                # each call below processes (events_in_chunk * n_fs) rows in one shot -- reusing
                # sample_chunk_size unchanged here would mean e.g. 50000*50 = 2.5M rows per call;
                # dividing by n_fs keeps each call close to the per-call scale used elsewhere in
                # this script (same empirically-tuned approach as evaluate_polvec.py).
                err_chunk_size = max(1, sample_chunk_size // n_fs)
                n_events = len(results_df)
                phiCP_err = np.full(n_events, np.nan)
                n_sample_failures = 0
                for start in tqdm(range(0, n_events, err_chunk_size), desc="Processing chunks (phiCP uncertainty)"):
                    end = min(start + err_chunk_size, n_events)
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
                                samples_norm = model.sample(num_samples=n_fs, context=X_test[start:end])  # [C, n_fs, F]
                            break
                        except AssertionError:
                            continue
                    if samples_norm is None:
                        n_sample_failures += 1
                        continue

                    samples = test_dataset.destandardize_outputs(samples_norm).cpu().numpy().reshape(C * n_fs, -1)

                    rep = lambda arr: np.repeat(arr[start:end], n_fs, axis=0)
                    conv_kwargs_chunk = dict(coordinates=coordinates, output_features=output_features,
                                              tau1_charged=rep(taup_charged_full), tau1_pi0=rep(taup_pi0_full),
                                              tau2_charged=rep(taun_charged_full), tau2_pi0=rep(taun_pi0_full),
                                              leptonic_mode=leptonic_mode)
                    samples = convert_coordinates_pred(samples, **conv_kwargs_chunk)
                    samples = add_energies_pair(samples)

                    tau_plus_samp  = samples[:, 0:4] + rep(taup_charged_full) + rep(taup_pi0_full)
                    tau_minus_samp = samples[:, 4:8] + rep(taun_charged_full) + rep(taun_pi0_full)

                    tmp_df = pd.DataFrame(np.repeat(pion_vals_full[start:end], n_fs, axis=0), columns=pion_cols)
                    dm_chunk = np.repeat(dm_vals_full[start:end], n_fs, axis=0)
                    tmp_df['taup_DM'] = dm_chunk[:, 0]
                    tmp_df['taun_DM'] = dm_chunk[:, 1]
                    tmp_df['samp_tau_plus_E'],  tmp_df['samp_tau_plus_px']  = tau_plus_samp[:, 0],  tau_plus_samp[:, 1]
                    tmp_df['samp_tau_plus_py'], tmp_df['samp_tau_plus_pz']  = tau_plus_samp[:, 2],  tau_plus_samp[:, 3]
                    tmp_df['samp_tau_minus_E'],  tmp_df['samp_tau_minus_px'] = tau_minus_samp[:, 0], tau_minus_samp[:, 1]
                    tmp_df['samp_tau_minus_py'], tmp_df['samp_tau_minus_pz'] = tau_minus_samp[:, 2], tau_minus_samp[:, 3]

                    phiCP_samples = _phiCP_from_df(tmp_df, 'samp').reshape(C, n_fs)
                    phiCP_err[start:end] = circular_std(phiCP_samples, axis=1)

                if n_sample_failures > 0:
                    print(f"  WARNING: flow sampling failed on 5 retries for {n_sample_failures} chunk(s) "
                          f"({n_sample_failures * err_chunk_size} events, upper bound); phiCP_err set to NaN for those events.")

                results_df['pred_phiCP_err'] = phiCP_err
                del pion_vals_full, dm_vals_full, taup_charged_full, taup_pi0_full, taun_charged_full, taun_pi0_full, phiCP_err

        # write the results dataframe to a parquet file
        results_df.to_parquet(f"{output_dir}/{data_config['test_output_name']}.parquet")
    
        if args.make_root_output:
            # write as root file aswell
            with uproot.recreate(f"{output_dir}/{data_config['test_output_name']}.root") as f:
                f.mktree('tree', results_df.to_dict(orient="list"))

        # delete all remaining objects to free memory
        del results_df

if __name__ == "__main__":
    main()
