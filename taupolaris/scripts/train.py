import torch
import pandas as pd
import argparse
from taupolaris.python.DataProcessing import get_train_val_test_datasets, _resolve_polvec_feature_level, _resolve_frame_aligned_level
from taupolaris.python.NN_Tools import setup_model_and_training, train_model, get_device
import yaml
import os
import numpy as np
import shutil


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', help='path to the configuration file', type=str, default='taupolaris/config/LHC.yaml', required=True)
    argparser.add_argument('--useMLP', help='whether to use a simple MLP instead of a normalizing flow', action='store_true')
    argparser.add_argument('--useTransformerBaseline', help='whether to use a transformer encoder + regression head (MSE) instead of a normalizing flow', action='store_true')
    argparser.add_argument('--loadDS', help='whether to load existing train/val datasets or recreate', action='store_true')

    args = argparser.parse_args()

    config_file = args.config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']

    # make output directory called outputs_{model_name}, with plots subdirectory
    output_dir = f"outputs_{nn_config['model_name']}"
    output_plots_dir = f"{output_dir}/plots"
    os.makedirs(output_plots_dir, exist_ok=True)

    # save a copy of the config
    shutil.copy(config_file, f"{output_dir}/config.yaml")

    # gpu or cpu
    device = get_device()

    ## Get train and validation datasets

    datasets = data_config['datasets']
    data_config['use_transformer'] = nn_config.get('use_transformer', False)  # label data
    train_dataset, val_dataset, input_features, output_features = get_train_val_test_datasets(datasets, data_config, load_existing=args.loadDS)

    # store the means and stds used for normalization
    in_mean, in_std = train_dataset.input_mean, train_dataset.input_std
    out_mean, out_std = train_dataset.output_mean, train_dataset.output_std

    np.savez(f'{output_dir}/normalization_params.npz',
             input_mean=in_mean, input_std=in_std,
             output_mean=out_mean, output_std=out_std)

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # Get hyperparameters
    use_transformer = nn_config.get('use_transformer', False)  # transformer or ResNet
    if args.useMLP:
        hp = nn_config['MLP_hyperparams']
    elif args.useTransformerBaseline:
        hp = nn_config['TransformerBaseline_hyperparams']
    else:
        hp = nn_config['hyperparams']
    hp['num_epochs'] = nn_config['n_epochs']
    print(f"Hyperparameters: {hp}")

    # Setup model
    model, optimizer, train_loader, val_loader, scheduler, es, start_epoch, initial_history = setup_model_and_training(hp, train_dataset, val_dataset, input_features, output_features, nn_config['model_name'], reload=nn_config['reload'], reload_scheduler=nn_config['reload_scheduler'], reset_training=nn_config.get('reset_training', False), batch_norm=False, useMLP=args.useMLP, useTransformer=use_transformer, useTransformerMLP=args.useTransformerBaseline, leptonic_mode=data_config['leptonic_mode'],
                                                                                                                  polvec_feature_level=_resolve_polvec_feature_level(data_config),
                        frame_aligned_level=_resolve_frame_aligned_level(data_config),
                        condition_net_init=nn_config.get('condition_net_init'),
                        condition_net_freeze=bool(nn_config.get('condition_net_freeze', False)),
                        context_hidden_layer=bool(nn_config.get('condition_net_hidden_layer', False)),
                        init_from=nn_config.get('init_from'))
    print("Model and training setup complete.")

    init_from = nn_config.get('init_from')
    if init_from and not nn_config['reload']:
        # the starting weights only make sense with the standardisation they were trained with
        src_norm = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(init_from))), 'normalization_params.npz')
        if os.path.exists(src_norm):
            src = np.load(src_norm)
            here = {'input_mean': in_mean, 'input_std': in_std, 'output_mean': out_mean, 'output_std': out_std}
            worst = max(float(np.max(np.abs(np.asarray(here[k]) - src[k]) / (np.abs(src[k]) + 1e-6))) for k in here)
            print(f">> init_from normalisation check vs {src_norm}: max relative difference {worst:.2e}")
            if worst > 1e-4:
                raise SystemExit('init_from: this run\'s input/output standardisation differs from the source model\'s -- '
                                 'the starting weights would see shifted inputs/outputs. Use the same datasets and split.')
        else:
            print(f">> WARNING: no normalization_params.npz next to {init_from}; cannot check the standardisation matches")

    # optional phiCP energy-score term (taupolaris/python/phicp_loss.py): loss = NLL + lambda * ES
    es_term = None
    if float(hp.get('es_phicp_lambda', 0) or 0) > 0:
        from taupolaris.python.phicp_loss import PhiCPEnergyScore
        es_term = dict(fn=PhiCPEnergyScore(input_features, output_features, in_mean, in_std, out_mean, out_std,
                                           n_samples=int(hp.get('es_phicp_samples', 2))),
                       lam=float(hp['es_phicp_lambda']), fraction=float(hp.get('es_phicp_fraction', 1.0)),
                       rampup_epochs=float(hp.get('es_phicp_rampup_epochs', 0)),
                       target=hp.get('es_phicp_target', 'all'))
        es_term['fn'].perp_eps = float(hp.get('es_phicp_perp_eps', 0.05))
        es_term['fn'].min_one_minus_beta2 = float(hp.get('es_phicp_min_one_minus_beta2', 1e-3))
        print(f">> phiCP energy-score term ON: lambda {es_term['lam']}, {hp.get('es_phicp_samples', 2)} samples/event, "
              f"fraction {es_term['fraction']}, ramp-up {es_term['rampup_epochs']} epochs, gradient into: {es_term['target']}, "
              f"softening perp_eps {es_term['fn'].perp_eps}, 1-beta^2 >= {es_term['fn'].min_one_minus_beta2}")

    # Train — TransformerBaseline uses MSE loss, same as useMLP
    use_mse_loss = args.useMLP or args.useTransformerBaseline
    best_val_loss, history = train_model(model, optimizer, train_loader, val_loader, num_epochs=nn_config['n_epochs'], device=device, verbose=True, output_plots_dir=output_plots_dir, save_every_N=1, scheduler=scheduler, early_stopper=es, useMLP=use_mse_loss, start_epoch=start_epoch, initial_history=initial_history, clip_grad_norm=hp.get('clip_grad_norm'), es_term=es_term)

    model_name = nn_config['model_name']
    torch.save(model.state_dict(), f'{output_dir}/{model_name}.pth')
