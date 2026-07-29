import argparse
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from taupolaris.python.NN_Models import ConditionalFlow, MLP, TransformerRegressor
from taupolaris.python.Plotting import plot_loss
import torch.nn as nn
import torch.optim as optim


def _unpack_batch(batch, device):
    """RegressionDataset yields (X, y) normally, or (X, y, w) when a per-event
    training weight was configured (see get_train_val_test_datasets). Handle both."""
    if len(batch) == 3:
        X, y, w = batch
        w = w.to(device)
    else:
        X, y = batch
        w = None
    return X.to(device), y.to(device), w


def _compute_loss(model, X, y, w, useMLP, mlp_loss_fn):
    """Weighted or unweighted loss, for either the flow (NLL) or MLP (MSE) baseline.
    Weighted loss is a weighted mean (sum(w*loss_i)/sum(w)), i.e. the weights rescale
    each event's contribution rather than acting like extra/fewer effective events."""
    if not useMLP:
        log_p = model.log_prob(inputs=y, context=X)
        if w is not None:
            return -(w * log_p).sum() / w.sum()
        return -log_p.mean()
    else:
        predictions = model(X)
        if w is not None:
            per_event_loss = ((predictions - y) ** 2).mean(dim=1)
            return (w * per_event_loss).sum() / w.sum()
        return mlp_loss_fn(predictions, y)


def get_device():
    """Select the best available compute device: CUDA GPU, Apple Silicon GPU (MPS), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_legacy_pizero_proj_checkpoint(state_dict):
    """True if state_dict predates commit 2a901355 ('add Npizero'), which added
    condition_net.pizero_proj and the transformer's final LayerNorm to
    ParticleTransformerCondition. Pass the result as legacy_pizero_proj= to
    load_model so the constructed architecture matches the checkpoint."""
    return not any(k.endswith('condition_net.pizero_proj.weight') for k in state_dict)


def load_model(hp, input_features, output_features, batch_norm=False, useMLP=False, useTransformer=False, useTransformerMLP=False, leptonic_mode=0, legacy_pizero_proj=False):
    if useMLP:
        model = MLP(input_size=len(input_features), output_size=len(output_features), num_blocks=hp['num_blocks'],
                    hidden_size=hp['hidden_size'], activation=nn.GELU())
    elif useTransformerMLP:
        model = TransformerRegressor(
            input_features=input_features,
            leptonic_mode=leptonic_mode,
            output_dim=len(output_features),
            context_dim=hp['context_dim'],
            d_model=hp['d_model'], nhead=hp['nhead'],
            num_layers=hp['num_transformer_layers'],
            dropout=hp['dropout'],
        )
    elif useTransformer:
        model = ConditionalFlow(input_dim=len(output_features),
                                input_features=input_features,
                                leptonic_mode=leptonic_mode,
                                context_dim=hp['context_dim'],
                                use_transformer=True,
                                d_model=hp['d_model'], nhead=hp['nhead'],
                                num_transformer_layers=hp['num_transformer_layers'],
                                dropout=hp['dropout'],
                                legacy_pizero_proj=legacy_pizero_proj,
                                num_layers=hp['num_layers'], num_bins=hp['num_bins'],
                                tail_bound=hp['tail_bound'], hidden_size=hp['hidden_size'],
                                num_blocks=hp['num_blocks'], activation=nn.GELU())
    else:
        model = ConditionalFlow(input_dim=len(output_features), raw_condition_dim=len(input_features),
                                context_dim=hp['condition_net_output_size'],
                                cond_hidden_dim=hp['condition_net_hidden_size'],
                                cond_num_blocks=hp['condition_net_num_blocks'],
                                num_layers=hp['num_layers'], num_bins=hp['num_bins'], tail_bound=hp['tail_bound'],
                                hidden_size=hp['hidden_size'], num_blocks=hp['num_blocks'],
                                batch_norm=batch_norm, activation=nn.LeakyReLU(0.05))
    return model


def setup_model_and_training(hp, train_dataset, test_dataset, input_features, output_features, model_name, verbose=True, reload=False, reload_scheduler=False, reset_training=False, batch_norm=False, useMLP=False, useTransformer=False, useTransformerMLP=False, leptonic_mode=0):
    train_dataloader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)

    if train_dataset.weights is not None:
        print(f">> Training WITH per-event weights (train mean={train_dataset.weights.mean().item():.4f}, "
              f"val mean={test_dataset.weights.mean().item():.4f}); loss will be a weighted mean.")
    else:
        print(">> Training WITHOUT per-event weights (all events weighted equally).")

    if useMLP:
        model = MLP(input_size=len(input_features), output_size=len(output_features), num_blocks=hp['num_blocks'],
                    hidden_size=hp['hidden_size'], activation=nn.GELU())
    elif useTransformerMLP:
        model = TransformerRegressor(
            input_features=input_features,
            leptonic_mode=leptonic_mode,
            output_dim=len(output_features),
            context_dim=hp['context_dim'],
            d_model=hp['d_model'], nhead=hp['nhead'],
            num_layers=hp['num_transformer_layers'],
            dropout=hp['dropout'],
        )
    elif useTransformer:
        model = ConditionalFlow(input_dim=len(output_features),
                                input_features=input_features,
                                leptonic_mode=leptonic_mode,
                                context_dim=hp['context_dim'],
                                use_transformer=True,
                                d_model=hp['d_model'], nhead=hp['nhead'],
                                num_transformer_layers=hp['num_transformer_layers'],
                                dropout=hp['dropout'],
                                num_layers=hp['num_layers'], num_bins=hp['num_bins'],
                                tail_bound=hp['tail_bound'], hidden_size=hp['hidden_size'],
                                num_blocks=hp['num_blocks'], activation=nn.GELU())
    else:
        model = ConditionalFlow(input_dim=len(output_features), raw_condition_dim=len(input_features),
                                context_dim=hp['condition_net_output_size'],
                                cond_hidden_dim=hp['condition_net_hidden_size'],
                                cond_num_blocks=hp['condition_net_num_blocks'],
                                num_layers=hp['num_layers'], num_bins=hp['num_bins'], tail_bound=hp['tail_bound'],
                                hidden_size=hp['hidden_size'], num_blocks=hp['num_blocks'],
                                batch_norm=batch_norm, activation=nn.LeakyReLU(0.05))

    if verbose:
        # print number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of model parameters: {total_params}")
    scheduler = None
    es = None

    # use cos annealing schedule with warm up period:
    total_steps = hp['num_epochs'] * len(train_dataloader)
    warmup_steps = int(0.05 * total_steps)  # e.g. 5% warmup

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'])
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,   # start at 0.1% of lr
        total_iters=warmup_steps
    )
    
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=hp['lr']*0.01
    )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    output_plots_dir = f'outputs_{model_name}/plots'

    if reload_scheduler:
        scheduler_path = f'{output_plots_dir}/partial_scheduler.pth'
        if os.path.exists(scheduler_path):
            print(f"Reloading scheduler from {scheduler_path}...")
            try:
                scheduler.load_state_dict(torch.load(scheduler_path))
            except:
                print(f"Loading scheduler from {scheduler_path} failed. Trying to load from CPU.")
                scheduler.load_state_dict(torch.load(scheduler_path, map_location=torch.device('cpu')))
        else:
            print(f"Scheduler path {scheduler_path} does not exist. Can't reload scheduler.")
    #es = EarlyStopper(patience=10, min_delta=0.)

    # check if reload is true, if so load model from output_dir if it exists
    start_epoch = 0
    initial_history = None

    if reload:
        device = get_device()
        partial_model_path = f'{output_plots_dir}/partial_model.pth'
        if os.path.exists(partial_model_path):
            model_path = partial_model_path
            print(f"Reloading model from {model_path}...")
        else:
            print(f"Model path {partial_model_path} does not exist. Can't reload. Exiting.")
            exit(1)
        copied_name = model_path.replace('.pth', '_copy.pth')
        os.system(f'cp {model_path} {copied_name}')
        model.load_state_dict(torch.load(model_path, map_location=device))

        if reset_training:
            print("reset_training=True: using reloaded model weights only; optimizer, scheduler, epoch, and history will be reset.")
        else:
            optimizer_path = f'{output_plots_dir}/partial_optimizer.pth'
            if os.path.exists(optimizer_path):
                print(f"Reloading optimizer from {optimizer_path}...")
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            else:
                print("No optimizer checkpoint found; optimizer state will be reset.")

            meta_path = f'{output_plots_dir}/partial_meta.pth'
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location=torch.device('cpu'))
                start_epoch = meta.get('epoch', 0)
                initial_history = meta.get('history', None)
                print(f"Resuming from epoch {start_epoch + 1}.")

    return model, optimizer, train_dataloader, test_dataloader, scheduler, es, start_epoch, initial_history



def train_model(model, optimizer, train_dataloader, test_dataloader, num_epochs=10, device="cpu", verbose=True, output_plots_dir=None,
    save_every_N=None, recompute_train_loss=True, scheduler=None, early_stopper=None, useMLP=False, optuna_trial=None,
    start_epoch=0, initial_history=None):
    # nflows' StandardNormal distribution registers a float64 buffer (_log_z)
    # internally; MPS doesn't support float64, so cast to float32 before moving device.
    model.float().to(device)
    history = initial_history if initial_history is not None else {"train_loss": [], "val_loss": []}
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")

    if early_stopper: early_stopper.reset()

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train() # need to set model back to train mode, otherise dropout (and batchnorm if used) will not work
        running_loss=0
        mlp_loss_fn = nn.MSELoss()
        for batch, batch_data in enumerate(train_dataloader):
            X, y, w = _unpack_batch(batch_data, device)
            optimizer.zero_grad()
            loss = _compute_loss(model, X, y, w, useMLP, mlp_loss_fn)
            loss.backward()
            # Lucas experimented here
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            lr = scheduler.get_last_lr()
            # changing learning rate per batch - only if scheduler is defined and is not ReduceLROnPlateau
            if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            if verbose and epoch<5 and batch % 100 == 0:
                print(f'Batch {batch} | loss {loss.item()} | lr: {lr[0]} ' if scheduler else f'Batch {batch} | loss {loss.item()}')
            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)

        model.eval()
        if recompute_train_loss:
            # recompute train loss for better estimate, capped to same number of batches as validation
            n_batches = len(test_dataloader)
            sum_train_loss = 0
            with torch.no_grad():
                for i, batch_data in enumerate(train_dataloader):
                    if i >= n_batches:
                        break
                    X_train, y_train, w_train = _unpack_batch(batch_data, device)
                    train_loss = _compute_loss(model, X_train, y_train, w_train, useMLP, mlp_loss_fn)
                    sum_train_loss += train_loss.item()
            train_loss = sum_train_loss / n_batches

        history["train_loss"].append(train_loss)

        # validation phase
        val_running_loss = 0
        with torch.no_grad():
            for batch_data in test_dataloader:
                X_val, y_val, w_val = _unpack_batch(batch_data, device)
                val_loss = _compute_loss(model, X_val, y_val, w_val, useMLP, mlp_loss_fn)
                val_running_loss += val_loss.item()
        val_loss = val_running_loss / len(test_dataloader)
        history["val_loss"].append(val_loss)

        if optuna_trial is not None and ((epoch == 10) or (epoch == 15)):
            import optuna
            optuna_trial.report(val_loss, epoch)
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # save model if its loss is better than the previous best
        if val_loss < best_val_loss and output_plots_dir:
            print(f"New best model found at epoch {epoch} with val_loss {val_loss}. Saving model...")
            torch.save(model.state_dict(), f'{output_plots_dir}/best_model.pth')
        best_val_loss = min(best_val_loss, val_loss)

        if early_stopper and early_stopper.early_stop(val_loss):
            print(f"Early stopping triggered for epoch {epoch+1}")
            break

        if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr = scheduler.get_last_lr()
            scheduler.step(val_loss)

        if verbose and epoch % 1 == 0:
            LR_string = f" | LR: {lr[0]:.2e}" if scheduler else ""
            print(f'Epoch {epoch} | loss {train_loss} | val_loss {val_loss} | lr {LR_string}')

        if epoch > 0 and output_plots_dir: plot_loss(history["train_loss"], history["val_loss"], output_dir=output_plots_dir)

        if save_every_N and epoch % save_every_N == 0 and output_plots_dir:
            print(f"Saving model checkpoint at epoch {epoch}...")
            torch.save(model.state_dict(), f'{output_plots_dir}/partial_model.pth')
            torch.save(optimizer.state_dict(), f'{output_plots_dir}/partial_optimizer.pth')
            torch.save({'epoch': epoch, 'history': history}, f'{output_plots_dir}/partial_meta.pth')
            if scheduler:
                torch.save(scheduler.state_dict(), f'{output_plots_dir}/partial_scheduler.pth')

    print("Training Completed. Trained for {} epochs.".format(epoch))

    return best_val_loss, history
