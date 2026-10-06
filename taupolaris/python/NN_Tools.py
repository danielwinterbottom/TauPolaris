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


def _compute_flow_loss_with_es(model, X, y, w, es_fn, lam, fraction=1.0, max_tries=3):
    """NLL + lam * (phiCP energy score) for the flow, sharing one conditioner pass between the two.
    The ES is evaluated on the first `fraction` of the batch (batches are shuffled, so this is a
    random subset) and weighted like the NLL. Returns (total, nll, es); es is None when the rare
    nflows spline-inverse assertion hits every retry, and the batch then trains on the NLL alone."""
    emb = model.condition_net(X)
    log_p = model.flow.log_prob(inputs=y, context=emb)
    w_ = torch.ones_like(log_p) if w is None else w
    nll = -(w_ * log_p).sum() / w_.sum()
    n = max(2, int(round(fraction * len(X))))
    for _ in range(max_tries):
        try:
            es_ev = es_fn.per_event(model.flow, emb[:n], X[:n], y[:n])
            break
        except AssertionError:
            continue
    else:
        return nll, nll, None
    es = (w_[:n] * es_ev).sum() / w_[:n].sum()
    return nll + lam * es, nll, es


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


def load_model(hp, input_features, output_features, batch_norm=False, useMLP=False, useTransformer=False, useTransformerMLP=False, leptonic_mode=0, legacy_pizero_proj=False, polvec_feature_level=0, frame_aligned_level=0, context_hidden_layer=False):
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
            polvec_feature_level=polvec_feature_level,
            frame_aligned_level=frame_aligned_level,
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
                                polvec_feature_level=polvec_feature_level,
                                frame_aligned_level=frame_aligned_level,
                                context_hidden_layer=context_hidden_layer,
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


def build_lr_scheduler(optimizer, hp, steps_per_epoch):
    """Per-batch LR schedule, selected by hp['lr_schedule'].

    'cosine' (default): linear warmup over warmup_frac of the run, then cosine
        annealing to lr*lr_min_factor. This is the schedule every existing model
        was trained with and is built from the same objects, so leaving the key
        out of a config changes nothing.

    'wsd' (warmup-stable-decay): the same warmup, then the LR is held constant
        until the last decay_frac of the run, where it decays linearly to
        lr*lr_min_factor. The constant middle is what makes a run extendable:
        raising n_epochs on a reloaded model continues at the plateau LR instead
        of re-warming to the peak (a fresh cosine) or stepping past T_max (a
        reloaded cosine, which sends CosineAnnealingLR's recursive update to
        absurd values). The final drop in loss comes from the decay phase, so
        it happens whenever the run is actually ended rather than at a T_max
        fixed when the scheduler was built.
    """
    schedule = hp.get('lr_schedule', 'cosine')
    n_epochs = hp['num_epochs']
    total_steps = n_epochs * steps_per_epoch
    warmup_frac = hp.get('warmup_frac', 0.05)
    lr_min_factor = hp.get('lr_min_factor', 0.01)
    warmup_steps = int(warmup_frac * total_steps)
    lr = hp['lr']
    start_factor = 1e-3   # warmup starts at 0.1% of lr

    if schedule == 'cosine':
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=start_factor, total_iters=warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=lr * lr_min_factor)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        print(f">> LR schedule: cosine -- warmup to {lr:.2e} over {warmup_steps / steps_per_epoch:.1f} epochs, "
              f"cosine to {lr * lr_min_factor:.2e} at epoch {n_epochs}")
    elif schedule == 'wsd':
        decay_frac = hp.get('decay_frac', 0.2)
        decay_steps = int(decay_frac * total_steps)
        decay_start = total_steps - decay_steps

        def factor(step):
            if step < warmup_steps:
                return start_factor + (1.0 - start_factor) * step / max(warmup_steps, 1)
            if step < decay_start:
                return 1.0
            t = min((step - decay_start) / max(decay_steps, 1), 1.0)
            return 1.0 + (lr_min_factor - 1.0) * t

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
        print(f">> LR schedule: wsd -- warmup to {lr:.2e} over {warmup_steps / steps_per_epoch:.1f} epochs, "
              f"constant until epoch {decay_start / steps_per_epoch:.1f}, "
              f"linear decay to {lr * lr_min_factor:.2e} at epoch {n_epochs}")
    else:
        raise ValueError(f"Unknown lr_schedule '{schedule}' -- expected 'cosine' or 'wsd'")
    return scheduler


def init_condition_net_from(model, path, verbose=True):
    """Initialise model.condition_net (the ParticleTransformerCondition of a
    ConditionalFlow) from another checkpoint, leaving the flow itself at its
    random initialisation. Nothing is frozen: this only sets the starting point.

    Accepts either
      * a TransformerRegressor checkpoint (encoder pretrained with a supervised
        multi-target MSE, keys 'encoder.*'; the 'head.*' keys are ignored), or
      * a ConditionalFlow checkpoint (keys 'condition_net.*'), e.g. the nu-method
        model, whose conditioner is architecturally identical up to its input set.

    The load is partial by design: every tensor whose name and shape match is
    copied; a type_emb table with a different number of rows (a checkpoint with
    fewer/more input tokens, e.g. without the hadronic-current tokens) has its
    leading rows copied and the rest left at random init; anything else is left
    untouched and reported. Fails loudly only if nothing at all could be copied.
    """
    sd = torch.load(path, map_location='cpu')
    for prefix in ('encoder.', 'condition_net.'):
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if sub:
            break
    else:
        raise ValueError(f"{path} holds neither 'encoder.*' nor 'condition_net.*' tensors")
    if prefix == 'encoder.':
        # a regressor's first head layer (Linear context->context, then GELU) is the flow
        # conditioner's optional context_hidden layer; only used if the model has one
        for suffix in ('weight', 'bias'):
            if f'head.0.{suffix}' in sd:
                sub[f'context_hidden.0.{suffix}'] = sd[f'head.0.{suffix}']
    target = model.condition_net.state_dict()
    copied, partial, skipped = [], [], []
    with torch.no_grad():
        for name, dst in target.items():
            if name not in sub:
                skipped.append(f'{name} (absent)')
                continue
            src = sub[name]
            if src.shape == dst.shape:
                dst.copy_(src); copied.append(name)
            elif name.endswith('type_emb.weight') and src.shape[1:] == dst.shape[1:]:
                n = min(src.shape[0], dst.shape[0])
                dst[:n].copy_(src[:n]); partial.append(f'{name} ({n}/{dst.shape[0]} rows)')
            else:
                skipped.append(f'{name} ({tuple(src.shape)} vs {tuple(dst.shape)})')
    if not copied:
        raise ValueError(f'no tensor of {path} matches model.condition_net')
    model.condition_net.load_state_dict(target)
    if verbose:
        print(f">> condition_net initialised from {path} ('{prefix}' tensors): "
              f"{len(copied)} copied, {len(partial)} partially copied, {len(skipped)} left at random init")
        for m in partial + skipped:
            print(f">>     {m}")
    return copied, partial, skipped


def setup_model_and_training(hp, train_dataset, test_dataset, input_features, output_features, model_name, verbose=True, reload=False, reload_scheduler=False, reset_training=False, batch_norm=False, useMLP=False, useTransformer=False, useTransformerMLP=False, leptonic_mode=0, polvec_feature_level=0, frame_aligned_level=0, condition_net_init=None, condition_net_freeze=False,
                             context_hidden_layer=False, init_from=None):
    """init_from: path to a full state_dict (e.g. another run's best_model.pth) to start from -- for
    fine-tuning. Unlike reload it starts a fresh optimizer, scheduler, epoch count and history,
    and it takes precedence over condition_net_init (which only covers the conditioner)."""
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
            polvec_feature_level=polvec_feature_level,
            frame_aligned_level=frame_aligned_level,
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
                                polvec_feature_level=polvec_feature_level,
                                frame_aligned_level=frame_aligned_level,
                                context_hidden_layer=context_hidden_layer,
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

    if init_from and not reload:
        model.load_state_dict(torch.load(init_from, map_location='cpu'))
        print(f">> All model weights initialised from {init_from} (fresh optimizer/scheduler/epochs)")
        if condition_net_init:
            print(">> condition_net_init ignored because init_from already sets every weight")
            condition_net_init = None
    elif init_from and reload:
        print(">> init_from ignored because reload=True (resuming this run's own partial checkpoint)")

    if condition_net_init:
        if reload:
            print(f">> condition_net_init ignored because reload=True (the partial checkpoint already holds the weights)")
        elif not hasattr(model, 'condition_net'):
            raise ValueError('condition_net_init needs a ConditionalFlow model (use_transformer: True)')
        else:
            init_condition_net_from(model, condition_net_init, verbose=verbose)

    if condition_net_freeze:
        # keep the (pretrained) conditioner fixed: no gradients, and left out of the
        # optimizer altogether so AdamW's weight decay cannot touch it either.
        if not hasattr(model, 'condition_net'):
            raise ValueError('condition_net_freeze needs a ConditionalFlow model (use_transformer: True)')
        if not condition_net_init and not reload:
            print(">> WARNING: condition_net_freeze=True without condition_net_init -- freezing a randomly initialised conditioner")
        for p_ in model.condition_net.parameters():
            p_.requires_grad_(False)
        print(f">> condition_net FROZEN: {sum(p_.numel() for p_ in model.condition_net.parameters())} parameters excluded from training")

    if verbose:
        # print number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of model parameters: {total_params} (trainable: {trainable})")
    scheduler = None
    es = None

    optimizer = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=hp['lr'])
    scheduler = build_lr_scheduler(optimizer, hp, len(train_dataloader))

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
    start_epoch=0, initial_history=None, clip_grad_norm=None, es_term=None):
    """es_term: optional dict(fn=PhiCPEnergyScore, lam=float, fraction=float, rampup_epochs=int) adding
    lam * (phiCP energy score) to the flow NLL (taupolaris/python/phicp_loss.py). During training lam
    ramps linearly from 0 over the first rampup_epochs epochs of this run; validation always uses the
    full lam, so val_loss (and the best-model choice) has one fixed definition. The NLL and ES parts
    are also stored separately in history (train_nll/train_es/val_nll/val_es)."""
    # nflows' StandardNormal distribution registers a float64 buffer (_log_z)
    # internally; MPS doesn't support float64, so cast to float32 before moving device.
    model.float().to(device)
    history = initial_history if initial_history is not None else {"train_loss": [], "val_loss": []}
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    use_es = es_term is not None and not useMLP
    if use_es:
        for k in ('train_nll', 'train_es', 'val_nll', 'val_es'):
            history.setdefault(k, [])
        steps_per_epoch = len(train_dataloader)
        ramp_steps = max(1, int(es_term.get('rampup_epochs', 0) * steps_per_epoch))
        first_step = start_epoch * steps_per_epoch

    def flow_es_loss(X, y, w, lam):
        return _compute_flow_loss_with_es(model, X, y, w, es_term['fn'], lam, es_term.get('fraction', 1.0))

    if early_stopper: early_stopper.reset()

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train() # need to set model back to train mode, otherise dropout (and batchnorm if used) will not work
        running_loss=0
        run_nll, run_es, n_es, n_es_fail = 0.0, 0.0, 0, 0
        mlp_loss_fn = nn.MSELoss()
        for batch, batch_data in enumerate(train_dataloader):
            X, y, w = _unpack_batch(batch_data, device)
            optimizer.zero_grad()
            if use_es:
                step = (epoch - 1) * steps_per_epoch + batch - first_step
                lam = es_term['lam'] * min(1.0, (step + 1) / ramp_steps) if es_term.get('rampup_epochs', 0) > 0 else es_term['lam']
                loss, nll_b, es_b = flow_es_loss(X, y, w, lam)
                run_nll += nll_b.item()
                if es_b is None:
                    n_es_fail += 1
                else:
                    run_es += es_b.item(); n_es += 1
            else:
                loss = _compute_loss(model, X, y, w, useMLP, mlp_loss_fn)
            loss.backward()
            if clip_grad_norm:
                # bounds the rare huge gradients from the spline tails that otherwise
                # show up as one-epoch loss spikes; off unless hp['clip_grad_norm'] > 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

            lr = scheduler.get_last_lr()
            # changing learning rate per batch - only if scheduler is defined and is not ReduceLROnPlateau
            if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            if verbose and epoch<5 and batch % 100 == 0:
                print(f'Batch {batch} | loss {loss.item()} | lr: {lr[0]} ' if scheduler else f'Batch {batch} | loss {loss.item()}')
            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        if use_es:
            train_nll, train_es = run_nll / len(train_dataloader), run_es / max(n_es, 1)
            if n_es_fail:
                print(f'>> ES skipped on {n_es_fail} batches (nflows spline-inverse assertion on every retry)')

        model.eval()
        if recompute_train_loss:
            # recompute train loss for better estimate, capped to same number of batches as validation
            n_batches = len(test_dataloader)
            sum_train_loss, sum_nll, sum_es, k_es = 0, 0.0, 0.0, 0
            with torch.no_grad():
                for i, batch_data in enumerate(train_dataloader):
                    if i >= n_batches:
                        break
                    X_train, y_train, w_train = _unpack_batch(batch_data, device)
                    if use_es:
                        train_loss, nll_b, es_b = flow_es_loss(X_train, y_train, w_train, es_term['lam'])
                        sum_nll += nll_b.item()
                        if es_b is not None:
                            sum_es += es_b.item(); k_es += 1
                    else:
                        train_loss = _compute_loss(model, X_train, y_train, w_train, useMLP, mlp_loss_fn)
                    sum_train_loss += train_loss.item()
            train_loss = sum_train_loss / n_batches
            if use_es:
                train_nll, train_es = sum_nll / n_batches, sum_es / max(k_es, 1)

        history["train_loss"].append(train_loss)

        # validation phase
        val_running_loss, val_nll, val_es, k_es = 0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch_data in test_dataloader:
                X_val, y_val, w_val = _unpack_batch(batch_data, device)
                if use_es:
                    val_loss, nll_b, es_b = flow_es_loss(X_val, y_val, w_val, es_term['lam'])
                    val_nll += nll_b.item()
                    if es_b is not None:
                        val_es += es_b.item(); k_es += 1
                else:
                    val_loss = _compute_loss(model, X_val, y_val, w_val, useMLP, mlp_loss_fn)
                val_running_loss += val_loss.item()
        val_loss = val_running_loss / len(test_dataloader)
        history["val_loss"].append(val_loss)
        if use_es:
            val_nll, val_es = val_nll / len(test_dataloader), val_es / max(k_es, 1)
            for k, v in (('train_nll', train_nll), ('train_es', train_es), ('val_nll', val_nll), ('val_es', val_es)):
                history[k].append(v)
            print(f'Epoch {epoch} | NLL train {train_nll:.4f} val {val_nll:.4f} | phiCP ES train {train_es:.5f} val {val_es:.5f} '
                  f'| lambda {es_term["lam"]:.4g}')

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
