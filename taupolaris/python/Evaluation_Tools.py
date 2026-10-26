import torch
from tqdm import tqdm
import numpy as np
import awkward as ak
from taupolaris.utils.kinematic_helpers import polarimetric_vector_tau, compute_spin_angles, boost_vector, boost, compute_spin_density_vars
from taupolaris.utils.coordinate_conversions import ConvertFromOrthonormalNRK_Predictions, convert_coordinates_pred
from taupolaris.utils.acoplanarity_tools import get_ditau_polarimetric, decay_mode_codes
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from iminuit import Minuit
import time

def flow_map_predict(
    model,
    X,
    test_dataset=None,
    num_draws=100,
    chunk_size=5000,
    method='gradient',
    n_steps=200,
    lr=1e-2,
    return_log_prob=False,
    verbose=False,
    n_random_starts=10,
    seed=12345,
    momentum_groups=((0, 1, 2), (3, 4, 5)),
    momentum_power=2,
):
    """
    Compute MAP (maximum log-probability) predictions from a normalizing flow.

    Parameters
    ----------
    model : ConditionalFlow
        The trained normalizing flow model.
    X : torch.Tensor
        Conditioning features of shape [B, context_dim].
    test_dataset : object, optional
        Must supply .destandardize_outputs(tensor). If None, no destandardization is performed.
    num_draws : int
        Number of samples per event. Used by method='stochastic' and 'gradient_warmstart'.
    chunk_size : int
        Number of events to process at once (controls memory usage).
    method : str
        'stochastic'          — draw num_draws samples per event and pick the highest log-prob
                                one. Uses sample_and_log_prob() for a single forward pass.
        'latent_zero'         — decode the latent origin z=0 for each event. Deterministic,
                                single forward pass. Best for unimodal posteriors.
        'gradient'            — z-space MAP: optimise z via Adam using the change-of-variables
                                identity log p = log p_Z(z) - log|det J_decode|. Initialises
                                from z=0. Single decode pass per step.
        'stable_gradient'     — same as 'gradient', but maximises the density in spherical
                                momentum coordinates (|p|, angles) for each neutrino instead of
                                Cartesian, i.e. log p(x) + 2 log|p| per neutrino. The Cartesian
                                density picks up a 1/|p|^2 volume factor at p=0, which drags the
                                'gradient' mode to p_nu ~ 0 for any posterior with support at low
                                energy; the 2 log|p| term removes it (map_minimizer_study.py
                                --jac_power 2). Needs test_dataset (to get |p| in physical units)
                                and Cartesian-like outputs (standard/onorm). The returned log p
                                is the Cartesian one, without the 2 log|p| terms.
        'gradient_multistart' — 'gradient' from z=0 plus n_random_starts starts z ~ N(0, 1), all run
                                in parallel; the highest final log p per event is kept (multi200 of
                                map_minimizer_study.py: the same starts, from one generator seeded
                                with `seed`, so they don't depend on the chunking). chunk_size counts
                                optimised rows, i.e. chunk_size // (1 + n_random_starts) events per
                                chunk, so the memory is that of 'gradient' with the same chunk_size.
        'stable_gradient_multistart' — 'gradient_multistart' with the objective of 'stable_gradient'
                                (log p + 2 log|p| per neutrino), which also picks the best start; the
                                returned log p is the Cartesian one of that start.
        'gradient_warmstart'  — same as 'gradient' but initialises z from the best of
                                num_draws stochastic samples (encoded back to z-space).
                                Handles multimodal posteriors: sampling finds the right basin,
                                gradient descent finds the exact mode within it.
        'minuit'              — same z-space change-of-variables objective as 'gradient', but
                                optimised with Minuit's MIGRAD instead of Adam. Not vectorised
                                over the batch, so events are processed one at a time; much
                                slower than the gradient methods but a more reliable benchmark
                                for the true MAP.
    n_steps : int
        Number of Adam steps. Only used by gradient methods.
    n_random_starts, seed : int
        Random starts per event and their generator seed. Only used by 'gradient_multistart'.
    lr : float
        Adam learning rate. Only used by gradient methods.
    momentum_groups : tuple of index tuples
        Output indices of each neutrino's 3-momentum components. Only used by the 'stable_gradient' methods;
        the default is the hadronic layout (taup_nu_{n,r,k}, taun_nu_{n,r,k}).
    momentum_power : float
        k in the 'stable_gradient' objective log p + k log|p| per neutrino. 2 (default): density per
        d|p| dOmega. 1: density per d|p| dphi dm^2 (neutrino energy, azimuth about the visible tau and
        visible+neutrino invariant mass), since d^3p = |p| / (2 |p_vis|) d|p| dphi dm^2.
    verbose : bool
        If True, show a running average of time-per-event and iterations-per-event in the
        progress bar, updated as more events are processed (so the average keeps refining
        rather than the running total).

    Returns
    -------
    samples_norm_alt : torch.Tensor, shape [B, features]
        MAP-selected samples in normalized (flow) space.
    samples_alt : np.ndarray or None
        Destandardized samples, or None if test_dataset not provided.
    """

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    B = X.shape[0]
    all_best_samples = []
    all_best_log_probs = [] if return_log_prob else None

    # running totals for the verbose progress-bar averages; averaged over events
    # processed so far, so they keep refining as more events are processed
    _total_time = 0.0
    _total_iters = 0
    _total_events = 0
    _total_log_p = 0.0

    def _update_postfix(pbar, dt, iters, n_events, log_p_sum):
        nonlocal _total_time, _total_iters, _total_events, _total_log_p
        _total_time += dt
        _total_iters += iters
        _total_events += n_events
        _total_log_p += log_p_sum
        if verbose:
            pbar.set_postfix({
                'avg_s/event': f"{_total_time / _total_events:.4f}",
                'avg_iters/event': f"{_total_iters / _total_events:.1f}",
                'avg_log_p': f"{_total_log_p / _total_events:.3f}",
            })

    spherical = method.startswith('stable_gradient')
    if spherical:
        if test_dataset is None:
            raise ValueError(f"method='{method}' needs test_dataset to compute |p| in physical units")
        out_mean = test_dataset.output_mean.to(X.device)
        out_std = test_dataset.output_std.to(X.device)

    def log_volume(x_norm):
        # momentum_power * log|p| per neutrino: 2 converts the Cartesian density to one per d|p| dOmega
        # (eps keeps it finite at p=0, where the term is very negative, so never selected)
        x_phys = x_norm * out_std + out_mean
        return sum(momentum_power * torch.log(x_phys[:, list(g)].norm(dim=-1) + 1e-6) for g in momentum_groups)

    if method == 'latent_zero':
        # this doesn't work very well, but might as well leave it in
        latent_dim = model.flow._distribution._shape[0]
        pbar = tqdm(range(0, B, chunk_size), desc="Processing chunks (latent_zero)")
        for start in pbar:
            t0 = time.time()
            end = min(start + chunk_size, B)
            X_chunk = X[start:end]
            C = X_chunk.shape[0]
            z_zero = torch.zeros(C, latent_dim, device=X_chunk.device)
            with torch.no_grad():
                x_map, _ = model.decode(z_zero, context=X_chunk)
                cond_embed_chunk = model.condition_net(X_chunk)
                log_p = model.flow.log_prob(inputs=x_map, context=cond_embed_chunk)
            all_best_samples.append(x_map.cpu())
            _update_postfix(pbar, time.time() - t0, C, C, log_p.sum().item())  # single decode pass per event

    elif method == 'stochastic':
        # sample from the pdf
        pbar = tqdm(range(0, B, chunk_size), desc="Processing chunks (stochastic)")
        for start in pbar:
            t0 = time.time()
            end = min(start + chunk_size, B)
            X_chunk = X[start:end]
            C = X_chunk.shape[0]
            with torch.no_grad():
                # single pass get both [C, num_draws, F] and [C, num_draws]
                samples_norm_chunk, log_probs = model.sample_and_log_prob(
                    num_samples=num_draws, context=X_chunk
                )
            best_idx = torch.argmax(log_probs, dim=1)              # [C]
            best_samples_chunk = samples_norm_chunk[torch.arange(C), best_idx]  # [C, F]
            best_log_p_sum = log_probs[torch.arange(C), best_idx].sum().item()
            # free the large [C, num_draws, F] tensor before appending to avoid two chunks in memory at once
            del samples_norm_chunk, log_probs, best_idx
            all_best_samples.append(best_samples_chunk.cpu())
            _update_postfix(pbar, time.time() - t0, C * num_draws, C, best_log_p_sum)

    elif method in ('gradient', 'stable_gradient'):
        # optimisation of the MAP estimate in z-space
        latent_dim = model.flow._distribution._shape[0]

        pbar = tqdm(range(0, B, chunk_size), desc=f"Processing chunks ({method})")
        for start in pbar:
            t0 = time.time()
            end = min(start + chunk_size, B)
            X_chunk = X[start:end]
            C = X_chunk.shape[0]

            # pre-compute the embedding once for this chunk
            with torch.no_grad():
                cond_embed_chunk = model.condition_net(X_chunk)

            # optimise z directly using the change-of-variables log p(x|c) = log p_Z(z) - log|det J_decode(z)|
            # (z-space geometry is isotropic so easier to optimise)
            z = torch.zeros(C, latent_dim, device=X_chunk.device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=lr)

            for step in range(n_steps):
                optimizer.zero_grad()
                
                x, logabsdet = model.flow._transform.inverse(z, context=cond_embed_chunk)
                
                log_pz = -0.5 * (z ** 2).sum(dim=-1)
                log_p = log_pz - logabsdet 
                #if step == 0: initial_log_p = log_p.mean().item()
                if spherical:
                    (-(log_p + log_volume(x)).sum()).backward()  # avoid divergence
                else:
                    (-log_p.sum()).backward()

                optimizer.step()

                #if step % 10 == 0:
                #    print(
                #    f"step={step:4d} "
                #    f"mean_logp={log_p.mean().item():.6f} "
                #    f"max_logp={log_p.max().item():.6f} "
                #    f"min_logp={log_p.min().item():.6f}"
                #    )

            #t1 = time.time()
            #print(f"Time taken for maximizing log p: {t1 - t0:.2f} s")
            #final_log_p = log_p.mean().item()
            #print(f"Initial mean log p: {initial_log_p:.6f}, Final mean log p: {final_log_p:.6f}, Gain: {final_log_p - initial_log_p:.6f}")

            if return_log_prob:
                all_best_log_probs.append(log_p.detach().cpu())

            with torch.no_grad():
                x_map, _ = model.flow._transform.inverse(z.detach(), context=cond_embed_chunk)
            #print(f"x_map[0]={x_map[0].cpu().numpy()}")
            all_best_samples.append(x_map.cpu())
            _update_postfix(pbar, time.time() - t0, C * n_steps, C, log_p.detach().sum().item())

    elif method in ('gradient_multistart', 'stable_gradient_multistart'):
        latent_dim = model.flow._distribution._shape[0]
        S = 1 + n_random_starts
        ev_chunk = max(1, chunk_size // S)
        rng = np.random.default_rng(seed)

        pbar = tqdm(range(0, B, ev_chunk), desc=f"Processing chunks (gradient, z=0 + {n_random_starts} random starts)")
        for start in pbar:
            t0 = time.time()
            end = min(start + ev_chunk, B)
            X_chunk = X[start:end]
            C = X_chunk.shape[0]

            with torch.no_grad():
                cond_embed_chunk = model.condition_net(X_chunk).repeat_interleave(S, 0)

            z0 = torch.zeros(C, S, latent_dim)
            z0[:, 1:] = torch.from_numpy(rng.standard_normal((C, n_random_starts, latent_dim)).astype(np.float32))
            z = z0.reshape(C * S, latent_dim).to(X_chunk.device).requires_grad_(True)
            optimizer = torch.optim.Adam([z], lr=lr)

            for step in range(n_steps + 1):
                optimizer.zero_grad()
                x, logabsdet = model.flow._transform.inverse(z, context=cond_embed_chunk)
                log_p = -0.5 * (z ** 2).sum(dim=-1) - logabsdet
                objective = log_p + log_volume(x) if spherical else log_p
                if step == n_steps:   # log p at the end point of every start, to pick the best
                    break
                (-objective.sum()).backward()
                optimizer.step()

            log_p = log_p.detach().reshape(C, S)
            best = objective.detach().reshape(C, S).argmax(1)
            z_best = z.detach().reshape(C, S, latent_dim)[torch.arange(C, device=z.device), best]
            log_p_best = log_p[torch.arange(C, device=log_p.device), best]
            if return_log_prob:
                all_best_log_probs.append(log_p_best.cpu())

            with torch.no_grad():
                x_map, _ = model.flow._transform.inverse(z_best, context=cond_embed_chunk.reshape(C, S, -1)[:, 0])
            all_best_samples.append(x_map.cpu())
            _update_postfix(pbar, time.time() - t0, C * S * n_steps, C, log_p_best.sum().item())

    elif method == 'gradient_forward':

        latent_dim = model.flow._distribution._shape[0]

        pbar = tqdm(range(0, B, chunk_size), desc="Processing chunks (forward gradient)")
        for start in pbar:
            t0 = time.time()
            end = min(start + chunk_size, B)
            X_chunk = X[start:end]
            C = X_chunk.shape[0]

            # Pre-compute the embedding once for this chunk 
            with torch.no_grad():
                cond_embed_chunk = model.condition_net(X_chunk)

            # Initialize optimization variable x
            z_init = torch.zeros(C, latent_dim, device=X_chunk.device)
            with torch.no_grad():
                # use the pre-computed embedding here to save a decode calculation step
                x_init, _ = model.flow._transform.inverse(z_init, context=cond_embed_chunk)
            
            x = x_init.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=lr)

            for step in range(n_steps):
                optimizer.zero_grad()
                
                log_p = model.flow.log_prob(inputs=x, context=cond_embed_chunk)

                final_log_p = log_p.mean().item()
                if step == 0: initial_log_p = final_log_p
                
                loss = - log_p.sum()
                loss.backward()
                optimizer.step()

                #if (step+1) % 10 == 0 or step == 0:
                #    print(
                #        f"step={step:4d} "
                #        f"mean_logp={log_p.mean().item():.6f} "
                #        f"max_logp={log_p.max().item():.6f} "
                #        f"min_logp={log_p.min().item():.6f} "
                #        f"x[0]={x[0].detach().cpu().numpy()}"
                #    )

            #t1 = time.time()
            #print(f"Time taken for maximizing log p {t1 - t0:.2f} s")
            #print(f"Initial mean log p: {initial_log_p:.6f}, Final mean log p: {final_log_p:.6f}, Gain: {final_log_p - initial_log_p:.6f}")
            all_best_samples.append(x.detach().cpu())
            _update_postfix(pbar, time.time() - t0, C * n_steps, C, log_p.detach().sum().item())

    elif method == 'minuit':
        # Minuit MAP: optimise z via MIGRAD using the same change-of-variables
        # identity as method='gradient' (log p = log p_Z(z) - log|det J_decode|).
        # Minuit isn't vectorised over a batch, so events are processed one at a time.
        latent_dim = model.flow._distribution._shape[0]

        pbar = tqdm(range(B), desc="Processing events (minuit)")
        for i in pbar:
            t0 = time.time()
            X_i = X[i:i + 1]

            with torch.no_grad():
                cond_embed_i = model.condition_net(X_i)

            z_init_np = np.zeros(latent_dim)

            def neg_log_p(z_params):
                z = torch.as_tensor(z_params, dtype=X_i.dtype, device=X_i.device).unsqueeze(0)
                with torch.no_grad():
                    _, logabsdet = model.flow._transform.inverse(z, context=cond_embed_i)
                    log_pz = -0.5 * (z ** 2).sum(dim=-1)
                    log_p = log_pz - logabsdet
                return -log_p.item()

            m = Minuit(neg_log_p, z_init_np)
            m.errordef = Minuit.LIKELIHOOD
            m.migrad()

            z_map = torch.tensor(m.values[:], dtype=X_i.dtype, device=X_i.device).unsqueeze(0)
            with torch.no_grad():
                x_map, _ = model.flow._transform.inverse(z_map, context=cond_embed_i)
            all_best_samples.append(x_map.cpu())

            if return_log_prob:
                all_best_log_probs.append(torch.tensor([-m.fval]))

            _update_postfix(pbar, time.time() - t0, m.nfcn, 1, -m.fval)  # m.nfcn = number of MIGRAD function calls; m.fval = -log_p at the minimum

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'stochastic', 'latent_zero', 'gradient', 'stable_gradient', 'gradient_multistart', 'stable_gradient_multistart', 'gradient_forward', or 'minuit'.")

    samples_norm_alt = torch.cat(all_best_samples, dim=0)

    if test_dataset is not None:
        samples_alt = test_dataset.destandardize_outputs(samples_norm_alt).cpu().numpy()
    else:
        samples_alt = None

    if return_log_prob:
        return samples_norm_alt, samples_alt, torch.cat(all_best_log_probs, dim=0).numpy()
    return samples_norm_alt, samples_alt


def spin_angles_from_polarimetric(h_plus, k_plus, h_minus, k_minus):
    """The spin projections cos theta_{n,r,k}^+- (and cosTheta) of kinematic_helpers.compute_spin_angles from the
    polarimetric vectors and unit tau directions in the ditau rest frame, as returned by
    acoplanarity_tools.get_ditau_polarimetric (the vectors phiCP is computed from). Also returns h+ and h- as (N, 3)
    arrays, zero for legs of no known decay mode."""
    as_np = lambda v: np.stack([ak.to_numpy(v.x), ak.to_numpy(v.y), ak.to_numpy(v.z)], axis=1)
    hp, hm = as_np(h_plus), as_np(h_minus)
    hp[hp[:, 0] == -9999.] = 0.
    hm[hm[:, 0] == -9999.] = 0.
    kp = as_np(k_plus)
    # compute_spin_angles takes four-vectors and uses only the direction of the tau+
    taup4 = np.column_stack([np.ones(len(kp)), kp])
    return compute_spin_angles(taup4, taup4, hp, hm), hp, hm


def compute_spin_vars(df, tau_pred_prefix='true_', tau_vis_prefix='', store_polvecs=False):
    """Adds {tau_pred_prefix}cos{n,r,k}_{plus,minus} and cosTheta (and with store_polvecs the polarimetric vectors
    {tau_pred_prefix}h_{plus,minus}_{x,y,z}: tau rest frames reached through the ditau frame, lab-aligned axes) for the
    taus {tau_pred_prefix}tau_{plus,minus} and the visible decay products {tau_vis_prefix}{taup,taun}_*.
    With reco_/true_ decay products and their decay-mode flags, the polarimetric vectors are those phiCP is computed
    from (acoplanarity_tools.get_ditau_polarimetric, with the 3h+pi0 current for DM 11); otherwise the older
    implementation (_compute_spin_vars_legacy)."""
    vis = tau_vis_prefix.rstrip('_')
    needed = [f'{tau_vis_prefix}{t}_{c}' for t in ('taup', 'taun') for c in ('pi2_px', 'pi3_px', 'ishadronic', 'npizero', 'is3prong')]
    if vis not in ('reco', 'true') or not all(c in df.columns for c in needed):
        return _compute_spin_vars_legacy(df, tau_pred_prefix, tau_vis_prefix, store_polvecs)
    h_plus, k_plus, h_minus, k_minus = get_ditau_polarimetric(
        df, tau_prefix=tau_pred_prefix.rstrip('_'), reco_pions=(vis == 'reco'), dm=decay_mode_codes(df, vis), verbose=False)
    spin_angles, hp, hm = spin_angles_from_polarimetric(h_plus, k_plus, h_minus, k_minus)
    for key, values in spin_angles.items():
        df[f'{tau_pred_prefix}{key}'] = values
    if store_polvecs:
        for c, k in enumerate('xyz'):
            df[f'{tau_pred_prefix}h_plus_{k}'] = hp[:, c]
            df[f'{tau_pred_prefix}h_minus_{k}'] = hm[:, c]
    return df


def _compute_spin_vars_legacy(df, tau_pred_prefix='true_', tau_vis_prefix='', store_polvecs=False):

    taup = df[[f'{tau_pred_prefix}tau_plus_E', f'{tau_pred_prefix}tau_plus_px', f'{tau_pred_prefix}tau_plus_py', f'{tau_pred_prefix}tau_plus_pz']].values
    taun = df[[f'{tau_pred_prefix}tau_minus_E', f'{tau_pred_prefix}tau_minus_px', f'{tau_pred_prefix}tau_minus_py', f'{tau_pred_prefix}tau_minus_pz']].values
    taup_pi1 = df[[f'{tau_vis_prefix}taup_pi1_E', f'{tau_vis_prefix}taup_pi1_px', f'{tau_vis_prefix}taup_pi1_py', f'{tau_vis_prefix}taup_pi1_pz']].values
    taup_pizero1 = df[[f'{tau_vis_prefix}taup_pizero1_E', f'{tau_vis_prefix}taup_pizero1_px', f'{tau_vis_prefix}taup_pizero1_py', f'{tau_vis_prefix}taup_pizero1_pz']].values
    taun_pi1 = df[[f'{tau_vis_prefix}taun_pi1_E', f'{tau_vis_prefix}taun_pi1_px', f'{tau_vis_prefix}taun_pi1_py', f'{tau_vis_prefix}taun_pi1_pz']].values
    taun_pizero1 = df[[f'{tau_vis_prefix}taun_pizero1_E', f'{tau_vis_prefix}taun_pizero1_px', f'{tau_vis_prefix}taun_pizero1_py', f'{tau_vis_prefix}taun_pizero1_pz']].values

    # Semileptonic support: if the ishadronic flag is available, use the
    # charged-leg momentum (== the single lepton, for a leptonically-decaying
    # tau) and the lepton-direction polarimetric approximation for those legs,
    # since taup_pi1/taup_pizero1 aren't meaningful for a leptonic decay.
    # Older (hadronic-only) dataframes without this flag fall back to the
    # original pion-only behaviour unchanged.
    has_lepton_info = (
        f'{tau_vis_prefix}taup_ishadronic' in df.columns and
        f'{tau_vis_prefix}taun_ishadronic' in df.columns
    )
    if has_lepton_info:
        taup_is_leptonic = df[f'{tau_vis_prefix}taup_ishadronic'].values == 0
        taun_is_leptonic = df[f'{tau_vis_prefix}taun_ishadronic'].values == 0
        taup_lep = df[[f'{tau_vis_prefix}taup_charged_E', f'{tau_vis_prefix}taup_charged_px', f'{tau_vis_prefix}taup_charged_py', f'{tau_vis_prefix}taup_charged_pz']].values
        taun_lep = df[[f'{tau_vis_prefix}taun_charged_E', f'{tau_vis_prefix}taun_charged_px', f'{tau_vis_prefix}taun_charged_py', f'{tau_vis_prefix}taun_charged_pz']].values

    # a1 (3-prong) support: if is3prong is available, use the real number of
    # charged pions (1 or 3) instead of hardcoding 1 for every row, and load
    # pi2/pi3 so the correct a1 polarimetric formula (polarimetric_vector_a1)
    # can be used for 3-prong legs instead of the 1-prong-only pi1/pizero1
    # formulas. pi1 is assumed to be the opposite-charge pion and pi2/pi3 the
    # two same-charge pions (SortPions convention in run_delphes.py); this
    # only holds for dataframes produced after that ordering fix, so older
    # reco-level a1 legs may still be mislabeled.
    has_a1_info = (
        f'{tau_vis_prefix}taup_is3prong' in df.columns and
        f'{tau_vis_prefix}taun_is3prong' in df.columns
    )
    if has_a1_info:
        taup_npi = np.where(df[f'{tau_vis_prefix}taup_is3prong'].values == 1, 3, 1)
        taun_npi = np.where(df[f'{tau_vis_prefix}taun_is3prong'].values == 1, 3, 1)
        taup_pi2 = df[[f'{tau_vis_prefix}taup_pi2_E', f'{tau_vis_prefix}taup_pi2_px', f'{tau_vis_prefix}taup_pi2_py', f'{tau_vis_prefix}taup_pi2_pz']].values
        taup_pi3 = df[[f'{tau_vis_prefix}taup_pi3_E', f'{tau_vis_prefix}taup_pi3_px', f'{tau_vis_prefix}taup_pi3_py', f'{tau_vis_prefix}taup_pi3_pz']].values
        taun_pi2 = df[[f'{tau_vis_prefix}taun_pi2_E', f'{tau_vis_prefix}taun_pi2_px', f'{tau_vis_prefix}taun_pi2_py', f'{tau_vis_prefix}taun_pi2_pz']].values
        taun_pi3 = df[[f'{tau_vis_prefix}taun_pi3_E', f'{tau_vis_prefix}taun_pi3_px', f'{tau_vis_prefix}taun_pi3_py', f'{tau_vis_prefix}taun_pi3_pz']].values
        taup_charge = np.ones(len(df))
        taun_charge = -np.ones(len(df))
    else:
        taup_npi = np.ones_like(df[f'{tau_vis_prefix}taup_haspizero'].values)
        taun_npi = np.ones_like(df[f'{tau_vis_prefix}taun_haspizero'].values)

    com_boost_vec = boost_vector(taup + taun)
    taup = boost(taup, -com_boost_vec)
    taun = boost(taun, -com_boost_vec)
    # boost decay products as well
    taup_pi1 = boost(taup_pi1, -com_boost_vec)
    taup_pizero1 = boost(taup_pizero1, -com_boost_vec)
    taun_pi1 = boost(taun_pi1, -com_boost_vec)
    taun_pizero1 = boost(taun_pizero1, -com_boost_vec)
    if has_lepton_info:
        taup_lep = boost(taup_lep, -com_boost_vec)
        taun_lep = boost(taun_lep, -com_boost_vec)
    if has_a1_info:
        taup_pi2 = boost(taup_pi2, -com_boost_vec)
        taup_pi3 = boost(taup_pi3, -com_boost_vec)
        taun_pi2 = boost(taun_pi2, -com_boost_vec)
        taun_pi3 = boost(taun_pi3, -com_boost_vec)

    taup_s = polarimetric_vector_tau(
        taup, taup_pi1, taup_pizero1,
        taup_npi, df[f'{tau_vis_prefix}taup_haspizero'].values,
        lep=taup_lep if has_lepton_info else None,
        is_leptonic=taup_is_leptonic if has_lepton_info else None,
        pi2=taup_pi2 if has_a1_info else None,
        pi3=taup_pi3 if has_a1_info else None,
        taucharge=taup_charge if has_a1_info else None,
    )
    taun_s = polarimetric_vector_tau(
        taun, taun_pi1, taun_pizero1,
        taun_npi, df[f'{tau_vis_prefix}taun_haspizero'].values,
        lep=taun_lep if has_lepton_info else None,
        is_leptonic=taun_is_leptonic if has_lepton_info else None,
        pi2=taun_pi2 if has_a1_info else None,
        pi3=taun_pi3 if has_a1_info else None,
        taucharge=taun_charge if has_a1_info else None,
    )

    spin_angles = compute_spin_angles(
        taup, taun,
        taup_s, taun_s,
        p_axis=None
    )

    # now add these to the dataframe
    for key, values in spin_angles.items():
        df[f'{tau_pred_prefix}{key}'] = values
    if store_polvecs:
        # the polarimetric vectors themselves (tau rest frames reached through the ditau frame, lab-aligned axes),
        # e.g. to project them onto a different (n, r, k) basis -- see taupolaris/python/spin_basis.py
        for c, k in enumerate('xyz'):
            df[f'{tau_pred_prefix}h_plus_{k}'] = taup_s[:, c]
            df[f'{tau_pred_prefix}h_minus_{k}'] = taun_s[:, c]

    return df

def save_sampled_pdfs(
    model,
    dataset,
    device,
    df,
    output_features,
    event_number,
    num_samples=50000,
    bins=100,
    outdir="pdf_slices_sampled",
    use_polar=False,
    use_onorm=False,
):
    """
    Estimate 1D marginals p(x_i | context) by directly sampling the conditional flow
    and plotting *normalized histograms* (no KDE).
    """

    os.makedirs(outdir, exist_ok=True)

    model.eval()

    X = dataset.X
    y = dataset.y
    X, y = X.to(device), y.to(device)

    # select just one event from row = event_number

    X = X[event_number].unsqueeze(0)

    with torch.no_grad():
        predictions_norm = model.sample(num_samples=num_samples, context=X).squeeze()

    predictions = dataset.destandardize_outputs(predictions_norm).cpu().numpy()

    if use_polar:
        # convert predictions to cartesian coordinates
        predictions = ConvertPredictionsToCartesian(predictions, output_features)
    if use_onorm:
        predictions = ConvertFromOrthonormalNRK_Predictions(
            predictions,
            taup_pi=df[['reco_taup_pi1_E', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz']].values,
            taup_pi0=df[['reco_taup_pizero1_E', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz']].values,
            taun_pi=df[['reco_taun_pi1_E', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz']].values,
            taun_pi0=df[['reco_taun_pizero1_E', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz']].values,
        )


    n_bins = bins   # preserve the integer

    for i, v in enumerate(output_features):

        # find binning to show 99% of the PDF distribution
        v_values = predictions[:, i]
        lower_bound = np.percentile(v_values, 0.5)
        upper_bound = np.percentile(v_values, 99.5)
        bins = np.linspace(lower_bound, upper_bound, n_bins)

        pred_i = predictions[:, i]
        plt.figure(figsize=(6, 4))
        plt.hist(pred_i, bins=bins, density=True, histtype='step', linewidth=2)
        # draw true value as an arrow
        true_value = dataset.destandardize_outputs(y[event_number].unsqueeze(0)).cpu().numpy()[0, i]
        if use_polar:
            # convert true value to cartesian coordinates
            true_value = ConvertPredictionsToCartesian(
                np.array([true_value]),
                output_features
            )[0, i]
        if use_onorm:
            true_value = ConvertFromOrthonormalNRK_Predictions(
                np.array([true_value]),
                taup_pi=df[['reco_taup_pi1_E', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz']].values[event_number:event_number+1],
                taup_pi0=df[['reco_taup_pizero1_E', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz']].values[event_number:event_number+1],
                taun_pi=df[['reco_taun_pi1_E', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz']].values[event_number:event_number+1],
                taun_pi0=df[['reco_taun_pizero1_E', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz']].values[event_number:event_number+1],
            )[0, i]
        plt.axvline(true_value, color='r', linestyle='--', linewidth=2, label='True Value')

        # also get analytical solutions if available
        analytical_col_0 = f'ana_pred_{v}'
        analytical_col_1 = f'ana_alt_pred_{v}'

        if analytical_col_0 in df.columns and analytical_col_1 in df.columns:
            analytical_sol_0 = df.iloc[event_number][analytical_col_0]
            analytical_sol_1 = df.iloc[event_number][analytical_col_1]
            plt.axvline(analytical_sol_0, color='g', linestyle='--', linewidth=2, label='Preferred Analytical Solution')
            plt.axvline(analytical_sol_1, color='b', linestyle=':', linewidth=2, label='Alternative Analytical Solution')
            plt.legend()

        plt.xlabel(v)
        plt.ylabel("pdf (sampled)")
        plt.title(f"Sampled p({v} | context of event {event_number})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"event{event_number}_{v}.pdf"))
        plt.close()

def _plot_pdf(v_pred, true_val, map_val, label, outdir, event_number, bins, clip, xlabel=None):
    if xlabel is None:
        xlabel = label
    bins = np.linspace(np.percentile(v_pred, 0.1), np.percentile(v_pred, 99.9), bins) if clip else bins
    plt.figure(figsize=(6, 4))
    plt.hist(v_pred, bins=bins, density=True, histtype='step', linewidth=2)
    plt.axvline(true_val, color='r', linestyle='--', linewidth=2, label='True value')
    if map_val is not None:
        plt.axvline(map_val, color='orange', linestyle='--', linewidth=2, label='MAP estimate')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("pdf (sampled)")
    plt.title(f"Sampled p({label} | context), event {event_number}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"event{event_number}_{label}.pdf"))
    plt.close()


def save_sampled_pdfs_LHC(
    model,
    dataset,
    device,
    output_features,
    event_number,
    num_samples=50000,
    bins=100,
    outdir="pdf_slices_sampled",
    map_value=None,
    df=None,
    coordinates='onorm',
    leptonic_mode=0,
    clip=True,
):
    """
    Sample 1D marginals p(x_i | context) from the conditional flow for a single
    LHC event and save a histogram per output feature. Plots are in the native
    coordinate space of the model (e.g. n/r/k for onorm).

    map_value : np.ndarray of shape [n_output_features], optional
        MAP estimate in the same destandardized coordinate space as the samples.
        If provided, overlaid as an orange dashed line on each plot.
    df : pd.DataFrame, optional
        Test dataframe with visible tau decay products. If provided, also produces
        Cartesian (px,py,pz) and energy plots after converting from native coordinates.
    """
    os.makedirs(outdir, exist_ok=True)
    model.eval()

    X = dataset.X[event_number].unsqueeze(0).to(device)
    y = dataset.y[event_number].unsqueeze(0)

    with torch.no_grad():
        predictions_norm = model.sample(num_samples=num_samples, context=X).squeeze()

    predictions = dataset.destandardize_outputs(predictions_norm).cpu().numpy()
    true_values = dataset.destandardize_outputs(y).cpu().numpy()[0]

    if clip==True:
        print("WARNING: This script clips 0.1% outliers on variables by default")
    for i, v in enumerate(output_features):
        _plot_pdf(predictions[:, i], true_values[i],
                  map_value[i] if map_value is not None else None,
                  v, outdir, event_number, bins, clip)

    # plot cartesian components and energy
    if df is not None and coordinates!='standard':
        print('recomputing 4 components in cartesian')
        tau1_prefix = 'taup' if 'taup_nu_px' in df.columns else 'tau1'
        tau2_prefix = 'taun' if tau1_prefix == 'taup' else 'tau2'

        reco_taup_charged = df[[f'reco_{tau1_prefix}_charged_e', f'reco_{tau1_prefix}_charged_px', f'reco_{tau1_prefix}_charged_py', f'reco_{tau1_prefix}_charged_pz']].values[event_number:event_number+1]
        reco_taun_charged = df[[f'reco_{tau2_prefix}_charged_e', f'reco_{tau2_prefix}_charged_px', f'reco_{tau2_prefix}_charged_py', f'reco_{tau2_prefix}_charged_pz']].values[event_number:event_number+1]
        reco_taup_pizero = df[[f'reco_{tau1_prefix}_pizero1_e', f'reco_{tau1_prefix}_pizero1_px', f'reco_{tau1_prefix}_pizero1_py', f'reco_{tau1_prefix}_pizero1_pz']].values[event_number:event_number+1]
        reco_taun_pizero = df[[f'reco_{tau2_prefix}_pizero1_e', f'reco_{tau2_prefix}_pizero1_px', f'reco_{tau2_prefix}_pizero1_py', f'reco_{tau2_prefix}_pizero1_pz']].values[event_number:event_number+1]

        single_kwargs = dict(coordinates=coordinates, output_features=output_features,
                             tau1_charged=reco_taup_charged, tau1_pi0=reco_taup_pizero,
                             tau2_charged=reco_taun_charged, tau2_pi0=reco_taun_pizero,
                             leptonic_mode=leptonic_mode)
        n = len(predictions)
        pred_conv_kwargs = dict(coordinates=coordinates, output_features=output_features,
                                tau1_charged=np.tile(reco_taup_charged, (n, 1)),
                                tau1_pi0=np.tile(reco_taup_pizero, (n, 1)),
                                tau2_charged=np.tile(reco_taun_charged, (n, 1)),
                                tau2_pi0=np.tile(reco_taun_pizero, (n, 1)),
                                leptonic_mode=leptonic_mode)

        pred_cart = convert_coordinates_pred(predictions, **pred_conv_kwargs)
        true_cart = convert_coordinates_pred(true_values[np.newaxis, :], **single_kwargs)[0]
        map_cart = convert_coordinates_pred(map_value[np.newaxis, :], **single_kwargs)[0] if map_value is not None else None

        cart_features = ['taup_nu_px', 'taup_nu_py', 'taup_nu_pz','taun_nu_px', 'taun_nu_py', 'taun_nu_pz']
        nu_p_slices = {'taup_nu': (0, 3), 'taun_nu': (3, 6)}

        # plot px, py, pz
        for i, v in enumerate(cart_features):
            _plot_pdf(pred_cart[:, i], true_cart[i],
                      map_cart[i] if map_cart is not None else None,
                      v, outdir, event_number, bins, clip)

        # plot energy
        for nu_name, (s, e) in nu_p_slices.items():
            E_pred = np.sqrt(np.sum(pred_cart[:, s:e]**2, axis=1))
            E_true = np.sqrt(np.sum(true_cart[s:e]**2))
            E_map = np.sqrt(np.sum(map_cart[s:e]**2)) if map_cart is not None else None
            _plot_pdf(E_pred, E_true, E_map, f'{nu_name}_E', outdir, event_number, bins, clip,
                      xlabel=f'{nu_name}_E [GeV]')





def plot_spin_density_matrix(results, dm_category, outdir):
    os.makedirs(outdir, exist_ok=True)

    labels = list(results.keys())
    n = len(labels)
    axes_labels = ['n', 'r', 'k']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def _draw_row_separators(ax, n_rows, n_cols):
        for row in range(1, n_rows):
            ax.hlines(row - 0.5, -0.5, n_cols - 0.5, colors='white', linewidths=2)

    # C matrix
    vmax = max(np.abs(results[l][2]).max() for l in labels)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        C = results[label][2]
        im = ax.imshow(C, cmap='RdBu', norm=norm)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(axes_labels)
        ax.set_yticklabels(axes_labels)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{C[i, j]:.3f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(C[i, j]) > 0.6 * vmax else 'black')
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Spin correlation matrix C  —  {dm_category}', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'C_matrix_{dm_category}.pdf'), bbox_inches='tight')
    plt.close()

    # B+ and B-
    bplus_mat  = np.array([results[l][0] for l in labels])   # [n, 3]
    bminus_mat = np.array([results[l][1] for l in labels])   # [n, 3]
    b_vmax = max(np.abs(bplus_mat).max(), np.abs(bminus_mat).max())
    b_norm = TwoSlopeNorm(vmin=-b_vmax, vcenter=0, vmax=b_vmax)

    fig, axes = plt.subplots(1, 2, figsize=(7, 0.6 * n + 1.5))
    for ax, mat, title in zip(axes, [bplus_mat, bminus_mat], ['$B^+$', '$B^-$']):
        im = ax.imshow(mat, cmap='RdBu', norm=b_norm, aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels(axes_labels)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(3):
                ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(mat[i, j]) > 0.6 * b_vmax else 'black')
        _draw_row_separators(ax, n, 3)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Polarisation vectors  —  {dm_category}', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'B_vectors_{dm_category}.pdf'), bbox_inches='tight')
    plt.close()

    # Entanglement info
    ent_mat = np.array([[results[l][3].real, results[l][4].real] for l in labels])  # [n, 2]
    e_vmax = np.abs(ent_mat).max()
    e_norm = TwoSlopeNorm(vmin=-e_vmax, vcenter=0, vmax=e_vmax) if e_vmax > 0 else None

    fig, ax = plt.subplots(figsize=(4, 0.6 * n + 1.5))
    im = ax.imshow(ent_mat, cmap='RdBu', norm=e_norm, aspect='auto')
    ax.set_xticks(range(2))
    ax.set_xticklabels(['concurrence', '$m_{12}$'])
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(2):
            ax.text(j, i, f'{ent_mat[i, j]:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if e_vmax > 0 and abs(ent_mat[i, j]) > 0.6 * e_vmax else 'black')
    _draw_row_separators(ax, n, 2)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Entanglement variables  —  {dm_category}', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'entanglement_{dm_category}.pdf'), bbox_inches='tight')
    plt.close()
