import torch
from tqdm import tqdm
import numpy as np
from taupolaris.utils.kinematic_helpers import polarimetric_vector_tau, compute_spin_angles, boost_vector, boost, compute_spin_density_vars
from taupolaris.utils.coordinate_conversions import ConvertFromOrthonormalNRK_Predictions, convert_coordinates_pred, ConvertToOrthonormalNRK
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MultipleLocator
from iminuit import Minuit
import time


import mplhep as hep
plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18})


# fixed tick spacing for _plot_pdf, applied to every PDF plot regardless of its data range
pdf_x_tick_step = 1.0
pdf_y_tick_step = 0.2


truth_color = "#EE8A3D" # '#C9B583'
sampled_color = '#0072B2'
pred_color = "#e42536" #'#CC3311'
transformer_color = '#2CA02C'
# truth_line_color = '#CC5801'


# Pretty x-axis labels for _plot_pdf, keyed by the raw feature name. Falls
# back to the raw name itself for anything not listed here (e.g. leptonic
# tau1/tau2 cartesian components, which reuse the taup/taun cart_features
# list further down and so aren't covered by name).
pretty_labels = {
    # hadronic onorm (taup/taun)
    # 'taup_nu_n': r"$\bar{\nu_\tau}$ ($n$ component)", 'taup_nu_r': r"$\bar{\nu_\tau}$ ($r$ component)", 'taup_nu_k': r"$\bar{\nu_\tau}$ ($k$ component)",
    # 'taun_nu_n': r"$\nu_\tau$ ($n$ component)", 'taun_nu_r': r"$\nu_\tau$ ($r$ component)", 'taun_nu_k': r"$\nu_\tau$ ($k$ component)",

    # 'taup_nu_n': r"$\mathbf{\nu}_{\mathbf{\tau}_{\mathbf{1}}}$ ($\mathbf{n}_{\mathbf{1}}$ component)", 'taup_nu_r': r"$\mathbf{\nu}_{\mathbf{\tau}_{\mathbf{1}}}$ ($\mathbf{r}_{\mathbf{1}}$ component)", 'taup_nu_k': r"$\mathbf{\nu}_{\mathbf{\tau}_{\mathbf{1}}}$ ($\mathbf{k}_{\mathbf{1}}$ component)",
    # 'taun_nu_n': r"$\mathbf{\nu}_{\mathbf{\tau}_{\mathbf{2}}}$ ($\mathbf{n}_{\mathbf{2}}$ component)", 'taun_nu_r': r"$\mathbf{\nu}_{\mathbf{\tau}_{\mathbf{2}}}$ ($\mathbf{r}_{\mathbf{2}}$ component)", 'taun_nu_k': r"$\mathbf{\nu}_{\mathbf{\tau}_{\mathbf{2}}}$ ($\mathbf{k}_{\mathbf{2}}$ component)",

    'taup_nu_n': r"${\nu}_{{\tau}_{{1}}}$ (${n}_{{1}}$ component)", 'taup_nu_r': r"${\nu}_{{\tau}_{{1}}}$ (${r}_{{1}}$ component)", 'taup_nu_k': r"${\nu}_{{\tau}_{{1}}}$ (${k}_{{1}}$ component)",
    'taun_nu_n': r"${\nu}_{{\tau}_{{2}}}$ (${n}_{{2}}$ component)", 'taun_nu_r': r"${\nu}_{{\tau}_{{2}}}$ (${r}_{{2}}$ component)", 'taun_nu_k': r"${\nu}_{{\tau}_{{2}}}$ (${k}_{{2}}$ component)",

    # semileptonic onorm (tau1/tau2)
    'tau1_nu_m': r"$\nu\nu_{\tau_1}$ (mass)",
    'tau1_nu_n': r"$\nu_{\tau_1}$ ($n$ component)", 'tau1_nu_r': r"$\nu_{\tau_1}$ ($r$ component)", 'tau1_nu_k': r"$\nu_{\tau_1}$ ($k$ component)",
    'tau2_nu_n': r"$\nu_{\tau_2}$ ($n$ component)", 'tau2_nu_r': r"$\nu_{\tau_2}$ ($r$ component)", 'tau2_nu_k': r"$\nu_{\tau_2}$ ($k$ component)",
    # cartesian neutrino momenta
    'taup_nu_px': r"$\bar{\nu_\tau}$ ($p_x$ component)", 'taup_nu_py': r"$\bar{\nu_\tau}$ ($p_y$ component)", 'taup_nu_pz': r"$\bar{\nu_\tau}$ ($p_z$ component)",
    'taun_nu_px': r"$\nu_\tau$ ($p_x$ component)", 'taun_nu_py': r"$\nu_\tau$ ($p_y$ component)", 'taun_nu_pz': r"$\nu_\tau$ ($p_z$ component)",
    # neutrino energy
    'taup_nu_E': r"$\bar{\nu_\tau}$ (energy) [GeV]", 'taun_nu_E': r"$\nu_\tau$ (energy) [GeV]",
}

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
    lr : float
        Adam learning rate. Only used by gradient methods.
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

    elif method == 'gradient':
        # optimisation of the MAP estimate in z-space
        latent_dim = model.flow._distribution._shape[0]

        pbar = tqdm(range(0, B, chunk_size), desc="Processing chunks (gradient)")
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
                
                _, logabsdet = model.flow._transform.inverse(z, context=cond_embed_chunk)
                
                log_pz = -0.5 * (z ** 2).sum(dim=-1)
                log_p = log_pz - logabsdet 
                #if step == 0: initial_log_p = log_p.mean().item()
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

            # final evaluation at the post-update z, so x_map and log_p are in sync
            # (log_p inside the loop is computed *before* that step's optimizer.step(),
            # so using it directly here would be one step stale relative to x_map)
            with torch.no_grad():
                x_map, logabsdet = model.flow._transform.inverse(z, context=cond_embed_chunk)
                log_p = -0.5 * (z ** 2).sum(dim=-1) - logabsdet

            if return_log_prob:
                all_best_log_probs.append(log_p.cpu())

            with torch.no_grad():
                x_map, _ = model.flow._transform.inverse(z.detach(), context=cond_embed_chunk)
            #print(f"x_map[0]={x_map[0].cpu().numpy()}")
            all_best_samples.append(x_map.cpu())
            _update_postfix(pbar, time.time() - t0, C * n_steps, C, log_p.detach().sum().item())

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
        raise ValueError(f"Unknown method '{method}'. Choose 'stochastic', 'latent_zero', 'gradient', 'gradient_forward', or 'minuit'.")

    samples_norm_alt = torch.cat(all_best_samples, dim=0)

    if test_dataset is not None:
        samples_alt = test_dataset.destandardize_outputs(samples_norm_alt).cpu().numpy()
    else:
        samples_alt = None

    if return_log_prob:
        return samples_norm_alt, samples_alt, torch.cat(all_best_log_probs, dim=0).numpy()
    return samples_norm_alt, samples_alt


def compute_spin_vars(df, tau_pred_prefix='true_', tau_vis_prefix=''):

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
        plt.figure(figsize=(8, 6))
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
        # plt.title(f"Sampled p({v} | context of event {event_number})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"event{event_number}_{v}.pdf"))
        plt.close()

def _plot_pdf(v_pred, true_val, map_val, label, outdir, event_number, bins, clip, xlabel=None, map_val2=None, transformer_val=None):
    if xlabel is None:
        xlabel = pretty_labels.get(label, label)
    bins = np.linspace(np.percentile(v_pred, 0.1), np.percentile(v_pred, 99.9), bins) if clip else bins
    plt.figure(figsize=(6, 4))
    plt.axvline(true_val, color=truth_color, linestyle='solid', linewidth=2, label='True value')
    weight = np.ones_like(v_pred) / len(v_pred)
    counts, _, _ = plt.hist(v_pred, bins=bins, weights=weight, histtype='step', linewidth=2, color=sampled_color, label='TauPolaris (sampled)')
    if map_val is not None:
        plt.axvline(map_val, color=pred_color, linestyle='solid', linewidth=2, label='TauPolaris (MAP)')
    # if map_val2 is not None:
    #     plt.axvline(map_val2, color=map2_color, linestyle=':', linewidth=2, label='TauPolaris (stochastic mode)')
    if transformer_val is not None:
        plt.axvline(transformer_val, color=transformer_color, linestyle='solid', linewidth=2, label='Transformer')
    ax = plt.gca()
    _, ymax = ax.get_ylim()
    ax.set_ylim(top=max(ymax, counts.max() * 1.15))

    # LUCAS uncomment
    ax.xaxis.set_major_locator(MultipleLocator(pdf_x_tick_step))
    # ax.yaxis.set_major_locator(MultipleLocator(pdf_y_tick_step))
    if "nu_n" in label:
        plt.xlim(-2.2, 2.2)
    elif "nu_r" in label:
        plt.xlim(-2.6, 2.6)


    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel("Normalized counts")
    # plt.title(f"Sampled p({label} | context), event {event_number}")
    # plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"event{event_number}_{label}.pdf"), bbox_inches='tight')
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
    map_value2=None,
    transformer_value=None,
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
        MAP estimate (gradient method) in the same destandardized coordinate space
        as the samples. If provided, overlaid as an orange dashed line on each plot.
    map_value2 : np.ndarray of shape [n_output_features], optional
        Alternative MAP estimate (stochastic/maximum-sampling method), in the same
        destandardized coordinate space. If provided, overlaid as a purple dotted
        line on each plot.
    transformer_value : np.ndarray of shape [6], optional
        Transformer prediction for [taup_nu_px, taup_nu_py, taup_nu_pz, taun_nu_px,
        taun_nu_py, taun_nu_pz], in physical Cartesian neutrino-momentum space.
        If provided, overlaid as a green dash-dot line on the regressed-neutrino
        (Cartesian) plots directly, and also on the train_coords plots after
        converting into the model's native onorm (n/r/k) basis (requires df and
        coordinates=='onorm' — otherwise it's only shown on the Cartesian plots).
    df : pd.DataFrame, optional
        Test dataframe with visible tau decay products. If provided, also produces
        Cartesian (px,py,pz) and energy plots after converting from native coordinates.
    """
    train_coords_dir = os.path.join(outdir, "train_coords")
    regressed_neutrinos_dir = os.path.join(outdir, "regressed_neutrinos")
    os.makedirs(train_coords_dir, exist_ok=True)
    os.makedirs(regressed_neutrinos_dir, exist_ok=True)
    model.eval()

    X = dataset.X[event_number].unsqueeze(0).to(device)
    y = dataset.y[event_number].unsqueeze(0)

    with torch.no_grad():
        predictions_norm = model.sample(num_samples=num_samples, context=X).squeeze()

    predictions = dataset.destandardize_outputs(predictions_norm).cpu().numpy()
    true_values = dataset.destandardize_outputs(y).cpu().numpy()[0]

    if clip==True:
        print("WARNING: This script clips 0.1% outliers on variables by default")

    # Recover the visible tau decay products needed to build the onorm (n/r/k) basis
    # and to convert to Cartesian — shared by the train_coords transformer overlay
    # below and the regressed-neutrino Cartesian plots further down.
    tau1_prefix = tau2_prefix = reco_taup_charged = reco_taun_charged = None
    reco_taup_pizero = reco_taun_pizero = None
    if df is not None and coordinates != 'standard':
        tau1_prefix = 'taup' if 'taup_nu_px' in df.columns else 'tau1'
        tau2_prefix = 'taun' if tau1_prefix == 'taup' else 'tau2'

        reco_taup_charged = df[[f'reco_{tau1_prefix}_charged_e', f'reco_{tau1_prefix}_charged_px', f'reco_{tau1_prefix}_charged_py', f'reco_{tau1_prefix}_charged_pz']].values[event_number:event_number+1]
        reco_taun_charged = df[[f'reco_{tau2_prefix}_charged_e', f'reco_{tau2_prefix}_charged_px', f'reco_{tau2_prefix}_charged_py', f'reco_{tau2_prefix}_charged_pz']].values[event_number:event_number+1]
        reco_taup_pizero = df[[f'reco_{tau1_prefix}_pizero1_e', f'reco_{tau1_prefix}_pizero1_px', f'reco_{tau1_prefix}_pizero1_py', f'reco_{tau1_prefix}_pizero1_pz']].values[event_number:event_number+1]
        reco_taun_pizero = df[[f'reco_{tau2_prefix}_pizero1_e', f'reco_{tau2_prefix}_pizero1_px', f'reco_{tau2_prefix}_pizero1_py', f'reco_{tau2_prefix}_pizero1_pz']].values[event_number:event_number+1]

    transformer_value_onorm = None
    if transformer_value is not None and reco_taup_charged is not None and coordinates == 'onorm':
        onorm_df = pd.DataFrame({
            f'{tau1_prefix}_nu_px': [transformer_value[0]],
            f'{tau1_prefix}_nu_py': [transformer_value[1]],
            f'{tau1_prefix}_nu_pz': [transformer_value[2]],
            f'{tau2_prefix}_nu_px': [transformer_value[3]],
            f'{tau2_prefix}_nu_py': [transformer_value[4]],
            f'{tau2_prefix}_nu_pz': [transformer_value[5]],
            f'reco_{tau1_prefix}_charged_px': reco_taup_charged[:, 1],
            f'reco_{tau1_prefix}_charged_py': reco_taup_charged[:, 2],
            f'reco_{tau1_prefix}_charged_pz': reco_taup_charged[:, 3],
            f'reco_{tau1_prefix}_pizero1_px': reco_taup_pizero[:, 1],
            f'reco_{tau1_prefix}_pizero1_py': reco_taup_pizero[:, 2],
            f'reco_{tau1_prefix}_pizero1_pz': reco_taup_pizero[:, 3],
            f'reco_{tau2_prefix}_charged_px': reco_taun_charged[:, 1],
            f'reco_{tau2_prefix}_charged_py': reco_taun_charged[:, 2],
            f'reco_{tau2_prefix}_charged_pz': reco_taun_charged[:, 3],
            f'reco_{tau2_prefix}_pizero1_px': reco_taun_pizero[:, 1],
            f'reco_{tau2_prefix}_pizero1_py': reco_taun_pizero[:, 2],
            f'reco_{tau2_prefix}_pizero1_pz': reco_taun_pizero[:, 3],
        })
        onorm_df = ConvertToOrthonormalNRK(
            onorm_df, prefix_to_convert=f'{tau1_prefix}_nu_',
            charged_prefix=f'reco_{tau1_prefix}_charged_',
            pi0_prefix=f'reco_{tau1_prefix}_pizero1_',
            drop_xyz=False,
        )
        onorm_df = ConvertToOrthonormalNRK(
            onorm_df, prefix_to_convert=f'{tau2_prefix}_nu_',
            charged_prefix=f'reco_{tau2_prefix}_charged_',
            pi0_prefix=f'reco_{tau2_prefix}_pizero1_',
            drop_xyz=False,
        )
        onorm_row = onorm_df.iloc[0]
        transformer_value_onorm = np.array(
            [onorm_row[feat] if feat in onorm_row else None for feat in output_features]
        )

    for i, v in enumerate(output_features):
        _plot_pdf(predictions[:, i], true_values[i],
                  map_value[i] if map_value is not None else None,
                  v, train_coords_dir, event_number, bins, clip,
                  map_val2=map_value2[i] if map_value2 is not None else None,
                  transformer_val=transformer_value_onorm[i] if transformer_value_onorm is not None else None)

    # plot cartesian components and energy
    if df is not None and coordinates!='standard':
        print('recomputing 4 components in cartesian')

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
        map_cart2 = convert_coordinates_pred(map_value2[np.newaxis, :], **single_kwargs)[0] if map_value2 is not None else None

        cart_features = ['taup_nu_px', 'taup_nu_py', 'taup_nu_pz','taun_nu_px', 'taun_nu_py', 'taun_nu_pz']
        nu_p_slices = {'taup_nu': (0, 3), 'taun_nu': (3, 6)}

        # plot px, py, pz
        for i, v in enumerate(cart_features):
            _plot_pdf(pred_cart[:, i], true_cart[i],
                      map_cart[i] if map_cart is not None else None,
                      v, regressed_neutrinos_dir, event_number, bins, clip,
                      map_val2=map_cart2[i] if map_cart2 is not None else None,
                      transformer_val=transformer_value[i] if transformer_value is not None else None)

        # plot energy
        for nu_name, (s, e) in nu_p_slices.items():
            E_pred = np.sqrt(np.sum(pred_cart[:, s:e]**2, axis=1))
            E_true = np.sqrt(np.sum(true_cart[s:e]**2))
            E_map = np.sqrt(np.sum(map_cart[s:e]**2)) if map_cart is not None else None
            E_map2 = np.sqrt(np.sum(map_cart2[s:e]**2)) if map_cart2 is not None else None
            E_transformer = np.sqrt(np.sum(transformer_value[s:e]**2)) if transformer_value is not None else None
            _plot_pdf(E_pred, E_true, E_map, f'{nu_name}_E', regressed_neutrinos_dir, event_number, bins, clip,
                      map_val2=E_map2, transformer_val=E_transformer)





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
