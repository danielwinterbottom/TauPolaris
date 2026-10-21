import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)


plt.rcParams.update({"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18})
legendfont = 18


spin_vars = {
    "cosn_plus":  r"$\cos\theta_n^+$",
    "cosr_plus":  r"$\cos\theta_r^+$",
    "cosk_plus":  r"$\cos\theta_k^+$",
    "cosn_minus": r"$\cos\theta_n^-$",
    "cosr_minus": r"$\cos\theta_r^-$",
    "cosk_minus": r"$\cos\theta_k^-$",
    "cosTheta":   r"$\cos\Theta$",
}


pred_taunu_vars = {
    "nubar_E":        r"$\nu_{\tau_1}$ Energy (GeV)",
    "nubar_px":       r"$\nu_{\tau_1}$ $p_x$ (GeV)",
    "nubar_py":       r"$\nu_{\tau_1}$ $p_y$ (GeV)",
    "nubar_pz":       r"$\nu_{\tau_1}$ $p_z$ (GeV)",
    "nu_E":           r"$\nu_{\tau_2}$ Energy (GeV)",
    "nu_px":          r"$\nu_{\tau_2}$ $p_x$ (GeV)",
    "nu_py":          r"$\nu_{\tau_2}$ $p_y$ (GeV)",
    "nu_pz":          r"$\nu_{\tau_2}$ $p_z$ (GeV)",
    "tau_plus_E":     r"$\tau_1$ Energy (GeV)",
    "tau_plus_px":    r"$\tau_1$ $p_x$ (GeV)",
    "tau_plus_py":    r"$\tau_1$ $p_y$ (GeV)",
    "tau_plus_pz":    r"$\tau_1$ $p_z$ (GeV)",
    "tau_minus_E":    r"$\tau_2$ Energy (GeV)",
    "tau_minus_px":   r"$\tau_2$ $p_x$ (GeV)",
    "tau_minus_py":   r"$\tau_2$ $p_y$ (GeV)",
    "tau_minus_pz":   r"$\tau_2$ $p_z$ (GeV)",
    "tau_plus_mass":  r"$m_{\tau_1}$ (GeV)",
    "tau_minus_mass": r"$m_{\tau_2}$ (GeV)",
    "boson_mass":     r"$m_{\tau_1\tau_2}$ (GeV)",
}


spin_density_vars = {
    "C11": r"$\cos\theta_n^+ \cos\theta_n^-$",
    "C22": r"$\cos\theta_r^+ \cos\theta_r^-$",
    "C33": r"$\cos\theta_k^+ \cos\theta_k^-$",
    "C12": r"$\cos\theta_n^+ \cos\theta_r^-$",
    "C13": r"$\cos\theta_n^+ \cos\theta_k^-$",
    "C23": r"$\cos\theta_r^+ \cos\theta_k^-$",
    "C21": r"$\cos\theta_r^+ \cos\theta_n^-$",
    "C31": r"$\cos\theta_k^+ \cos\theta_n^-$",
    "C32": r"$\cos\theta_k^+ \cos\theta_r^-$",
}


spin_density_products = {
    "C11": ("cosn_plus", "cosn_minus"),
    "C22": ("cosr_plus", "cosr_minus"),
    "C33": ("cosk_plus", "cosk_minus"),
    "C12": ("cosn_plus", "cosr_minus"),
    "C13": ("cosn_plus", "cosk_minus"),
    "C23": ("cosr_plus", "cosk_minus"),
    "C21": ("cosr_plus", "cosn_minus"),
    "C31": ("cosk_plus", "cosn_minus"),
    "C32": ("cosk_plus", "cosr_minus"),
}


derived_vars = {
    "tau_plus_pt":  r"$\tau_1$ $p_T$ (GeV)",
    "tau_minus_pt": r"$\tau_2$ $p_T$ (GeV)",
    "ditau_pt":     r"$p_T^{\tau_1\tau_2}$",
    "ditau_px":     r"$p_x^{\tau_1\tau_2}$",
    "ditau_py":     r"$p_y^{\tau_1\tau_2}$",
    "ditau_pz":     r"$p_z^{\tau_1\tau_2}$",
    "ditau_E":      r"$E^{\tau_1\tau_2}$",
    "nu_pt":        r"$\nu_{\tau_2}$ $p_T$ (GeV)",
    "nu_eta":       r"$\eta^{\nu_{\tau_2}}$",
    "nu_phi":       r"$\phi^{\nu_{\tau_2}}$",
    "nubar_eta":    r"$\eta^{\nu_{\tau_1}}$",
    "nubar_phi":    r"$\phi^{\nu_{\tau_1}}$",
}


angle_vars = {
    "opening_angle":       r"$\Delta\theta(\tau_1,\tau_2)$ [rad]",
    "dR":                  r"$\Delta R(\tau_1,\tau_2)$",
    "nu_nubar_deta":       r"$\Delta\eta(\nu_{\tau_2},\nu_{\tau_1})$",
    "nu_nubar_dphi":       r"$\Delta\phi(\nu_{\tau_2},\nu_{\tau_1})$",
    "nu_nubar_dR":         r"$\Delta R(\nu_{\tau_2},\nu_{\tau_1})$",
    "nu_taum_dphi":        r"$\Delta\phi(\nu_{\tau_2},\tau_2)$",
    "nu_taum_dR":          r"$\Delta R(\nu_{\tau_2},\tau_2)$",
    "nubar_taup_dphi":     r"$\Delta\phi(\nu_{\tau_1},\tau_1)$",
    "nubar_taup_dR":       r"$\Delta R(\nu_{\tau_1},\tau_1)$",
}



m_tau = 1.77686
m_boson = 125.0

truth_color = "#EE8A3D"
truth_alpha = 0.6
sampled_color = '#0072B2'
pred_color = "#e42536"
transformer_color = '#2CA02C'
truth_line_color = truth_color



def add_spin_density_cols(df):
    """Add cos-product spin density columns for every true_/map_pred_/pred_/transformer_pred_ prefix."""
    df = df.copy()
    for name, (var1, var2) in spin_density_products.items():
        for prefix in ("true_", "map_pred_", "pred_", "transformer_pred_"):
            col1, col2 = f"{prefix}{var1}", f"{prefix}{var2}"
            if col1 in df.columns and col2 in df.columns:
                df[f"{prefix}{name}"] = df[col1] * df[col2]
    return df


def _pt_eta_phi(px, py, pz):
    """Transverse momentum, pseudorapidity, and azimuthal angle."""
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px)
    return pt, eta, phi


def add_derived_kinematic_cols(df):
    """Add tau pT, ditau (four-)momentum, and neutrino pT/eta/phi columns for
    every true_/map_pred_/pred_/transformer_pred_ prefix."""
    df = df.copy()
    for prefix in ("true_", "map_pred_", "pred_", "transformer_pred_"):
        tp_px, tp_py, tp_pz, tp_E = (f"{prefix}tau_plus_{c}" for c in ("px", "py", "pz", "E"))
        tm_px, tm_py, tm_pz, tm_E = (f"{prefix}tau_minus_{c}" for c in ("px", "py", "pz", "E"))

        if tp_px in df.columns and tp_py in df.columns:
            df[f"{prefix}tau_plus_pt"] = np.hypot(df[tp_px], df[tp_py])
        if tm_px in df.columns and tm_py in df.columns:
            df[f"{prefix}tau_minus_pt"] = np.hypot(df[tm_px], df[tm_py])

        if tp_px in df.columns and tm_px in df.columns:
            df[f"{prefix}ditau_px"] = df[tp_px] + df[tm_px]
        if tp_py in df.columns and tm_py in df.columns:
            df[f"{prefix}ditau_py"] = df[tp_py] + df[tm_py]
        if tp_pz in df.columns and tm_pz in df.columns:
            df[f"{prefix}ditau_pz"] = df[tp_pz] + df[tm_pz]
        if tp_E in df.columns and tm_E in df.columns:
            df[f"{prefix}ditau_E"] = df[tp_E] + df[tm_E]

        ditau_px, ditau_py = f"{prefix}ditau_px", f"{prefix}ditau_py"
        if ditau_px in df.columns and ditau_py in df.columns:
            df[f"{prefix}ditau_pt"] = np.hypot(df[ditau_px], df[ditau_py])

        for nu_name in ("nu", "nubar"):
            nu_px, nu_py, nu_pz = (f"{prefix}{nu_name}_{c}" for c in ("px", "py", "pz"))
            if nu_px in df.columns and nu_py in df.columns and nu_pz in df.columns:
                pt, eta, phi = _pt_eta_phi(df[nu_px], df[nu_py], df[nu_pz])
                df[f"{prefix}{nu_name}_pt"] = pt
                df[f"{prefix}{nu_name}_eta"] = eta
                df[f"{prefix}{nu_name}_phi"] = phi
    return df


def _opening_angle(px1, py1, pz1, px2, py2, pz2):
    """3D opening angle (radians) between two momentum vectors."""
    dot = px1 * px2 + py1 * py2 + pz1 * pz2
    mag1 = np.sqrt(px1**2 + py1**2 + pz1**2)
    mag2 = np.sqrt(px2**2 + py2**2 + pz2**2)
    cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    return np.arccos(cos_angle)


def _delta_eta_phi(px1, py1, pz1, px2, py2, pz2):
    """(deta, dphi) between two momentum vectors, dphi wrapped to [-pi, pi]."""
    _, eta1, phi1 = _pt_eta_phi(px1, py1, pz1)
    _, eta2, phi2 = _pt_eta_phi(px2, py2, pz2)
    dphi = np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi
    return eta1 - eta2, dphi


def _delta_r(px1, py1, pz1, px2, py2, pz2):
    """Standard collider dR = sqrt(deta^2 + dphi^2) between two momentum vectors."""
    deta, dphi = _delta_eta_phi(px1, py1, pz1, px2, py2, pz2)
    return np.hypot(deta, dphi)


def add_angle_cols(df):
    """Add the opening angle and dR between the two taus, deta/dphi/dR
    between the two neutrinos, and dphi/dR between each neutrino and its
    parent tau (nu <-> tau_minus, nubar <-> tau_plus), for every
    true_/map_pred_/pred_/transformer_pred_ prefix."""
    df = df.copy()
    for prefix in ("true_", "map_pred_", "pred_", "transformer_pred_"):
        tau_needed = [f"{prefix}tau_plus_px", f"{prefix}tau_plus_py", f"{prefix}tau_plus_pz",
                      f"{prefix}tau_minus_px", f"{prefix}tau_minus_py", f"{prefix}tau_minus_pz"]
        if all(c in df.columns for c in tau_needed):
            tau_args = [df[c] for c in tau_needed]
            df[f"{prefix}opening_angle"] = _opening_angle(*tau_args)
            df[f"{prefix}dR"] = _delta_r(*tau_args)

        nu_needed = [f"{prefix}nu_px", f"{prefix}nu_py", f"{prefix}nu_pz",
                     f"{prefix}nubar_px", f"{prefix}nubar_py", f"{prefix}nubar_pz"]
        if all(c in df.columns for c in nu_needed):
            nu_args = [df[c] for c in nu_needed]
            deta, dphi = _delta_eta_phi(*nu_args)
            df[f"{prefix}nu_nubar_deta"] = deta
            df[f"{prefix}nu_nubar_dphi"] = dphi
            df[f"{prefix}nu_nubar_dR"] = np.hypot(deta, dphi)

        nu_taum_needed = [f"{prefix}nu_px", f"{prefix}nu_py", f"{prefix}nu_pz",
                          f"{prefix}tau_minus_px", f"{prefix}tau_minus_py", f"{prefix}tau_minus_pz"]
        if all(c in df.columns for c in nu_taum_needed):
            _, dphi = _delta_eta_phi(*(df[c] for c in nu_taum_needed))
            df[f"{prefix}nu_taum_dphi"] = dphi
            df[f"{prefix}nu_taum_dR"] = _delta_r(*(df[c] for c in nu_taum_needed))

        nubar_taup_needed = [f"{prefix}nubar_px", f"{prefix}nubar_py", f"{prefix}nubar_pz",
                             f"{prefix}tau_plus_px", f"{prefix}tau_plus_py", f"{prefix}tau_plus_pz"]
        if all(c in df.columns for c in nubar_taup_needed):
            _, dphi = _delta_eta_phi(*(df[c] for c in nubar_taup_needed))
            df[f"{prefix}nubar_taup_dphi"] = dphi
            df[f"{prefix}nubar_taup_dR"] = _delta_r(*(df[c] for c in nubar_taup_needed))
    return df


def plot_angle_vars(df, output_dir, useMAP=True, tag='', weight_col=None):
    df = add_angle_cols(df)
    suffix = f'{tag}_' if tag else ''
    plot_two_panel_vars(df, angle_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=False, weight_col=weight_col)


def load_transformer_predictions(df, transformer_path):
    """Attach predictions from a second (transformer) parquet file under a
    transformer_pred_ prefix. That file is expected to use the same pred_*
    column naming as the sampled predictions, and to be row-aligned with df."""
    transformer_df = pd.read_parquet(transformer_path)
    if len(transformer_df) != len(df):
        raise ValueError(
            f"Transformer predictions file has {len(transformer_df)} rows, "
            f"but the main input has {len(df)} rows — they must be row-aligned."
        )
    df = df.copy()
    all_vars = list(pred_taunu_vars) + list(spin_vars)
    for var in all_vars:
        col = f"pred_{var}"
        if col in transformer_df.columns:
            df[f"transformer_pred_{var}"] = transformer_df[col].values
    return df


def _resolution_diff(true_vals, other_vals):
    """Per-event (other - true) resolution, aligned onto true_vals' index
    (predictions are never NaN, so other_vals always covers that index)."""
    return other_vals.loc[true_vals.index] - true_vals


def _weighted_mean(values, weights=None):
    if weights is None:
        return values.mean()
    return np.average(values, weights=weights)


def _weighted_quantile(values, weights, q):
    # for IQR
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sorter = np.argsort(values)
    values, weights = values[sorter], weights[sorter]
    cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(q, cdf, values)


def _quantile(series, q, weights=None):
    # for IQR
    if weights is None:
        return series.quantile(q)
    return _weighted_quantile(series, weights, q)


def _iqr(series, weights=None):
    # IQR instead of standard deviation
    return _quantile(series, 0.75, weights) - _quantile(series, 0.25, weights)


def _count_norm_weights(n_events, weights=None):
    # weight by 1/N instead of using density so not divided by bin width
    if weights is None:
        return np.full(n_events, 1.0 / n_events)
    return np.asarray(weights) / n_events


def _draw_resolution(ax_res, true_vals, pred_vals, sampled_vals, show_sampled,
                      transformer_vals, show_transformer, label, weights=None):
    # per event resolution (pred-true)
    w = weights.loc[true_vals.index] if weights is not None else None
    hist_w = _count_norm_weights(len(true_vals), w)
    diff_pred = _resolution_diff(true_vals, pred_vals)
    diff_parts = [diff_pred]
    if show_sampled:
        diff_sampled = _resolution_diff(true_vals, sampled_vals)
        diff_parts.append(diff_sampled)
    if show_transformer:
        diff_transformer = _resolution_diff(true_vals, transformer_vals)
        diff_parts.append(diff_transformer)
    combined_diff = pd.concat(diff_parts)
    if w is not None:
        combined_weight = pd.concat([w] * len(diff_parts))
        bin_range = (_weighted_quantile(combined_diff, combined_weight, 0.01),
                     _weighted_quantile(combined_diff, combined_weight, 0.99))
    else:
        bin_range = (combined_diff.quantile(0.01), combined_diff.quantile(0.99))
    diff_bins = np.histogram_bin_edges(combined_diff, bins=50, range=bin_range)

    pred_mu, pred_iqr = _weighted_mean(diff_pred, w), _iqr(diff_pred, w)
    if show_transformer:
        transformer_mu, transformer_iqr = _weighted_mean(diff_transformer, w), _iqr(diff_transformer, w)
        ax_res.hist(diff_transformer, bins=diff_bins, histtype="step", weights=hist_w, linewidth=2, linestyle='solid', color=transformer_color,
                    label=fr"Transformer: $\mu$={transformer_mu:.2f}, IQR={transformer_iqr:.2f}")
    if show_sampled:
        sampled_mu, sampled_iqr = _weighted_mean(diff_sampled, w), _iqr(diff_sampled, w)
        ax_res.hist(diff_sampled, bins=diff_bins, histtype="step", weights=hist_w, linewidth=2, color=sampled_color,
                    label=fr"TauPolaris (sampled): $\mu$={sampled_mu:.2f}, IQR={sampled_iqr:.2f}")

    ax_res.hist(diff_pred, bins=diff_bins, histtype="step", weights=hist_w, linewidth=2, color=pred_color,
                label=fr"TauPolaris (MAP): $\mu$={pred_mu:.2f}, IQR={pred_iqr:.2f}")
    ax_res.set_xlabel(f"{label} Resolution (Pred. - Truth)")
    ax_res.set_ylabel("Normalized counts")
    ax_res.set_xlim(diff_bins[0], diff_bins[-1])
    ymin, ymax = ax_res.get_ylim()
    ymax_mult = 1.35 + 0.15 * (int(show_sampled) + int(show_transformer))
    ax_res.set_ylim(ymin, ymax * ymax_mult)
    ax_res.legend(loc="upper left", fontsize=legendfont) #, frameon=True)


def _draw_distribution_ratio(ax, ax_ratio, true_vals, pred_vals, sampled_vals, show_sampled,
                              transformer_vals, show_transformer, bins, ymax_mult,
                              weight_true=None, weight_sampled=None, weight_transformer=None):
    """Draw the gray-filled truth / blue predicted / orange sampled / green transformer
    overlay and its pred-over-truth ratio panel, shared by all distribution plots."""
    n = len(true_vals)
    hist_weight_true = _count_norm_weights(n, weight_true)
    hist_weight_sampled = _count_norm_weights(n, weight_sampled) if show_sampled else None
    hist_weight_transformer = _count_norm_weights(n, weight_transformer) if show_transformer else None

    true_counts, _ = np.histogram(true_vals, bins=bins)
    true_hist, _ = np.histogram(true_vals, bins=bins, weights=hist_weight_true)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Relative sqrt count uncertainty on truth
        true_rel_err = np.where(true_counts > 0, 1.0 / np.sqrt(true_counts), np.nan)
    if show_sampled:
        sampled_hist, _ = np.histogram(sampled_vals, bins=bins, weights=hist_weight_sampled)
        with np.errstate(divide="ignore", invalid="ignore"):
            sampled_ratio = np.where(true_hist > 0, sampled_hist / true_hist, np.nan)
    if show_transformer:
        transformer_hist, _ = np.histogram(transformer_vals, bins=bins, weights=hist_weight_transformer)
        with np.errstate(divide="ignore", invalid="ignore"):
            transformer_ratio = np.where(true_hist > 0, transformer_hist / true_hist, np.nan)
    ax.hist(true_vals, bins=bins, histtype="stepfilled", weights=hist_weight_true,
            linewidth=0, color=truth_color, alpha=truth_alpha,
            label="Generator Truth")
    ax.hist(true_vals, bins=bins, histtype="step", weights=hist_weight_true,
            linewidth=0.5, color="black")
    if show_transformer:
        ax.hist(transformer_vals, bins=bins, histtype="step", weights=hist_weight_transformer,
                linewidth=2, linestyle="solid", color=transformer_color,
                label="Transformer")
    if show_sampled:
        ax.hist(sampled_vals, bins=bins, histtype="step", weights=hist_weight_sampled,
                linewidth=2, linestyle="solid", color=sampled_color,
                label="TauPolaris (sampled)")
    ax.set_ylabel("Normalized counts")
    ax.set_xlim(bins[0], bins[-1])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * ymax_mult)
    ax.legend(loc="upper right", fontsize=legendfont) #, frameon=True)
    ax.tick_params(labelbottom=False)

    band_lower = np.where(true_counts > 0, 1.0 - true_rel_err, np.nan)
    band_upper = np.where(true_counts > 0, 1.0 + true_rel_err, np.nan)
    ax_ratio.fill_between(bins, np.append(band_lower, band_lower[-1]), np.append(band_upper, band_upper[-1]),
                           step="post", color="lightgrey", alpha=0.6, linewidth=0, zorder=0)
    ax_ratio.axhline(1.0, color="gray", linestyle="dashed", linewidth=1)
    if show_transformer:
        ax_ratio.stairs(transformer_ratio, bins, linewidth=2, color=transformer_color, linestyle='solid')
    if show_sampled:
        ax_ratio.stairs(sampled_ratio, bins, linewidth=2, color=sampled_color)
    ax_ratio.set_ylabel("Pred. / Truth")
    ax_ratio.set_xlim(bins[0], bins[-1])
    ax_ratio.set_ylim(0.75, 1.25)


def _var_bins(var, true_vals, pred_vals, sampled_vals, show_sampled,
               transformer_vals, show_transformer, bounded, weights=None):
    # set bin edges
    if bounded:
        return np.linspace(-1.0, 1.0, 51)
    combined_parts = [true_vals, pred_vals]
    if show_sampled:
        combined_parts.append(sampled_vals)
    if show_transformer:
        combined_parts.append(transformer_vals)
    combined = pd.concat(combined_parts)
    combined_weights = weights.loc[combined.index] if weights is not None else None
    if var in ('nu_taum_dR', 'nubar_taup_dR'):
        bin_range = (0.0, 0.25)
    elif var in ('nu_taum_dphi', 'nubar_taup_dphi'):
        bin_range = (-0.25, 0.25)
    elif var.endswith('_E') or var.endswith('_mass') or var.endswith('_pt'):
        bin_range = (combined.min(), _quantile(combined, 0.98, combined_weights))
    else:
        bin_range = (_quantile(combined, 0.01, combined_weights), _quantile(combined, 0.99, combined_weights))
    return np.histogram_bin_edges(combined, bins=50, range=bin_range)


def plot_two_panel_vars(df, var_dict, output_dir, subdir, useMAP=True, bounded=False, weight_col=None):
    # plot distribution and ratio
    dist_dir = os.path.join(output_dir, subdir, "distributions")
    res_dir = os.path.join(output_dir, subdir, "resolutions")
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    weights = df[weight_col] if weight_col else None

    for var, label in var_dict.items():
        true_col = f"true_{var}"
        pred_col = f"map_pred_{var}" if useMAP else f"pred_{var}"
        sampled_col = f"pred_{var}"
        transformer_col = f"transformer_pred_{var}"

        if pred_col not in df.columns or true_col not in df.columns:
            print(f"Warning: Missing columns for {var}, skipping")
            continue

        # Only overlay the raw sampled prediction when the main line is the MAP estimate;
        # when --sampled is used the main line already is pred_*, so it would be a duplicate.
        show_sampled = useMAP and sampled_col != pred_col and sampled_col in df.columns
        show_transformer = transformer_col in df.columns

        # if for whatever reason the truh is missing skip that event
        true_vals = df[true_col].dropna()
        pred_vals = df[pred_col].loc[true_vals.index]
        sampled_vals = df[sampled_col].loc[true_vals.index] if show_sampled else None
        transformer_vals = df[transformer_col].loc[true_vals.index] if show_transformer else None
        # weights
        weight_true = weights.loc[true_vals.index] if weights is not None else None
        weight_sampled = weight_true if show_sampled else None
        weight_transformer = weight_true if show_transformer else None
        # max y range dependant on number of histos
        ymax_mult = 1.35 + 0.10 * (int(show_sampled) + int(show_transformer))
        bins = _var_bins(var, true_vals, pred_vals, sampled_vals, show_sampled,
                          transformer_vals, show_transformer, bounded, weights)

        # Make agreement figure
        fig_dist = plt.figure(figsize=(8, 6))
        gs = fig_dist.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
        ax = fig_dist.add_subplot(gs[0])
        ax_ratio = fig_dist.add_subplot(gs[1], sharex=ax)
        _draw_distribution_ratio(ax, ax_ratio, true_vals, pred_vals, sampled_vals, show_sampled,
                                  transformer_vals, show_transformer, bins, ymax_mult,
                                  weight_true, weight_sampled, weight_transformer)
        ax_ratio.set_xlabel(label)
        fig_dist.savefig(os.path.join(dist_dir, f"{var}.pdf"), bbox_inches="tight")
        plt.close(fig_dist)

        # make resolution figure
        fig_res, ax_res = plt.subplots(figsize=(8, 6))
        _draw_resolution(ax_res, true_vals, pred_vals, sampled_vals, show_sampled,
                          transformer_vals, show_transformer, label, weights=weights)
        fig_res.savefig(os.path.join(res_dir, f"{var}_res.pdf"), bbox_inches="tight")
        plt.close(fig_res)

        print(f">> Saved paper plots {var}.pdf (distributions/ and resolutions/)")

def plot_spin_vars(df, output_dir, useMAP=True, tag='', weight_col=None):
    suffix = f'{tag}_' if tag else ''
    plot_two_panel_vars(df, spin_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=True, weight_col=weight_col)


def plot_spin_density_vars(df, output_dir, useMAP=True, tag='', weight_col=None):
    df = add_spin_density_cols(df)
    suffix = f'{tag}_' if tag else ''
    plot_two_panel_vars(df, spin_density_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=True, weight_col=weight_col)


def plot_derived_vars(df, output_dir, useMAP=True, tag='', weight_col=None):
    df = add_derived_kinematic_cols(df)
    suffix = f'{tag}_' if tag else ''
    plot_two_panel_vars(df, derived_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=False, weight_col=weight_col)


mass_vars = {"tau_plus_mass", "tau_minus_mass", "boson_mass"}


def plot_mass_vars(df, output_dir, useMAP=True, tag='', weight_col=None):
    # Single-panel plots for the tau/boson masses since only one truth value
    suffix = f'{tag}_' if tag else ''
    subdir = f'{suffix}paper_plots'
    dist_dir = os.path.join(output_dir, subdir, "distributions")
    res_dir = os.path.join(output_dir, subdir, "resolutions")
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    weights = df[weight_col] if weight_col else None

    for var in mass_vars:
        label = pred_taunu_vars[var]
        true_col = f"true_{var}"
        pred_col = f"map_pred_{var}" if useMAP else f"pred_{var}"
        sampled_col = f"pred_{var}"
        transformer_col = f"transformer_pred_{var}"

        if pred_col not in df.columns:
            print(f"Warning: Missing columns for {var}, skipping")
            continue

        show_sampled = useMAP and sampled_col != pred_col and sampled_col in df.columns
        show_transformer = transformer_col in df.columns
        has_true = true_col in df.columns
        pred_vals = df[pred_col]
        sampled_vals = df[sampled_col] if show_sampled else None
        transformer_vals = df[transformer_col] if show_transformer else None
        weight_sampled = weights.loc[sampled_vals.index] if weights is not None and show_sampled else None
        weight_transformer = weights.loc[transformer_vals.index] if weights is not None and show_transformer else None
        ymax_mult = 1.35 + 0.10 * (int(show_sampled) + int(show_transformer))

        is_tau_mass = var in ("tau_plus_mass", "tau_minus_mass")
        truth_val = m_tau if is_tau_mass else m_boson
        bin_range = (0.5, 2.5) if is_tau_mass else (70,150)
        bins = np.linspace(bin_range[0], bin_range[1], 51)

        fig_dist, ax = plt.subplots(figsize=(8, 6))

        ax.axvline(truth_val, color=truth_line_color, linestyle="solid", linewidth=2,
                   label=fr"Generator Truth")
        if show_transformer:
            transformer_mu, transformer_iqr = _weighted_mean(transformer_vals, weight_transformer), _iqr(transformer_vals, weight_transformer)
            ax.hist(transformer_vals, bins=bins, histtype="step", weights=_count_norm_weights(len(transformer_vals), weight_transformer),
                    linewidth=2, linestyle="solid", color=transformer_color,
                    label=fr"Transformer")
        if show_sampled:
            sampled_mu, sampled_iqr = _weighted_mean(sampled_vals, weight_sampled), _iqr(sampled_vals, weight_sampled)
            ax.hist(sampled_vals, bins=bins, histtype="step", weights=_count_norm_weights(len(sampled_vals), weight_sampled),
                    linewidth=2, linestyle="solid", color=sampled_color,
                    label=fr"TauPolaris (sampled)")
        ax.set_xlabel(label)
        ax.set_ylabel("Normalized counts")
        ax.set_xlim(bins[0], bins[-1])
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * ymax_mult)
        ax.legend(loc="upper left", fontsize=legendfont) #, frameon=True)

        fig_dist.savefig(os.path.join(dist_dir, f"{var}.pdf"), bbox_inches="tight")
        plt.close(fig_dist)

        if has_true:
            fig_res, ax_res = plt.subplots(figsize=(8, 6))
            true_vals = df[true_col].dropna()
            _draw_resolution(ax_res, true_vals, pred_vals, sampled_vals, show_sampled,
                              transformer_vals, show_transformer, label, weights=weights)
            fig_res.savefig(os.path.join(res_dir, f"{var}_res.pdf"), bbox_inches="tight")
            plt.close(fig_res)

        print(f">> Saved paper plots {var}.pdf (distributions/{' and resolutions/' if has_true else ''})")


def paper_plot(df, output_dir, useMAP=True, tag='', weight_col=None):
    suffix = f'{tag}_' if tag else ''
    kinematic_vars = {v: l for v, l in pred_taunu_vars.items() if v not in mass_vars}
    plot_mass_vars(df, output_dir, useMAP=useMAP, tag=tag, weight_col=weight_col)
    plot_two_panel_vars(df, kinematic_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=False, weight_col=weight_col)


def main():

    parser = argparse.ArgumentParser(description="Plot kinematics and spin variables")
    parser.add_argument("--input", type=str, help="Path to input parquet file")
    parser.add_argument('--transformer_input', type=str, default=None,
                        help="Path to a second parquet file with transformer predictions "
                             "(same pred_* column naming as the sampled predictions, row-aligned "
                             "with --input); overlaid as a green line if given")
    parser.add_argument('--useMLP', action='store_true')
    parser.add_argument('--sampled', action='store_true')
    parser.add_argument('--replaceFailed', action='store_true', help="If using MAP estimate, replace failed MAP predictions with the sampled prediction")
    parser.add_argument('--tag', type=str, default='', help="Tag appended to output subdirectory names")
    parser.add_argument('--weight', type=str, default=None,
                        help="Name of a column in the input parquet to use as the per-event "
                             "weight for every histogram (distributions, ratios, resolutions, and masses)")
    args = parser.parse_args()
    use_map = not args.useMLP
    if args.sampled:
        use_map = False
    df = pd.read_parquet(args.input)
    output_dir = args.input.replace('.parquet', '')
    print(f">> Loaded {len(df)} events from {args.input}")

    if args.transformer_input:
        df = load_transformer_predictions(df, args.transformer_input)
        print(f">> Loaded transformer predictions from {args.transformer_input}")

    if args.weight and args.weight not in df.columns:
        raise ValueError(f"--weight column '{args.weight}' not found in input parquet")

    # cos vars
    plot_spin_vars(df, output_dir, use_map, tag=args.tag, weight_col=args.weight)
    # kinematics
    paper_plot(df, output_dir, use_map, tag=args.tag, weight_col=args.weight)
    # spin matrix
    plot_spin_density_vars(df, output_dir, use_map, tag=args.tag, weight_col=args.weight)
    # extra derivde things
    plot_derived_vars(df, output_dir, use_map, tag=args.tag, weight_col=args.weight)
    plot_angle_vars(df, output_dir, use_map, tag=args.tag, weight_col=args.weight)


if __name__ == "__main__":
    main()