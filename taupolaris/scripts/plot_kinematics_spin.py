import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import numpy as np
import argparse
import os


plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 16})


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
    "nubar_E":        r"$E^\bar{\nu}$",
    "nubar_px":       r"$p_x^\bar{\nu}$",
    "nubar_py":       r"$p_y^\bar{\nu}$",
    "nubar_pz":       r"$p_z^\bar{\nu}$",
    "nu_E":           r"$E^{\nu}$",
    "nu_px":          r"$p_x^\nu$",
    "nu_py":          r"$p_y^\nu$",
    "nu_pz":          r"$p_z^\nu$",
    "tau_plus_E":     r"$E^{\tau^+}$",
    "tau_plus_px":    r"$p_x^\tau$",
    "tau_plus_py":    r"$p_y^\tau$",
    "tau_plus_pz":    r"$p_z^\tau$",
    "tau_minus_E":    r"$E^{\tau^-}$",
    "tau_minus_px":   r"$p_x^\tau$",
    "tau_minus_py":   r"$p_y^\tau$",
    "tau_minus_pz":   r"$p_z^\tau$",
    "tau_plus_mass":  r"$m_{\tau^+}$",
    "tau_minus_mass": r"$m_{\tau^-}$",
    "boson_mass":     r"$m_{\tau^+\tau^-}$",
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
    "tau_plus_pt":  r"$p_T^{\tau^+}$",
    "tau_minus_pt": r"$p_T^{\tau^-}$",
    "ditau_pt":     r"$p_T^{\tau^+\tau^-}$",
    "ditau_px":     r"$p_x^{\tau^+\tau^-}$",
    "ditau_py":     r"$p_y^{\tau^+\tau^-}$",
    "ditau_pz":     r"$p_z^{\tau^+\tau^-}$",
    "ditau_E":      r"$E^{\tau^+\tau^-}$",
    "nu_pt":        r"$p_T^{\nu}$",
    "nu_eta":       r"$\eta^{\nu}$",
    "nu_phi":       r"$\phi^{\nu}$",
    "nubar_pt":     r"$p_T^{\bar{\nu}}$",
    "nubar_eta":    r"$\eta^{\bar{\nu}}$",
    "nubar_phi":    r"$\phi^{\bar{\nu}}$",
}


m_tau = 1.77686
m_boson = 125.0

truth_color = "#FFDBBB" # '#C9B583'
sampled_color = '#0072B2'
pred_color = "#e42536" #'#CC3311'
transformer_color = '#2CA02C'
truth_line_color = '#CC5801'


# more purple style
# truth_color = "#C8BCC3"
# sampled_color = "#0E8BD3" # "#BA2D0B"
# transformer_color = "#50A733"



def replace_failed_map(df, threshold=1.0):
    """Replace map_pred_* values with the sampled pred_* prediction for events where
    the MAP optimiser failed (spike at 0 GeV)."""
    df = df.copy()
    nu_failed = df["map_pred_nu_E"] < threshold if "map_pred_nu_E" in df.columns else pd.Series(False, index=df.index)
    nubar_failed = df["map_pred_nubar_E"] < threshold if "map_pred_nubar_E" in df.columns else pd.Series(False, index=df.index)
    failed = nu_failed | nubar_failed
    frac_failed = failed.sum() / len(df) if len(df) > 0 else 0.0
    print(f">> replaceFailed: {failed.sum()}/{len(df)} ({frac_failed:.2%}) events had failed MAP estimates, replacing with sampled predictions")

    map_cols = [c for c in df.columns if c.startswith("map_pred_")]
    for map_col in map_cols:
        pred_col = "pred_" + map_col[len("map_pred_"):]
        if pred_col in df.columns:
            df[map_col] = np.where(failed, df[pred_col], df[map_col])
    return df


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


angle_vars = {
    "opening_angle":       r"$\Delta\theta(\tau^+,\tau^-)$ [rad]",
    "dR":                  r"$\Delta R(\tau^+,\tau^-)$",
    "nu_nubar_deta":       r"$\Delta\eta(\nu,\bar{\nu})$",
    "nu_nubar_dphi":       r"$\Delta\phi(\nu,\bar{\nu})$",
    "nu_nubar_dR":         r"$\Delta R(\nu,\bar{\nu})$",
    "nu_taum_dphi":        r"$\Delta\phi(\nu,\tau^-)$",
    "nu_taum_dR":          r"$\Delta R(\nu,\tau^-)$",
    "nubar_taup_dphi":     r"$\Delta\phi(\bar{\nu},\tau^+)$",
    "nubar_taup_dR":       r"$\Delta R(\bar{\nu},\tau^+)$",
}


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


def plot_angle_vars(df, output_dir, useMAP=True, tag=''):
    df = add_angle_cols(df)
    suffix = f'{tag}_' if tag else ''
    plot_two_panel_vars(df, angle_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=False)


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
    """Per-event (other - true) resolution, aligned on the shared event index."""
    common_idx = true_vals.index.intersection(other_vals.index)
    return other_vals.loc[common_idx] - true_vals.loc[common_idx]


def _iqr(series):
    """Interquartile range (Q3 - Q1) — more robust to outlier tails than std."""
    return series.quantile(0.75) - series.quantile(0.25)


def _draw_resolution(ax_res, true_vals, pred_vals, sampled_vals, show_sampled,
                      transformer_vals, show_transformer):
    """Draw the per-event (pred - truth) resolution histogram on the right,
    x-range capped to the 1st-99th percentile so outlier tails don't dominate."""
    diff_pred = _resolution_diff(true_vals, pred_vals)
    diff_parts = [diff_pred]
    if show_sampled:
        diff_sampled = _resolution_diff(true_vals, sampled_vals)
        diff_parts.append(diff_sampled)
    if show_transformer:
        diff_transformer = _resolution_diff(true_vals, transformer_vals)
        diff_parts.append(diff_transformer)
    combined_diff = pd.concat(diff_parts)
    bin_range = (combined_diff.quantile(0.01), combined_diff.quantile(0.99))
    diff_bins = np.histogram_bin_edges(combined_diff, bins=50, range=bin_range)

    pred_mu, pred_iqr = diff_pred.mean(), _iqr(diff_pred)
    ax_res.axvline(0.0, color="gray", linestyle="dashed", linewidth=1)
    if show_transformer:
        transformer_mu, transformer_iqr = diff_transformer.mean(), _iqr(diff_transformer)
        ax_res.hist(diff_transformer, bins=diff_bins, histtype="step", density=True, linewidth=2.2, color=transformer_color,
                    label=fr"Transformer: $\mu$={transformer_mu:.3f}, IQR={transformer_iqr:.3f}")
    if show_sampled:
        sampled_mu, sampled_iqr = diff_sampled.mean(), _iqr(diff_sampled)
        ax_res.hist(diff_sampled, bins=diff_bins, histtype="step", density=True, linewidth=2.2, color=sampled_color,
                    label=fr"TauPolaris (sampled): $\mu$={sampled_mu:.3f}, IQR={sampled_iqr:.3f}")

    ax_res.hist(diff_pred, bins=diff_bins, histtype="step", density=True, linewidth=2.2, color=pred_color,
                label=fr"TauPolaris (MAP): $\mu$={pred_mu:.3f}, IQR={pred_iqr:.3f}")
    ax_res.set_xlabel("Resolution (Pred. - Truth)")
    ax_res.set_ylabel("a.u.")
    ax_res.set_xlim(diff_bins[0], diff_bins[-1])
    ymin, ymax = ax_res.get_ylim()
    ymax_mult = 1.35 + 0.15 * (int(show_sampled) + int(show_transformer))
    ax_res.set_ylim(ymin, ymax * ymax_mult)
    ax_res.legend(loc="upper right", frameon=True)


def _draw_distribution_ratio(ax, ax_ratio, true_vals, pred_vals, sampled_vals, show_sampled,
                              transformer_vals, show_transformer, bins, ymax_mult):
    """Draw the gray-filled truth / blue predicted / orange sampled / green transformer
    overlay and its pred-over-truth ratio panel, shared by all distribution plots."""
    true_counts, _ = np.histogram(true_vals, bins=bins)
    true_hist, _ = np.histogram(true_vals, bins=bins, density=True)
    # TODO: remove pred_hist (MAP), it's not needed now
    pred_hist, _ = np.histogram(pred_vals, bins=bins, density=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Relative statistical (Poisson) uncertainty on the truth histogram itself,
        # shown as a band around 1.0 in the ratio panel.
        true_rel_err = np.where(true_counts > 0, 1.0 / np.sqrt(true_counts), np.nan)
    if show_sampled:
        sampled_hist, _ = np.histogram(sampled_vals, bins=bins, density=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            sampled_ratio = np.where(true_hist > 0, sampled_hist / true_hist, np.nan)
    if show_transformer:
        transformer_hist, _ = np.histogram(transformer_vals, bins=bins, density=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            transformer_ratio = np.where(true_hist > 0, transformer_hist / true_hist, np.nan)

    ax.hist(true_vals, bins=bins, histtype="stepfilled", density=True,
            linewidth=0, color=truth_color, alpha=0.8,
            label="Generator Truth")
    ax.hist(true_vals, bins=bins, histtype="step", density=True,
            linewidth=0.5, color="black")
    if show_transformer:
        ax.hist(transformer_vals, bins=bins, histtype="step", density=True,
                linewidth=2, linestyle="solid", color=transformer_color,
                label="Transformer")
    if show_sampled:
        ax.hist(sampled_vals, bins=bins, histtype="step", density=True,
                linewidth=2, linestyle="solid", color=sampled_color,
                label="TauPolaris (sampled)")
    ax.set_ylabel("a.u.")
    ax.set_xlim(bins[0], bins[-1])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * ymax_mult)
    ax.legend(loc="upper right", frameon=True)
    ax.tick_params(labelbottom=False)

    band_lower = np.where(true_counts > 0, 1.0 - true_rel_err, np.nan)
    band_upper = np.where(true_counts > 0, 1.0 + true_rel_err, np.nan)
    ax_ratio.fill_between(bins, np.append(band_lower, band_lower[-1]), np.append(band_upper, band_upper[-1]),
                           step="post", color="lightgrey", alpha=0.6, linewidth=0, zorder=0)
    ax_ratio.axhline(1.0, color="gray", linestyle="dashed", linewidth=1)
    if show_transformer:
        ax_ratio.stairs(transformer_ratio, bins, linewidth=2, color=transformer_color)
    if show_sampled:
        ax_ratio.stairs(sampled_ratio, bins, linewidth=2, color=sampled_color)
    ax_ratio.set_ylabel("Pred. / Truth")
    ax_ratio.set_xlim(bins[0], bins[-1])
    ax_ratio.set_ylim(0.8, 1.2)


def _var_bins(var, true_vals, pred_vals, sampled_vals, show_sampled,
               transformer_vals, show_transformer, bounded):
    """Histogram bin edges for a two-panel plot: fixed [-1, 1] for bounded
    (cosine) vars, otherwise a data-driven quantile range."""
    if bounded:
        return np.linspace(-1.0, 1.0, 51)
    combined_parts = [true_vals, pred_vals]
    if show_sampled:
        combined_parts.append(sampled_vals)
    if show_transformer:
        combined_parts.append(transformer_vals)
    combined = pd.concat(combined_parts)
    if var in ('nu_taum_dR', 'nubar_taup_dR'):
        bin_range = (0.0, 0.4)
    elif var in ('nu_taum_dphi', 'nubar_taup_dphi'):
        bin_range = (-0.25, 0.25)
    elif var.endswith('_E') or var.endswith('_mass') or var.endswith('_pt'):
        bin_range = (combined.min(), combined.quantile(0.98))
    else:
        bin_range = (combined.quantile(0.01), combined.quantile(0.99))
    return np.histogram_bin_edges(combined, bins=50, range=bin_range)


def plot_two_panel_vars(df, var_dict, output_dir, subdir, useMAP=True, bounded=False):
    """Truth/pred/sampled distribution+ratio plot and a separate resolution plot
    for every var in var_dict, saved into distributions/ and resolutions/
    subdirectories respectively. bounded=True fixes the histogram range to
    [-1, 1] (spin_vars, spin_density_vars); otherwise the range is data-driven
    (pred_taunu_vars kinematics)."""
    dist_dir = os.path.join(output_dir, subdir, "distributions")
    res_dir = os.path.join(output_dir, subdir, "resolutions")
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

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

        true_vals = df[true_col].dropna()
        pred_vals = df[pred_col].dropna()
        sampled_vals = df[sampled_col].dropna() if show_sampled else None
        transformer_vals = df[transformer_col].dropna() if show_transformer else None
        ymax_mult = 1.35 + 0.10 * (int(show_sampled) + int(show_transformer))
        bins = _var_bins(var, true_vals, pred_vals, sampled_vals, show_sampled,
                          transformer_vals, show_transformer, bounded)

        fig_dist = plt.figure(figsize=(7, 7))
        gs = fig_dist.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax = fig_dist.add_subplot(gs[0])
        ax_ratio = fig_dist.add_subplot(gs[1], sharex=ax)
        _draw_distribution_ratio(ax, ax_ratio, true_vals, pred_vals, sampled_vals, show_sampled,
                                  transformer_vals, show_transformer, bins, ymax_mult)
        ax_ratio.set_xlabel(label)
        fig_dist.savefig(os.path.join(dist_dir, f"{var}.pdf"))
        plt.close(fig_dist)

        fig_res, ax_res = plt.subplots(figsize=(7, 7))
        _draw_resolution(ax_res, true_vals, pred_vals, sampled_vals, show_sampled,
                          transformer_vals, show_transformer)
        fig_res.savefig(os.path.join(res_dir, f"{var}.pdf"))
        plt.close(fig_res)

        print(f">> Saved paper plots {var}.pdf (distributions/ and resolutions/)")


def plot_spin_vars(df, output_dir, useMAP=True, tag=''):
    suffix = f'{tag}_' if tag else ''
    plot_two_panel_vars(df, spin_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=True)


def plot_spin_density_vars(df, output_dir, useMAP=True, tag=''):
    df = add_spin_density_cols(df)
    suffix = f'{tag}_' if tag else ''
    plot_two_panel_vars(df, spin_density_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=True)


def plot_derived_vars(df, output_dir, useMAP=True, tag=''):
    df = add_derived_kinematic_cols(df)
    suffix = f'{tag}_' if tag else ''
    plot_two_panel_vars(df, derived_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=False)


mass_vars = {"tau_plus_mass", "tau_minus_mass", "boson_mass"}


def plot_mass_vars(df, output_dir, useMAP=True, tag=''):
    """Single-panel plots for the tau/boson masses: their truth is a known
    fixed value (m_tau or m_boson), so there's no truth distribution to
    overlay — just a dashed truth line against the predicted histogram(s)."""
    suffix = f'{tag}_' if tag else ''
    subdir = f'{suffix}paper_plots'
    dist_dir = os.path.join(output_dir, subdir, "distributions")
    res_dir = os.path.join(output_dir, subdir, "resolutions")
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

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
        pred_vals = df[pred_col].dropna()
        sampled_vals = df[sampled_col].dropna() if show_sampled else None
        transformer_vals = df[transformer_col].dropna() if show_transformer else None
        ymax_mult = 1.35 + 0.10 * (int(show_sampled) + int(show_transformer))

        is_tau_mass = var in ("tau_plus_mass", "tau_minus_mass")
        truth_val = m_tau if is_tau_mass else m_boson
        bin_range = (1.0, 2.5) if is_tau_mass else (m_boson - 50, m_boson + 50)
        bins = np.linspace(bin_range[0], bin_range[1], 51)

        pred_mu, pred_sigma = pred_vals.mean(), pred_vals.std()

        fig_dist, ax = plt.subplots(figsize=(7, 7))

        ax.axvline(truth_val, color=truth_line_color, linestyle="dashed", linewidth=2.2,
                   label=fr"Generator Truth ($\mu$={truth_val:.2f} GeV)")
        if show_sampled:
            sampled_mu, sampled_sigma = sampled_vals.mean(), sampled_vals.std()
            ax.hist(sampled_vals, bins=bins, histtype="step", density=True,
                    linewidth=2.2, linestyle="solid", color=sampled_color,
                    label=fr"TauPolaris Sampled $\mu$={sampled_mu:.2f}, $\sigma$={sampled_sigma:.2f}")
        if show_transformer:
            transformer_mu, transformer_sigma = transformer_vals.mean(), transformer_vals.std()
            ax.hist(transformer_vals, bins=bins, histtype="step", density=True,
                    linewidth=2.2, linestyle="solid", color=transformer_color,
                    label=fr"Transformer: $\mu$={transformer_mu:.2f}, $\sigma$={transformer_sigma:.2f}")
        # ax.hist(pred_vals, bins=bins, histtype="step", density=True,
        #         linewidth=2.2, linestyle="solid", color=pred_color,
        #         label=fr"TauPolaris (MAP): $\mu$={pred_mu:.2f}, $\sigma$={pred_sigma:.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("a.u.")
        ax.set_xlim(bins[0], bins[-1])
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * ymax_mult)
        ax.legend(loc="upper right", frameon=True)

        fig_dist.savefig(os.path.join(dist_dir, f"{var}.pdf"))
        plt.close(fig_dist)

        if has_true:
            fig_res, ax_res = plt.subplots(figsize=(7, 7))
            true_vals = df[true_col].dropna()
            _draw_resolution(ax_res, true_vals, pred_vals, sampled_vals, show_sampled,
                              transformer_vals, show_transformer)
            fig_res.savefig(os.path.join(res_dir, f"{var}.pdf"))
            plt.close(fig_res)

        print(f">> Saved paper plots {var}.pdf (distributions/{' and resolutions/' if has_true else ''})")


def paper_plot(df, output_dir, useMAP=True, tag=''):
    suffix = f'{tag}_' if tag else ''
    kinematic_vars = {v: l for v, l in pred_taunu_vars.items() if v not in mass_vars}
    plot_mass_vars(df, output_dir, useMAP=useMAP, tag=tag)
    plot_two_panel_vars(df, kinematic_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=False)


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

    if args.replaceFailed:
        if use_map:
            df = replace_failed_map(df)
        else:
            print(">> Warning: --replaceFailed has no effect when not using MAP estimates")

    plot_spin_vars(df, output_dir, use_map, tag=args.tag)
    paper_plot(df, output_dir, use_map, tag=args.tag)
    plot_spin_density_vars(df, output_dir, use_map, tag=args.tag)
    plot_derived_vars(df, output_dir, use_map, tag=args.tag)
    plot_angle_vars(df, output_dir, use_map, tag=args.tag)


if __name__ == "__main__":
    main()