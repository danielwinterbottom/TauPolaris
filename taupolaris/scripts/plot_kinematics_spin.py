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


# ---------------------------------------------------------------------------
# Variables to plot (column suffix -> axis label)
# ---------------------------------------------------------------------------

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

# masses get single-panel plots against a fixed truth value
mass_vars = ("tau_plus_mass", "tau_minus_mass", "boson_mass")
kinematic_vars = {v: l for v, l in pred_taunu_vars.items() if v not in mass_vars}


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

# (output name prefix, object 1, object 2, quantities to store) for add_angle_cols
angle_pairs = (
    ("",            "tau_plus", "tau_minus", ("opening_angle", "dR")),
    ("nu_nubar_",   "nu",       "nubar",     ("deta", "dphi", "dR")),
    ("nu_taum_",    "nu",       "tau_minus", ("dphi", "dR")),
    ("nubar_taup_", "nubar",    "tau_plus",  ("dphi", "dR")),
)


# ---------------------------------------------------------------------------
# Constants and styling
# ---------------------------------------------------------------------------

# column prefixes for truth, MAP prediction, sampled prediction, transformer and SVfit predictions
prefixes = ("true_", "map_pred_", "pred_", "transformer_pred_", "svfit_pred_")

m_tau = 1.77686
m_boson = 125.0

truth_color = "#EE8A3D"
truth_alpha = 0.6
sampled_color = '#0072B2'
pred_color = "#e42536"
transformer_color = '#2CA02C'
svfit_color = '#9467BD'

# overlay key -> (color, legend label)
overlay_styles = {
    "svfit":       (svfit_color, "SVfit"),
    "transformer": (transformer_color, "Transformer"),
    "sampled":     (sampled_color, "TauPolaris (sampled)"),
    "map":         (pred_color, "TauPolaris (MAP)"),
}


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _pt_eta_phi(px, py, pz):
    """Transverse momentum, pseudorapidity, and azimuthal angle."""
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px)
    return pt, eta, phi


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


# ---------------------------------------------------------------------------
# Derived columns (added for every prefix whose inputs are present)
# ---------------------------------------------------------------------------

def _has(df, *cols):
    return all(c in df.columns for c in cols)


def _p3_cols(prefix, obj):
    return [f"{prefix}{obj}_{c}" for c in ("px", "py", "pz")]


def spin_density_cols(df):
    """Cos-product spin density columns."""
    new = {}
    for name, (var1, var2) in spin_density_products.items():
        for prefix in prefixes:
            col1, col2 = f"{prefix}{var1}", f"{prefix}{var2}"
            if _has(df, col1, col2):
                new[f"{prefix}{name}"] = df[col1] * df[col2]
    return new


def derived_kinematic_cols(df):
    """Tau pT, ditau (four-)momentum, and neutrino pT/eta/phi columns."""
    new = {}
    for prefix in prefixes:
        for tau in ("tau_plus", "tau_minus"):
            px, py = f"{prefix}{tau}_px", f"{prefix}{tau}_py"
            if _has(df, px, py):
                new[f"{prefix}{tau}_pt"] = np.hypot(df[px], df[py])

        for c in ("px", "py", "pz", "E"):
            plus, minus = f"{prefix}tau_plus_{c}", f"{prefix}tau_minus_{c}"
            if _has(df, plus, minus):
                new[f"{prefix}ditau_{c}"] = df[plus] + df[minus]

        ditau_px, ditau_py = f"{prefix}ditau_px", f"{prefix}ditau_py"
        if ditau_px in new and ditau_py in new:
            new[f"{prefix}ditau_pt"] = np.hypot(new[ditau_px], new[ditau_py])

        for nu in ("nu", "nubar"):
            cols = _p3_cols(prefix, nu)
            if _has(df, *cols):
                pt, eta, phi = _pt_eta_phi(*(df[c] for c in cols))
                new[f"{prefix}{nu}_pt"] = pt
                new[f"{prefix}{nu}_eta"] = eta
                new[f"{prefix}{nu}_phi"] = phi
    return new


def angle_cols(df):
    """Angular separations listed in angle_pairs (taus, neutrinos, and each
    neutrino with its parent tau: nu <-> tau_minus, nubar <-> tau_plus)."""
    new = {}
    for prefix in prefixes:
        for name, obj1, obj2, quantities in angle_pairs:
            cols = _p3_cols(prefix, obj1) + _p3_cols(prefix, obj2)
            if not _has(df, *cols):
                continue
            p = [df[c] for c in cols]
            deta, dphi = _delta_eta_phi(*p)
            values = {"deta": deta, "dphi": dphi, "dR": np.hypot(deta, dphi)}
            if "opening_angle" in quantities:
                values["opening_angle"] = _opening_angle(*p)
            for q in quantities:
                new[f"{prefix}{name}{q}"] = values[q]
    return new


def add_all_derived_cols(df):
    """Return df with the spin density, derived kinematic and angle columns appended
    for every prefix whose inputs are present."""
    new = {**spin_density_cols(df), **derived_kinematic_cols(df), **angle_cols(df)}
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


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
    for var in list(pred_taunu_vars) + list(spin_vars):
        col = f"pred_{var}"
        if col in transformer_df.columns:
            df[f"transformer_pred_{var}"] = transformer_df[col].values
    return df


def load_svfit_predictions(df, svfit_path):
    """Attach SVfit predictions from an evaluate_svfit.py output file (pred_SVfit_* columns,
    row-aligned with df) under a svfit_pred_ prefix. Failed SVfit fits keep their NaN predictions:
    the SVfit curves skip those events, while every other curve still uses all events."""
    svfit_df = pd.read_parquet(svfit_path)
    pred_cols = [f"pred_SVfit_{var}" for var in list(pred_taunu_vars) + list(spin_vars) if f"pred_SVfit_{var}" in svfit_df.columns]
    check_cols = [c for c in ("true_nu_px", "true_nubar_px") if c in svfit_df.columns and c in df.columns]
    if len(svfit_df) != len(df):
        raise ValueError(
            f"SVfit predictions file has {len(svfit_df)} rows, "
            f"but the main input has {len(df)} rows — they must be row-aligned."
        )
    # the SVfit results come from a separate script, so check the events really line up
    for col in check_cols:
        if not np.allclose(svfit_df[col].values, df[col].values, equal_nan=True):
            raise ValueError(f"{col} differs between the SVfit file and the main input — they are not row-aligned.")
    df = df.copy()
    for col in pred_cols:
        vals = svfit_df[col].values
        if col.endswith("_E"):
            # SVfit can return unphysical negative (neutrino) energies; set them to 0
            n_negative = int((vals < 0).sum())
            if n_negative:
                print(f">> Setting {n_negative} negative {col} values to 0")
            vals = np.where(vals < 0, 0.0, vals)  # keeps NaN for failed fits
        df[col.replace("pred_SVfit_", "svfit_pred_")] = vals
    n_failed = int((~svfit_df["svfit_valid"].values.astype(bool)).sum())
    print(f">> {n_failed} of {len(df)} events have no valid SVfit fit; they are left out of the SVfit curves only")
    return df


# ---------------------------------------------------------------------------
# Weighted statistics (weights=None means unweighted)
# ---------------------------------------------------------------------------

def _weighted_mean(values, weights=None):
    if weights is None:
        return values.mean()
    return np.average(values, weights=weights)


def _weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sorter = np.argsort(values)
    values, weights = values[sorter], weights[sorter]
    cdf = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(q, cdf, values)


def _quantile(series, q, weights=None):
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


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _overlays(df, var, use_map, index=None):
    """[(style key, values)] of the extra predictions to overlay, in draw order.
    The sampled prediction is only overlaid when the main line is the MAP
    estimate; otherwise the main line already is pred_*, so it would be a duplicate.
    Each overlay keeps only its non-NaN events (SVfit is NaN for failed fits), so the
    drawing functions normalise every curve over its own events."""
    overlays = []
    if f"svfit_pred_{var}" in df.columns:
        overlays.append(("svfit", df[f"svfit_pred_{var}"]))
    if f"transformer_pred_{var}" in df.columns:
        overlays.append(("transformer", df[f"transformer_pred_{var}"]))
    if use_map and f"pred_{var}" in df.columns:
        overlays.append(("sampled", df[f"pred_{var}"]))
    if index is not None:
        overlays = [(key, vals.loc[index]) for key, vals in overlays]
    return [(key, vals.dropna()) for key, vals in overlays]


def _weights_for(vals, weights=None):
    """Per-event weights matching vals' events (None if unweighted)."""
    return weights.loc[vals.index] if weights is not None else None


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_resolution(ax_res, true_vals, main_vals, overlays, label, weights=None):
    """Per-event (pred - truth) histograms for the main prediction and each overlay."""
    # the main prediction is never NaN, so it covers true_vals' index; overlays may cover fewer
    # events (SVfit failed fits), so each diff only keeps the events where both values exist
    diff_main = main_vals.loc[true_vals.index] - true_vals
    diffs = [(key, (vals - true_vals).dropna()) for key, vals in overlays]
    all_diffs = diffs + [("map", diff_main)]

    combined_diff = pd.concat([d for _, d in all_diffs])
    if weights is not None:
        combined_weight = pd.concat([_weights_for(d, weights) for _, d in all_diffs])
        bin_range = (_weighted_quantile(combined_diff, combined_weight, 0.01),
                     _weighted_quantile(combined_diff, combined_weight, 0.99))
    else:
        bin_range = (combined_diff.quantile(0.01), combined_diff.quantile(0.99))
    diff_bins = np.histogram_bin_edges(combined_diff, bins=50, range=bin_range)

    for key, diff in all_diffs:
        color, name = overlay_styles[key]
        w = _weights_for(diff, weights)
        ax_res.hist(diff, bins=diff_bins, histtype="step", weights=_count_norm_weights(len(diff), w), linewidth=2, color=color,
                    label=fr"{name}: $\mu$={_weighted_mean(diff, w):.2f}, IQR={_iqr(diff, w):.2f}")

    ax_res.set_xlabel(f"{label} Resolution (Pred. - Truth)")
    ax_res.set_ylabel("Normalized counts")
    ax_res.set_xlim(diff_bins[0], diff_bins[-1])
    ymin, ymax = ax_res.get_ylim()
    ax_res.set_ylim(ymin, ymax * (1.35 + 0.15 * len(overlays)))
    ax_res.legend(loc="upper left", fontsize=legendfont)


def _draw_distribution_ratio(ax, ax_ratio, true_vals, overlays, bins, weights=None):
    """Filled truth histogram with each overlay on top, plus a pred/truth ratio panel
    with the truth statistical uncertainty as a grey band."""
    hist_w = _count_norm_weights(len(true_vals), weights)
    true_counts, _ = np.histogram(true_vals, bins=bins)
    true_hist, _ = np.histogram(true_vals, bins=bins, weights=hist_w)

    ax.hist(true_vals, bins=bins, histtype="stepfilled", weights=hist_w,
            linewidth=0, color=truth_color, alpha=truth_alpha, label="Generator Truth")
    ax.hist(true_vals, bins=bins, histtype="step", weights=hist_w, linewidth=0.5, color="black")

    with np.errstate(divide="ignore", invalid="ignore"):
        # relative sqrt count uncertainty on truth
        true_rel_err = np.where(true_counts > 0, 1.0 / np.sqrt(true_counts), np.nan)
    band_lower = np.where(true_counts > 0, 1.0 - true_rel_err, np.nan)
    band_upper = np.where(true_counts > 0, 1.0 + true_rel_err, np.nan)
    ax_ratio.fill_between(bins, np.append(band_lower, band_lower[-1]), np.append(band_upper, band_upper[-1]),
                          step="post", color="lightgrey", alpha=0.6, linewidth=0, zorder=0)
    ax_ratio.axhline(1.0, color="gray", linestyle="dashed", linewidth=1)

    for key, vals in overlays:
        color, name = overlay_styles[key]
        # normalised over this overlay's own events (fewer than truth for SVfit)
        overlay_w = _count_norm_weights(len(vals), _weights_for(vals, weights))
        ax.hist(vals, bins=bins, histtype="step", weights=overlay_w, linewidth=2, color=color, label=name)
        overlay_hist, _ = np.histogram(vals, bins=bins, weights=overlay_w)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(true_hist > 0, overlay_hist / true_hist, np.nan)
        ax_ratio.stairs(ratio, bins, linewidth=2, color=color)

    ax.set_ylabel("Normalized counts")
    ax.set_xlim(bins[0], bins[-1])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * (1.35 + 0.10 * len(overlays)))
    ax.legend(loc="upper right", fontsize=legendfont)
    ax.tick_params(labelbottom=False)

    ax_ratio.set_ylabel("Pred. / Truth")
    ax_ratio.set_xlim(bins[0], bins[-1])
    ax_ratio.set_ylim(0.75, 1.25)


def _var_bins(var, all_vals, bounded, weights=None):
    """50 bins: [-1, 1] for bounded (cosine) variables, otherwise a range
    set from the quantiles of all the values to be drawn."""
    if bounded:
        return np.linspace(-1.0, 1.0, 51)
    combined = pd.concat(all_vals)
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


# ---------------------------------------------------------------------------
# Plot drivers
# ---------------------------------------------------------------------------

def plot_two_panel_vars(df, var_dict, dist_dir, res_dir, use_map=True, bounded=False, weights=None):
    """For each variable: distribution + ratio plot, and resolution plot."""
    for var, label in var_dict.items():
        true_col = f"true_{var}"
        main_col = f"map_pred_{var}" if use_map else f"pred_{var}"
        if not _has(df, true_col, main_col):
            print(f"Warning: Missing columns for {var}, skipping")
            continue

        # if for whatever reason the truth is missing skip that event
        true_vals = df[true_col].dropna()
        main_vals = df[main_col].loc[true_vals.index]
        overlays = _overlays(df, var, use_map, index=true_vals.index)
        weight_true = weights.loc[true_vals.index] if weights is not None else None
        bins = _var_bins(var, [true_vals, main_vals] + [v for _, v in overlays], bounded, weights)

        fig_dist = plt.figure(figsize=(8, 6))
        gs = fig_dist.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
        ax = fig_dist.add_subplot(gs[0])
        ax_ratio = fig_dist.add_subplot(gs[1], sharex=ax)
        _draw_distribution_ratio(ax, ax_ratio, true_vals, overlays, bins, weight_true)
        ax_ratio.set_xlabel(label)
        _save(fig_dist, os.path.join(dist_dir, f"{var}.pdf"))

        fig_res, ax_res = plt.subplots(figsize=(8, 6))
        _draw_resolution(ax_res, true_vals, main_vals, overlays, label, weights)
        _save(fig_res, os.path.join(res_dir, f"{var}_res.pdf"))

        print(f">> Saved paper plots {var}.pdf (distributions/ and resolutions/)")


def plot_mass_vars(df, dist_dir, res_dir, use_map=True, weights=None):
    """Single-panel plots for the tau/boson masses, with the fixed truth value as a line."""
    for var in mass_vars:
        label = pred_taunu_vars[var]
        true_col = f"true_{var}"
        main_col = f"map_pred_{var}" if use_map else f"pred_{var}"
        if main_col not in df.columns:
            print(f"Warning: Missing columns for {var}, skipping")
            continue

        overlays = _overlays(df, var, use_map)
        is_tau_mass = var in ("tau_plus_mass", "tau_minus_mass")
        truth_val = m_tau if is_tau_mass else m_boson
        bins = np.linspace(0.5, 2.5, 51) if is_tau_mass else np.linspace(70, 150, 51)

        fig_dist, ax = plt.subplots(figsize=(8, 6))
        ax.axvline(truth_val, color=truth_color, linestyle="solid", linewidth=2, label="Generator Truth")
        for key, vals in overlays:
            color, name = overlay_styles[key]
            ax.hist(vals, bins=bins, histtype="step", weights=_count_norm_weights(len(vals), _weights_for(vals, weights)),
                    linewidth=2, color=color, label=name)
        ax.set_xlabel(label)
        ax.set_ylabel("Normalized counts")
        ax.set_xlim(bins[0], bins[-1])
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * (1.35 + 0.10 * len(overlays)))
        ax.legend(loc="upper left", fontsize=legendfont)
        _save(fig_dist, os.path.join(dist_dir, f"{var}.pdf"))

        has_true = true_col in df.columns
        if has_true:
            fig_res, ax_res = plt.subplots(figsize=(8, 6))
            _draw_resolution(ax_res, df[true_col].dropna(), df[main_col], overlays, label, weights)
            _save(fig_res, os.path.join(res_dir, f"{var}_res.pdf"))

        print(f">> Saved paper plots {var}.pdf (distributions/{' and resolutions/' if has_true else ''})")


def main():

    parser = argparse.ArgumentParser(description="Plot kinematics and spin variables")
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file")
    parser.add_argument('--transformer_input', type=str, default=None,
                        help="Path to a second parquet file with transformer predictions "
                             "(same pred_* column naming as the sampled predictions, row-aligned "
                             "with --input); overlaid as a green line if given")
    parser.add_argument('--svfit_input', type=str, default=None,
                        help="Path to an evaluate_svfit.py output parquet (pred_SVfit_* columns, row-aligned "
                             "with --input); overlaid as a purple line if given. Events with a failed "
                             "SVfit fit are left out of the SVfit curves only")
    parser.add_argument('--sampled', action='store_true',
                        help="Use the sampled prediction (pred_*) as the main line instead of the MAP estimate")
    parser.add_argument('--useMLP', action='store_true', help=argparse.SUPPRESS)  # old alias for --sampled
    parser.add_argument('--tag', type=str, default='', help="Tag prepended to the output subdirectory name")
    parser.add_argument('--weight', type=str, default=None,
                        help="Name of a column in the input parquet to use as the per-event "
                             "weight for every histogram (distributions, ratios, resolutions, and masses)")
    args = parser.parse_args()
    use_map = not (args.sampled or args.useMLP)

    df = pd.read_parquet(args.input)
    print(f">> Loaded {len(df)} events from {args.input}")
    if args.transformer_input:
        df = load_transformer_predictions(df, args.transformer_input)
        print(f">> Loaded transformer predictions from {args.transformer_input}")
    if args.svfit_input:
        df = load_svfit_predictions(df, args.svfit_input)
        print(f">> Loaded SVfit predictions from {args.svfit_input}")
    if args.weight and args.weight not in df.columns:
        raise ValueError(f"--weight column '{args.weight}' not found in input parquet")
    df = add_all_derived_cols(df)
    weights = df[args.weight] if args.weight else None

    subdir = f"{args.tag}_paper_plots" if args.tag else "paper_plots"
    output_dir = os.path.join(args.input.replace('.parquet', ''), subdir)
    dist_dir = os.path.join(output_dir, "distributions")
    res_dir = os.path.join(output_dir, "resolutions")
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    plot_two_panel_vars(df, spin_vars, dist_dir, res_dir, use_map, bounded=True, weights=weights)
    plot_mass_vars(df, dist_dir, res_dir, use_map, weights=weights)
    plot_two_panel_vars(df, kinematic_vars, dist_dir, res_dir, use_map, weights=weights)
    plot_two_panel_vars(df, spin_density_vars, dist_dir, res_dir, use_map, bounded=True, weights=weights)
    plot_two_panel_vars(df, derived_vars, dist_dir, res_dir, use_map, weights=weights)
    plot_two_panel_vars(df, angle_vars, dist_dir, res_dir, use_map, weights=weights)


if __name__ == "__main__":
    main()
