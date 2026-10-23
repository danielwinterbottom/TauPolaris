"""2D per-event resolution plots (prediction vs truth) for the variables in
plot_kinematics_spin.py, with one panel per prediction: MAP, sampled, and stable MAP /
transformer / SVfit if --stable_map_input / --transformer_input / --svfit_input are given."""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# also sets up the matplotlib backend and CMS style
from plot_kinematics_spin import (
    spin_vars, kinematic_vars, training_basis_vars, spin_density_vars, derived_vars, angle_vars,
    overlay_styles, load_transformer_predictions, load_svfit_predictions, load_stable_map_predictions,
    add_all_derived_cols,
    add_map_nu_E_cut_args, select_events, output_subdir, _var_bins, _weights_for,
)


# (variable dict, bounded) for every group plotted; masses are skipped since their truth is fixed
plot_groups = (
    (spin_vars, True),
    (kinematic_vars, False),
    (training_basis_vars, False),
    (spin_density_vars, True),
    (derived_vars, False),
    (angle_vars, False),
)

cmap = "Blues"
vmax_percentile = 99


def _panels(df, var, index):
    """[(style key, values)] for each available prediction, aligned to index.
    Each panel keeps only its non-NaN events (SVfit is NaN for failed fits)."""
    cols = (("map", f"map_pred_{var}"), ("stable_map", f"stable_map_pred_{var}"), ("sampled", f"pred_{var}"),
            ("transformer", f"transformer_pred_{var}"), ("svfit", f"svfit_pred_{var}"))
    return [(key, df[col].loc[index].dropna()) for key, col in cols if col in df.columns]


def _corr(x, y, weights=None):
    """(Weighted) Pearson correlation coefficient. Written out by hand rather than
    np.cov(aweights=...) since that rejects negative weights. Non-finite entries are ignored."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[ok], y[ok], w[ok]
    dx, dy = x - np.average(x, weights=w), y - np.average(y, weights=w)
    return np.sum(w * dx * dy) / np.sqrt(np.sum(w * dx**2) * np.sum(w * dy**2))


def _conditional_hist(true_vals, pred_vals, bins, weights=None):
    """2D histogram with each truth bin normalised to 1, i.e. P(pred | truth).
    Indexed [truth bin, pred bin]; empty truth bins are NaN."""
    h, _, _ = np.histogram2d(true_vals, pred_vals, bins=[bins, bins], weights=weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        return h / h.sum(axis=1, keepdims=True)


def plot_2d_vars(df, var_dict, out_dir, bounded, weights=None, note=None):
    for var, label in var_dict.items():
        true_col = f"true_{var}"
        if true_col not in df.columns or f"map_pred_{var}" not in df.columns:
            print(f"Warning: Missing columns for {var}, skipping")
            continue

        # if for whatever reason the truth is missing skip that event
        true_vals = df[true_col].dropna()
        panels = _panels(df, var, true_vals.index)
        bins = _var_bins(var, [true_vals] + [v for _, v in panels], bounded, weights)

        # each panel uses only its own events, so pair truth and weights with them
        hists, corrs = [], []
        for _, vals in panels:
            panel_true, panel_w = true_vals.loc[vals.index], _weights_for(vals, weights)
            hists.append(_conditional_hist(panel_true, vals, bins, panel_w))
            corrs.append(_corr(panel_true, vals, panel_w))
        # shared colour scale; cap at the 99th percentile of filled bins so a few
        # saturated bins (e.g. at cos = +-1) don't wash out the core
        filled = np.concatenate([h[np.isfinite(h) & (h > 0)] for h in hists])
        vmax = np.percentile(filled, vmax_percentile)

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharex=True, sharey=True,
                                 constrained_layout=True, squeeze=False)
        axes = axes[0]
        for ax, (key, _), h, r in zip(axes, panels, hists, corrs):
            mesh = ax.pcolormesh(bins, bins, h.T, cmap=cmap, vmin=0, vmax=vmax, rasterized=True)
            ax.plot(bins[[0, -1]], bins[[0, -1]], color="gray", linestyle="dashed", linewidth=1.5)
            ax.set_title(overlay_styles[key][1], fontsize=18)
            box = dict(facecolor="white", alpha=0.8, edgecolor="none")
            ax.text(0.05, 0.93, f"r = {r:.3f}", transform=ax.transAxes, fontsize=16, bbox=box)
            if note:
                ax.text(0.05, 0.85, note, transform=ax.transAxes, fontsize=16, bbox=box)
            ax.set_xlabel(f"True {label}")
            ax.set_xlim(bins[0], bins[-1])
            ax.set_ylim(bins[0], bins[-1])
        axes[0].set_ylabel(f"Predicted {label}")
        fig.colorbar(mesh, ax=list(axes), label="P(pred. | truth)", extend="max")

        fig.savefig(os.path.join(out_dir, f"{var}_2D.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f">> Saved 2D plot {var}_2D.pdf ({n} panels)")


def main():
    parser = argparse.ArgumentParser(description="Plot 2D prediction vs truth for kinematics and spin variables")
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file")
    parser.add_argument('--transformer_input', type=str, default=None,
                        help="Path to a second parquet file with transformer predictions "
                             "(same pred_* column naming as the sampled predictions, row-aligned "
                             "with --input); shown as an extra panel if given")
    parser.add_argument('--svfit_input', type=str, default=None,
                        help="Path to an evaluate_svfit.py output parquet (pred_SVfit_* columns, row-aligned "
                             "with --input); shown as an extra panel if given. Events with a failed "
                             "SVfit fit are left out of the SVfit panel only")
    parser.add_argument('--stable_map_input', type=str, default=None,
                        help="Path to an evaluate.py output run with map_method: stable_gradient "
                             "(stable_map_pred_* columns, row-aligned with --input); shown as an extra panel if given")
    parser.add_argument('--tag', type=str, default='', help="Tag prepended to the output subdirectory name")
    parser.add_argument('--weight', type=str, default=None,
                        help="Name of a column in the input parquet to use as the per-event weight")
    add_map_nu_E_cut_args(parser)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f">> Loaded {len(df)} events from {args.input}")
    if args.transformer_input:
        df = load_transformer_predictions(df, args.transformer_input)
        print(f">> Loaded transformer predictions from {args.transformer_input}")
    if args.svfit_input:
        df = load_svfit_predictions(df, args.svfit_input)
        print(f">> Loaded SVfit predictions from {args.svfit_input}")
    if args.stable_map_input:
        df = load_stable_map_predictions(df, args.stable_map_input)
        print(f">> Loaded stable MAP predictions from {args.stable_map_input}")
    if args.weight and args.weight not in df.columns:
        raise ValueError(f"--weight column '{args.weight}' not found in input parquet")
    df, cut_tag, note = select_events(df, args)
    df = add_all_derived_cols(df)
    weights = df[args.weight] if args.weight else None

    out_dir = os.path.join(args.input.replace('.parquet', ''), output_subdir(args.tag, cut_tag), "2D")
    os.makedirs(out_dir, exist_ok=True)

    for var_dict, bounded in plot_groups:
        plot_2d_vars(df, var_dict, out_dir, bounded, weights, note)


if __name__ == "__main__":
    main()
