"""
Plot toy distributions of con and m12 from two NLL-fit output files
(standard entanglement vs no-entanglement) and estimate the p-value
for separating them, assuming data lands at the median of the signal.
"""

import argparse
import math
import numpy as np
import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import mplhep as hep

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 18,
                      "ytick.labelsize": 18, "legend.fontsize": 18})
# The CMS style maps \mathcal (mathtext.cal) to the same sans-serif font as
# regular text, so \mathcal{C} renders as a plain C with no script styling.
# main.tex loads no special math-font package, so the paper's C (concurrence)
# is default LaTeX Computer Modern; 'cm' reproduces that specific glyph here.
plt.rcParams['mathtext.fontset'] = 'cm'

# Collaborator's shared palette (models.tex figures); reused here so the
# with-/no-entanglement colors sit in the same family as the rest of the paper.
sig_color = '#0072B2'   # "With entanglement"  (== sampled_color)
bkg_color = '#e42536'   # "No entanglement"    (== pred_color)

parser = argparse.ArgumentParser()
parser.add_argument("--sig-file",
                    default="nll_fits_aug7/toys_tree.root",
                    help="ROOT file with entanglement toys (signal hypothesis)")
parser.add_argument("--bkg-file",
                    default="nll_fits_no_entanglement_aug7/toys_tree.root",
                    help="ROOT file with no-entanglement toys (null hypothesis)")
parser.add_argument("--tree", default="toy_tree")
parser.add_argument("--n-bins", type=int, default=50)
parser.add_argument("--outdir", default=".")
parser.add_argument("--cl", type=float, default=0.95,
                    help="Confidence level used for the p-value/significance limit "
                         "quoted when zero null toys land beyond the signal median.")
args = parser.parse_args()

# --- normal-distribution helpers (no scipy dependency) ----------------------
def _norm_sf(z):
    """One-sided p-value for a given significance Z: P(X > z), X ~ N(0,1)."""
    return 0.5 * math.erfc(z / math.sqrt(2))

def _norm_isf(p):
    """Inverse of _norm_sf: significance Z corresponding to one-sided p-value p."""
    if p <= 0:
        return math.inf
    if p >= 1:
        return -math.inf
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _norm_sf(mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def p_value_report(n_beyond, n_total, cl=0.95):
    """
    Summarise a one-sided toy-counting p-value estimate p_hat = n_beyond/n_total.

    If n_beyond > 0, returns the point estimate with its (Gaussian-propagated)
    Poisson uncertainty, both in p and in the corresponding significance Z.

    If n_beyond == 0, p_hat = 0 is not a meaningful point estimate -- instead
    returns the exact Clopper-Pearson one-sided upper limit on p at confidence
    level `cl` (i.e. p < p_UL at that CL), and the corresponding *lower* limit
    on Z (Z > Z_LL at that CL). This replaces the naive floor p < 1/n_total,
    which is not a real confidence statement.
    """
    out = {"n_beyond": n_beyond, "n_total": n_total, "p_hat": n_beyond / n_total}
    if n_beyond > 0:
        p_hat = n_beyond / n_total
        Z_hat = _norm_isf(p_hat)
        sigma_p = math.sqrt(p_hat * (1 - p_hat) / n_total)
        phi_Z = math.exp(-Z_hat**2 / 2) / math.sqrt(2 * math.pi)
        sigma_Z = sigma_p / phi_Z if phi_Z > 0 else math.inf
        out.update(kind="measured", Z_hat=Z_hat, sigma_p=sigma_p, sigma_Z=sigma_Z)
    else:
        p_UL = 1 - (1 - cl) ** (1 / n_total)   # exact zero-count Clopper-Pearson bound
        Z_LL = _norm_isf(p_UL)
        out.update(kind="limit", cl=cl, p_UL=p_UL, Z_LL=Z_LL)
    return out

def format_p_report(rep):
    if rep["kind"] == "measured":
        return (f"p = {rep['p_hat']:.2e} +/- {rep['sigma_p']:.1e}   "
                f"(Z = {rep['Z_hat']:.2f} +/- {rep['sigma_Z']:.2f})   "
                f"[{rep['n_beyond']}/{rep['n_total']} toys beyond]")
    else:
        return (f"0/{rep['n_total']} toys beyond  ->  p < {rep['p_UL']:.2e} "
                f"at {100*rep['cl']:.0f}% CL   (Z > {rep['Z_LL']:.2f} at {100*rep['cl']:.0f}% CL)")

variables = [
    {"name": "con",  "xlabel": r"$\mathcal{C}$",  "logy": True, "xrange": (0, 1.0), "show_err": False},
    {"name": "m12",  "xlabel": r"$m_{12}$",        "logy": True, "xrange": (0, 4.5), "show_err": False},
]

def load(path, tree, variables):
    rdf = ROOT.RDataFrame(tree, path)
    return {v: rdf.AsNumpy([v])[v] for v in variables}

_C_branches = ["Cnn","Cnr","Cnk","Crn","Crr","Crk","Ckn","Ckr","Ckk"]
_B_branches = ["Bpn","Bpr","Bpk","Bmn","Bmr","Bmk"]

def branches_present(path, tree, names):
    rdf = ROOT.RDataFrame(tree, path)
    present = set(rdf.GetColumnNames())
    return [n for n in names if n in present]

print("Loading toys...")
var_names = [v["name"] for v in variables]

sig_C = branches_present(args.sig_file, args.tree, _C_branches)
sig_B = branches_present(args.sig_file, args.tree, _B_branches)
all_extra = sig_C + sig_B

sig = load(args.sig_file, args.tree, var_names + all_extra)
bkg = load(args.bkg_file, args.tree, var_names + all_extra)

for var in variables:
    sig_vals = sig[var["name"]]
    bkg_vals = bkg[var["name"]]

    all_vals = np.concatenate([sig_vals, bkg_vals])
    lo = var["xrange"][0] if var["xrange"][0] is not None else np.percentile(all_vals, 1)
    hi = var["xrange"][1] if var["xrange"][1] is not None else np.percentile(all_vals, 99)
    bins = np.linspace(lo, hi, args.n_bins + 1)

    sig_counts, _ = np.histogram(sig_vals, bins=bins)
    bkg_counts, _ = np.histogram(bkg_vals, bins=bins)
    if len(sig_vals) != len(bkg_vals):
        print(f"  NOTE: unequal toy counts (sig={len(sig_vals)}, bkg={len(bkg_vals)}) "
              f"-- raw-count heights are not directly comparable between the two")

    # Median of signal distribution as proxy for "observed data"
    median_sig = np.median(sig_vals)

    # p-value: fraction of null (no-entanglement) toys more extreme than median_sig.
    # "More extreme" means further toward signal median relative to bkg median.
    bkg_median = np.median(bkg_vals)
    if median_sig >= bkg_median:
        more_extreme = bkg_vals >= median_sig
    else:
        more_extreme = bkg_vals <= median_sig
    n_beyond = int(np.sum(more_extreme))
    n_total = len(bkg_vals)
    p_value = n_beyond / n_total
    p_report = p_value_report(n_beyond, n_total, cl=args.cl)
    print(f"  {var['name']}: {n_beyond}/{n_total} no-entanglement toys at least as "
          f"extreme as the signal median ({median_sig:.4g})")
    print(f"    {format_p_report(p_report)}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    def filled_step(ax, bins, counts, color, label):
        x = np.repeat(bins, 2)
        y = np.concatenate([[0], np.repeat(counts, 2), [0]])
        ax.fill_between(x, y, color=color, alpha=0.35, step=None)
        ax.step(bins, np.append(counts, 0), where='post',
                color=color, linewidth=1.5, label=label)

    filled_step(ax, bins, sig_counts, sig_color, 'With entanglement')
    filled_step(ax, bins, bkg_counts, bkg_color, 'No entanglement')

    # Arrow at signal median
    ymax = max(sig_counts.max(), bkg_counts.max())
    if var["logy"]:
        ax.set_yscale('log')
        ymin_log = min(sig_counts[sig_counts > 0].min(), bkg_counts[bkg_counts > 0].min())
        yaxis_min = ymin_log * 0.5
        # Generous headroom above the tallest bin so the legend box (which
        # matplotlib's auto-placement doesn't treat the shaded fill as an
        # obstacle for) always lands in clear space above the histograms.
        ax.set_ylim(yaxis_min, ymax * 100)
        arrow_bot = yaxis_min
        arrow_top = ymin_log * 50
        text_y = arrow_top
    else:
        arrow_bot = 0
        arrow_top = ymax * 0.28
        text_y = arrow_top
        ax.set_ylim(0, ymax * 1.45)

    ax.annotate('', xy=(median_sig, arrow_bot),
                xytext=(median_sig, arrow_top),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
    ax.text(median_sig, text_y, f'median\n({median_sig:.3f})',
            color='black', ha='center', va='bottom')

    # CL is quoted in the terminal output and caption text, not on the plot:
    # it overflows the axes at font size 18, and it's ambiguous at a glance
    # whether it qualifies the p-value or the significance.
    cmp = r'\geq' if median_sig >= bkg_median else r'\leq'
    if p_report["kind"] == "measured":
        if var.get("show_err", True):
            p_label = (f'p(no-ent. ${cmp}$ median) = {p_report["p_hat"]:.1e} '
                       f'$\\pm$ {p_report["sigma_p"]:.1e}  '
                       f'(${p_report["Z_hat"]:.1f}\\sigma$)')
        else:
            p_label = (f'p(no-ent. ${cmp}$ median) = {p_report["p_hat"]:.1e}  '
                       f'(${p_report["Z_hat"]:.1f}\\sigma$)')
    else:
        p_label = (f'p(no-ent. ${cmp}$ median) $<$ {p_report["p_UL"]:.1e}  '
                   f'($>{p_report["Z_LL"]:.1f}\\sigma$)')

    # Legend swatches as filled boxes (shaded fill + solid edge) matching the
    # histogram style, rather than bare colored lines.
    def swatch(color, label):
        return Patch(facecolor=mcolors.to_rgba(color, alpha=0.35),
                     edgecolor=color, linewidth=1.5, label=label)

    ax.legend(handles=[
        swatch(sig_color, 'With entanglement'),
        swatch(bkg_color, 'No entanglement'),
        Patch(color='none', label=p_label),
    ], loc='upper left')

    ax.set_xlim(lo, hi)
    ax.set_xlabel(var["xlabel"])
    ax.set_ylabel('Counts')

    fname = f"{args.outdir}/toy_separation_{var['name']}.pdf"
    fig.savefig(fname, bbox_inches='tight')
    print(f"  Saved {fname}")
    plt.close(fig)

def print_1sigma(label, branches, sig, bkg):
    if not branches:
        return
    print(f"\n{'─'*55}")
    print(f"  1-sigma ranges for {label}")
    print(f"  {'Branch':<8}  {'sig: median [−1σ, +1σ]':^28}  {'bkg: median [−1σ, +1σ]':^28}")
    print(f"{'─'*55}")
    for br in branches:
        for tag, vals in [('sig', sig), ('bkg', bkg)]:
            if br not in vals:
                continue
        s = sig[br]
        b = bkg[br]
        def fmt(v):
            med = np.median(v)
            lo  = np.percentile(v, 16)
            hi  = np.percentile(v, 84)
            return f"{med:+.3f} ({lo-med:+.3f} / {hi-med:+.3f})"
        print(f"  {br:<8}  {fmt(s):^28}  {fmt(b):^28}")

print_1sigma("C matrix", sig_C, sig, bkg)
print_1sigma("B vectors", sig_B, sig, bkg)
