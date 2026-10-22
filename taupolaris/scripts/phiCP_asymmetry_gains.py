"""Per-channel phiCP asymmetry gain of TauPolaris over the approximate (Run-3) methods, and its mean over channels,
from the logs/DM*_<tag>.npz histograms written by plot_phiCP.py. The asymmetry is plot_phiCP.py's:
sqrt(sum_bins (odd - even)^2) of the CP-odd and CP-even phiCP histograms, each normalised to unit sum; gain = A_nu / A_run3 - 1.
Several TauPolaris inputs can be compared against the same Run-3 reference.

    python taupolaris/scripts/phiCP_asymmetry_gains.py --run3 <phiCP dir>/logs \\
        --nu 'paper=<phiCP dir>/logs:RecoNu_Smeared' --nu 'multi200=<other phiCP dir>/logs:RecoNu_Smeared' [--gen <logs>]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def asym(path):
    # each histogram normalised to unit sum: older logs (e.g. the paper's) store densities, i.e. /bin width
    d = np.load(path)
    even, odd = d['even_counts'] / d['even_counts'].sum(), d['odd_counts'] / d['odd_counts'].sum()
    return float(np.sqrt(((odd - even) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run3', required=True, help='logs dir with DM*_RecoRun3.npz')
    ap.add_argument('--nu', action='append', required=True, help="name=<logs dir>[:tag], tag default RecoNu_Smeared")
    ap.add_argument('--gen', default=None, help='logs dir with DM*_POL_GEN.npz (optional, for reference)')
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.run3, 'DM*_RecoRun3.npz'))):
        ch = os.path.basename(f).split('_')[0]
        row = {'channel': ch, 'A run3': asym(f)}
        if a.gen and os.path.exists(os.path.join(a.gen, f'{ch}_POL_GEN.npz')):
            row['A gen'] = asym(os.path.join(a.gen, f'{ch}_POL_GEN.npz'))
        for spec in a.nu:
            name, rest = spec.split('=', 1)
            d, _, tag = rest.partition(':')
            p = os.path.join(d, f'{ch}_{tag or "RecoNu_Smeared"}.npz')
            row[f'A {name}'] = asym(p) if os.path.exists(p) else np.nan
            row[f'gain {name}'] = row[f'A {name}'] / row['A run3'] - 1
        rows.append(row)
    t = pd.DataFrame(rows)
    gains = [c for c in t.columns if c.startswith('gain ')]
    t = t.dropna(subset=gains)
    mean = {'channel': f'mean ({len(t)} ch)', **{c: t[c].mean() for c in t.columns if c != 'channel'}}
    t = pd.concat([t, pd.DataFrame([mean])], ignore_index=True)
    fmt = {c: (lambda v: f'{v:+.1%}') if c.startswith('gain') else (lambda v: f'{v:.4f}') for c in t.columns if c != 'channel'}
    print(t.to_string(index=False, formatters=fmt))
    if a.csv:
        t.to_csv(a.csv, index=False)


if __name__ == '__main__':
    main()
