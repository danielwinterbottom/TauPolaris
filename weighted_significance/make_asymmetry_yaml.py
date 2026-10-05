"""Write an asymmetry yaml (the input of total_asymmetry.py) from a per-event results
parquet, using exactly plot_phiCP.py's convention so the numbers are comparable with
asymmetry_had_Polaris.yaml / asymmetry_lep_Polaris.yaml:

  * one entry per ORDERED decay-mode combination (tau+ DM, tau- DM) from plot_phiCP's
    dm_combs list -- mixed pairs therefore count only the (p, n) ordering, not (n, p)
  * 20 phiCP bins on [0, 2pi], CP-even/CP-odd from the TauSpinner alpha=0/90 weights,
    each histogram normalised to unit sum (weights / sum(weights), density=False)
  * asymmetry = sqrt( sum_bins (odd - even)^2 ),  N = number of events in the combination

Works on an evaluate_polvec.py output (pred_phiCP, reco_taup_DM, ...) or an
evaluate_regressor.py output (pred_phiCP / pred_phiCP_nuvis). Values are written as plain
floats, so the file loads with yaml.safe_load.

    python weighted_significance/make_asymmetry_yaml.py --results <parquet> --column pred_phiCP \\
        --out weighted_significance/asymmetry_had_<label>.yaml [--leptonic]
"""
import argparse
import numpy as np
import pandas as pd
import yaml

HADRONIC = [[0, 0], [0, 1], [1, 1], [2, 2], [1, 2], [0, 2], [10, 10], [0, 10], [1, 10], [2, 10],
            [0, 11], [1, 11], [2, 11], [10, 11], [11, 11]]
LEPTONIC = [[100, 0], [100, 1], [100, 2], [100, 10], [100, 11]]


def asymmetry(phi, w_even, w_odd, nbins=20):
    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    e = np.histogram(phi, bins=edges, weights=w_even / w_even.sum())[0]
    o = np.histogram(phi, bins=edges, weights=w_odd / w_odd.sum())[0]
    return float(np.sqrt(np.sum((o - e) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True)
    ap.add_argument('--column', default='pred_phiCP')
    ap.add_argument('--out', required=True)
    ap.add_argument('--leptonic', action='store_true', help='use the tau_l combinations instead of the hadronic ones')
    ap.add_argument('--dm_prefix', default='reco', help="'reco' -> reco_taup_DM / reco_taun_DM columns (default), 'gen' -> gen_*")
    ap.add_argument('--min_events', type=int, default=20)
    args = ap.parse_args()

    cols = [args.column, f'{args.dm_prefix}_taup_DM', f'{args.dm_prefix}_taun_DM', 'tauspinner_wt_alpha0', 'tauspinner_wt_alpha90']
    df = pd.read_parquet(args.results, columns=cols)
    phi = df[args.column].to_numpy(dtype=float)
    dm_p, dm_n = df[cols[1]].to_numpy(), df[cols[2]].to_numpy()
    w_e, w_o = df['tauspinner_wt_alpha0'].to_numpy(dtype=float), df['tauspinner_wt_alpha90'].to_numpy(dtype=float)
    ok = np.isfinite(phi)
    out = {}
    for p, n in (LEPTONIC if args.leptonic else HADRONIC):
        m = ok & (dm_p == p) & (dm_n == n)          # ordered, as in plot_phiCP.py
        if m.sum() < args.min_events:
            continue
        out[f'DM{p}DM{n}'] = {'N': int(m.sum()), 'asymmetry': asymmetry(phi[m], w_e[m], w_o[m])}
        print(f"DM{p}DM{n}: N={m.sum()}, asymmetry={out[f'DM{p}DM{n}']['asymmetry']:.4f}")
    with open(args.out, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=True)
    print(f'>> wrote {args.out} ({len(out)} combinations, {sum(v["N"] for v in out.values())} events)')


if __name__ == '__main__':
    main()
