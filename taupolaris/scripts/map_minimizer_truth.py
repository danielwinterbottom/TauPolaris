"""Compare MAP solutions from map_minimizer_study.py runs with the generator-level neutrinos: how often a neutrino ends
up at |p| ~ 0 (the 1/|p|^2 spike of the Cartesian density), and the neutrino momentum, energy and direction resolution.

The test file must be the one the study ran on (row-aligned, first N events); the truth is the output features
(taup_nu_n/r/k, taun_nu_n/r/k) in the same (n, r, k) frame as the MAP.

    python taupolaris/scripts/map_minimizer_truth.py --run 'grad200 k=0=<dir>/k0/grad200.npz' --run 'multi200 k=2=<dir>/k2/multi200.npz' ...
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def summary(x, t, logp):
    p1, p2 = np.linalg.norm(x[:, :3], axis=1), np.linalg.norm(x[:, 3:], axis=1)
    t1, t2 = np.linalg.norm(t[:, :3], axis=1), np.linalg.norm(t[:, 3:], axis=1)
    dp = np.linalg.norm(x[:, :3] - t[:, :3], axis=1) + np.linalg.norm(x[:, 3:] - t[:, 3:], axis=1)
    dE = np.concatenate([p1 - t1, p2 - t2])
    # direction: only where both the true and the MAP neutrino have |p| > 5 GeV
    a, b = np.concatenate([x[:, :3], x[:, 3:]]), np.concatenate([t[:, :3], t[:, 3:]])
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 5) & (nb > 5)
    ang = np.degrees(np.arccos(np.clip((a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok]), -1, 1)))
    return {'either |p|<1 GeV': ((p1 < 1) | (p2 < 1)).mean(), '|p|<0.5 GeV (per nu)': np.concatenate([p1, p2]).__lt__(0.5).mean(),
            'sum|dp| median': np.median(dp), 'sum|dp| RMS': np.sqrt((dp ** 2).mean()),
            '|dE| median': np.median(np.abs(dE)), 'dE RMS': np.sqrt((dE ** 2).mean()),
            'angle median [deg]': np.median(ang) if ok.any() else np.nan, 'mean log p (Cartesian)': np.mean(logp) if logp is not None else np.nan}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help="'label=path/to/method.npz'")
    ap.add_argument('--test_file', default='prepared_LHC_data_sep01/ppToHToTauTau_AllDM_UnCorr_Jul20/test_dataframe__leptonic_mode_0_transformer.parquet')
    ap.add_argument('--csv', default=None)
    a = ap.parse_args()
    runs = [r.rsplit('=', 1) for r in a.run]
    R = {lab: np.load(path) for lab, path in runs}
    n = min(len(d['x_best']) for d in R.values())
    names = [str(c) for c in next(iter(R.values()))['output_names']]
    t = next(pq.ParquetFile(a.test_file).iter_batches(batch_size=n, columns=names)).to_pandas()[names].to_numpy(float)
    rows = [{'method': 'truth', **summary(t, t, None)}]
    for lab, d in R.items():
        if [str(c) for c in d['output_names']] != names:
            raise SystemExit(f'{lab}: different output columns')
        logp = d['logp_cart_best'][:n] if 'logp_cart_best' in d else d['logp_starts'][:n].max(1)
        rows.append({'method': lab, **summary(d['x_best'][:n].astype(float), t, logp)})
    tab = pd.DataFrame(rows)
    pct = {'either |p|<1 GeV', '|p|<0.5 GeV (per nu)'}
    fmt = {c: (lambda v: f'{100 * v:.2f}%') if c in pct else (lambda v: f'{v:.2f}') for c in tab.columns if c != 'method'}
    print(f'=== {n} events; momenta in GeV; truth row: the generator neutrinos themselves')
    print(tab.to_string(index=False, formatters=fmt))
    if a.csv:
        tab.to_csv(a.csv, index=False)


if __name__ == '__main__':
    main()
