"""Evaluate a TransformerRegressor (train.py --useTransformerBaseline) that regresses
polarimetric vectors, neutrinos and taus in the onorm basis -- the supervised
pretraining stage for the flow's conditioner (config_polvec_hadonly_onorm_newvars_pretrain_regressor.yaml).

Reports, on the test dataset, per leg decay mode and per decay-mode pair:
  - tau momentum: relative |p| bias/IQR, lab direction error, transverse (n,r) error
  - neutrino: same, plus the internal consistency tau_pred vs nu_pred + reco visible
  - polarimetric vector: angle to truth, |h| (the regressor is not constrained to unit norm)
  - derived quantities: tau mass from nu_pred + visible (truth = 1.777 GeV by construction),
    ditau mass from the regressed taus and from nu_pred + visible, and the phiCP CP-even/odd
    asymmetry from the regressed h and taus (same metric and code path as evaluate_polvec)
Writes tables (CSV) and resolution plots to outputs_<model>/plots_eval_regressor/<test_output_name>/.

    python taupolaris/scripts/evaluate_regressor.py -c <pretrain config> [--max_events N] [--useCPU]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from taupolaris.python.DataProcessing import get_test_dataset, _resolve_polvec_feature_level, _resolve_frame_aligned_level
from taupolaris.python.NN_Tools import load_model, get_device
from taupolaris.utils.coordinate_conversions import ConvertFromOrthonormalNRK_Predictions_PolVec
from taupolaris.scripts.evaluate_polvec import (add_DM, phicp_from_cart, normalised_phicp_counts, plot_true_vs_pred_2d,
                                                asymmetry_quadrature, print_asymmetry_table, plot_phiCP_cp_comparison,
                                                phicp_from_regressed_taus, columns_for_regressed_tau_phicp,
                                                PHICP_TABLE_HADRONIC, PHICP_BINS, M_TAU)

LEG_DMS = (0, 1, 2, 10, 11, 100)     # 100 = leptonic leg (semileptonic / all-channel models)
DM_PAIRS = [k for k, _ in PHICP_TABLE_HADRONIC]


def to_cartesian(native, df, tau_labels, h_col, mom_col):
    """Inverse onorm transform for a [h(6), momentum(6)] block; the momentum can be the
    neutrino or the tau (same basis transform). h_col/mom_col map (tau, component) to the
    native column name. Returns (h_t1, h_t2, p_t1, p_t2) as (N,3) lab-frame arrays."""
    t1, t2 = tau_labels
    cols = [h_col(t, c) for t in (t1, t2) for c in 'nrk'] + [mom_col(t, c) for t in (t1, t2) for c in 'nrk']
    vis = [df[[f'reco_{t}_{part}_p{c}' for c in 'xyz']].values for t in (t1, t2) for part in ('charged', 'pizero1')]
    cart = ConvertFromOrthonormalNRK_Predictions_PolVec(native[cols].values, *vis)
    return cart[:, 0:3], cart[:, 3:6], cart[:, 6:9], cart[:, 9:12]


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=1), -1, 1)))


def iqr(x):
    return float(np.subtract(*np.percentile(x, [75, 25])))


def mass(E, p):
    return np.sqrt(np.clip(E ** 2 - np.sum(p ** 2, axis=1), 0, None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c', required=True)
    ap.add_argument('--max_events', type=int, default=0, help='<= 0 means all')
    ap.add_argument('--useCPU', action='store_true')
    ap.add_argument('--batch_size', type=int, default=8192)
    ap.add_argument('--checkpoint', default=None, help='weights to evaluate (default: <outdir>/plots/best_model.pth). '
                    'Use a copy when the training is still running, so the file is not overwritten mid-read.')
    ap.add_argument('--tag', default='', help='suffix for the output subdirectory, e.g. the epoch the checkpoint came from')
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    data_config, nn_config = config['Data'], config['SetupNN']
    device = torch.device('cpu') if args.useCPU else get_device()
    output_dir = f"outputs_{nn_config['model_name']}"
    coordinates = data_config['coordinates']
    assert coordinates == 'onorm', 'this script expects coordinates: onorm'
    output_features = data_config['Features']['output_features'][coordinates]
    leptonic_mode = data_config.get('leptonic_mode', -1)
    tau_labels = ('tau1', 'tau2') if leptonic_mode == 1 else ('taup', 'taun')
    t1, t2 = tau_labels
    have_h = all(f'ts_hh_{t}_{c}' in output_features for t in tau_labels for c in 'nrk')
    have_nu = all(f'{t}_nu_{c}' in output_features for t in tau_labels for c in 'nrk')
    have_tau = all(f'undecayed_{t}_{c}' in output_features for t in tau_labels for c in 'nrk')
    print(f'>> outputs: h={have_h} nu={have_nu} tau={have_tau} ({len(output_features)} columns)')

    norm_data = np.load(f'{output_dir}/normalization_params.npz')
    hp = nn_config['TransformerBaseline_hyperparams']
    model = load_model(hp, data_config['Features']['input_features'], output_features, useTransformerMLP=True,
                       leptonic_mode=leptonic_mode,
                       polvec_feature_level=_resolve_polvec_feature_level(data_config),
                       frame_aligned_level=_resolve_frame_aligned_level(data_config))
    ckpt = args.checkpoint or os.path.join(output_dir, 'plots', 'best_model.pth')
    if not os.path.exists(ckpt):
        ckpt = os.path.join(output_dir, f"{nn_config['model_name']}.pth")
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.to(device).eval()
    print(f'>> loaded {ckpt}')

    test_datasets = data_config['test_dataset'] if isinstance(data_config['test_dataset'], list) else [data_config['test_dataset']]
    test_names = data_config['test_output_name'] if isinstance(data_config['test_output_name'], list) else [data_config['test_output_name']]
    for test_path, test_name in zip(test_datasets, test_names):
        print(f'>> evaluating {test_path}')
        data_config['test_dataset'] = test_path
        available = pq.ParquetFile(test_path).schema_arrow.names
        wanted = set(data_config['Features']['input_features']) | set(output_features)
        for t in tau_labels:
            wanted |= {f'reco_{t}_{part}_{c}' for part in ('charged', 'pizero1') for c in ('px', 'py', 'pz', 'e')}
            wanted |= {f'undecayed_{t}_{c}' for c in ('px', 'py', 'pz', 'e')}
            wanted |= {f'{t}_nu_{c}' for c in ('px', 'py', 'pz')}
            wanted |= {f'ts_hh_{t}_{c}' for c in 'xyz'}
            for pre in ('', 'reco_'):
                wanted |= {f'{pre}{t}_{f}' for f in ('ishadronic', 'npizero', 'is3prong')}
        wanted |= {'tauspinner_wt_alpha0', 'tauspinner_wt_alpha45', 'tauspinner_wt_alpha90'}
        wanted |= columns_for_regressed_tau_phicp(tau_labels)   # reco pion four-vectors for the nu + visible -> analytic h route
        dataset, df, _, _ = get_test_dataset(data_config, norm_data, columns=sorted(wanted & set(available)))
        if args.max_events > 0 and len(df) > args.max_events:
            df = df.iloc[:args.max_events].reset_index(drop=True)
        n = len(df)
        X, _ = dataset[:]
        X = X[:n]
        preds = []
        with torch.no_grad():
            for s in range(0, n, args.batch_size):
                preds.append(model(X[s:s + args.batch_size].to(device)).cpu())
        native = pd.DataFrame(dataset.destandardize_outputs(torch.cat(preds)).numpy(), columns=output_features)
        outdir = os.path.join(output_dir, 'plots_eval_regressor', test_name + (f'_{args.tag}' if args.tag else ''))
        os.makedirs(outdir, exist_ok=True)

        # ---- truth (Cartesian, lab)
        true_p = {t: df[[f'undecayed_{t}_p{c}' for c in 'xyz']].values.astype(float) for t in tau_labels}
        true_E = {t: df[f'undecayed_{t}_e'].values.astype(float) for t in tau_labels}
        true_nu = {t: df[[f'{t}_nu_p{c}' for c in 'xyz']].values.astype(float) for t in tau_labels}
        true_h = {t: unit(df[[f'ts_hh_{t}_{c}' for c in 'xyz']].values.astype(float)) for t in tau_labels}
        vis_p = {t: (df[[f'reco_{t}_charged_p{c}' for c in 'xyz']].values + df[[f'reco_{t}_pizero1_p{c}' for c in 'xyz']].values).astype(float) for t in tau_labels}
        vis_E = {t: (df[f'reco_{t}_charged_e'].values + df[f'reco_{t}_pizero1_e'].values).astype(float) for t in tau_labels}
        dm = dict(zip(tau_labels, add_DM(df, tau_labels, dm_prefix='reco')))
        w_e, w_o = df['tauspinner_wt_alpha0'].values, df['tauspinner_wt_alpha90'].values

        # ---- predictions (Cartesian, lab)
        pred_h, pred_tau, pred_nu = {}, {}, {}
        h_col = lambda t, c: f'ts_hh_{t}_{c}'
        if have_h and have_tau:
            h1, h2, p1, p2 = to_cartesian(native, df, tau_labels, h_col, lambda t, c: f'undecayed_{t}_{c}')
            pred_h, pred_tau = {t1: h1, t2: h2}, {t1: p1, t2: p2}
        if have_h and have_nu:
            h1, h2, p1, p2 = to_cartesian(native, df, tau_labels, h_col, lambda t, c: f'{t}_nu_{c}')
            pred_h = pred_h or {t1: h1, t2: h2}
            pred_nu = {t1: p1, t2: p2}
        tau_from_nu = {t: pred_nu[t] + vis_p[t] for t in pred_nu}
        E_from_nu = {t: np.linalg.norm(pred_nu[t], axis=1) + vis_E[t] for t in pred_nu}

        # ---- per-leg table
        rows = []
        for d in LEG_DMS:
            r = {'leg DM': d}
            rel, ang, trans, nurel, nuang, hang, hnorm, mtau, cons = ([] for _ in range(9))
            for t in tau_labels:
                m = dm[t] == d
                if m.sum() == 0:
                    continue
                pt = true_p[t][m]
                if pred_tau:
                    pp = pred_tau[t][m]
                    rel.append((np.linalg.norm(pp, axis=1) - np.linalg.norm(pt, axis=1)) / np.linalg.norm(pt, axis=1))
                    ang.append(angle_deg(pp, pt))
                    k = unit(vis_p[t][m]); dv = pp - pt
                    trans.append(np.linalg.norm(dv - np.sum(dv * k, axis=1, keepdims=True) * k, axis=1))
                if pred_nu:
                    nt, npred = true_nu[t][m], pred_nu[t][m]
                    nurel.append((np.linalg.norm(npred, axis=1) - np.linalg.norm(nt, axis=1)) / np.maximum(np.linalg.norm(nt, axis=1), 1e-6))
                    nuang.append(angle_deg(npred, nt))
                    mtau.append(mass(E_from_nu[t][m], tau_from_nu[t][m]))
                    if pred_tau:
                        cons.append(np.linalg.norm(pred_tau[t][m] - tau_from_nu[t][m], axis=1))
                if pred_h:
                    hang.append(angle_deg(pred_h[t][m], true_h[t][m]))
                    hnorm.append(np.linalg.norm(pred_h[t][m], axis=1))
            r['N_legs'] = int(sum(len(x) for x in (rel or nurel or hang)))
            if rel:
                rel, ang, trans = map(np.concatenate, (rel, ang, trans))
                r.update({'tau |p| rel bias': np.median(rel), 'tau |p| rel IQR': iqr(rel),
                          'tau dir err med [deg]': np.median(ang), 'tau dir err p90 [deg]': np.percentile(ang, 90),
                          'tau transverse err med [GeV]': np.median(trans)})
            if nurel:
                nurel, nuang, mtau = map(np.concatenate, (nurel, nuang, mtau))
                r.update({'nu |p| rel IQR': iqr(nurel), 'nu dir err med [deg]': np.median(nuang),
                          'm(nu_pred+vis) med [GeV]': np.median(mtau), 'm(nu_pred+vis) IQR [GeV]': iqr(mtau)})
                if cons:
                    r['|tau_pred - (nu_pred+vis)| med [GeV]'] = np.median(np.concatenate(cons))
            if hang:
                hang, hnorm = map(np.concatenate, (hang, hnorm))
                r.update({'h angle med [deg]': np.median(hang), 'h <cos>': float(np.mean(np.cos(np.radians(hang)))),
                          '|h| med': np.median(hnorm), '|h| IQR': iqr(hnorm)})
            rows.append(r)
        leg = pd.DataFrame(rows)
        pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60)
        print('\n=== per-leg regression quality (test sample) ===')
        fmt = lambda v: f'{v:.4f}'
        blocks = [[c for c in leg.columns if c.startswith('tau ') or c in ('leg DM', 'N_legs')],
                  [c for c in leg.columns if c.startswith(('nu ', 'm(nu', '|tau_pred')) or c == 'leg DM'],
                  [c for c in leg.columns if c.startswith(('h ', '|h|')) or c == 'leg DM']]
        for cols in blocks:
            if len(cols) > 1:
                print(leg[cols].to_string(index=False, float_format=fmt)); print()
        leg.to_csv(os.path.join(outdir, 'per_leg_dm.csv'), index=False)

        # ---- event-level derived quantities: ditau mass, phiCP
        E_t = {t: np.sqrt(np.sum(pred_tau[t] ** 2, axis=1) + M_TAU ** 2) for t in pred_tau}
        m_true = mass(true_E[t1] + true_E[t2], true_p[t1] + true_p[t2])
        series = {}
        if pred_tau:
            series['m(tau_pred, tau_pred)'] = mass(E_t[t1] + E_t[t2], pred_tau[t1] + pred_tau[t2])
        if pred_nu:
            series['m(nu_pred+vis, nu_pred+vis)'] = mass(E_from_nu[t1] + E_from_nu[t2], tau_from_nu[t1] + tau_from_nu[t2])
        print('\n=== ditau mass [GeV] (truth median %.2f) ===' % np.median(m_true))
        for k, v in series.items():
            rel = (v - m_true) / m_true
            print(f'  {k:32s} median {np.median(v):7.2f}  rel bias {np.median(rel):+.4f}  rel IQR {iqr(rel):.4f}')
        fig, ax = plt.subplots(figsize=(6.5, 5)); bins = np.linspace(0, 250, 101)
        for k, v in series.items():
            ax.hist(v, bins=bins, histtype='step', lw=1.4, label=k)
        ax.set_xlabel('ditau mass [GeV]'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'ditau_mass.pdf')); plt.close(fig)
        if pred_nu:
            fig, ax = plt.subplots(figsize=(6.5, 5)); bins = np.linspace(0, 4, 101)
            for d in LEG_DMS:
                vals = np.concatenate([mass(E_from_nu[t][dm[t] == d], tau_from_nu[t][dm[t] == d]) for t in tau_labels])
                ax.hist(vals, bins=bins, histtype='step', lw=1.3, density=True, label=f'DM{d}')
            ax.axvline(M_TAU, color='k', ls='--', lw=1); ax.set_xlabel('m(nu_pred + reco visible) [GeV]'); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(outdir, 'tau_mass_from_nu.pdf')); plt.close(fig)

        # ---- phiCP: the same plot suite evaluate_polvec produces for the flows
        # ("All events" + every decay-mode pair, CP-even vs CP-odd via the TauSpinner weights,
        # asymmetry table in paper order), for two constructions:
        #   phiCP            regressed h + regressed taus (the direct-method observable)
        #   phiCP_nuvis      neutrino + reco visible -> tau, analytic h (the nu-method route,
        #                    get_ditau_polarimetric, fed the regressor's neutrinos)
        true_cart = pd.DataFrame(np.concatenate([true_h[t1], true_h[t2], true_p[t1], true_p[t2]], axis=1),
                                 columns=[f'ts_hh_{t}_{c}' for t in tau_labels for c in 'xyz'] + [f'undecayed_{t}_p{c}' for t in tau_labels for c in 'xyz'])
        true_phi = phicp_from_cart(true_cart, true_E[t1], true_E[t2], tau_labels)
        variants = {}
        if pred_h and pred_tau:
            E_t = {t: np.sqrt(np.sum(pred_tau[t] ** 2, axis=1) + M_TAU ** 2) for t in pred_tau}
            pred_cart = pd.DataFrame(np.concatenate([unit(pred_h[t1]), unit(pred_h[t2]), pred_tau[t1], pred_tau[t2]], axis=1), columns=true_cart.columns)
            variants['phiCP'] = phicp_from_cart(pred_cart, E_t[t1], E_t[t2], tau_labels)
        if pred_nu:
            phi_nuvis, _, _ = phicp_from_regressed_taus(df, tau_from_nu[t1], tau_from_nu[t2], E_from_nu[t1], E_from_nu[t2],
                                                        dm[t1], dm[t2], tau_labels)
            variants['phiCP_nuvis'] = phi_nuvis
        dm_pairs = [[100, 0], [100, 1], [100, 2], [100, 10], [100, 11], [100, 100],
                    [0, 0], [0, 1], [1, 1], [2, 2], [1, 2], [0, 2], [10, 10], [0, 10], [1, 10], [2, 10],
                    [0, 11], [1, 11], [2, 11], [10, 11], [11, 11]]
        edges = np.linspace(0, 2 * np.pi, PHICP_BINS + 1)
        for stem, phi in variants.items():
            plot_phiCP_cp_comparison(true_phi, phi, w_e, w_o, f'All events ({stem})', os.path.join(outdir, f'{stem}.pdf'), PHICP_BINS)
            dm_dir = os.path.join(outdir, f'{stem}_by_dm'); os.makedirs(dm_dir, exist_ok=True)
            asym, rows = {}, []
            for a, b in [('all', 'all')] + dm_pairs:
                m = np.ones(n, bool) if a == 'all' else (((dm[t1] == a) & (dm[t2] == b)) | ((dm[t1] == b) & (dm[t2] == a)))
                if m.sum() < 20:
                    continue
                if a == 'all':
                    at = asymmetry_quadrature(normalised_phicp_counts(true_phi[m], w_e[m], edges), normalised_phicp_counts(true_phi[m], w_o[m], edges))
                    ap = asymmetry_quadrature(normalised_phicp_counts(phi[m], w_e[m], edges), normalised_phicp_counts(phi[m], w_o[m], edges))
                else:
                    at, ap = plot_phiCP_cp_comparison(true_phi[m], phi[m], w_e[m], w_o[m], f'DM{a} - DM{b} ({m.sum()} events)',
                                                      os.path.join(dm_dir, f'{stem}_DM{a}_DM{b}.pdf'), PHICP_BINS)
                    asym[tuple(sorted((a, b)))] = ap
                dphi = (phi[m] - true_phi[m] + np.pi) % (2 * np.pi) - np.pi
                rows.append({'pair': f'{a}-{b}', 'N': int(m.sum()), 'asym true': at, 'asym pred': ap, '<cos dphiCP>': float(np.mean(np.cos(dphi)))})
            tab = pd.DataFrame(rows)
            print(f'\n=== {stem}: ' + ('regressed h + regressed taus' if stem == 'phiCP' else 'neutrino + visible -> tau, analytic h') + ' ===')
            print(tab.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
            tab.to_csv(os.path.join(outdir, f'{stem}_by_pair.csv'), index=False)
            print_asymmetry_table(asym, f'{stem}: predicted asymmetry by decay mode')
        if len(variants) == 2:
            a, b = variants['phiCP'], variants['phiCP_nuvis']
            print(f"\n   per-event agreement phiCP vs phiCP_nuvis: <cos(diff)> = {np.mean(np.cos(a - b)):.4f}")

        # ---- per-event results, for downstream scripts (weighted_significance/, compare_polvec_methods)
        res = {'true_phiCP': true_phi, 'reco_taup_DM': dm[t1], 'reco_taun_DM': dm[t2],
               'tauspinner_wt_alpha0': w_e, 'tauspinner_wt_alpha90': w_o}
        for stem, phi in variants.items():
            res[f'pred_{stem}'] = phi
        for t, lab in zip(tau_labels, ('plus', 'minus')):
            for i, c in enumerate('xyz'):
                res[f'true_tau_{lab}_p{c}'] = true_p[t][:, i]
                if pred_tau: res[f'pred_tau_{lab}_p{c}'] = pred_tau[t][:, i]
                if pred_nu: res[f'pred_nu_{lab}_p{c}'] = pred_nu[t][:, i]
                if pred_h: res[f'pred_ts_hh_{t}_{c}'] = pred_h[t][:, i]
                res[f'true_ts_hh_{t}_{c}'] = true_h[t][:, i]
        if 'tauspinner_wt_alpha45' in df.columns:
            res['tauspinner_wt_alpha45'] = df['tauspinner_wt_alpha45'].values
        res_path = os.path.join(output_dir, f'regressor_eval_results_{test_name}' + (f'_{args.tag}' if args.tag else '') + '.parquet')
        pd.DataFrame(res).to_parquet(res_path, index=False)
        print(f'>> per-event results ({n} rows) saved to {res_path}')

        # ---- one resolution plot per NATIVE output (what the MSE is computed on):
        # true-vs-pred 2D panel + residual histogram split by the leg's decay mode
        native_dir = os.path.join(outdir, 'native_outputs')
        os.makedirs(native_dir, exist_ok=True)
        summary = []
        for col in output_features:
            tv, pv = df[col].values.astype(float), native[col].values.astype(float)
            leg = next((t for t in tau_labels if f'{t}_' in col), None)
            res = pv - tv
            lim = np.percentile(np.abs(tv), 99.5)
            rlim = 4 * np.percentile(np.abs(res), 68)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
            axes[0].hist2d(tv, pv, bins=80, range=[[-lim, lim], [-lim, lim]] if tv.min() < 0 else [[0, lim], [0, lim]], cmin=1)
            lo, hi = axes[0].get_xlim(); axes[0].plot([lo, hi], [lo, hi], 'r--', lw=1)
            axes[0].set_xlabel(f'true {col}'); axes[0].set_ylabel(f'pred {col}')
            bins = np.linspace(-rlim, rlim, 101)
            axes[1].hist(res, bins=bins, histtype='step', lw=1.6, color='k', density=True, label=f'all (med {np.median(res):+.3g}, IQR {iqr(res):.3g})')
            if leg is not None:
                for d in LEG_DMS:
                    m = dm[leg] == d
                    if m.sum() > 50:
                        axes[1].hist(res[m], bins=bins, histtype='step', lw=1.1, density=True, label=f'DM{d} (IQR {iqr(res[m]):.3g})')
            axes[1].set_xlabel(f'pred - true  [{col}]'); axes[1].legend(fontsize=7)
            fig.suptitle(col); fig.tight_layout(); fig.savefig(os.path.join(native_dir, f'{col}.pdf'), dpi=120); plt.close(fig)
            row = {'output': col, 'bias (median)': np.median(res), 'IQR': iqr(res), 'RMS': float(np.sqrt(np.mean(res ** 2))),
                   'corr': float(np.corrcoef(tv, pv)[0, 1])}
            if leg is not None:
                for d in LEG_DMS:
                    m = dm[leg] == d
                    row[f'IQR DM{d}'] = iqr(res[m]) if m.sum() > 50 else np.nan
            summary.append(row)
        summary = pd.DataFrame(summary)
        print('\n=== per-output residuals in native (onorm) units, pred - true ===')
        print(summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
        summary.to_csv(os.path.join(outdir, 'native_output_residuals.csv'), index=False)
        # all residuals on one page for a quick look
        ncol = 6; nrow = int(np.ceil(len(output_features) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow))
        for ax, col in zip(axes.ravel(), output_features):
            res = native[col].values - df[col].values; rlim = 4 * np.percentile(np.abs(res), 68)
            ax.hist(res, bins=np.linspace(-rlim, rlim, 81), histtype='step', color='k', lw=1.2)
            ax.set_title(col, fontsize=8); ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(output_features):]:
            ax.axis('off')
        fig.suptitle('pred - true, native onorm outputs'); fig.tight_layout(); fig.savefig(os.path.join(outdir, 'native_residuals_overview.pdf'), dpi=120); plt.close(fig)

        # ---- Cartesian quantities, true vs pred (same helper/style as evaluate_polvec)
        for t in tau_labels:
            if pred_tau:
                E_pred = np.sqrt(np.sum(pred_tau[t] ** 2, axis=1) + M_TAU ** 2)
                plot_true_vs_pred_2d([true_p[t][:, 0], true_p[t][:, 1], true_p[t][:, 2], true_E[t]],
                                     [pred_tau[t][:, 0], pred_tau[t][:, 1], pred_tau[t][:, 2], E_pred],
                                     [f'{t}_px', f'{t}_py', f'{t}_pz', f'{t}_E'], f'Tau 4-vector: {t}',
                                     os.path.join(outdir, f'tau4vec_{t}.pdf'), 60, sym_range='auto')
            if pred_nu:
                plot_true_vs_pred_2d([true_nu[t][:, 0], true_nu[t][:, 1], true_nu[t][:, 2], np.linalg.norm(true_nu[t], axis=1)],
                                     [pred_nu[t][:, 0], pred_nu[t][:, 1], pred_nu[t][:, 2], np.linalg.norm(pred_nu[t], axis=1)],
                                     [f'{t}_nu_px', f'{t}_nu_py', f'{t}_nu_pz', f'{t}_nu_|p|'], f'Neutrino: {t}',
                                     os.path.join(outdir, f'nu3vec_{t}.pdf'), 60, sym_range='auto')
            if pred_h:
                plot_true_vs_pred_2d([true_h[t][:, i] for i in range(3)], [unit(pred_h[t])[:, i] for i in range(3)],
                                     [f'h_{t}_{c}' for c in 'xyz'], f'Polarimetric vector (unit-normalised): {t}',
                                     os.path.join(outdir, f'polvec_{t}.pdf'), 60, sym_range=(-1, 1))

        # ---- resolution plots per leg DM
        if pred_tau:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for d in LEG_DMS:
                sel = [(t, dm[t] == d) for t in tau_labels]
                rel = np.concatenate([(np.linalg.norm(pred_tau[t][m], axis=1) - np.linalg.norm(true_p[t][m], axis=1)) / np.linalg.norm(true_p[t][m], axis=1) for t, m in sel])
                ang = np.concatenate([angle_deg(pred_tau[t][m], true_p[t][m]) for t, m in sel])
                axes[0].hist(rel, bins=np.linspace(-1, 1, 101), histtype='step', density=True, label=f'DM{d}')
                axes[1].hist(ang, bins=np.linspace(0, 2, 101), histtype='step', density=True, label=f'DM{d}')
                if pred_h:
                    axes[2].hist(np.concatenate([angle_deg(pred_h[t][m], true_h[t][m]) for t, m in sel]), bins=np.linspace(0, 180, 91), histtype='step', density=True, label=f'DM{d}')
            axes[0].set_xlabel('(|p|_pred - |p|_true)/|p|_true'); axes[1].set_xlabel('tau direction error [deg]'); axes[2].set_xlabel('angle(h_pred, h_true) [deg]')
            for ax in axes: ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(outdir, 'resolutions_by_dm.pdf')); plt.close(fig)
        print(f'>> tables and plots in {outdir}')


if __name__ == '__main__':
    main()
