"""Diagrams for why the Cartesian MAP of a neutrino momentum is pulled onto p = 0 (the 'bowl' / 1/|p|^n spike).

fig_bowl_toy:   a one-event toy. The tau-mass constraint m_tau^2 - m_vis^2 ~ E_vis E_nu theta^2 puts the neutrino on a
                paraboloid around the visible axis k, p_T ~ sqrt(E_nu); with a flat E_nu distribution the same probability
                per GeV is squeezed into a shrinking cross-section towards the apex, so the density per unit volume
                grows there even though the probability does not.
fig_flow_event: the trained flow's posterior for one study event where multi-start jumps to p ~ 0: samples in the
                (k, p_T) plane with the truth and the grad200 / multi200 MAPs, the |p| distribution of the samples, and
                the flow log-density along the line between the two MAPs for k = 0, 1, 2 (+k log|p| per neutrino).

    python taupolaris/scripts/plot_map_spike_diagrams.py --outdir <dir> [--event 101 --nu 0]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'taupolaris', 'scripts'))

BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e5e4e0'
M_TAU = 1.777


def style(ax):
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


def toy_event(n=200_000, e_vis=30.0, m_vis=0.14, e_max=60.0, smear=0.25, seed=1):
    """Neutrino momenta for one toy event: E_nu flat in [0, e_max], direction from the tau-mass constraint (exact
    massless 2-body relation for the opening angle), uniform azimuth, plus Gaussian transverse smearing."""
    rng = np.random.default_rng(seed)
    p_vis = np.sqrt(e_vis ** 2 - m_vis ** 2)
    M = M_TAU ** 2 - m_vis ** 2
    e = rng.uniform(M / (2 * (e_vis + p_vis)), e_max, n)
    cos = np.clip((e_vis - M / (2 * e)) / p_vis, -1, 1)
    pt, pk = e * np.sqrt(1 - cos ** 2), e * cos
    phi = rng.uniform(0, 2 * np.pi, n)
    pn, pr = pt * np.cos(phi) + rng.normal(0, smear, n), pt * np.sin(phi) + rng.normal(0, smear, n)
    return pn, pr, pk, e, np.sqrt(M / e_vis)


def fig_bowl_toy(outdir):
    pn, pr, pk, e, c = toy_event()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={'width_ratios': [1.35, 1.0, 1.1]})

    # (a) side view: n vs k, a thin slice |r| < 0.15 GeV through the axis
    ax = axs[0]; s = np.abs(pr) < 0.15
    ax.scatter(pk[s][:6000], pn[s][:6000], s=2, color=BLUE, alpha=0.35, lw=0, rasterized=True)
    kk = np.linspace(0, 40, 400)
    for sgn in (1, -1):
        ax.plot(kk, sgn * c * np.sqrt(kk), color=INK, lw=1.2, ls='--')
    ax.annotate('apex: p = 0\nhighest density,\nalmost no probability', xy=(0, 0), xytext=(6, -3.2), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=INK, lw=0.8))
    ax.text(26, 2.35, r'$p_T \approx \sqrt{(m_\tau^2-m_{vis}^2)\,E_\nu / E_{vis}}$', color=INK, fontsize=10)
    ax.set_xlim(-1, 40); ax.set_ylim(-4, 4)
    ax.set_xlabel(r'$p_k$ [GeV]  (along the visible $\tau$)', color=INK); ax.set_ylabel(r'$p_n$ [GeV]', color=INK)
    ax.set_title('(a) side view of the allowed neutrino momenta', fontsize=11, color=INK, loc='left')
    style(ax)

    # (b) end-on views at three energies, the same number of points (= same probability) in each
    ax = axs[1]
    for (lo, hi), col, off in (((0.8, 1.2), ORANGE, 0.0), ((4.6, 5.4), AQUA, 0.0), ((19, 21), VIOLET, 0.0)):
        s = (e > lo) & (e < hi)
        idx = np.where(s)[0][:700]
        ax.scatter(pn[idx], pr[idx], s=4, color=col, alpha=0.5, lw=0, label=f'$E_\\nu$ = {0.5 * (lo + hi):g} GeV')
    ax.set_aspect('equal'); ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel(r'$p_n$ [GeV]', color=INK); ax.set_ylabel(r'$p_r$ [GeV]', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper right', labelcolor=INK, markerscale=3)
    ax.set_title('(b) end-on: same probability per slice', fontsize=11, color=INK, loc='left')
    style(ax)

    # (c) probability vs density along the bowl
    ax = axs[2]
    edges = np.logspace(-1.3, np.log10(60), 30); mid = np.sqrt(edges[1:] * edges[:-1])
    h, _ = np.histogram(e, edges)
    prob = h / len(e) / np.diff(edges)                                  # probability per GeV of |p|
    # occupied volume per |p| bin: annulus of radius p_T(E) and the smearing width, times the bin length
    pt_mid = c * np.sqrt(mid); w = 0.25
    area = np.pi * ((pt_mid + 2 * w) ** 2 - np.clip(pt_mid - 2 * w, 0, None) ** 2)
    dens = prob / area
    ax.plot(mid, prob / prob[-5:].mean(), color=BLUE, lw=2, label='probability per GeV of |p|  (flat)')
    ax.plot(mid, dens / dens[-5:].mean(), color=ORANGE, lw=2, label='density per unit volume  (Cartesian)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$|p_\nu|$ [GeV]', color=INK); ax.set_ylabel('relative to its value at large |p|', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper right', labelcolor=INK)
    ax.set_title('(c) the density rises towards p = 0', fontsize=11, color=INK, loc='left')
    style(ax)

    fig.suptitle(r'Toy event: $\tau\to\pi\nu$, $E_{vis}$ = 30 GeV, $E_\nu$ flat in 0-60 GeV, 0.25 GeV transverse smearing',
                 fontsize=11, color=MUTED, x=0.01, ha='left')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_bowl_toy.{ext}'), dpi=140)
    plt.close(fig)


def fig_flow_event(outdir, study, test_file, cfg, event, nu, n_samples):
    from map_minimizer_study import load
    model_path = os.path.join(os.path.dirname(cfg), 'plots', 'best_model.pth')
    model, X, norm, out = load(cfg, model_path, test_file, event + 1)
    mu, sd = norm['output_mean'].reshape(-1), norm['output_std'].reshape(-1)
    torch.manual_seed(3)
    with torch.no_grad():
        xs, lps = model.sample_and_log_prob(num_samples=n_samples, context=X[event:event + 1])
    xs = xs[0].numpy() * sd + mu; lps = lps[0].numpy()
    g, m = np.load(os.path.join(study, 'grad200.npz')), np.load(os.path.join(study, 'multi200.npz'))
    t = next(pq.ParquetFile(test_file).iter_batches(batch_size=event + 1, columns=out)).to_pandas()[out].to_numpy()[event]
    sl = slice(3 * nu, 3 * nu + 3); other = slice(3 * (1 - nu), 3 * (1 - nu) + 3)
    pts = {'truth': t, 'grad200 MAP': g['x_best'][event], 'multi200 MAP': m['x_best'][event]}
    lp_map = {'grad200 MAP': g['logp_starts'][event, 0], 'multi200 MAP': m['logp_starts'][event].max()}
    mark = {'truth': dict(marker='*', s=220, color=INK), 'grad200 MAP': dict(marker='o', s=90, color=AQUA),
            'multi200 MAP': dict(marker='D', s=80, color=ORANGE)}
    leg = 'tau+' if nu == 0 else 'tau-'

    pk, pT = xs[:, sl][:, 2], np.hypot(xs[:, sl][:, 0], xs[:, sl][:, 1])
    pabs = np.linalg.norm(xs[:, sl], axis=1)
    pabs_o = np.linalg.norm(xs[:, other], axis=1)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axs[0]
    ax.scatter(pk, pT, s=3, color=BLUE, alpha=0.25, lw=0, rasterized=True, label=f'{n_samples} flow samples')
    for k, v in pts.items():
        ax.scatter(v[sl][2], np.hypot(v[sl][0], v[sl][1]), edgecolor='white', lw=1.0, zorder=5, label=k, **mark[k])
    ax.set_xlim(-2, np.quantile(pk, 0.995) * 1.05); ax.set_ylim(0, np.quantile(pT, 0.995) * 1.1)
    ax.set_xlabel(r'$p_k$ [GeV]  (along the visible $\tau$)', color=INK); ax.set_ylabel(r'$p_T=\sqrt{p_n^2+p_r^2}$ [GeV]', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK, markerscale=1)
    ax.set_title(f'(a) flow posterior, {leg} neutrino, event {event}', fontsize=11, color=INK, loc='left')
    style(ax)

    ax = axs[1]
    h, edges = np.histogram(pabs, np.linspace(0, np.quantile(pabs, 0.995), 50))
    ax.stairs(h / h.sum() / np.diff(edges), edges, color=BLUE, lw=2, label='flow samples')
    for k in pts:
        ax.axvline(np.linalg.norm(pts[k][sl]), color=mark[k]['color'], lw=1.5, ls='--' if k != 'truth' else '-', label=k)
    ax.set_xlabel(r'$|p_\nu|$ [GeV]', color=INK); ax.set_ylabel('probability per GeV', color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper right', labelcolor=INK)
    ax.set_title('(b) where the probability is', fontsize=11, color=INK, loc='left')
    style(ax)

    # (c) the flow density along the straight line (in all 6 outputs) from the multi200 MAP to the grad200 MAP,
    # with and without the |p| terms: which of the two peaks is higher depends on the coordinates
    ax = axs[2]
    tt = np.concatenate([[0.0], np.logspace(-4, 0, 400)])
    xa, xb = (pts['multi200 MAP'] - mu) / sd, (pts['grad200 MAP'] - mu) / sd
    xp = torch.tensor(xa[None] + tt[:, None] * (xb - xa)[None], dtype=torch.float32)
    with torch.no_grad():
        ctx = model.condition_net(X[event:event + 1]).repeat(len(tt), 1)
        lp = model.flow.log_prob(inputs=xp, context=ctx).numpy()
    phys = xp.numpy() * sd + mu
    p1, p2 = np.linalg.norm(phys[:, :3], axis=1), np.linalg.norm(phys[:, 3:], axis=1)
    jac = np.log(p1 ** 2 + 0.01) + np.log(p2 ** 2 + 0.01)
    ref = -1
    for kpow, col, lab in ((0, BLUE, 'k = 0: Cartesian density'), (1, AQUA, 'k = 1'), (2, VIOLET, 'k = 2: per unit energy and solid angle')):
        y = lp + 0.5 * kpow * jac
        ax.plot(np.linalg.norm(phys[:, sl], axis=1), y - y[ref], color=col, lw=2, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(r'$|p_\nu|$ along the path [GeV]', color=INK)
    ax.set_ylabel('log density relative to the grad200 MAP', color=INK)
    ax.annotate('multi200 MAP', xy=(np.linalg.norm(phys[0, sl]), 0), xytext=(0.05, 0.93), textcoords='axes fraction', color=ORANGE, fontsize=9)
    ax.annotate('grad200 MAP', xy=(1, 0), xytext=(0.72, 0.93), textcoords='axes fraction', color=AQUA, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc='lower center', labelcolor=INK)
    ax.set_title('(c) density along the line between the two MAPs', fontsize=11, color=INK, loc='left')
    style(ax)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(outdir, f'fig_flow_event{event}_nu{nu}.{ext}'), dpi=140)
    plt.close(fig)
    print(f'event {event} {leg} nu: |p| truth {np.linalg.norm(t[sl]):.2f}, grad200 {np.linalg.norm(pts["grad200 MAP"][sl]):.2f}, '
          f'multi200 {np.linalg.norm(pts["multi200 MAP"][sl]):.2f} GeV; samples with |p| < 1 GeV: {100 * (pabs < 1).mean():.2f}%')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--config', default='outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/config.yaml')
    ap.add_argument('--study', default='outputs_Flow_Uncorr_Masked_Hadronic_100e_July28/map_minimizer_study')
    ap.add_argument('--test_file', default='prepared_LHC_data_sep01/ppToHToTauTau_AllDM_UnCorr_Jul20/test_dataframe__leptonic_mode_0_transformer.parquet')
    ap.add_argument('--event', type=int, default=101)
    ap.add_argument('--nu', type=int, default=0, choices=[0, 1], help='0: tau+ neutrino, 1: tau- neutrino')
    ap.add_argument('--n_samples', type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    torch.set_num_threads(2)
    fig_bowl_toy(a.outdir)
    fig_flow_event(a.outdir, a.study, a.test_file, a.config, a.event, a.nu, a.n_samples)
    print(f'>> wrote {a.outdir}/fig_bowl_toy.(pdf|png), fig_flow_event{a.event}_nu{a.nu}.(pdf|png)')


if __name__ == '__main__':
    main()
