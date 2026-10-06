"""Differentiable phiCP and the energy-score loss term for the direct pol-vec flow.

The flow's NLL trains the full 12-dimensional density (two pol-vecs and two tau momenta in the
per-leg visible (n,r,k) frames), which is dominated by directions phiCP barely depends on. The
energy score (Gneiting & Raftery 2007) on phiCP adds a term that targets the spin-relevant
directions while staying a strictly proper scoring rule, so it does not reward collapsing the
predicted distribution (unlike a plain sample-vs-truth MSE):

    ES = E|X - y| - 1/2 E|X - X'|      X, X' ~ flow(.|event), y = true phiCP

with |.| the chord distance between angles on the unit circle, |(cos a, sin a) - (cos b, sin b)|
= 2|sin((a-b)/2)|. With m >= 2 samples per event the unbiased estimator is

    ES ~= (1/m) sum_i |x_i - y| - 1/(m(m-1)) sum_{i<j} |x_i - x_j|

phiCP is computed exactly as evaluate_polvec does (convert_native_to_cartesian ->
phicp_from_cart -> compute_aco_polarimetric) but in torch, returning (cos phiCP, sin phiCP)
directly: cos = u1.u2 and sin = P2.(u1 x u2) for the unit transverse pol-vecs u1, u2. This
avoids arccos (whose gradient diverges at +-1) and gives the point on the circle the score needs.
The sign convention is the signed phiCP in [0, 2pi) of compute_aco_polarimetric.

Only coordinates='onorm' with the 12 pol-vec + tau-momentum outputs (onorm_output_order) and
hadronic taup/taun legs are supported; the per-leg basis is rebuilt from the reco pi1+pi2+pi3
(= the 'charged' vector) + pizero1 inputs, which therefore must be in input_features.
"""
import torch

M_TAU = 1.77686


def build_nrk_basis(vis, eps=1e-12):
    """torch version of coordinate_conversions._build_nrk_basis_from_visible_tau.
    vis: (N,3) visible momentum. Returns n, r, k, each (N,3)."""
    k_norm = vis.norm(dim=1, keepdim=True)
    k = torch.where(k_norm > eps, vis / k_norm.clamp_min(eps), torch.zeros_like(vis))
    p = torch.zeros_like(vis); p[:, 2] = -1.0
    n = torch.cross(p, k, dim=1)
    n_norm = n.norm(dim=1, keepdim=True)
    n = torch.where(n_norm > eps, n / n_norm.clamp_min(eps), torch.zeros_like(n))
    r = p - k * (p * k).sum(1, keepdim=True)
    r_norm = r.norm(dim=1, keepdim=True)
    r = torch.where(r_norm > eps, r / r_norm.clamp_min(eps), torch.zeros_like(r))
    zero = ((n_norm <= eps) & (k_norm <= eps)).squeeze(1)
    if zero.any():
        n[zero] = torch.tensor([1.0, 0.0, 0.0], device=vis.device, dtype=vis.dtype)
        r[zero] = torch.tensor([0.0, 1.0, 0.0], device=vis.device, dtype=vis.dtype)
        k[zero] = torch.tensor([0.0, 0.0, 1.0], device=vis.device, dtype=vis.dtype)
    return n, r, k


def _unit(v, eps=1e-12):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def phicp_cos_sin(native, basis1, basis2):
    """(cos phiCP, sin phiCP) from native onorm outputs.

    native : (..., 12) in onorm_output_order: ts_hh_t1_{n,r,k}, ts_hh_t2_{n,r,k},
             undecayed_t1_{n,r,k}, undecayed_t2_{n,r,k}   (physical units)
    basis1, basis2 : (n, r, k) tuples of (..., 3) tensors for legs t1 and t2, broadcastable to native.
    Returns cos, sin each of shape native.shape[:-1].
    """
    def to_cart(v, b):
        n, r, k = b
        return v[..., 0:1] * n + v[..., 1:2] * r + v[..., 2:3] * k

    h_t1 = to_cart(native[..., 0:3], basis1)
    h_t2 = to_cart(native[..., 3:6], basis2)
    p_t1 = to_cart(native[..., 6:9], basis1)
    p_t2 = to_cart(native[..., 9:12], basis2)
    E_t1 = torch.sqrt((p_t1 ** 2).sum(-1, keepdim=True) + M_TAU ** 2)
    E_t2 = torch.sqrt((p_t2 ** 2).sum(-1, keepdim=True) + M_TAU ** 2)
    # tau_directions_com: boost tau1 into the ditau rest frame (kinematic_helpers.boost with -beta)
    beta = (p_t1 + p_t2) / (E_t1 + E_t2)
    b2 = (beta ** 2).sum(-1, keepdim=True)
    gamma = 1.0 / torch.sqrt((1.0 - b2).clamp_min(1e-12))
    bp = (beta * p_t1).sum(-1, keepdim=True)
    gamma2 = torch.where(b2 > 0, (gamma - 1.0) / b2.clamp_min(1e-30), torch.zeros_like(b2))
    p1_com = p_t1 + gamma2 * (-bp) * (-beta) + gamma * (-beta) * E_t1
    kx = _unit(p1_com)
    # phicp_from_cart's labelling: R1 = h_t2, P1 = -kx (tau2 direction); R2 = h_t1, P2 = kx
    R1, P1, R2, P2 = h_t2, -kx, h_t1, kx
    u1 = _unit(R1 - (R1 * P1).sum(-1, keepdim=True) * P1)
    u2 = _unit(R2 - (R2 * P2).sum(-1, keepdim=True) * P2)
    cos = (u1 * u2).sum(-1)
    sin = (P2 * torch.cross(u1, u2, dim=-1)).sum(-1)
    return cos, sin


def energy_score_circle(cs_samples, cs_true, eps=1e-12):
    """Per-event energy score (beta=1) on the unit circle.
    cs_samples : (B, m, 2) sampled (cos, sin);  cs_true : (B, 2).  Returns (B,)."""
    m = cs_samples.shape[1]
    if m < 2:
        raise ValueError('the energy score needs at least 2 samples per event')
    d_true = torch.sqrt(((cs_samples - cs_true[:, None, :]) ** 2).sum(-1) + eps).mean(1)
    i, j = torch.triu_indices(m, m, offset=1, device=cs_samples.device)
    d_pair = torch.sqrt(((cs_samples[:, i] - cs_samples[:, j]) ** 2).sum(-1) + eps).mean(1)
    return d_true - 0.5 * d_pair


class PhiCPEnergyScore:
    """Holds what is needed to go from a training batch (standardised X, y) to the energy score.

    input_features / output_features : the model's feature lists (as in the config)
    input_mean/std, output_mean/std   : the dataset's standardisation tensors
    """

    def __init__(self, input_features, output_features, input_mean, input_std, output_mean, output_std,
                 n_samples=2, tau_labels=('taup', 'taun')):
        t1, t2 = tau_labels
        expected = [f'ts_hh_{t}_{c}' for t in (t1, t2) for c in 'nrk'] + \
                   [f'undecayed_{t}_{c}' for t in (t1, t2) for c in 'nrk']
        if list(output_features) != expected:
            raise ValueError(f'phiCP energy score needs output_features == {expected}, got {list(output_features)}')
        self.idx = {}
        for t in (t1, t2):
            for obj in ('pi1', 'pi2', 'pi3', 'pizero1'):
                cols = [f'reco_{t}_{obj}_{c}' for c in ('px', 'py', 'pz')]
                missing = [c for c in cols if c not in input_features]
                if missing:
                    raise ValueError(f'phiCP energy score needs {missing} in input_features to rebuild the (n,r,k) basis')
                self.idx[(t, obj)] = [list(input_features).index(c) for c in cols]
        self.tau_labels = tau_labels
        self.in_mean, self.in_std = input_mean.reshape(-1), input_std.reshape(-1)
        self.out_mean, self.out_std = output_mean.reshape(-1), output_std.reshape(-1)
        self.n_samples = n_samples

    def bases(self, X):
        """Per-leg (n,r,k) bases from the standardised inputs X (B, n_in); no gradient."""
        out = []
        with torch.no_grad():
            m, s = self.in_mean.to(X.device), self.in_std.to(X.device)
            for t in self.tau_labels:
                vis = 0
                for obj in ('pi1', 'pi2', 'pi3', 'pizero1'):
                    ix = self.idx[(t, obj)]
                    vis = vis + X[:, ix] * s[ix] + m[ix]
                out.append(build_nrk_basis(vis))
        return out

    def native(self, y_std):
        return y_std * self.out_std.to(y_std.device) + self.out_mean.to(y_std.device)

    def true_cos_sin(self, X, y_std, bases=None):
        b1, b2 = bases or self.bases(X)
        return phicp_cos_sin(self.native(y_std), b1, b2)

    def per_event(self, flow, cond_embed, X, y_std):
        """Per-event ES (B,) for the flow's distribution given the conditioner output cond_embed.
        Samples are drawn with gradient (nflows' spline inverse is differentiable)."""
        b1, b2 = self.bases(X)
        samples = flow.sample(num_samples=self.n_samples, context=cond_embed)      # (B, m, 12), standardised
        nat = self.native(samples)
        bb1 = tuple(v[:, None, :] for v in b1); bb2 = tuple(v[:, None, :] for v in b2)
        cs_s = torch.stack(phicp_cos_sin(nat, bb1, bb2), dim=-1)                   # (B, m, 2)
        cs_t = torch.stack(phicp_cos_sin(self.native(y_std), b1, b2), dim=-1)      # (B, 2)
        return energy_score_circle(cs_s, cs_t)
