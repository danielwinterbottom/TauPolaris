import numpy as np
import pandas as pd


def ConvertToPolar(df,prefix):
    # convert the px, py, pz columns with given prefix to polar coordinates (pt, eta, phi)
    px = df[f'{prefix}x']
    py = df[f'{prefix}y']
    pz = df[f'{prefix}z']

    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    # to avoid division by zero
    theta = np.arccos(np.clip(pz / p, -1.0, 1.0))
    eta = -np.log(np.tan(theta / 2))
    phi = np.arctan2(py, px)

    df[f'{prefix}_pt'] = pt
    df[f'{prefix}_eta'] = eta
    df[f'{prefix}_phi'] = phi

    # drop the original x, y, z columns
    df = df.drop(columns=[f'{prefix}x', f'{prefix}y', f'{prefix}z'])

    return df

def ConvertToCartesian(df,prefix):
    # convert the pt, eta, phi columns with given prefix to cartesian coordinates (px, py, pz)
    pt = df[f'{prefix}_pt']
    eta = df[f'{prefix}_eta']
    phi = df[f'{prefix}_phi']

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    df[f'{prefix}x'] = px
    df[f'{prefix}y'] = py
    df[f'{prefix}z'] = pz

    # drop the original pt, eta, phi columns
    df = df.drop(columns=[f'{prefix}_pt', f'{prefix}_eta', f'{prefix}_phi'])

    return df

# make a similar function for converting back the predictions which aren't labelled with a prefix
def ConvertPredictionsToCartesian(predictions, output_features):
    # predictions is a numpy array of shape [N_events, N_features]
    predictions_df = pd.DataFrame(predictions, columns=output_features)

    # find all prefixes by removing the _pt, _eta, _phi suffixes
    prefixes = set()
    for col in predictions_df.columns:
        if col.endswith('_pt'):
            prefixes.add(col[:-3])
        elif col.endswith('_eta'):
            prefixes.add(col[:-4])
        elif col.endswith('_phi'):
            prefixes.add(col[:-4])

    for prefix in prefixes:
        predictions_df = ConvertToCartesian(predictions_df, prefix)

    # return as numpy array
    return predictions_df.values

def _get_vec(df, pref: str, suffixes=("px", "py", "pz")) -> np.ndarray:
    """Return (N,3) array from columns {pref}{suffixes[0]}, {pref}{suffixes[1]}, {pref}{suffixes[2]}."""
    return np.stack(
        [df[f"{pref}{suffixes[0]}"].to_numpy(),
         df[f"{pref}{suffixes[1]}"].to_numpy(),
         df[f"{pref}{suffixes[2]}"].to_numpy()],
        axis=1
    ).astype(float)

def _build_nrk_basis_from_visible_tau(
    df,
    charged_prefix: str,
    pi0_prefix: str,
    eps: float = 1e-12,
):
    """
    Build event-by-event orthonormal basis (n_hat, r_hat, k_hat) using:
      p_hat = (0,0,-1)
      k_hat = unit(pi + pi0)
      n_hat = unit(p_hat x k_hat)
      r_hat = unit(p_hat - k_hat (p_hat·k_hat))

    Returns:
      n, r, k as (N,3) arrays
    """

    v_vis = _get_vec(df, charged_prefix) + _get_vec(df, pi0_prefix)

    # --- k_hat ---
    k_norm = np.linalg.norm(v_vis, axis=1, keepdims=True)

    k = np.zeros_like(v_vis)
    np.divide(v_vis, k_norm, out=k, where=(k_norm > eps))

    # beam axis
    N = len(df)
    p = np.zeros((N, 3), dtype=float)
    p[:, 2] = -1.0

    n = np.cross(p, k)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)

    n_unit = np.zeros_like(n)
    np.divide(n, n_norm, out=n_unit, where=(n_norm > eps))
    n = n_unit

    cosTheta = np.sum(p * k, axis=1, keepdims=True)

    r = p - k * cosTheta
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)

    r_unit = np.zeros_like(r)
    np.divide(r, r_norm, out=r_unit, where=(r_norm > eps))
    r = r_unit

    # fallback basis if everything vanished
    zero_mask = ((n_norm <= eps) & (k_norm <= eps)).ravel()

    n[zero_mask] = [1.0, 0.0, 0.0]
    r[zero_mask] = [0.0, 1.0, 0.0]
    k[zero_mask] = [0.0, 0.0, 1.0]

    return n, r, k

def ConvertToOrthonormalNRK(
    df,
    prefix_to_convert: str,
    charged_prefix: str,
    pi0_prefix: str,
    out_prefix = None,
    drop_xyz: bool = True,
    eps: float = 1e-12,
    keep_basis: bool = False,
    suffixes=("px", "py", "pz"),
):
    if out_prefix is None:
        out_prefix = prefix_to_convert

    n, r, k = _build_nrk_basis_from_visible_tau(df, charged_prefix, pi0_prefix, eps=eps)
    v = _get_vec(df, prefix_to_convert, suffixes=suffixes)

    # Assign the new columns in ONE operation rather than one at a time. Each
    # single-column assignment on a wide frame is a separate block insert, and
    # _process_chunk calls this a dozen-odd times per chunk -- enough to trip
    # pandas' "DataFrame is highly fragmented" PerformanceWarning and flood the
    # prep log. Same columns, same order, same values.
    new_cols = {
        f"{out_prefix}n": np.sum(v * n, axis=1),
        f"{out_prefix}r": np.sum(v * r, axis=1),
        f"{out_prefix}k": np.sum(v * k, axis=1),
    }
    if keep_basis:
        for i, comp in enumerate(["x", "y", "z"]):
            new_cols[f"{out_prefix}n{comp}"] = n[:, i]
            new_cols[f"{out_prefix}r{comp}"] = r[:, i]
            new_cols[f"{out_prefix}k{comp}"] = k[:, i]
    new_df = pd.DataFrame(new_cols, index=df.index)
    # concat APPENDS, where the old per-column assignment OVERWROTE, so drop any
    # pre-existing names first -- otherwise a repeat call on the same prefix
    # would silently leave duplicate columns behind. (Column order shifts for
    # overwritten names; everything here is indexed by name, never position.)
    clash = [c for c in new_df.columns if c in df.columns]
    if clash:
        df = df.drop(columns=clash)
    df = pd.concat([df, new_df], axis=1)

    if drop_xyz:
        df = df.drop(columns=[f"{prefix_to_convert}{suffixes[0]}", f"{prefix_to_convert}{suffixes[1]}", f"{prefix_to_convert}{suffixes[2]}"])

    return df


def ConvertFromOrthonormalNRK(
    df,
    prefix_to_convert: str,  # expects {prefix}n,{prefix}r,{prefix}k
    charged_prefix: str,
    pi0_prefix: str,
    out_prefix = None,  # writes {out}{suffixes[0]},{out}{suffixes[1]},{out}{suffixes[2]}
    drop_nrk: bool = False,
    eps: float = 1e-12,
    suffixes=("px", "py", "pz"),
):
    if out_prefix is None:
        out_prefix = prefix_to_convert

    n, r, k = _build_nrk_basis_from_visible_tau(df, charged_prefix, pi0_prefix, eps=eps)

    vn = df[f"{prefix_to_convert}n"].to_numpy(dtype=float)[:, None]
    vr = df[f"{prefix_to_convert}r"].to_numpy(dtype=float)[:, None]
    vk = df[f"{prefix_to_convert}k"].to_numpy(dtype=float)[:, None]

    v = vn * n + vr * r + vk * k

    df[f"{out_prefix}{suffixes[0]}"] = v[:, 0]
    df[f"{out_prefix}{suffixes[1]}"] = v[:, 1]
    df[f"{out_prefix}{suffixes[2]}"] = v[:, 2]

    if drop_nrk:
        df = df.drop(columns=[f"{prefix_to_convert}n", f"{prefix_to_convert}r", f"{prefix_to_convert}k"])

    return df


def ConvertNRKToAngular(df, prefix, drop_nrk=True, norm_col_suffix="norm", eps=1e-12):
    """
    Collapse a (nominally unit-length) vector's (n,r,k) onorm components into
    its 2 genuine direction degrees of freedom: costheta = k-component of the
    *normalized* direction (k is the polar axis, chosen because it's built
    from the visible-tau flight direction) and phi = the azimuthal angle in
    the n-r plane. Unlike the raw (n,r,k) triplet, these 2 numbers have no
    hidden n^2+r^2+k^2=1 constraint between them, which is otherwise a
    degenerate (measure-zero) target for a density model like a normalizing
    flow to learn.

    In practice this vector isn't always exactly unit-length -- e.g. the
    ts_hh_taup/taun polarimetric vectors have a rare (~0.1-0.5% of events),
    heavy-tailed deviation from |h|=1 traced to numerical singularities in
    calculate_hh.py's per-channel kinematic formulas, not a physics effect.
    Rather than silently baking that deviation into costheta (which would
    make it wrong specifically for events near the poles, where the missing
    normalization is most amplified), this function explicitly normalizes
    before splitting into angles, and separately stores the raw
    pre-normalization magnitude in a `{prefix}{norm_col_suffix}` column
    (not used for training) so events with a bad magnitude can be identified
    and cut downstream without that information being lost at prep time.
    """
    n = df[f"{prefix}n"].to_numpy(dtype=float)
    r = df[f"{prefix}r"].to_numpy(dtype=float)
    k = df[f"{prefix}k"].to_numpy(dtype=float)

    norm = np.sqrt(n ** 2 + r ** 2 + k ** 2)
    safe_norm = np.where(norm > eps, norm, 1.0)

    df[f"{prefix}costheta"] = k / safe_norm
    df[f"{prefix}phi"] = np.arctan2(r, n)  # ratio r/n is unaffected by the overall scale
    df[f"{prefix}{norm_col_suffix}"] = norm

    if drop_nrk:
        df = df.drop(columns=[f"{prefix}n", f"{prefix}r", f"{prefix}k"])

    return df


def ConvertAngularToNRK(df, prefix, drop_angular=True):
    """Inverse of ConvertNRKToAngular: (costheta, phi) -> (n, r, k)."""
    costheta = df[f"{prefix}costheta"].to_numpy(dtype=float)
    phi = df[f"{prefix}phi"].to_numpy(dtype=float)

    sintheta = np.sqrt(np.clip(1.0 - costheta ** 2, 0.0, None))
    df[f"{prefix}n"] = sintheta * np.cos(phi)
    df[f"{prefix}r"] = sintheta * np.sin(phi)
    df[f"{prefix}k"] = costheta

    if drop_angular:
        df = df.drop(columns=[f"{prefix}costheta", f"{prefix}phi"])

    return df


def convert_coordinates_pred(arr, coordinates, output_features, tau1_charged, tau1_pi0, tau2_charged, tau2_pi0, leptonic_mode=0):
    if coordinates == 'polar':
        if leptonic_mode !=0:
            raise ValueError("Polar coordinates are currently only implemented for the hadronic tau case (leptonic_mode=0)")
        return ConvertPredictionsToCartesian(arr, output_features)
    elif coordinates == 'onorm':
        return ConvertFromOrthonormalNRK_Predictions(arr, tau1_charged=tau1_charged, tau1_pi0=tau1_pi0,
                                                     tau2_charged=tau2_charged, tau2_pi0=tau2_pi0, leptonic_mode=leptonic_mode)
    return arr


def ConvertFromOrthonormalNRK_Predictions(
    predictions,
    tau1_charged,
    tau1_pi0,
    tau2_charged,
    tau2_pi0,
    eps: float = 1e-12,
    leptonic_mode: int = 0,
):

    charged_name = "charged"

    # create a dataframe with predictions and visible tau decay products
    
    tau1_prefix = 'taup'
    tau2_prefix = 'taun'

    if leptonic_mode == 1:
        column_names = ['tau1_nu_m','tau1_nu_n', 'tau1_nu_r', 'tau1_nu_k','tau2_nu_n', 'tau2_nu_r', 'tau2_nu_k']
        tau1_prefix = 'tau1'
        tau2_prefix = 'tau2'
    elif leptonic_mode == 2:
        column_names = ['tau1_nu_m','tau1_nu_n', 'tau1_nu_r', 'tau1_nu_k','tau2_nu_m','tau2_nu_n', 'tau2_nu_r', 'tau2_nu_k']
    else:
        column_names = ['taup_nu_n', 'taup_nu_r', 'taup_nu_k','taun_nu_n', 'taun_nu_r', 'taun_nu_k']

    tau1_charged_columns = [f'reco_{tau1_prefix}_{charged_name}_{comp}' for comp in ['E', 'px', 'py', 'pz']]
    tau1_pi0_columns = [f'reco_{tau1_prefix}_pizero1_{comp}' for comp in ['E', 'px', 'py', 'pz']]
    tau2_charged_columns = [f'reco_{tau2_prefix}_{charged_name}_{comp}' for comp in ['E', 'px', 'py', 'pz']]
    tau2_pi0_columns = [f'reco_{tau2_prefix}_pizero1_{comp}' for comp in ['E', 'px', 'py', 'pz']]
    visible_data = np.concatenate([tau1_charged, tau1_pi0, tau2_charged, tau2_pi0], axis=1)
    visible_column_names = tau1_charged_columns + tau1_pi0_columns + tau2_charged_columns + tau2_pi0_columns
    all_data = np.concatenate([predictions, visible_data], axis=1)
    column_names += visible_column_names
    df = pd.DataFrame(all_data, columns=column_names)


    # now call ConvertFromOrthonormalNRK on the dataframe, first for nubar tau+ neutrino
    charged_prefix = f'reco_{tau1_prefix}_{charged_name}_'
    pi0_prefix = f'reco_{tau1_prefix}_pizero1_'
    df = ConvertFromOrthonormalNRK(
        df,
        prefix_to_convert=f'{tau1_prefix}_nu_',
        charged_prefix=charged_prefix,
        pi0_prefix=pi0_prefix,
        drop_nrk=True,
        eps=eps,
    )
    # then for nubar tau- neutrino
    charged_prefix = f'reco_{tau2_prefix}_{charged_name}_'
    pi0_prefix = f'reco_{tau2_prefix}_pizero1_'
    df = ConvertFromOrthonormalNRK(
        df,
        prefix_to_convert=f'{tau2_prefix}_nu_',
        charged_prefix=charged_prefix,
        pi0_prefix=pi0_prefix,
        drop_nrk=True,
        eps=eps,
    )

    # ensure it is ordered correctly
    if leptonic_mode==1:
        ordered_columns = [
            'tau1_nu_m', 'tau1_nu_px', 'tau1_nu_py', 'tau1_nu_pz',
            'tau2_nu_px', 'tau2_nu_py', 'tau2_nu_pz'
        ]
    elif leptonic_mode==2:
        ordered_columns = [
            'taup_nu_m', 'taup_nu_px', 'taup_nu_py', 'taup_nu_pz',
            'taun_nu_m', 'taun_nu_px', 'taun_nu_py', 'taun_nu_pz'
        ]
    else:
        ordered_columns = [
            'taup_nu_px', 'taup_nu_py', 'taup_nu_pz',
            'taun_nu_px', 'taun_nu_py', 'taun_nu_pz'
        ]
    df = df[ordered_columns]
    # return as numpy array
    return df.values


def ConvertFromOrthonormalNRK_Predictions_PolVec(
    predictions,
    reco_taup_charged,
    reco_taup_pizero,
    reco_taun_charged,
    reco_taun_pizero,
    eps: float = 1e-12,
):
    """
    Inverse onorm -> Cartesian transform for the polvec model's 12 outputs:
    ts_hh_taup/taun (polarimetric unit vectors) + undecayed_taup/taun (tau momenta),
    each projected onto the per-tau visible-momentum (n,r,k) basis (see
    ConvertToOrthonormalNRK / _build_nrk_basis_from_visible_tau).

    Parameters
    ----------
    predictions : np.ndarray, shape [N, 12]
        Columns in the order of output_features['onorm'] in config_polvec.yaml:
        ts_hh_taup_{n,r,k}, ts_hh_taun_{n,r,k}, undecayed_taup_{n,r,k}, undecayed_taun_{n,r,k}
    reco_taup_charged, reco_taup_pizero, reco_taun_charged, reco_taun_pizero : np.ndarray, shape [N, 3]
        Visible tau (px,py,pz) momentum components used to build the same basis as at
        training time (must match the charged/pi0 vectors used there).

    Returns
    -------
    np.ndarray, shape [N, 12]: ts_hh_taup_{x,y,z}, ts_hh_taun_{x,y,z},
    undecayed_taup_{px,py,pz}, undecayed_taun_{px,py,pz}
    """
    column_names = [
        'ts_hh_taup_n', 'ts_hh_taup_r', 'ts_hh_taup_k',
        'ts_hh_taun_n', 'ts_hh_taun_r', 'ts_hh_taun_k',
        'undecayed_taup_n', 'undecayed_taup_r', 'undecayed_taup_k',
        'undecayed_taun_n', 'undecayed_taun_r', 'undecayed_taun_k',
    ]
    visible_column_names = (
        [f'reco_taup_charged_{c}' for c in ['px', 'py', 'pz']] +
        [f'reco_taup_pizero1_{c}' for c in ['px', 'py', 'pz']] +
        [f'reco_taun_charged_{c}' for c in ['px', 'py', 'pz']] +
        [f'reco_taun_pizero1_{c}' for c in ['px', 'py', 'pz']]
    )
    visible_data = np.concatenate(
        [reco_taup_charged, reco_taup_pizero, reco_taun_charged, reco_taun_pizero], axis=1
    )
    df = pd.DataFrame(
        np.concatenate([predictions, visible_data], axis=1),
        columns=column_names + visible_column_names,
    )

    df = ConvertFromOrthonormalNRK(
        df, prefix_to_convert='ts_hh_taup_',
        charged_prefix='reco_taup_charged_', pi0_prefix='reco_taup_pizero1_',
        drop_nrk=True, eps=eps, suffixes=('x', 'y', 'z'),
    )
    df = ConvertFromOrthonormalNRK(
        df, prefix_to_convert='ts_hh_taun_',
        charged_prefix='reco_taun_charged_', pi0_prefix='reco_taun_pizero1_',
        drop_nrk=True, eps=eps, suffixes=('x', 'y', 'z'),
    )
    df = ConvertFromOrthonormalNRK(
        df, prefix_to_convert='undecayed_taup_',
        charged_prefix='reco_taup_charged_', pi0_prefix='reco_taup_pizero1_',
        drop_nrk=True, eps=eps,
    )
    df = ConvertFromOrthonormalNRK(
        df, prefix_to_convert='undecayed_taun_',
        charged_prefix='reco_taun_charged_', pi0_prefix='reco_taun_pizero1_',
        drop_nrk=True, eps=eps,
    )

    ordered_columns = [
        'ts_hh_taup_x', 'ts_hh_taup_y', 'ts_hh_taup_z',
        'ts_hh_taun_x', 'ts_hh_taun_y', 'ts_hh_taun_z',
        'undecayed_taup_px', 'undecayed_taup_py', 'undecayed_taup_pz',
        'undecayed_taun_px', 'undecayed_taun_py', 'undecayed_taun_pz',
    ]
    return df[ordered_columns].values


def ConvertFromOrthonormalNRK_Predictions_PolVec_Angular(
    predictions,
    reco_taup_charged,
    reco_taup_pizero,
    reco_taun_charged,
    reco_taun_pizero,
    eps: float = 1e-12,
):
    """
    Inverse onorm_angular -> Cartesian transform for the polvec model's 10
    outputs: ts_hh_taup/taun (polarimetric unit vectors), each parameterized
    by (costheta, phi) instead of the raw (n,r,k) triplet, plus
    undecayed_taup/taun (tau momenta) as full (n,r,k) triplets -- see
    ConvertNRKToAngular / ConvertToOrthonormalNRK.

    Parameters
    ----------
    predictions : np.ndarray, shape [N, 10]
        Columns in the order of output_features['onorm_angular'] in
        config_polvec.yaml: ts_hh_taup_{costheta,phi}, ts_hh_taun_{costheta,phi},
        undecayed_taup_{n,r,k}, undecayed_taun_{n,r,k}
    reco_taup_charged, reco_taup_pizero, reco_taun_charged, reco_taun_pizero : np.ndarray, shape [N, 3]
        Visible tau (px,py,pz) momentum components used to build the same basis as at
        training time (must match the charged/pi0 vectors used there).

    Returns
    -------
    np.ndarray, shape [N, 12]: ts_hh_taup_{x,y,z}, ts_hh_taun_{x,y,z},
    undecayed_taup_{px,py,pz}, undecayed_taun_{px,py,pz}
    """
    column_names = [
        'ts_hh_taup_costheta', 'ts_hh_taup_phi',
        'ts_hh_taun_costheta', 'ts_hh_taun_phi',
        'undecayed_taup_n', 'undecayed_taup_r', 'undecayed_taup_k',
        'undecayed_taun_n', 'undecayed_taun_r', 'undecayed_taun_k',
    ]
    visible_column_names = (
        [f'reco_taup_charged_{c}' for c in ['px', 'py', 'pz']] +
        [f'reco_taup_pizero1_{c}' for c in ['px', 'py', 'pz']] +
        [f'reco_taun_charged_{c}' for c in ['px', 'py', 'pz']] +
        [f'reco_taun_pizero1_{c}' for c in ['px', 'py', 'pz']]
    )
    visible_data = np.concatenate(
        [reco_taup_charged, reco_taup_pizero, reco_taun_charged, reco_taun_pizero], axis=1
    )
    df = pd.DataFrame(
        np.concatenate([predictions, visible_data], axis=1),
        columns=column_names + visible_column_names,
    )

    df = ConvertAngularToNRK(df, prefix='ts_hh_taup_', drop_angular=True)
    df = ConvertAngularToNRK(df, prefix='ts_hh_taun_', drop_angular=True)

    df = ConvertFromOrthonormalNRK(
        df, prefix_to_convert='ts_hh_taup_',
        charged_prefix='reco_taup_charged_', pi0_prefix='reco_taup_pizero1_',
        drop_nrk=True, eps=eps, suffixes=('x', 'y', 'z'),
    )
    df = ConvertFromOrthonormalNRK(
        df, prefix_to_convert='ts_hh_taun_',
        charged_prefix='reco_taun_charged_', pi0_prefix='reco_taun_pizero1_',
        drop_nrk=True, eps=eps, suffixes=('x', 'y', 'z'),
    )
    df = ConvertFromOrthonormalNRK(
        df, prefix_to_convert='undecayed_taup_',
        charged_prefix='reco_taup_charged_', pi0_prefix='reco_taup_pizero1_',
        drop_nrk=True, eps=eps,
    )
    df = ConvertFromOrthonormalNRK(
        df, prefix_to_convert='undecayed_taun_',
        charged_prefix='reco_taun_charged_', pi0_prefix='reco_taun_pizero1_',
        drop_nrk=True, eps=eps,
    )

    ordered_columns = [
        'ts_hh_taup_x', 'ts_hh_taup_y', 'ts_hh_taup_z',
        'ts_hh_taun_x', 'ts_hh_taun_y', 'ts_hh_taun_z',
        'undecayed_taup_px', 'undecayed_taup_py', 'undecayed_taup_pz',
        'undecayed_taun_px', 'undecayed_taun_py', 'undecayed_taun_pz',
    ]
    return df[ordered_columns].values
