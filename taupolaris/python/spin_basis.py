"""Spin-correlation observables cos(theta_i^+-) with the polarimetric vectors and the (n, r, k) basis taken from
possibly different sources -- e.g. reconstructed polarimetric vectors projected onto the basis built from the
generator-level taus, to separate the effect of the reconstructed basis (k = tau+ direction in the ditau rest frame)
from that of the reconstructed polarimetric vectors.

The basis follows kinematic_helpers.compute_spin_angles (k = tau+ direction in the ditau rest frame of the given taus,
n = p x k and r = p - k (p.k) with the beam axis p = (0,0,-1)); the polarimetric vectors are unit vectors in the tau
rest frames with lab-aligned axes, as stored by compute_spin_vars(store_polvecs=True) and as ts_hh_{leg}_{x,y,z}
(TAUOLA's polarimetric vectors at generator level in the prepared dataframes, which the wt_hp_*_hm_* spin weights
follow exactly).
"""
import numpy as np

from taupolaris.utils.kinematic_helpers import compute_spin_angles, boost, boost_vector

COS_KEYS = [f'cos{a}_{s}' for a in 'nrk' for s in ('plus', 'minus')]


def _unit(v):
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def spin_cosines(h_plus, h_minus, taup4, taun4):
    """h_plus/h_minus: (N,3) polarimetric vectors; taup4/taun4: (N,4) (E, px, py, pz) lab four-momenta of the taus
    defining the basis. Returns a dict with cos{n,r,k}_{plus,minus}."""
    bv = boost_vector(taup4 + taun4)
    sa = compute_spin_angles(boost(taup4, -bv), boost(taun4, -bv), _unit(h_plus), _unit(h_minus))
    return {k: sa[k] for k in COS_KEYS}


def taus_from(df, prefix):
    """(taup4, taun4) from {prefix}tau_plus_{E,px,py,pz} / {prefix}tau_minus_{...} columns."""
    get = lambda s: df[[f'{prefix}tau_{s}_{c}' for c in ('E', 'px', 'py', 'pz')]].to_numpy(float)
    return get('plus'), get('minus')


def polvecs_from(df, prefix):
    """(h_plus, h_minus) from {prefix}h_{plus,minus}_{x,y,z} (compute_spin_vars(store_polvecs=True))."""
    get = lambda s: df[[f'{prefix}h_{s}_{c}' for c in 'xyz']].to_numpy(float)
    return get('plus'), get('minus')


def gen_polvecs_from_test_file(path, n_rows):
    """Generator-level polarimetric vectors (TAUOLA's ts_hh) from a prepared test dataframe that is row-aligned with
    the results file. The legs are as labelled in the test file: taup/taun (fully-hadronic) or tau1/tau2 (semileptonic,
    tau1 = the leptonic tau, i.e. the stored 'taup' of the results file before the physical-charge swap). Returns
    (h_first, h_second, px_first, px_second), the last two being undecayed_{leg}_px for the caller's alignment check."""
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    legs = ('taup', 'taun') if 'ts_hh_taup_x' in names else ('tau1', 'tau2')
    cols = [f'ts_hh_{t}_{c}' for t in legs for c in 'xyz'] + [f'undecayed_{t}_px' for t in legs]
    t = pq.read_table(path, columns=cols).to_pandas()
    if len(t) != n_rows:
        raise ValueError(f'{path}: {len(t)} rows, but the results file has {n_rows} -- not row-aligned')
    return (t[cols[0:3]].to_numpy(float), t[cols[3:6]].to_numpy(float),
            t[cols[6]].to_numpy(float), t[cols[7]].to_numpy(float))
