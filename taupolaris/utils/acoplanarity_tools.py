import warnings
import awkward as ak
import vector
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in multiply")
vector.register_awkward()
from taupolaris.utils.PolarimetricA1 import PolarimetricA1_vectorised
from taupolaris.utils.Polarimetric3hpi0 import Polarimetric3hpi0_vectorised

def pt_direction_to_momentum4d(pt, direction, mass):
    # get p vector from pT and direction vector (cartesian)
    pt_dir = np.sqrt(direction.x**2 + direction.y**2)
    px = pt * direction.x / pt_dir
    py = pt * direction.y / pt_dir
    pz = pt * direction.z / pt_dir
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return ak.zip({"px": px, "py": py, "pz": pz, "E": E}, with_name="Momentum4D")

def spatial(v, return_E=False):
    p3 = ak.zip({"x": v.px, "y": v.py, "z": v.pz}, with_name="Vector3D")
    if return_E:
        return p3, v.E
    else:
        return p3

def boost_vec(v):
    return ak.zip({"x": v.px / v.E, "y": v.py / v.E, "z": v.pz / v.E}, with_name="Vector3D")

def boost4(v, bv):
    r = v.boost(-bv)
    return ak.zip({"px": r.x, "py": r.y, "pz": r.z, "E": r.t}, with_name="Momentum4D")

def invMass(t1, t2):
    s = t1 + t2
    return np.sqrt(np.maximum(s.dot(s), 0.0))

def _quadratic(a, b, c):
    """Numerically stable quadratic formula returning both roots (smaller, larger)."""
    D = b**2 - 4.0 * a * c
    masked_D = ak.where(D < 0, 0.0, D)
    q = ak.where(D < 0, -0.5 * b, -0.5 * (b + np.copysign(np.sqrt(masked_D), b)))
    return c / q, q / a

def polarimetric_vec_dm0(H_pi, boost_vec_tau):
    return spatial(boost4(H_pi, boost_vec_tau)).unit()

def polarimetric_vec_leptonic(H_lep, boost_vec_tau, times_by_bl=False):
    # get muon in tau rest frame
    H_lep_rf = boost4(H_lep, boost_vec_tau)
    # compute x = 2*E_lep/m_tau in tau rest frame
    x_lep = 2*H_lep_rf.E/1.777

    # bound x_lep between 0 and 1
    x_lep = ak.where(x_lep < 0, 0, ak.where(x_lep > 1, 1, x_lep))
    bl = (1-2*x_lep)/(3-2*x_lep)

    if times_by_bl:
        return spatial(bl * H_lep_rf).unit()
    else: 
        return spatial(-1 * H_lep_rf).unit()

def polarimetric_vec_dm1(H_pi, H_pizero, H_tau, boost_vec_tau):
    """Compute the DM1/2 (rho/a11pr) polarimetric vector in the tau rest frame."""
    q = H_pi - H_pizero
    P = H_tau
    N = H_tau - H_pi - H_pizero
    qN = q.dot(N)
    qP = q.dot(P)
    NP = N.dot(P)
    q2 = q.dot(q)
    massP = np.sqrt(np.maximum(P.dot(P), 0.0))
    coeff = massP / (2 * qN * qP - q2 * NP)
    pv = coeff * (2 * qN * q - q2 * N)
    return spatial(boost4(pv, boost_vec_tau)).unit()

def polarimetric_vec_dm10(tau_rf, os_pi_rf, ss1_pi_rf, ss2_pi_rf, taucharge):
    """Compute the DM10 (a1) polarimetric vector in the tau rest frame."""
    bv_tau = boost_vec(tau_rf)
    tau_trf   = boost4(tau_rf,    bv_tau)
    os_trf    = boost4(os_pi_rf,  bv_tau)
    ss1_trf   = boost4(ss1_pi_rf, bv_tau)
    ss2_trf   = boost4(ss2_pi_rf, bv_tau)
    pv = PolarimetricA1_vectorised(tau_trf, os_trf, ss1_trf, ss2_trf, taucharge).PVC()
    # note the sign reversal here by convention
    pv_ak = ak.zip({"x": -pv.x, "y": -pv.y, "z": -pv.z}, with_name="Vector3D")
    return pv_ak.unit()

def polarimetric_vec_dm11(tau_rf, os_pi_rf, ss1_pi_rf, ss2_pi_rf, pizero_rf, taucharge):
    """Compute the DM11 (3h+pi0) polarimetric vector in the tau rest frame, using the
    vectorised CLEO 4-pion (tauola curr_cleo MNUM=1) hadronic current -- see
    Polarimetric3hpi0.py. Same calling convention as polarimetric_vec_dm10."""
    bv_tau = boost_vec(tau_rf)
    tau_trf   = boost4(tau_rf,     bv_tau)
    os_trf    = boost4(os_pi_rf,   bv_tau)
    ss1_trf   = boost4(ss1_pi_rf,  bv_tau)
    ss2_trf   = boost4(ss2_pi_rf,  bv_tau)
    piz_trf   = boost4(pizero_rf,  bv_tau)
    pv = Polarimetric3hpi0_vectorised(tau_trf, os_trf, ss1_trf, ss2_trf, piz_trf, taucharge).PVC()
    # note the sign reversal here by convention (same as polarimetric_vec_dm10)
    pv_ak = ak.zip({"x": -pv.x, "y": -pv.y, "z": -pv.z}, with_name="Vector3D")
    return pv_ak.unit()

def rotate_to_GJMax(visible_tau, tau):
    tau_dir = spatial(tau).unit()
    vis_dir = spatial(visible_tau).unit()

    mass_tau = np.sqrt(np.maximum(tau.dot(tau), 0.0))
    mass_vis = np.sqrt(np.maximum(visible_tau.dot(visible_tau), 0.0))
    tau_p_mag = np.sqrt(tau.px**2 + tau.py**2 + tau.pz**2)
    vis_p_mag = np.sqrt(visible_tau.px**2 + visible_tau.py**2 + visible_tau.pz**2)

    theta_GJ = np.arccos(np.clip(tau_dir.dot(vis_dir), -1, 1))
    theta_GJ_max = np.arcsin(np.clip((mass_tau**2 - mass_vis**2) / (2 * mass_tau * vis_p_mag), -1, 1))

    mask = theta_GJ > theta_GJ_max

    n_1_x = 1 / np.sqrt(1 + (visible_tau.px / visible_tau.py)**2)
    n_1_y = -n_1_x * visible_tau.px / visible_tau.py
    n_1 = ak.zip({"x": n_1_x, "y": n_1_y, "z": ak.zeros_like(n_1_x)}, with_name="Vector3D")
    n_2 = n_1.cross(vis_dir)

    phi_opt_1 = np.arctan(tau_dir.dot(n_2) / tau_dir.dot(n_1))
    new_dir_1 = np.cos(theta_GJ_max) * vis_dir + np.sin(theta_GJ_max) * (np.cos(phi_opt_1) * n_1 + np.sin(phi_opt_1) * n_2)
    phi_opt_2 = phi_opt_1 + np.pi
    new_dir_2 = np.cos(theta_GJ_max) * vis_dir + np.sin(theta_GJ_max) * (np.cos(phi_opt_2) * n_1 + np.sin(phi_opt_2) * n_2)

    mask_dir = new_dir_1.dot(tau_dir) > new_dir_2.dot(tau_dir)
    new_dir = ak.where(mask_dir, new_dir_1, new_dir_2)

    new_tau = ak.zip({
        "px": tau_p_mag * new_dir.x,
        "py": tau_p_mag * new_dir.y,
        "pz": tau_p_mag * new_dir.z,
        "E":  np.sqrt(tau_p_mag**2 + mass_tau**2),
    }, with_name="Momentum4D")

    return ak.where(mask, new_tau, tau)

def estimate_PV_tau_momentum_magnitude(df, tau_prefix, use_map=True):
    """
    tau prefix is taun or taup
    """
    predicted_prefix = 'map_pred' if use_map else 'pred'
    # Visible tau
    piOS = ak.zip({"px": df[f"reco_{tau_prefix}_pi1_px"], "py": df[f"reco_{tau_prefix}_pi1_py"], "pz": df[f"reco_{tau_prefix}_pi1_pz"], "E": df[f"reco_{tau_prefix}_pi1_E"]}, with_name="Momentum4D")
    piSS1 = ak.zip({"px": df[f"reco_{tau_prefix}_pi2_px"], "py": df[f"reco_{tau_prefix}_pi2_py"], "pz": df[f"reco_{tau_prefix}_pi2_pz"], "E": df[f"reco_{tau_prefix}_pi2_E"]}, with_name="Momentum4D")
    piSS2 = ak.zip({"px": df[f"reco_{tau_prefix}_pi3_px"], "py": df[f"reco_{tau_prefix}_pi3_py"], "pz": df[f"reco_{tau_prefix}_pi3_pz"], "E": df[f"reco_{tau_prefix}_pi3_E"]}, with_name="Momentum4D")
    vis_tau = piOS + piSS1 + piSS2

    # Taus estimated from norm flow momentum and SV direction
    if f"FastMTT_pt_{tau_prefix}_constraint" in df.columns:
        # use fastMTT
        sv_direction = ak.zip({"x": df[f"reco_{tau_prefix}_sv_x"], "y": df[f"reco_{tau_prefix}_sv_y"], "z": df[f"reco_{tau_prefix}_sv_z"]}, with_name="Vector3D")
        fastmtt_pt = df[f"FastMTT_pt_{tau_prefix}_constraint"]
        tau = pt_direction_to_momentum4d(fastmtt_pt, sv_direction, 1.777)
    else:
        # fallback on predicted momentum magnitude
        print(f"Warning: FastMTT pT column FastMTT_pt_{tau_prefix}_constraint not found, using predicted momentum magnitude instead!")
        pred_tau_name = 'tau_minus' if tau_prefix == 'taun' else 'tau_plus'
        tau_mag = np.sqrt(df[f"{predicted_prefix}_{pred_tau_name}_px"]**2 + df[f"{predicted_prefix}_{pred_tau_name}_py"]**2 + df[f"{predicted_prefix}_{pred_tau_name}_pz"]**2)
        sv_mag = np.sqrt(df[f"reco_{tau_prefix}_sv_x"]**2 + df[f"reco_{tau_prefix}_sv_y"]**2 + df[f"reco_{tau_prefix}_sv_z"]**2)
        tau = ak.zip({"px": tau_mag * df[f"reco_{tau_prefix}_sv_x"] / sv_mag, "py": tau_mag * df[f"reco_{tau_prefix}_sv_y"] / sv_mag, "pz": tau_mag * df[f"reco_{tau_prefix}_sv_z"] / sv_mag, "E":  np.sqrt(tau_mag**2 + 1.777**2)}, with_name="Momentum4D")

    # Rotate to maximally allowed GJ angle
    tau = rotate_to_GJMax(vis_tau, tau)

    # boost to tau rest frames for polarimetric vector calculation
    frame_tau = boost_vec(tau)

    tau_rf = boost4(tau, frame_tau)
    piOS_rf  = boost4(piOS, frame_tau)
    piSS1_rf  = boost4(piSS1, frame_tau)
    piSS2_rf = boost4(piSS2, frame_tau)

    # get Pvecs
    charge = -1 if tau_prefix == 'taun' else +1
    pvec = polarimetric_vec_dm10(tau_rf, piOS_rf, piSS1_rf, piSS2_rf, charge)

    return tau, pvec

def tauMomentumSolutions(tauDir, a1LV):
    tauMass = 1.77682
    a1P2 = a1LV.px**2 + a1LV.py**2 + a1LV.pz**2
    a1M2 = a1LV.dot(a1LV)
    cosThetaGJ = np.clip(spatial(a1LV).unit().dot(tauDir), -1.0, 1.0)
    sin2ThetaGJ = 1.0 - cosThetaGJ**2

    a = 4.0 * (a1M2 + a1P2 * sin2ThetaGJ)
    b = -4.0 * (a1M2 + tauMass**2) * np.sqrt(a1P2) * cosThetaGJ
    c = 4.0 * tauMass**2 * (a1M2 + a1P2) - (a1M2 + tauMass**2)**2

    tauMomentumSmall, tauMomentumLarge = _quadratic(a, b, c)
    tauMomentumMean = (tauMomentumSmall + tauMomentumLarge) / 2.0

    def makeTauLV(p):
        return ak.zip({
            "px": p * tauDir.x,
            "py": p * tauDir.y,
            "pz": p * tauDir.z,
            "E":  np.sqrt(p**2 + tauMass**2),
        }, with_name="Momentum4D")

    return makeTauLV(tauMomentumSmall), makeTauLV(tauMomentumLarge), makeTauLV(tauMomentumMean)


def tauPairMomentumSolutions(tau1Dir, a1LV1, tau2Dir, a1LV2):
    Hmass = 125.10

    tau1Solutions = tauMomentumSolutions(tau1Dir, a1LV1)
    tau2Solutions = tauMomentumSolutions(tau2Dir, a1LV2)

    d00 = np.abs(invMass(tau1Solutions[0], tau2Solutions[0]) - Hmass)
    d01 = np.abs(invMass(tau1Solutions[0], tau2Solutions[1]) - Hmass)
    d10 = np.abs(invMass(tau1Solutions[1], tau2Solutions[0]) - Hmass)
    d11 = np.abs(invMass(tau1Solutions[1], tau2Solutions[1]) - Hmass)

    bestTau2ForSmall1 = ak.where(d00 < d01, tau2Solutions[0], tau2Solutions[1])
    bestDSmall1 = ak.where(d00 < d01, d00, d01)
    bestTau2ForLarge1 = ak.where(d10 < d11, tau2Solutions[0], tau2Solutions[1])
    bestDLarge1 = ak.where(d10 < d11, d10, d11)

    tau1PairConstraintLV = ak.where(bestDSmall1 < bestDLarge1, tau1Solutions[0], tau1Solutions[1])
    tau2PairConstraintLV = ak.where(bestDSmall1 < bestDLarge1, bestTau2ForSmall1, bestTau2ForLarge1)

    return tau1PairConstraintLV, tau2PairConstraintLV


def get_R_P_vectors_all(df, tau_prefix='tau', use_map=True):
    """Compute R and P vectors for all events, selecting IP (DM0) or pizero (DM1) for R."""
    P_pion = ak.zip({
        "px": df[f"reco_{tau_prefix}_pi1_px"],
        "py": df[f"reco_{tau_prefix}_pi1_py"],
        "pz": df[f"reco_{tau_prefix}_pi1_pz"],
        "E":  df[f"reco_{tau_prefix}_pi1_E"],
    }, with_name="Momentum4D")

    P_lep = ak.zip({
        "px": df[f"reco_{tau_prefix}_charged_px"],
        "py": df[f"reco_{tau_prefix}_charged_py"],
        "pz": df[f"reco_{tau_prefix}_charged_pz"],
        "E":  df[f"reco_{tau_prefix}_charged_E"],
    }, with_name="Momentum4D")

    R_ip = ak.zip({
        "px": df[f"reco_{tau_prefix}_pi1_ipx"],
        "py": df[f"reco_{tau_prefix}_pi1_ipy"],
        "pz": df[f"reco_{tau_prefix}_pi1_ipz"],
        "E":  ak.zeros_like(df[f"reco_{tau_prefix}_pi1_ipx"]),
    }, with_name="Momentum4D")

    R_lep_ip = ak.zip({
        "px": df[f"reco_{tau_prefix}_charged_ipx"],
        "py": df[f"reco_{tau_prefix}_charged_ipy"],
        "pz": df[f"reco_{tau_prefix}_charged_ipz"],
        "E":  ak.zeros_like(df[f"reco_{tau_prefix}_charged_ipx"]),
    }, with_name="Momentum4D")

    R_pizero = ak.zip({
        "px": df[f"reco_{tau_prefix}_pizero1_px"],
        "py": df[f"reco_{tau_prefix}_pizero1_py"],
        "pz": df[f"reco_{tau_prefix}_pizero1_pz"],
        "E":  df[f"reco_{tau_prefix}_pizero1_E"],
    }, with_name="Momentum4D")

    # tau momentum prediction from the flow (magnitude) and SV-PV direction
    P_tau, R_PV = estimate_PV_tau_momentum_magnitude(df, tau_prefix, use_map=use_map)
    R_PV_4d = ak.zip({"px": R_PV.x, "py": R_PV.y, "pz": R_PV.z, "E": ak.zeros_like(R_PV.x)}, with_name="Momentum4D")

    is_leptonic = df[f"{tau_prefix}_DM"].values == 100
    is_dm0 = df[f"{tau_prefix}_DM"].values == 0
    is_dm1dm2 = df[f"{tau_prefix}_DM"].isin([1, 2]).values
    is_dm10 = df[f"{tau_prefix}_DM"].values == 10

    R = ak.with_name(ak.where(is_dm0, R_ip, 
                        ak.where(is_leptonic, R_lep_ip, 
                                ak.where(is_dm1dm2, R_pizero, R_PV_4d))), "Momentum4D")
    P = ak.with_name(ak.where(is_dm10, P_tau, 
                        ak.where(is_leptonic, P_lep, P_pion)), "Momentum4D")

    return R, P, is_dm1dm2

def compute_aco_polarimetric(R1, P1, R2, P2):
    R1perp = R1 - ((R1.dot(P1)) / (P1.dot(P1))) * P1
    R2perp = R2 - ((R2.dot(P2)) / (P2.dot(P2))) * P2
    angle = np.arccos((R1perp.unit()).dot(R2perp.unit()))
    sign = (P2.unit()).dot((R1perp.unit()).cross(R2perp.unit()))
    angle = ak.where(sign >= 0, angle, 2 * np.pi - angle)
    return angle

def compute_aco_classic(R1, P1, R2, P2, leg1_is_dp, leg2_is_dp):
    """Compute the acoplanarity angle using the calculated R and P vectors, with shifts for meson polarisation with decay plane"""
    # Boost to visible charged decay product frame
    bv = boost_vec(P1 + P2)

    R1_p3, R1_E = spatial(boost4(R1, bv), return_E=True)
    R2_p3, R2_E = spatial(boost4(R2, bv), return_E=True)
    P1_p3, P1_E = spatial(boost4(P1, bv), return_E=True)
    P2_p3, P2_E = spatial(boost4(P2, bv), return_E=True)

    # Get perpendicular components
    R1perp = R1_p3 - ((R1_p3.dot(P1_p3)) / (P1_p3.dot(P1_p3))) * P1_p3
    R2perp = R2_p3 - ((R2_p3.dot(P2_p3)) / (P2_p3.dot(P2_p3))) * P2_p3

    # Define angle between planes and O*, then apply appropriate shift
    angle = np.arccos((R1perp.unit()).dot(R2perp.unit()))
    sign  = (P2_p3.unit()).dot((R1perp.unit()).cross(R2perp.unit()))
    angle = ak.where(sign >= 0, angle, 2 * np.pi - angle)

    # Apply shift for polarisation of mesons
    Y1 = (R1_E - P1_E) / (P1_E + R1_E)
    Y2 = (R2_E - P2_E) / (P2_E + R2_E)
    meson_sign = ak.where(leg1_is_dp, np.sign(Y1), ak.ones_like(angle))
    meson_sign = meson_sign * ak.where(leg2_is_dp, np.sign(Y2), ak.ones_like(angle))
    needs_shift = (leg1_is_dp | leg2_is_dp) & (meson_sign < 0)
    angle = ak.where(needs_shift, ak.where(angle < np.pi, angle + np.pi, angle - np.pi), angle)
    return angle

def compute_aco_classic_a1a1(df):
    # Tau plus
    piOS_p = ak.zip({"px": df[f"reco_taup_pi1_px"], "py": df[f"reco_taup_pi1_py"], "pz": df[f"reco_taup_pi1_pz"], "E": df[f"reco_taup_pi1_E"]}, with_name="Momentum4D")
    piSS1_p = ak.zip({"px": df[f"reco_taup_pi2_px"], "py": df[f"reco_taup_pi2_py"], "pz": df[f"reco_taup_pi2_pz"], "E": df[f"reco_taup_pi2_E"]}, with_name="Momentum4D")
    piSS2_p = ak.zip({"px": df[f"reco_taup_pi3_px"], "py": df[f"reco_taup_pi3_py"], "pz": df[f"reco_taup_pi3_pz"], "E": df[f"reco_taup_pi3_E"]}, with_name="Momentum4D")
    vis_tau_p = piOS_p + piSS1_p + piSS2_p
    dir_tau_p =  ak.zip({"x": df[f"reco_taup_sv_x"], "y": df[f"reco_taup_sv_y"], "z": df[f"reco_taup_sv_z"] }, with_name="Vector3D").unit()

    # Tau plus
    piOS_n = ak.zip({"px": df[f"reco_taun_pi1_px"], "py": df[f"reco_taun_pi1_py"], "pz": df[f"reco_taun_pi1_pz"], "E": df[f"reco_taun_pi1_E"]}, with_name="Momentum4D")
    piSS1_n = ak.zip({"px": df[f"reco_taun_pi2_px"], "py": df[f"reco_taun_pi2_py"], "pz": df[f"reco_taun_pi2_pz"], "E": df[f"reco_taun_pi2_E"]}, with_name="Momentum4D")
    piSS2_n = ak.zip({"px": df[f"reco_taun_pi3_px"], "py": df[f"reco_taun_pi3_py"], "pz": df[f"reco_taun_pi3_pz"], "E": df[f"reco_taun_pi3_E"]}, with_name="Momentum4D")
    vis_tau_n = piOS_n + piSS1_n + piSS2_n
    dir_tau_n =  ak.zip({"x": df[f"reco_taun_sv_x"], "y": df[f"reco_taun_sv_y"], "z": df[f"reco_taun_sv_z"] }, with_name="Vector3D").unit()

    # work out the tau momenta using pair mass constraint
    tau_p, tau_n = tauPairMomentumSolutions(dir_tau_p, vis_tau_p, dir_tau_n, vis_tau_n)

    # boost to this frame
    higgs_bv = boost_vec(tau_p + tau_n)

    # Boost the decay products to the Higgs rest frame
    tau_p_rf = boost4(tau_p, higgs_bv)
    tau_n_rf = boost4(tau_n, higgs_bv)
    piOS_p_rf  = boost4(piOS_p, higgs_bv)
    piSS1_p_rf  = boost4(piSS1_p, higgs_bv)
    piSS2_p_rf = boost4(piSS2_p, higgs_bv)
    piOS_n_rf  = boost4(piOS_n, higgs_bv)
    piSS1_n_rf  = boost4(piSS1_n, higgs_bv)
    piSS2_n_rf = boost4(piSS2_n, higgs_bv)

    # Get the polarimetric vectors
    taup_s = polarimetric_vec_dm10(tau_p_rf, piOS_p_rf, piSS1_p_rf, piSS2_p_rf, +1)
    taun_s = polarimetric_vec_dm10(tau_n_rf, piOS_n_rf, piSS1_n_rf, piSS2_n_rf, -1)

    # Get phiCP
    phicp = compute_aco_polarimetric(taup_s, spatial(tau_p_rf).unit(), taun_s, spatial(tau_n_rf).unit())

    return phicp



def get_ditau_polarimetric(df, tau_prefix='true', reco_pions=True, add_ghosts=True):
    """
    tau_prefix: sets whether reco or true tau is used ("true" or "map_pred")
    reco_pions: sets whether to use reco pions or true pions (some storage issues in gen)
    add_ghosts: whether to add a "ghost" particle with small momentum to taus to handel cases where neutrinos are predicted to have zero momentum which causes issues for the polarimetric vector calculation. 
    """

    # Get R and P (polarimetric vector edition)
    if reco_pions:
        print(">> Using reconstructed pions for polarimetric vector calculation.")
        pion_prefix = 'reco'
    else:
        print('>> Using true pions for polarimetric vector calculation.')
        pion_prefix = 'true'

    # check if pi2 exists if not return 0
    if f"{pion_prefix}_taup_pi2_px" not in df.columns or f"{pion_prefix}_taun_pi2_px" not in df.columns:
        print(">> pi2 information not found, returning zero polarimetric vectors.")
        num_events = len(df)
        zero_vector = ak.zip({"x": ak.zeros_like(df[f"{pion_prefix}_taup_pi1_px"]), "y": ak.zeros_like(df[f"{pion_prefix}_taup_pi1_px"]), "z": ak.zeros_like(df[f"{pion_prefix}_taup_pi1_px"])}, with_name="Vector3D")
        return zero_vector, zero_vector, zero_vector, zero_vector 

    # Tau plus decay mode
    taup_is_leptonic = df["taup_DM"].values == 100
    taup_is_dm0 = df["taup_DM"].values == 0
    taup_is_dm1or2 = df["taup_DM"].isin([1, 2]).values
    taup_is_dm10 = df["taup_DM"].values == 10
    taup_is_dm11 = df["taup_DM"].values == 11

    # Tau plus decay products
    piOS_p = ak.zip({"px": df[f"{pion_prefix}_taup_pi1_px"], "py": df[f"{pion_prefix}_taup_pi1_py"], "pz": df[f"{pion_prefix}_taup_pi1_pz"], "E": df[f"{pion_prefix}_taup_pi1_E"]}, with_name="Momentum4D")  # only actually OS in 3 prong case
    piSS1_p = ak.zip({"px": df[f"{pion_prefix}_taup_pi2_px"], "py": df[f"{pion_prefix}_taup_pi2_py"], "pz": df[f"{pion_prefix}_taup_pi2_pz"], "E": df[f"{pion_prefix}_taup_pi2_E"]}, with_name="Momentum4D")
    piSS2_p = ak.zip({"px": df[f"{pion_prefix}_taup_pi3_px"], "py": df[f"{pion_prefix}_taup_pi3_py"], "pz": df[f"{pion_prefix}_taup_pi3_pz"], "E": df[f"{pion_prefix}_taup_pi3_E"]}, with_name="Momentum4D")
    pizero_p = ak.zip({"px": df[f"{pion_prefix}_taup_pizero1_px"], "py": df[f"{pion_prefix}_taup_pizero1_py"], "pz": df[f"{pion_prefix}_taup_pizero1_pz"], "E": df[f"{pion_prefix}_taup_pizero1_E"]}, with_name="Momentum4D")
    lep_p = ak.zip({"px": df[f"{pion_prefix}_taup_charged_px"], "py": df[f"{pion_prefix}_taup_charged_py"], "pz": df[f"{pion_prefix}_taup_charged_pz"], "E": df[f"{pion_prefix}_taup_charged_E"]}, with_name="Momentum4D")

    # Tau plus
    tau_p = ak.zip({"px": df[f"{tau_prefix}_tau_plus_px"], "py": df[f"{tau_prefix}_tau_plus_py"], "pz": df[f"{tau_prefix}_tau_plus_pz"], "E": df[f"{tau_prefix}_tau_plus_E"]}, with_name="Momentum4D")

    # Tau minus decay mode
    taun_is_leptonic = df["taun_DM"].values == 100
    taun_is_dm0 = df["taun_DM"].values == 0
    taun_is_dm1or2 = df["taun_DM"].isin([1, 2]).values
    taun_is_dm10 = df["taun_DM"].values == 10
    taun_is_dm11 = df["taun_DM"].values == 11

    # Tau minus decay products

    piOS_n = ak.zip({"px": df[f"{pion_prefix}_taun_pi1_px"], "py": df[f"{pion_prefix}_taun_pi1_py"], "pz": df[f"{pion_prefix}_taun_pi1_pz"], "E": df[f"{pion_prefix}_taun_pi1_E"]}, with_name="Momentum4D")
    piSS1_n = ak.zip({"px": df[f"{pion_prefix}_taun_pi2_px"], "py": df[f"{pion_prefix}_taun_pi2_py"], "pz": df[f"{pion_prefix}_taun_pi2_pz"], "E": df[f"{pion_prefix}_taun_pi2_E"]}, with_name="Momentum4D")
    piSS2_n = ak.zip({"px": df[f"{pion_prefix}_taun_pi3_px"], "py": df[f"{pion_prefix}_taun_pi3_py"], "pz": df[f"{pion_prefix}_taun_pi3_pz"], "E": df[f"{pion_prefix}_taun_pi3_E"]}, with_name="Momentum4D")
    pizero_n = ak.zip({"px": df[f"{pion_prefix}_taun_pizero1_px"], "py": df[f"{pion_prefix}_taun_pizero1_py"], "pz": df[f"{pion_prefix}_taun_pizero1_pz"], "E": df[f"{pion_prefix}_taun_pizero1_E"]}, with_name="Momentum4D")
    lep_n = ak.zip({"px": df[f"{pion_prefix}_taun_charged_px"], "py": df[f"{pion_prefix}_taun_charged_py"], "pz": df[f"{pion_prefix}_taun_charged_pz"], "E": df[f"{pion_prefix}_taun_charged_E"]}, with_name="Momentum4D")

    #lep_n = ak.zip({"px": df[f"{tau_prefix}_taun_lep_px"], "py": df[f"{tau_prefix}_taun_lep_py"], "pz": df[f"{tau_prefix}_taun_lep_pz"], "E": df[f"{tau_prefix}_taun_lep_E"]}, with_name="Momentum4D")

    # Tau minus
    tau_n = ak.zip({"px": df[f"{tau_prefix}_tau_minus_px"], "py": df[f"{tau_prefix}_tau_minus_py"], "pz": df[f"{tau_prefix}_tau_minus_pz"], "E": df[f"{tau_prefix}_tau_minus_E"]}, with_name="Momentum4D")

    if add_ghosts:
        # add small "ghost" particle to the taus to handle cases where the tau momentum is exactly equal to the visible momentum (i.e when neutrino was estimated to have exactly zero momentum)
        # ghosts will be in the same direction as the tau and have total p=E=epsilon
        epsilon = 1e-3
        tau_p_pmag = np.sqrt(tau_p.px**2 + tau_p.py**2 + tau_p.pz**2)
        tau_n_pmag = np.sqrt(tau_n.px**2 + tau_n.py**2 + tau_n.pz**2)
        tau_p_unit = ak.zip({"x": tau_p.px / tau_p_pmag, "y": tau_p.py / tau_p_pmag, "z": tau_p.pz / tau_p_pmag}, with_name="Vector3D")
        tau_n_unit = ak.zip({"x": tau_n.px / tau_n_pmag, "y": tau_n.py / tau_n_pmag, "z": tau_n.pz / tau_n_pmag}, with_name="Vector3D")
        ghost_p = ak.zip({"px": tau_p_unit.x * epsilon, "py": tau_p_unit.y * epsilon, "pz": tau_p_unit.z * epsilon, "E": ak.full_like(tau_p.E, epsilon)}, with_name="Momentum4D")
        ghost_n = ak.zip({"px": tau_n_unit.x * epsilon, "py": tau_n_unit.y * epsilon, "pz": tau_n_unit.z * epsilon, "E": ak.full_like(tau_n.E, epsilon)}, with_name="Momentum4D")
        tau_p = tau_p + ghost_p
        tau_n = tau_n + ghost_n

    # Higgs rest frame boost vectors
    higgs_bv = boost_vec(tau_p + tau_n)

    # Boost the decay products to the Higgs rest frame
    tau_p_rf = boost4(tau_p, higgs_bv)
    tau_n_rf = boost4(tau_n, higgs_bv)
    piOS_p_rf  = boost4(piOS_p, higgs_bv)
    piSS1_p_rf  = boost4(piSS1_p, higgs_bv)
    piSS2_p_rf = boost4(piSS2_p, higgs_bv)
    pizero_p_rf = boost4(pizero_p, higgs_bv)
    lep_p_rf = boost4(lep_p, higgs_bv)
    piOS_n_rf  = boost4(piOS_n, higgs_bv)
    piSS1_n_rf  = boost4(piSS1_n, higgs_bv)
    piSS2_n_rf = boost4(piSS2_n, higgs_bv)
    pizero_n_rf = boost4(pizero_n, higgs_bv)
    lep_n_rf = boost4(lep_n, higgs_bv)

    # Tau rest frame boost vectors (from Higgs rest frame)
    bv_taup = boost_vec(tau_p_rf)
    bv_taun = boost_vec(tau_n_rf)

    # Get the polarimetric vectors
    default = ak.zip({"x": ak.full_like(piOS_p_rf.px, -9999.0),
                        "y": ak.full_like(piOS_p_rf.py, -9999.0),
                        "z": ak.full_like(piOS_p_rf.pz, -9999.0)}, with_name="Vector3D")

    taup_s = ak.where(taup_is_dm0, polarimetric_vec_dm0(piOS_p_rf, bv_taup),
                        ak.where(taup_is_dm1or2, polarimetric_vec_dm1(piOS_p_rf, pizero_p_rf, tau_p_rf, bv_taup),
                                 ak.where(taup_is_dm10, polarimetric_vec_dm10(tau_p_rf, piOS_p_rf, piSS1_p_rf, piSS2_p_rf, +1),
                                    ak.where(taup_is_dm11, polarimetric_vec_dm11(tau_p_rf, piOS_p_rf, piSS1_p_rf, piSS2_p_rf, pizero_p_rf, +1),
                                       ak.where(taup_is_leptonic, polarimetric_vec_leptonic(lep_p_rf, bv_taup),
                                             default)))))


    taun_s = ak.where(taun_is_dm0, polarimetric_vec_dm0(piOS_n_rf, bv_taun),
                        ak.where(taun_is_dm1or2, polarimetric_vec_dm1(piOS_n_rf, pizero_n_rf, tau_n_rf, bv_taun),
                                 ak.where(taun_is_dm10, polarimetric_vec_dm10(tau_n_rf, piOS_n_rf, piSS1_n_rf, piSS2_n_rf, -1),
                                    ak.where(taun_is_dm11, polarimetric_vec_dm11(tau_n_rf, piOS_n_rf, piSS1_n_rf, piSS2_n_rf, pizero_n_rf, -1),
                                       ak.where(taun_is_leptonic, polarimetric_vec_leptonic(lep_n_rf, bv_taun),
                                             default)))))

    return taup_s, spatial(tau_p_rf).unit(), taun_s, spatial(tau_n_rf).unit()
