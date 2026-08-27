from __future__ import annotations

import numpy as np
from taupolaris.utils.PolarimetricA1 import Vec4
from taupolaris.utils import tauola_fortran as _tf

# As in PolarimetricA1.py: ROOT is needed only by the scalar Polarimetric3hpi0
# class below, not by Polarimetric3hpi0_vectorised (pure numpy), so it must not
# be a hard import -- see the comment there.
try:
    from ROOT import TLorentzVector, TComplex
except ImportError:  # ROOT unavailable -- only the scalar class is affected
    TLorentzVector = TComplex = None

"""
Polarimetric vector for tau -> 3h + pi0 (DM=11: 3 charged hadrons + 1 neutral
pion), i.e. the TAUOLA "curr_cleo" MNUM=1 hadronic current (pi-pi-pi0pi+ and
charge conjugate), following the same pattern as PolarimetricA1.py for DM=10:

  - Polarimetric3hpi0            : single-event, ROOT TLorentzVector/TComplex
  - Polarimetric3hpi0_vectorised : array-backed (numpy/awkward Momentum4D), for
                                    processing many events at once

Both reuse the exact same CLVEC/CLAXI/normalisation formula as PolarimetricA1
(general neutrino 4-vector N = P_tau - visible, no frame rotation required),
only the hadronic current construction (curr_cleo instead of F3PI) differs.
The non-vectorised class calls tauola_fortran.curr_cleo directly (already
validated via calculate_hh.py/dam4pi); the vectorised class re-implements that
same current using numpy array arithmetic (validated against the scalar
version -- see tests).

Argument/pion-ordering convention matches PolarimetricA1: p4_os_pi is the
pion with charge opposite to the tau, p4_ss1_pi/p4_ss2_pi are the two pions
with the same charge as the tau, p4_pi0 is the neutral pion. All 4-vectors
must be given in the tau rest frame. taucharge is +1/-1 (or an array thereof
for the vectorised class).
"""


class Polarimetric3hpi0:
    def __init__(self,
                 p4_tau: TLorentzVector,
                 p4_os_pi: TLorentzVector,
                 p4_ss1_pi: TLorentzVector,
                 p4_ss2_pi: TLorentzVector,
                 p4_pi0: TLorentzVector,
                 taucharge: float) -> None:
        self.p4_tau    = p4_tau
        self.p4_os_pi  = p4_os_pi
        self.p4_ss1_pi = p4_ss1_pi
        self.p4_ss2_pi = p4_ss2_pi
        self.p4_pi0    = p4_pi0

        self.SIGN = -taucharge

    @staticmethod
    def _pxpypzE(p4: TLorentzVector):
        return [p4.Px(), p4.Py(), p4.Pz(), p4.E()]

    def PVC(self) -> TLorentzVector:
        P = self.p4_tau

        pim1 = self._pxpypzE(self.p4_ss1_pi)
        pim2 = self._pxpypzE(self.p4_ss2_pi)
        pim3 = self._pxpypzE(self.p4_pi0)
        pim4 = self._pxpypzE(self.p4_os_pi)

        hadcur = _tf.curr_cleo(1, pim1, pim2, pim3, pim4)  # [px,py,pz,E], python complex

        a1 = self.p4_ss1_pi + self.p4_ss2_pi + self.p4_pi0 + self.p4_os_pi
        N = P - a1

        H = [
            TComplex(hadcur[3].real, hadcur[3].imag),
            TComplex(hadcur[0].real, hadcur[0].imag),
            TComplex(hadcur[1].real, hadcur[1].imag),
            TComplex(hadcur[2].real, hadcur[2].imag),
        ]
        HC = [TComplex.Conjugate(h) for h in H]

        CLV = self.CLVEC(H, HC, N)
        CLA = self.CLAXI(H, HC, N)

        omega = P*CLV - P*CLA
        out = (P.M2() * (CLA-CLV) - P*(P*CLA - P*CLV))*(1/omega/P.M())

        return out

    def CLVEC(self, H: list, HC: list, N: TLorentzVector) -> TLorentzVector:
        HN  = H[0]*N.E() - H[1]*N.Px() - H[2]*N.Py() - H[3]*N.Pz()
        HCN = HC[0]*N.E() - HC[1]*N.Px() - HC[2]*N.Py() - HC[3]*N.Pz()
        HH  = (H[0]*HC[0] - H[1]*HC[1] - H[2]*HC[2] - H[3]*HC[3]).Re()

        PIVEC0 = 2*( 2*(HN*HC[0]).Re() - HH*N.E() )
        PIVEC1 = 2*( 2*(HN*HC[1]).Re() - HH*N.Px()     )
        PIVEC2 = 2*( 2*(HN*HC[2]).Re() - HH*N.Py()     )
        PIVEC3 = 2*( 2*(HN*HC[3]).Re() - HH*N.Pz()     )

        return TLorentzVector(PIVEC1, PIVEC2, PIVEC3, PIVEC0)

    def CLAXI(self, H: list, HC: list, N: TLorentzVector) -> TLorentzVector:
        a1 = HC[1]; a2 = HC[2]; a3 = HC[3]; a4 = HC[0]
        b1 =  H[1]; b2 =  H[2]; b3 =  H[3]; b4 =  H[0]
        c1 = N.Px(); c2 = N.Py(); c3 = N.Pz(); c4 = N.E()

        d34 = (a3*b4 - a4*b3).Im()
        d24 = (a2*b4 - a4*b2).Im()
        d23 = (a2*b3 - a3*b2).Im()
        d14 = (a1*b4 - a4*b1).Im()
        d13 = (a1*b3 - a3*b1).Im()
        d12 = (a1*b2 - a2*b1).Im()

        PIAX0 = -self.SIGN*2*(-c1*d23 + c2*d13 - c3*d12)
        PIAX1 = self.SIGN*2*(c2*d34 - c3*d24 + c4*d23)
        PIAX2 = self.SIGN*2*(-c1*d34 + c3*d14 - c4*d13)
        PIAX3 = self.SIGN*2*(c1*d24 - c2*d14 + c4*d12)

        return TLorentzVector(PIAX1, PIAX2, PIAX3, PIAX0)


# ---------------------------------------------------------------------------
# Vectorised curr_cleo (MNUM=1: pi-pi-pi0pi+ / charge conjugate)
# Convention: 4-momenta are tuples/lists (px, py, pz, E) of numpy arrays,
# matching tauola_fortran.py's own [px,py,pz,E] indexing.
# ---------------------------------------------------------------------------

def _mass2_arr(p):
    return p[3]**2 - p[2]**2 - p[1]**2 - p[0]**2


def _dot4_arr(a, b):
    return a[3]*b[3] - a[2]*b[2] - a[1]*b[1] - a[0]*b[0]


def _bwign_arr(A, XM, XG):
    return 1.0 / (A - XM**2 + 1j*XM*XG)


def _bwigm_arr(S, M, G, XM1, XM2):
    thresh = (XM1 + XM2)**2
    above  = S > thresh
    S_safe = np.where(above, S, 1.0)
    W      = np.sqrt(np.maximum(S_safe, 0.0))

    QS  = np.sqrt(np.abs((S_safe - (XM1+XM2)**2)*(S_safe - (XM1-XM2)**2))) / W
    QM  = np.sqrt(np.abs((M**2   - (XM1+XM2)**2)*(M**2   - (XM1-XM2)**2))) / M
    GS  = np.where(above, G*(M/W)**2*(QS/QM)**3, 0.0)

    return (M**2) / (M**2 - S - 1j*np.sqrt(np.maximum(S, 0.0))*GS)


def _curr_cleo_mnum1(PIM1, PIM2, PIM3, PIM4):
    """
    Vectorised re-implementation of tauola_fortran.curr_cleo(1, ...).
    PIM1, PIM2 : same-sign pions (px,py,pz,E arrays)
    PIM3       : neutral pion
    PIM4       : opposite-sign pion
    Returns hadcur[0..3] = (px, py, pz, E) complex arrays.
    """
    (AMRO2, GAMRO2, AMRO3, GAMRO3, AMOM, GAMOM,
     AMPL, ALF0, ALF1, ALF2, ALF3,
     LAM0, LAM1, LAM2, LAM3, BET1, BET2, BET3) = _tf._CC

    AMPI, AMPIZ, AMRO, GAMRO = _tf.AMPI, _tf.AMPIZ, _tf.AMRO, _tf.GAMRO

    PP  = [PIM1, PIM2, PIM3, PIM4]
    PAA = [PP[0][i] + PP[1][i] + PP[2][i] + PP[3][i] for i in range(4)]

    hadcur = [0j, 0j, 0j, 0j]

    QQ = _mass2_arr(PAA)
    FORM4 = (LAM0 + LAM1*_bwign_arr(QQ, AMRO, GAMRO)
                  + LAM2*_bwign_arr(QQ, AMRO2, GAMRO2)
                  + LAM3*_bwign_arr(QQ, AMRO3, GAMRO3))

    for K1, K2 in [(1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
        if K2 == 3:
            AMPR, AMPA = AMPL[3], AMPIZ
        elif K1 == 3:
            AMPR, AMPA = AMPL[4], AMPIZ
        else:
            AMPR, AMPA = AMPL[2], AMPI

        SK = _mass2_arr([PP[K1-1][i] + PP[K2-1][i] for i in range(4)])

        AA = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        for L in range(1, 5):
            if L == K1 or L == K2:
                continue
            PL = PP[L-1]
            DENOM = _mass2_arr([PAA[i] - PL[i] for i in range(4)])
            for i in range(4):
                for j in range(4):
                    SIG = 1.0 if j == 3 else -1.0
                    AA[i][j] = AA[i][j] - SIG*(PAA[i] - 2*PL[i])*(PAA[j] - PL[j])/DENOM

        FORM2PI = (BET1*_bwigm_arr(SK, AMRO,  GAMRO,  AMPA, AMPI)
                 + BET2*_bwigm_arr(SK, AMRO2, GAMRO2, AMPA, AMPI)
                 + BET3*_bwigm_arr(SK, AMRO3, GAMRO3, AMPA, AMPI))
        FORM1_val = AMPL[1] + AMPR*FORM2PI

        for i in range(4):
            s = 0j
            for j in range(4):
                s = s + FORM1_val*FORM4*AA[i][j]*(PP[K1-1][j] - PP[K2-1][j])
            hadcur[i] = hadcur[i] + s

    # omega current (AMPL[5] is always non-zero, cf. tauola_fortran.curr_cleo)
    FORM2_om = AMPL[5]*(ALF0 + ALF1*_bwign_arr(QQ, AMRO, GAMRO)
                             + ALF2*_bwign_arr(QQ, AMRO2, GAMRO2)
                             + ALF3*_bwign_arr(QQ, AMRO3, GAMRO3))
    PIM3v, PIM4v = PP[2], PP[3]
    for KK in (1, 2):
        PA = PP[KK-1]
        PB = PP[2-KK]
        diff = [PAA[i] - PA[i] for i in range(4)]

        QQA   = _mass2_arr(diff)
        QP1P2 = _dot4_arr(diff, PB)
        QP1P3 = _dot4_arr(diff, PIM3v)
        QP1P4 = _dot4_arr(diff, PIM4v)
        P1P2  = _dot4_arr(PA, PB)
        P1P3  = _dot4_arr(PA, PIM3v)
        P1P4  = _dot4_arr(PA, PIM4v)

        FORM3_om = _bwign_arr(QQA, AMOM, GAMOM)

        for K in range(4):
            hadcur[K] = hadcur[K] + FORM2_om*FORM3_om*(
                PB[K]  *(QP1P3*P1P4 - QP1P4*P1P3)
              + PIM3v[K]*(QP1P4*P1P2 - QP1P2*P1P4)
              + PIM4v[K]*(QP1P2*P1P3 - QP1P3*P1P2))

    return hadcur


class Polarimetric3hpi0_vectorised:
    def __init__(self,
                 p4_tau,
                 p4_os_pi,
                 p4_ss1_pi,
                 p4_ss2_pi,
                 p4_pi0,
                 taucharge) -> None:
        """
        Vectorised polarimetric vector for tau -> 3h+pi0 decay, operating on
        arrays of events. Inputs are Momentum4D awkward arrays (fields: E,
        px, py, pz) or any object whose .E / .px / .py / .pz attributes
        return array-like values. taucharge is an array of +/-1 values.
        """

        def _v(p4):
            return Vec4(np.asarray(p4.E), np.asarray(p4.px),
                        np.asarray(p4.py), np.asarray(p4.pz))

        self.p4_tau    = _v(p4_tau)
        self.p4_os_pi  = _v(p4_os_pi)
        self.p4_ss1_pi = _v(p4_ss1_pi)
        self.p4_ss2_pi = _v(p4_ss2_pi)
        self.p4_pi0    = _v(p4_pi0)

        self.SIGN = -np.asarray(taucharge, dtype=float)

    @staticmethod
    def _pxpypzE(v: Vec4):
        return (v.x, v.y, v.z, v.t)

    def hadronic_current(self):
        """The tau -> 3h+pi0 (CLEO 4-pion) hadronic current and its invariants.

        As in PolarimetricA1_vectorised.hadronic_current: depends only on the
        four pion 4-momenta, no tau or neutrino, so it is the complete visible
        half of the polarimetric-vector calculation and can be used directly as
        an engineered input feature.

        Returns (H, scalars) with H ordered [t, x, y, z].
        """
        pim1 = self._pxpypzE(self.p4_ss1_pi)
        pim2 = self._pxpypzE(self.p4_ss2_pi)
        pim3 = self._pxpypzE(self.p4_pi0)
        pim4 = self._pxpypzE(self.p4_os_pi)

        hadcur = _curr_cleo_mnum1(pim1, pim2, pim3, pim4)  # [px,py,pz,E]
        H = [hadcur[3], hadcur[0], hadcur[1], hadcur[2]]

        a1 = self.p4_ss1_pi + self.p4_ss2_pi + self.p4_pi0 + self.p4_os_pi
        os_ss1 = (self.p4_os_pi + self.p4_ss1_pi).M2()
        os_ss2 = (self.p4_os_pi + self.p4_ss2_pi).M2()
        ss_ss  = (self.p4_ss1_pi + self.p4_ss2_pi).M2()
        return H, {'s1': os_ss1, 's2': os_ss2, 's3': ss_ss, 'm2_vis': a1.M2()}

    def PVC(self) -> 'Vec4':
        P = self.p4_tau

        a1 = self.p4_ss1_pi + self.p4_ss2_pi + self.p4_pi0 + self.p4_os_pi
        N  = P - a1

        H, _ = self.hadronic_current()
        HC = [np.conj(h) for h in H]

        CLV = self._CLVEC(H, HC, N)
        CLA = self._CLAXI(H, HC, N)

        omega    = P.dot(CLV) - P.dot(CLA)
        pdotdiff = P.dot(CLA) - P.dot(CLV)
        out = ((CLA - CLV) * P.M2() - P * pdotdiff) * (1.0 / omega / P.M())

        return out

    def _CLVEC(self, H, HC, N):
        HN = H[0]*N.t  - H[1]*N.x  - H[2]*N.y  - H[3]*N.z
        HH = (H[0]*HC[0] - H[1]*HC[1] - H[2]*HC[2] - H[3]*HC[3]).real

        return Vec4(
            2*(2*(HN*HC[0]).real - HH*N.t),
            2*(2*(HN*HC[1]).real - HH*N.x),
            2*(2*(HN*HC[2]).real - HH*N.y),
            2*(2*(HN*HC[3]).real - HH*N.z),
        )

    def _CLAXI(self, H, HC, N):
        a1, a2, a3, a4 = HC[1], HC[2], HC[3], HC[0]
        b1, b2, b3, b4 =  H[1],  H[2],  H[3],  H[0]
        c1, c2, c3, c4 = N.x, N.y, N.z, N.t

        d34 = (a3*b4 - a4*b3).imag
        d24 = (a2*b4 - a4*b2).imag
        d23 = (a2*b3 - a3*b2).imag
        d14 = (a1*b4 - a4*b1).imag
        d13 = (a1*b3 - a3*b1).imag
        d12 = (a1*b2 - a2*b1).imag

        return Vec4(
            -self.SIGN * 2*(-c1*d23 + c2*d13 - c3*d12),
             self.SIGN * 2*( c2*d34 - c3*d24 + c4*d23),
             self.SIGN * 2*(-c1*d34 + c3*d14 - c4*d13),
             self.SIGN * 2*( c1*d24 - c2*d14 + c4*d12),
        )
