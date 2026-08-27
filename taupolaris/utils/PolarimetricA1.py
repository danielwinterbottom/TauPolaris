from __future__ import annotations

import numpy as np

# ROOT is needed only by the scalar, single-event PolarimetricA1 class below.
# PolarimetricA1_vectorised -- the one every production code path actually uses
# (kinematic_helpers.polarimetric_vector_a1, acoplanarity_tools.polarimetric_vec_dm10)
# -- is pure numpy and touches neither TLorentzVector nor TComplex. Importing
# ROOT unconditionally at module scope therefore made the vectorised class
# unusable in environments without a matching ROOT build (e.g. the conda env
# used for training, whose Python version differs from ROOT's), for no reason.
# `from __future__ import annotations` above keeps the TLorentzVector
# annotations on the scalar class from being evaluated at class-creation time.
try:
    from ROOT import TLorentzVector, TComplex
except ImportError:  # ROOT unavailable -- only the scalar class is affected
    TLorentzVector = TComplex = None


class PolarimetricA1:
    def __init__(self,
                 p4_tau: TLorentzVector,
                 p4_os_pi: TLorentzVector,
                 p4_ss1_pi: TLorentzVector,
                 p4_ss2_pi: TLorentzVector,
                 taucharge: float) -> None:        
        """
        Calculate Polarimetric vector for tau to a1 decay
        All the vectors in the arguments must be wrt the rest frame
        """

        self.p4_tau         =  p4_tau       # tau
        self.p4_os_pi       =  p4_os_pi     # os-pion
        self.p4_ss1_pi      =  p4_ss1_pi    # ss1-pion
        self.p4_ss2_pi      =  p4_ss2_pi    # ss2-pion

        self.mpi            =  0.13957018   # GeV
        self.mpi0           =  0.1349766    # GeV
        self.mtau           =  1.776        # GeV
        self.coscab         =  0.975
        self.mrho           =  0.773        # GeV
        self.mrhoprime      =  1.370        # GeV
        self.ma1            =  1.251        # GeV
        self.mpiprime       =  1.300        # GeV
        self.Gamma0rho      =  0.145        # GeV
        self.Gamma0rhoprime =  0.510        # GeV
        self.Gamma0a1       =  0.599        # GeV
        self.Gamma0piprime  =  0.3          # GeV
        self.fpi            =  0.093        # GeV
        self.fpiprime       =  0.08         # GeV
        self.gpiprimerhopi  =  5.8          # GeV
        self.grhopipi       =  6.08         # GeV
        self.beta           = -0.145
        self.COEF1          =  2.0*np.sqrt(2.)/3.0
        self.COEF2          = -2.0*np.sqrt(2.)/3.0
        # C AJW 2/98: Add in the D-wave and I=0 3pi substructure:
        self.COEF3          =  2.0*np.sqrt(2.)/3.0
        self.SIGN           = -taucharge
        self.doSystematic   =  False
        self.systType       =  "UP"


    def PVC(self) -> TLorentzVector:
        P  = self.p4_tau
        q1 = self.p4_ss1_pi
        q2 = self.p4_ss2_pi
        q3 = self.p4_os_pi

        a1 = q1 + q2 + q3

        N = P - a1

        s1 = (q2 + q3).M2()
        s2 = (q1 + q3).M2()
        s3 = (q1 + q2).M2()

        vec1 = q2 - q3 - a1*(a1*(q2-q3)/a1.M2())
        vec2 = q3 - q1 - a1*(a1*(q3-q1)/a1.M2())
        vec3 = q1 - q2 - a1*(a1*(q1-q2)/a1.M2())

        F1 = TComplex(self.COEF1)*self.F3PI(1, a1.M2(), s1, s2)
        F2 = TComplex(self.COEF2)*self.F3PI(2, a1.M2(), s2, s1)
        F3 = TComplex(self.COEF3)*self.F3PI(3, a1.M2(), s3, s1)

        HADCUR = []

        HADCUR.append(TComplex(vec1.E())*F1 + TComplex(vec2.E())*F2 + TComplex(vec3.E())*F3)
        HADCUR.append(TComplex(vec1.Px())*F1 + TComplex(vec2.Px())*F2 + TComplex(vec3.Px())*F3)
        HADCUR.append(TComplex(vec1.Py())*F1 + TComplex(vec2.Py())*F2 + TComplex(vec3.Py())*F3)
        HADCUR.append(TComplex(vec1.Pz())*F1 + TComplex(vec2.Pz())*F2 + TComplex(vec3.Pz())*F3)

        HADCURC = [TComplex.Conjugate(val) for val in HADCUR]

        CLV = self.CLVEC(HADCUR, HADCURC, N)
        CLA = self.CLAXI(HADCUR, HADCURC, N)

        omega   = P*CLV - P*CLA
        out = (P.M2() * (CLA-CLV) - P*(P*CLA - P*CLV))*(1/omega/P.M())

        return out


    def F3PI(self,
             IFORM: float,
             QQ: float,
             SA: float,
             SB: float):
        """
            Calculate the F3PIFactor.
        """
        MRO = 0.7743
        GRO = 0.1491
        MRP = 1.370
        GRP = 0.386
        MF2 = 1.275
        GF2 = 0.185
        MF0 = 1.186
        GF0 = 0.350
        MSG = 0.860
        GSG = 0.880
        MPIZ = self.mpi0
        MPIC = self.mpi

        M1 = 0
        M2 = 0
        M3 = 0

        IDK = 1  # It is 3pi

        if IDK == 1:
            M1 = MPIZ
            M2 = MPIZ
            M3 = MPIC
        elif IDK == 2:
            M1 = MPIC
            M2 = MPIC
            M3 = MPIC

        M1SQ = M1*M1
        M2SQ = M2*M2
        M3SQ = M3*M3


        # parameter varioation for
        # systematics from https://arxiv.org/pdf/hep-ex/9902022.pdf
        db2, dph2 = 0.094, 0.253
        db3, dph3 = 0.094, 0.104
        db4, dph4 = 0.296, 0.170
        db5, dph5 = 0.167, 0.104
        db6, dph6 = 0.284, 0.036
        db7, dph7 = 0.148, 0.063

        scale = 0.0
        if self.doSystematic:
            if self.systType == "UP":
                scale = 1
            elif self.systType == "DOWN":
                scale = -1

        # Breit-Wigner functions with isotropic decay angular distribution
        # Real part must be equal to one, stupid polar implemenation in root
        BT1 = TComplex(1., 0.)
        BT2 = TComplex(0.12  + scale*db2, 0.) * TComplex(1, (0.99   +  scale*dph2)*np.pi, True)
        BT3 = TComplex(0.37  + scale*db3, 0.) * TComplex(1, (-0.15  +  scale*dph3)*np.pi, True)
        BT4 = TComplex(0.87  + scale*db4, 0.) * TComplex(1, (0.53   +  scale*dph4)*np.pi, True)
        BT5 = TComplex(0.71  + scale*db5, 0.) * TComplex(1, (0.56   +  scale*dph5)*np.pi, True)
        BT6 = TComplex(2.10  + scale*db6, 0.) * TComplex(1, (0.23   +  scale*dph6)*np.pi, True)
        BT7 = TComplex(0.77  + scale*db7, 0.) * TComplex(1, (-0.54  +  scale*dph7)*np.pi, True)

        F3PIFactor = None

        if IDK == 2:
            if IFORM == 1 or IFORM == 2:
                S1 = SA
                S2 = SB
                S3 = QQ - SA - SB + M1SQ + M2SQ + M3SQ

                F134 = -(1 / 3.) * ((S3 - M3SQ) - (S1 - M1SQ))
                F15A = -(1 / 2.) * ((S2 - M2SQ) - (S3 - M3SQ))
                F15B = -(1 / 18.) * (QQ - M2SQ + S2) * (2 * M1SQ + 2 * M3SQ - S2) / S2
                F167 = -(2 / 3.)

                # Breit Wigners for all the contributions:
                FRO1 = self.BWIGML(S1, MRO, GRO, M2, M3, 1)
                FRP1 = self.BWIGML(S1, MRP, GRP, M2, M3, 1)
                FRO2 = self.BWIGML(S2, MRO, GRO, M3, M1, 1)
                FRP2 = self.BWIGML(S2, MRP, GRP, M3, M1, 1)
                FF21 = self.BWIGML(S1, MF2, GF2, M2, M3, 2)
                FF22 = self.BWIGML(S2, MF2, GF2, M3, M1, 2)
                FSG2 = self.BWIGML(S2, MSG, GSG, M3, M1, 0)
                FF02 = self.BWIGML(S2, MF0, GF0, M3, M1, 0)

                F3PIFactor = BT1*FRO1 \
                           + BT2*FRP1 \
                           + BT3*TComplex(F134, 0.)*FRO2 \
                           + BT4*TComplex(F134, 0.)*FRP2 \
                           - BT5*TComplex(F15A, 0.)*FF21 \
                           - BT5*TComplex(F15B, 0.)*FF22 \
                           - BT6*TComplex(F167, 0.)*FSG2 \
                           - BT7*TComplex(F167, 0.)*FF02

            elif IFORM == 3:
                S3 = SA
                S1 = SB
                S2 = QQ - SA - SB + M1SQ + M2SQ + M3SQ

                F34A =  (1 / 3) * ((S2 - M2SQ) - (S3 - M3SQ))
                F34B =  (1 / 3) * ((S3 - M3SQ) - (S1 - M1SQ))
                F35A = -(1 / 18) * (QQ - M1SQ + S1) * (2 * M2SQ + 2 * M3SQ - S1) / S1
                F35B =  (1 / 18) * (QQ - M2SQ + S2) * (2 * M3SQ + 2 * M1SQ - S2) / S2
                F36A = -(2 / 3)
                F36B =  (2 / 3)

                FRO1 = self.BWIGML(S1, MRO, GRO, M2, M3, 1)
                FRP1 = self.BWIGML(S1, MRP, GRP, M2, M3, 1)
                FRO2 = self.BWIGML(S2, MRO, GRO, M3, M1, 1)
                FRP2 = self.BWIGML(S2, MRP, GRP, M3, M1, 1)
                FF21 = self.BWIGML(S1, MF2, GF2, M2, M3, 2)
                FF22 = self.BWIGML(S2, MF2, GF2, M3, M1, 2)
                FSG1 = self.BWIGML(S1, MSG, GSG, M2, M3, 0)
                FSG2 = self.BWIGML(S2, MSG, GSG, M3, M1, 0)
                FF01 = self.BWIGML(S1, MF0, GF0, M2, M3, 0)
                FF02 = self.BWIGML(S2, MF0, GF0, M3, M1, 0)

                F3PIFactor = BT3*(TComplex(F34A, 0.)*FRO1 + TComplex(F34B, 0.)*FRO2) \
                           + BT4*(TComplex(F34A, 0.)*FRP1 + TComplex(F34B, 0.)*FRP2) \
                           - BT5*(TComplex(F35A, 0.)*FF21 + TComplex(F35B, 0.)*FF22) \
                           - BT6*(TComplex(F36A, 0.)*FSG1 + TComplex(F36B, 0.)*FSG2) \
                           - BT7*(TComplex(F36A, 0.)*FF01 + TComplex(F36B, 0.)*FF02)

        if IDK == 1:
            if IFORM == 1 or IFORM == 2:
                S1 = SA
                S2 = SB
                S3 = QQ - SA - SB + M1SQ + M2SQ + M3SQ

                # C it is 2pi0pi-
                # C Lorentz invariants for all the contributions:
                F134 = -(1 / 3.)  * ((S3 - M3SQ) - (S1 - M1SQ))                      # array
                F150 =  (1 / 18.) * (QQ - M3SQ + S3) * (2*M1SQ + 2*M2SQ - S3) / S3   # array
                F167 =  (2 / 3.)                                                     # scalar

                # FR**: all are TComplex
                FRO1 = self.BWIGML(S1, MRO, GRO, M2, M3, 1)
                FRP1 = self.BWIGML(S1, MRP, GRP, M2, M3, 1)
                FRO2 = self.BWIGML(S2, MRO, GRO, M3, M1, 1)
                FRP2 = self.BWIGML(S2, MRP, GRP, M3, M1, 1)
                FF23 = self.BWIGML(S3, MF2, GF2, M1, M2, 2)
                FSG3 = self.BWIGML(S3, MSG, GSG, M1, M2, 0)
                FF03 = self.BWIGML(S3, MF0, GF0, M1, M2, 0)


                F3PIFactor = BT1*FRO1 \
                           + BT2*FRP1 \
                           + BT3*TComplex(F134)*FRO2 \
                           + BT4*TComplex(F134)*FRP2 \
                           + BT5*TComplex(F150)*FF23 \
                           + BT6*TComplex(F167)*FSG3 \
                           + BT7*TComplex(F167)*FF03

            elif IFORM == 3:
                S3 = SA
                S1 = SB
                S2 = QQ - SA - SB + M1SQ + M2SQ + M3SQ

                F34A = (1 / 3.) * ((S2 - M2SQ) - (S3 - M3SQ)) # array
                F34B = (1 / 3.) * ((S3 - M3SQ) - (S1 - M1SQ)) # array
                F35 = -(1 / 2.) * ((S1 - M1SQ) - (S2 - M2SQ)) # array

                FRO1 = self.BWIGML(S1, MRO, GRO, M2, M3, 1)
                FRP1 = self.BWIGML(S1, MRP, GRP, M2, M3, 1)
                FRO2 = self.BWIGML(S2, MRO, GRO, M3, M1, 1)
                FRP2 = self.BWIGML(S2, MRP, GRP, M3, M1, 1)
                FF23 = self.BWIGML(S3, MF2, GF2, M1, M2, 2)

                F3PIFactor = BT3*(TComplex(F34A)*FRO1 + TComplex(F34B)*FRO2) \
                           + BT4*(TComplex(F34A)*FRP1 + TComplex(F34B)*FRP2) \
                           + BT5*TComplex(F35)*FF23

        FORMA1 = self.FA1A1P(QQ) # TComplex
        out = F3PIFactor*FORMA1

        return  out



    # ------- L-wave BreightWigner for rho
    # Breit-Wigner function with isotropic decay angular distribution
    def BWIGML(self,
               S: float,
               M: float,
               G: float,
               m1: float,
               m2: float,
               L: int):
        MP = (m1 + m2)**2
        MM = (m1 - m2)**2
        MSQ = M**2       
        W = np.sqrt(S)   

        if W > m1+m2:
            QS = np.sqrt(abs( (S  - MP)*(S  - MM)))/W;
            QM = np.sqrt(abs( (MSQ - MP)*(MSQ - MM)))/M;
            IPOW = 2*L +1;
            WGS=G*(MSQ/W)*(QS/QM)**IPOW
        else: WGS = 0.

        num = TComplex(MSQ, 0)
        den = TComplex((MSQ - S), -WGS)

        out = num/den

        return out

    def FA1A1P(self, XMSQ: float) -> TComplex:
        XM1 = 1.275000
        XG1 = 0.700
        XM2 = 1.461000
        XG2 = 0.250
        BET = TComplex(0.0, 0.0)

        GG1 = XM1*XG1/(1.3281*0.806)
        GG2 = XM2*XG2/(1.3281*0.806)
        XM1SQ = XM1*XM1
        XM2SQ = XM2*XM2

        GF = self.WGA1(XMSQ)
        FG1 = GG1*GF
        FG2 = GG2*GF

        F1 = TComplex(-XM1SQ)/TComplex(XMSQ - XM1SQ, FG1)
        F2 = TComplex(-XM2SQ)/TComplex(XMSQ - XM2SQ, FG2)
        FA1A1P = F1 + (BET*F2)

        return FA1A1P


    def WGA1(self, QQ: float):
        # C mass-dependent M*Gamma of a1 through its decays to
        # C.[(rho-pi S-wave) + (rho-pi D-wave) +
        # C.(f2 pi D-wave) + (f0pi S-wave)]
        # C.AND simple K*K S-wave
        MKST = 0.894
        MK   = 0.496
        MK1SQ = (MKST+MK)**2
        MK2SQ = (MKST-MK)**2
        # C coupling constants squared:
        C3PI = 0.2384*0.2384
        CKST = 4.7621*4.7621*C3PI
        # C Parameterization of numerical integral of total width of a1 to 3pi.
        # C From M. Schmidtler, CBX-97-64-Update.

        S = QQ
        WG3PIC = self.WGA1C(S)
        WG3PIN = self.WGA1N(S)

        # C Contribution to M*Gamma(m(3pi)^2) from S-wave K*K, if above threshold
        if S > MK1SQ: 
            GKST = np.sqrt((S-MK1SQ)*(S-MK2SQ))/(2.0*S)
        else: GKST = 0.0    

        out = C3PI*(WG3PIC+WG3PIN) + (CKST*GKST)

        return out



    def WGA1C(self, S: float):
        STH = 0.1753
        Q0  = 5.80900
        Q1  = -3.00980
        Q2  = 4.57920
        P0  = -13.91400
        P1  = 27.67900
        P2  = -13.39300
        P3  = 3.19240
        P4  = -0.10487

        if S < STH:
            G1_IM = 0.0
        elif S < 0.823:    
            G1_IM = Q0 * ((S - STH)*(S - STH)*(S - STH)) * (1.0 + Q1 * (S - STH) + Q2 * (S - STH)*(S - STH))
        else: 
            G1_IM = P0 + P1*S + P2*S*S + P3*S*S*S + P4*S*S*S*S

        return G1_IM


    def WGA1N(self, S: float):
        Q0 = 6.28450
        Q1 = -2.95950
        Q2 = 4.33550
        P0 = -15.41100
        P1 = 32.08800
        P2 = -17.66600
        P3 = 4.93550
        P4 = -0.37498
        STH = 0.1676

        if S < STH:
            G1_IM = 0.0
        elif S < 0.823:
            G1_IM = Q0 * ((S - STH)*(S - STH)*(S - STH)) * (1.0 + Q1 * (S - STH) + Q2 * (S - STH)*(S - STH))
        else:    
            G1_IM = P0 + P1*S + P2*S*S + P3*S*S*S + P4*S*S*S*S

        return G1_IM


    def CLVEC(self, H: list, HC: list, N: TLorentzVector) -> TLorentzVector:
        HN  = H[0]*N.E() - H[1]*N.Px() - H[2]*N.Py() - H[3]*N.Pz()        # TComplex
        HCN = HC[0]*N.E() - HC[1]*N.Px() - HC[2]*N.Py() - HC[3]*N.Pz()    # TComplex
        HH  = (H[0]*HC[0] - H[1]*HC[1] - H[2]*HC[2] - H[3]*HC[3]).Re() # ak.Array

        PIVEC0 = 2*( 2*(HN*HC[0]).Re() - HH*N.E() )
        PIVEC1 = 2*( 2*(HN*HC[1]).Re() - HH*N.Px()     )
        PIVEC2 = 2*( 2*(HN*HC[2]).Re() - HH*N.Py()     )
        PIVEC3 = 2*( 2*(HN*HC[3]).Re() - HH*N.Pz()     )

        out = TLorentzVector(PIVEC1, PIVEC2, PIVEC3, PIVEC0)

        return out


    def CLAXI(self, H: list, HC: list, N: TLorentzVector) -> TLorentzVector:
        a1 = HC[1]
        a2 = HC[2]
        a3 = HC[3]
        a4 = HC[0]

        b1 = H[1]
        b2 = H[2]
        b3 = H[3]
        b4 = H[0]

        c1 = N.Px()
        c2 = N.Py()
        c3 = N.Pz()
        c4 = N.E()

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

        out = TLorentzVector(PIAX1, PIAX2, PIAX3, PIAX0)

        return out


class Vec4:
    """Array-backed Lorentz 4-vector, metric (+,-,-,-)"""

    def __init__(self, t, x, y, z):
        self.t = t
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec4(self.t + other.t, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec4(self.t - other.t, self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Vec4(-self.t, -self.x, -self.y, -self.z)

    def __mul__(self, scalar):
        return Vec4(self.t * scalar, self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def dot(self, other):
        return self.t * other.t - self.x * other.x - self.y * other.y - self.z * other.z

    def M2(self):
        return self.dot(self)

    def M(self):
        return np.sqrt(np.abs(self.M2()))


class PolarimetricA1_vectorised:
    def __init__(self,
                 p4_tau,
                 p4_os_pi,
                 p4_ss1_pi,
                 p4_ss2_pi,
                 taucharge) -> None:
        """
        Vectorised polarimetric vector for tau -> a1 decay, operating on
        arrays of events.  Inputs are Momentum4D awkward arrays (fields: E,
        px, py, pz) or any object whose .E / .px / .py / .pz attributes
        return array-like values.  taucharge is an array of +/-1 values.
        """

        def _v(p4):
            return Vec4(np.asarray(p4.E), np.asarray(p4.px),
                        np.asarray(p4.py), np.asarray(p4.pz))

        self.p4_tau    = _v(p4_tau)
        self.p4_os_pi  = _v(p4_os_pi)
        self.p4_ss1_pi = _v(p4_ss1_pi)
        self.p4_ss2_pi = _v(p4_ss2_pi)

        self.mpi            =  0.13957018
        self.mpi0           =  0.1349766
        self.mtau           =  1.776
        self.coscab         =  0.975
        self.mrho           =  0.773
        self.mrhoprime      =  1.370
        self.ma1            =  1.251
        self.mpiprime       =  1.300
        self.Gamma0rho      =  0.145
        self.Gamma0rhoprime =  0.510
        self.Gamma0a1       =  0.599
        self.Gamma0piprime  =  0.3
        self.fpi            =  0.093
        self.fpiprime       =  0.08
        self.gpiprimerhopi  =  5.8
        self.grhopipi       =  6.08
        self.beta           = -0.145
        self.COEF1          =  2.0*np.sqrt(2.)/3.0
        self.COEF2          = -2.0*np.sqrt(2.)/3.0
        self.COEF3          =  2.0*np.sqrt(2.)/3.0
        self.SIGN           = -np.asarray(taucharge, dtype=float)
        self.doSystematic   =  False
        self.systType       =  "UP"


    def hadronic_current(self):
        """The a1 -> 3pi hadronic current, and the Dalitz scalars it is built from.

        Depends ONLY on the three pion 4-momenta -- no tau and no neutrino enter
        anywhere below. That is the whole reason this is split out of PVC(): the
        polarimetric vector factorises as h = f(hadronic current, visible, tau),
        so this method returns the complete *visible* half of the calculation and
        can be used as an engineered input feature (see
        acoplanarity_tools.hadronic_current_features).

        Because the form factors are Lorentz scalars and vec1/vec2/vec3 are
        covariant, this may be evaluated in any frame; the caller chooses. PVC()
        calls it with the pions in the tau rest frame (its own convention), while
        the feature code calls it in the a1 rest frame so the result is tau-free.

        Returns (HADCUR, scalars):
          HADCUR  : list of 4 complex arrays, ordered [t, x, y, z]
          scalars : dict with s1, s2, s3 (pair invariant masses squared -- s1 and
                    s2 are the two opposite-sign/same-sign rho candidates, s3 the
                    same-sign pair, which has no resonance) and m2_vis = m^2(a1).
        """
        q1 = self.p4_ss1_pi
        q2 = self.p4_ss2_pi
        q3 = self.p4_os_pi

        a1 = q1 + q2 + q3

        s1 = (q2 + q3).M2()
        s2 = (q1 + q3).M2()
        s3 = (q1 + q2).M2()

        a1_m2 = a1.M2()
        vec1 = (q2 - q3) - a1 * (a1.dot(q2 - q3) / a1_m2)
        vec2 = (q3 - q1) - a1 * (a1.dot(q3 - q1) / a1_m2)
        vec3 = (q1 - q2) - a1 * (a1.dot(q1 - q2) / a1_m2)

        F1 = self.COEF1 * self._F3PI(1, a1_m2, s1, s2)
        F2 = self.COEF2 * self._F3PI(2, a1_m2, s2, s1)
        F3 = self.COEF3 * self._F3PI(3, a1_m2, s3, s1)

        HADCUR = [
            vec1.t * F1 + vec2.t * F2 + vec3.t * F3,
            vec1.x * F1 + vec2.x * F2 + vec3.x * F3,
            vec1.y * F1 + vec2.y * F2 + vec3.y * F3,
            vec1.z * F1 + vec2.z * F2 + vec3.z * F3,
        ]
        return HADCUR, {'s1': s1, 's2': s2, 's3': s3, 'm2_vis': a1_m2}

    def PVC(self) -> 'Vec4':
        P  = self.p4_tau
        a1 = self.p4_ss1_pi + self.p4_ss2_pi + self.p4_os_pi
        N  = P - a1

        HADCUR, _ = self.hadronic_current()
        HADCURC = [np.conj(h) for h in HADCUR]

        CLV = self._CLVEC(HADCUR, HADCURC, N)
        CLA = self._CLAXI(HADCUR, HADCURC, N)

        omega    = P.dot(CLV) - P.dot(CLA)
        pdotdiff = P.dot(CLA) - P.dot(CLV)
        out = ((CLA - CLV) * P.M2() - P * pdotdiff) * (1.0 / omega / P.M())

        return out


    def _F3PI(self, IFORM: int, QQ, SA, SB):
        MRO = 0.7743;  GRO = 0.1491
        MRP = 1.370;   GRP = 0.386
        MF2 = 1.275;   GF2 = 0.185
        MF0 = 1.186;   GF0 = 0.350
        MSG = 0.860;   GSG = 0.880

        M1, M2, M3 = self.mpi0, self.mpi0, self.mpi
        M1SQ = M1*M1;  M2SQ = M2*M2;  M3SQ = M3*M3

        db2, dph2 = 0.094, 0.253
        db3, dph3 = 0.094, 0.104
        db4, dph4 = 0.296, 0.170
        db5, dph5 = 0.167, 0.104
        db6, dph6 = 0.284, 0.036
        db7, dph7 = 0.148, 0.063

        scale = 0.0
        if self.doSystematic:
            scale = 1.0 if self.systType == "UP" else -1.0

        BT1 = 1.0 + 0j
        BT2 = (0.12 + scale*db2) * np.exp(1j * (0.99   + scale*dph2) * np.pi)
        BT3 = (0.37 + scale*db3) * np.exp(1j * (-0.15  + scale*dph3) * np.pi)
        BT4 = (0.87 + scale*db4) * np.exp(1j * ( 0.53  + scale*dph4) * np.pi)
        BT5 = (0.71 + scale*db5) * np.exp(1j * ( 0.56  + scale*dph5) * np.pi)
        BT6 = (2.10 + scale*db6) * np.exp(1j * ( 0.23  + scale*dph6) * np.pi)
        BT7 = (0.77 + scale*db7) * np.exp(1j * (-0.54  + scale*dph7) * np.pi)

        if IFORM == 1 or IFORM == 2:
            S1 = SA
            S2 = SB
            S3 = QQ - SA - SB + M1SQ + M2SQ + M3SQ

            F134 = -(1/3.)  * ((S3 - M3SQ) - (S1 - M1SQ))
            F150 =  (1/18.) * (QQ - M3SQ + S3) * (2*M1SQ + 2*M2SQ - S3) / S3
            F167 =  (2/3.)

            FRO1 = self._BWIGML(S1, MRO, GRO, M2, M3, 1)
            FRP1 = self._BWIGML(S1, MRP, GRP, M2, M3, 1)
            FRO2 = self._BWIGML(S2, MRO, GRO, M3, M1, 1)
            FRP2 = self._BWIGML(S2, MRP, GRP, M3, M1, 1)
            FF23 = self._BWIGML(S3, MF2, GF2, M1, M2, 2)
            FSG3 = self._BWIGML(S3, MSG, GSG, M1, M2, 0)
            FF03 = self._BWIGML(S3, MF0, GF0, M1, M2, 0)

            F3PIFactor = (BT1*FRO1 + BT2*FRP1
                          + BT3*F134*FRO2 + BT4*F134*FRP2
                          + BT5*F150*FF23 + BT6*F167*FSG3 + BT7*F167*FF03)

        else:  # IFORM == 3
            S3 = SA
            S1 = SB
            S2 = QQ - SA - SB + M1SQ + M2SQ + M3SQ

            F34A = (1/3.)  * ((S2 - M2SQ) - (S3 - M3SQ))
            F34B = (1/3.)  * ((S3 - M3SQ) - (S1 - M1SQ))
            F35  = -(1/2.) * ((S1 - M1SQ) - (S2 - M2SQ))

            FRO1 = self._BWIGML(S1, MRO, GRO, M2, M3, 1)
            FRP1 = self._BWIGML(S1, MRP, GRP, M2, M3, 1)
            FRO2 = self._BWIGML(S2, MRO, GRO, M3, M1, 1)
            FRP2 = self._BWIGML(S2, MRP, GRP, M3, M1, 1)
            FF23 = self._BWIGML(S3, MF2, GF2, M1, M2, 2)

            F3PIFactor = (BT3*(F34A*FRO1 + F34B*FRO2)
                          + BT4*(F34A*FRP1 + F34B*FRP2)
                          + BT5*F35*FF23)

        return F3PIFactor * self._FA1A1P(QQ)


    def _BWIGML(self, S, M, G, m1, m2, L):
        MP  = (m1 + m2)**2
        MM  = (m1 - m2)**2
        MSQ = M**2
        W   = np.sqrt(np.maximum(S, 0.0))

        above  = W > (m1 + m2)
        safe_W = np.where(above, W, 1.0)

        QS  = np.where(above, np.sqrt(np.abs((S - MP)*(S - MM))) / safe_W, 0.0)
        QM  = np.sqrt(np.abs((MSQ - MP)*(MSQ - MM))) / M
        WGS = np.where(above, G * (MSQ / safe_W) * (QS / QM)**(2*L + 1), 0.0)

        return MSQ / ((MSQ - S) - 1j*WGS)


    def _FA1A1P(self, XMSQ):
        XM1 = 1.275000;  XG1 = 0.700
        XM2 = 1.461000;  XG2 = 0.250
        BET = 0.0 + 0j

        GG1   = XM1*XG1 / (1.3281*0.806)
        GG2   = XM2*XG2 / (1.3281*0.806)
        XM1SQ = XM1**2
        XM2SQ = XM2**2

        GF  = self._WGA1(XMSQ)
        FG1 = GG1 * GF
        FG2 = GG2 * GF

        F1 = -XM1SQ / ((XMSQ - XM1SQ) + 1j*FG1)
        F2 = -XM2SQ / ((XMSQ - XM2SQ) + 1j*FG2)

        return F1 + BET*F2


    def _WGA1(self, QQ):
        MKST  = 0.894;  MK = 0.496
        MK1SQ = (MKST + MK)**2
        MK2SQ = (MKST - MK)**2
        C3PI  = 0.2384**2
        CKST  = (4.7621**2) * C3PI

        S    = QQ
        GKST = np.where(S > MK1SQ,
                        np.sqrt(np.maximum((S - MK1SQ)*(S - MK2SQ), 0.0)) / (2.0*S),
                        0.0)

        return C3PI*(self._WGA1C(S) + self._WGA1N(S)) + CKST*GKST


    def _WGA1C(self, S):
        STH = 0.1753
        Q0 =  5.80900;  Q1 = -3.00980;  Q2 = 4.57920
        P0 = -13.91400;  P1 = 27.67900;  P2 = -13.39300;  P3 = 3.19240;  P4 = -0.10487

        dS = S - STH
        lo = Q0 * dS**3 * (1.0 + Q1*dS + Q2*dS**2)
        hi = P0 + P1*S + P2*S**2 + P3*S**3 + P4*S**4

        return np.where(S < STH, 0.0, np.where(S < 0.823, lo, hi))


    def _WGA1N(self, S):
        Q0 =  6.28450;  Q1 = -2.95950;  Q2 = 4.33550
        P0 = -15.41100;  P1 = 32.08800;  P2 = -17.66600;  P3 = 4.93550;  P4 = -0.37498
        STH = 0.1676

        dS = S - STH
        lo = Q0 * dS**3 * (1.0 + Q1*dS + Q2*dS**2)
        hi = P0 + P1*S + P2*S**2 + P3*S**3 + P4*S**4

        return np.where(S < STH, 0.0, np.where(S < 0.823, lo, hi))


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

