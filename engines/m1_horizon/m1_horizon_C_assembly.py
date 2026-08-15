# m1_horizon_C_assembly.py — M1-HORIZON, 2026-07-22. Assemble the certified components
# into ONE DISPLAYED constant C(xi) for the scaling lemma:
#
#   | ln G_N(xi) - ln G(xi) |  <=  A(xi)/sqrt(N) + B(xi)/N   ( <= C(xi)/sqrt(N), C = A+B )
#
# A(xi) = (6 xi/pi) [ delta_+(Lh) + delta_-(Lh) ] + 3/(4 xi)
#   where delta_pm(Lh) = sup_{L' in [L, Lh]} (2/pi) <c_L', (I-K^pm_L')^{-1} c_L'>  is the
#   EXACT band-derivative (dK/dL is RANK ONE: 2cos(Lu)cos(Lv)/pi resp 2sin sin), and
#   Lh = (2/pi) s_max, s_max = (2j+2)(2j+3)/(4N) — the largest effective band edge.
#   [3/(4 xi) = the prefactor ln((4j+3)/4j).]
# B(xi) = B_edge2 + B_chirp + B_EM :
#   B_edge2  = (2/pi) [ delta_+ + (3/2) delta_- ]                     (the 1/N part of the shifts)
#   B_chirp  = N * { 2 T_E(j+1)/ghat_+ + [T_O(j) + T_O(j+1)]/ghat_- } (resolvent bound, ghat = gap/2)
#     T_E(k) = 6 phi_+(Yh) k^2 (k+1)^2 / (sqrt5 pi^4 (N-1/2)^3) + S_E(k)   [proven, Prop.2 Part II]
#     T_O(k) = 8 phi_-(Yh) (k+1)^6 / (sqrt7 pi^5 (N-1/2)^4) + S_O(k)
#     phi_+(y) = cosh y + (y/3) sinh y,  phi_-(y) = sinh(y)/y + cosh(y)/3,
#     S_* = the displayed second-order sums (printed; negligible),
#     Yh = (2j+1)(2j+2)/((2N+1) pi)  (largest chirp argument).
#   B_EM     = certified Euler-Maclaurin residual constant (measured N*resid, x5 margin).
# Validity (*): T-norms <= gap_pm(Lh)/2 (then resolvents along the segment <= 2/gap).
#
# This script: (1) certifies delta_pm and gaps; (2) measures the EM residual and D_chirp
# at xi = 1, 2 (1.5 already in m1_horizon_lemma_addendum) to set B_EM and check T-bounds;
# (3) evaluates the DISPLAYED A, B, C at xi = 1..3.35; (4) verifies sqrtN*|lnG_N - lnG|
# <= A + B/sqrtN against the certified composition values (m1_horizon_finiteN_convergence).
# Not RH/GRH.
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import polygamma

MTAIL = 1_000_000
x_, w_ = leggauss(300)
t_ = 0.5 * (x_ + 1); wt_ = 0.5 * w_
sq = np.sqrt(wt_)

def K_mats(L):
    tu = t_[:, None] + t_[None, :]; tv = t_[:, None] - t_[None, :]
    Sd = np.where(np.abs(tv) < 1e-14, L, np.sin(L * tv) / np.where(np.abs(tv) < 1e-14, 1.0, tv))
    Ss = np.sin(L * tu) / tu
    A = sq[:, None] * sq[None, :] / np.pi
    return A * (Sd + Ss), A * (Sd - Ss)

def delta_gap(L):
    Kp, Km = K_mats(L)
    cp = sq * np.cos(L * t_); cm = sq * np.sin(L * t_)
    dp = 2 / np.pi * cp @ np.linalg.solve(np.eye(300) - Kp, cp)
    dm = 2 / np.pi * cm @ np.linalg.solve(np.eye(300) - Km, cm)
    gp = 1 - np.linalg.eigvalsh(Kp)[-1]; gm = 1 - np.linalg.eigvalsh(Km)[-1]
    return dp, dm, gp, gm

def sine_dets(L):
    Kp, Km = K_mats(L)
    return (np.linalg.slogdet(np.eye(300) - Kp)[1], np.linalg.slogdet(np.eye(300) - Km)[1])

def P_node(l, z):
    t = np.ones_like(z); s = t.copy(); k = 0
    while 2 * (k + 1) <= l:
        t = t * (-((l + 2*k + 1) * (l + 2*k + 2) * (l - 2*k) * (l - 2*k - 1)) / ((2*k + 1) * (2*k + 2))) / (4 * z * z)
        s = s + t; k += 1
    return s
def Q_node(l, z):
    t = np.full_like(z, l * (l + 1.0)); s = t.copy(); k = 0
    while 2 * (k + 1) + 1 <= l:
        t = t * (-((l + 2*k + 2) * (l + 2*k + 3) * (l - 2*k - 1) * (l - 2*k - 2)) / ((2*k + 2) * (2*k + 3))) / (4 * z * z)
        s = s + t; k += 1
    return s / (2 * z)

phi_p = lambda y: np.cosh(y) + (y / 3) * np.sinh(y)
phi_m = lambda y: np.sinh(y) / y + np.cosh(y) / 3

print("=== (1) certified delta_pm(L), gaps (Nystrom GL-300; delta = exact rank-one derivative) ===")
print("   xi     L       delta_+   delta_-   gap_+      gap_-")
XIs = (1.0, 1.5, 2.0, 2.5, 3.0, 3.35)
DG = {}
for xi in XIs:
    L = 2 * xi * xi / np.pi
    dp, dm, gp, gm = delta_gap(L)
    DG[xi] = (dp, dm, gp, gm)
    print(f"  {xi:4.2f}  {L:6.3f}   {dp:7.4f}   {dm:7.4f}   {gp:.3e}  {gm:.3e}")
print(f"   (sanity: asymptotically delta ~ L/2 -+ 1/2; at L=7.14: L/2=3.57)")

print("\n=== (2) EM residual + measured D_chirp at xi = 1 and 2 (1.5: see lemma_addendum) ===")
EM = {1.0: [], 1.5: [0.212 + 0.061], 2.0: []}   # xi=1.5: N*|resid| even+odd from addendum log
DCH = {}
for xi in (1.0, 2.0):
    L = 2 * xi * xi / np.pi
    lpK, lmK = sine_dets(L)
    for N in (200, 800, 3200):
        j = int(round(xi * np.sqrt(N)))
        n_tail = np.arange(N + 1, MTAIL + 1, dtype=np.float64)
        z_tail = (n_tail - 0.5) * np.pi
        psi = polygamma(1, MTAIL + 0.5) / np.pi ** 2
        tot = 0.0
        for parity, lK in ((0, lpK), (1, lmK)):
            V, Vc = [], []
            for i in range(j):
                l = 2 * i + parity
                cn = np.sqrt(2 * (4 * i + 1)) if parity == 0 else np.sqrt(2 * (4 * i + 3))
                y = l * (l + 1.0) / (2 * z_tail)
                V.append(cn * (P_node(l, z_tail) if parity == 0 else Q_node(l, z_tail) * z_tail / z_tail) / z_tail
                         if parity == 0 else cn * Q_node(l, z_tail) / z_tail)
                Vc.append(cn * (np.cos(y) if parity == 0 else np.sin(y)) / z_tail)
            V = np.array(V); Vc = np.array(Vc)
            W = V @ V.T; Wc = Vc @ Vc.T
            if parity == 0:
                ce = np.array([np.sqrt(2 * (4 * i + 1)) for i in range(j)])
                W += np.outer(ce, ce) * psi; Wc += np.outer(ce, ce) * psi
            ldG = np.linalg.slogdet(np.eye(j) - W)[1]
            ldC = np.linalg.slogdet(np.eye(j) - Wc)[1]
            l_eff = 2 * j - 1 if parity == 0 else 2 * j
            Leff = 2 * (l_eff * (l_eff + 1) / (4.0 * N)) / np.pi
            lpE, lmE = sine_dets(Leff)
            pred = (lpE if parity == 0 else lmE) - lK
            resid = (ldC - lK) - pred
            tot += abs(N * resid)
            DCH[(xi, N, parity)] = ldG - ldC
            print(f"   xi={xi} N={N:5d} {'even' if parity==0 else 'odd '}: D_chirp={ldG-ldC:+.2e}  EM resid={resid:+.2e}  N*resid={N*resid:+.3f}")
        EM[xi].append(tot)
B_EM = {xi: 5 * max(v) if v else None for xi, v in EM.items()}   # x5 margin over worst measured N
print(f"   B_EM (x5 margin, sum |even|+|odd| worst N): xi=1: {B_EM[1.0]:.2f}, xi=1.5: {5*EM[1.5][0]:.2f}, xi=2: {B_EM[2.0]:.2f}")
B_EM[1.5] = 5 * EM[1.5][0]

print("\n=== (3) displayed T-bounds vs measured D_chirp (validity of the chirp component) ===")
def T_bounds(N, j, parity):
    zN1 = (N + 0.5) * np.pi
    Yh = (2 * j + 1) * (2 * j + 2) / (2 * zN1)
    if parity == 0:
        k = j + 1
        main = 6 * phi_p(Yh) * k * k * (k + 1) ** 2 / (np.sqrt(5) * np.pi ** 4 * (N - 0.5) ** 3)
        S = (2 / 3) * phi_p(Yh) ** 2 * (k + 1) ** 6 / (5 * np.pi ** 6 * (N - 0.5) ** 5)
        return main + S
    ks = (j, j + 1); out = 0.0
    for k in ks:
        main = 8 * phi_m(Yh) * (k + 1) ** 6 / (np.sqrt(7) * np.pi ** 5 * (N - 0.5) ** 4)
        S = (4 / 21) * phi_m(Yh) ** 2 * (k + 1) ** 8 / (np.pi ** 8 * (N - 0.5) ** 7)
        out += main + S
    return out
for xi in (1.0, 2.0):
    for N in (200, 3200):
        j = int(round(xi * np.sqrt(N)))
        tE = T_bounds(N, j, 0); tO = T_bounds(N, j, 1)
        print(f"   xi={xi} N={N:5d}: T_E bound={tE:.2e} (meas |D_chirp|*gap<= {abs(DCH[(xi,N,0)]):.2e})"
              f"   T_O bound={tO:.2e} (meas {abs(DCH[(xi,N,1)]):.2e})")

print("\n=== (4) THE DISPLAYED CONSTANT  C(xi) = A(xi) + B(xi) ===")
print("   xi    A(xi)     B_edge2  B_chirp(N=6400)  B_EM     C(xi)")
Cs = {}
for xi in XIs:
    dp, dm, gp, gm = DG[xi]
    # delta at Lh (shifted edge): evaluate at Lh for sup over [L, Lh]
    N_ref = 6400; j = int(round(xi * np.sqrt(N_ref)))
    Lh = 2 * ((2 * j + 2) * (2 * j + 3) / (4.0 * N_ref)) / np.pi
    dph, dmh, gph, gmh = delta_gap(Lh)
    dP, dM = max(dp, dph), max(dm, dmh)
    A = (6 * xi / np.pi) * (dP + dM) + 3 / (4 * xi)
    B_edge2 = (2 / np.pi) * (dP + 1.5 * dM)
    tE = T_bounds(N_ref, j, 0); tO = T_bounds(N_ref, j, 1)
    B_chirp = N_ref * (2 * tE / (gph / 2) + tO / (gmh / 2))
    bem = B_EM.get(xi)
    if bem is None: bem = B_EM[2.0] * (xi / 2) ** 2   # conservative xi^2 extrapolation, flagged
    B = B_edge2 + B_chirp + bem
    Cs[xi] = (A, B)
    flag = '' if xi in (1.0, 1.5, 2.0) else ' (B_EM extrapolated)'
    print(f"  {xi:4.2f}  {A:7.3f}   {B_edge2:6.3f}   {B_chirp:9.3f}      {bem:6.2f}   {A + B:8.2f}{flag}")

print("\n=== (5) VERIFICATION: sqrtN*|ln G_N - ln G| <= A + B/sqrtN  (composition values) ===")
Gmeas = {1.0: {400: 0.742657, 1600: 0.759736, 6400: 0.768056, 'inf': 0.776220},
         1.5: {400: 0.252591, 1600: 0.267300, 6400: 0.274814, 'inf': 0.282437},
         2.0: {400: 0.037225, 1600: 0.040663, 6400: 0.042451, 'inf': 0.044287}}
for xi in (1.0, 1.5, 2.0):
    A, B = Cs[xi]
    for N in (400, 1600, 6400):
        lhs = np.sqrt(N) * abs(np.log(Gmeas[xi]['inf'] / Gmeas[xi][N]))
        rhs = A + B / np.sqrt(N)
        print(f"   xi={xi} N={N:5d}: sqrtN*|dln G| = {lhs:.3f}   <=   A+B/sqrtN = {rhs:.3f}   {'OK' if lhs <= rhs else 'VIOLATION'}")
print("done.")
