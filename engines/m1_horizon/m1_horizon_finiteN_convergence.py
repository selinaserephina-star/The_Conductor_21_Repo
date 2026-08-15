# m1_horizon_finiteN_convergence.py — M1-HORIZON, 2026-07-22 (evanescent session).
# Close the loop against the measured tables: the certified EXACT composition
#   ln g_j^2(N) = ln((4j+3)/(2N)) + 2 F_E(j+1) - F_O(j) - F_O(j+1)
# (F = ln det(I - CD-Gram on tail atoms), certified to 1e-10 in m1_horizon_downdate_check)
# is evaluated WITHOUT any Lanczos at N = 400 / 1600 / 6400, and compared to
#  (i)  the HP-measured G table at N=400 (mpmath dps 220 ODD-Lanczos, m1_horizon_hp_bridge),
#  (ii) the closed-form scaling limit G(xi) = 2 xi [det(I-K^+_L)/det(I-K^-_L)]^2, L=2xi^2/pi,
# showing G(N) -> G_closed with an O(1/sqrt N) rate. Not RH/GRH.
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import polygamma

MTAIL = 1_000_000

def P_node(l, z):
    t = np.ones_like(z); s = t.copy(); k = 0
    while 2 * (k + 1) <= l:
        fac = -((l + 2*k + 1) * (l + 2*k + 2) * (l - 2*k) * (l - 2*k - 1)) / ((2*k + 1) * (2*k + 2))
        t = t * fac / (4 * z * z); s = s + t; k += 1
    return s

def Q_node(l, z):
    t = np.full_like(z, l * (l + 1.0)); s = t.copy(); k = 0
    while 2 * (k + 1) + 1 <= l:
        fac = -((l + 2*k + 2) * (l + 2*k + 3) * (l - 2*k - 1) * (l - 2*k - 2)) / ((2*k + 2) * (2*k + 3))
        t = t * fac / (4 * z * z); s = s + t; k += 1
    return s / (2 * z)

def composition_lng2(N, jlist):
    jmax = max(jlist) + 1
    n_tail = np.arange(N + 1, MTAIL + 1, dtype=np.float64)
    z_tail = (n_tail - 0.5) * np.pi
    VE = np.array([np.sqrt(2 * (4 * i + 1)) * P_node(2 * i, z_tail) / z_tail for i in range(jmax + 1)])
    VO = np.array([np.sqrt(2 * (4 * i + 3)) * Q_node(2 * i + 1, z_tail) / z_tail for i in range(jmax + 1)])
    WE = VE @ VE.T
    ce = np.sqrt(2 * (4 * np.arange(jmax + 1) + 1))
    WE += np.outer(ce, ce) * (polygamma(1, MTAIL + 0.5) / np.pi ** 2)
    WO = VO @ VO.T
    def F(W, j):
        sgn, ld = np.linalg.slogdet(np.eye(j) - W[:j, :j])
        return ld if sgn > 0 else np.nan
    out = {}
    for j in jlist:
        out[j] = (np.log((4 * j + 3) / (2.0 * N)) + 2 * F(WE, j + 1) - F(WO, j) - F(WO, j + 1))
    return out

x_, w_ = leggauss(240)
t_ = 0.5 * (x_ + 1); wt_ = 0.5 * w_
def logG_closed(xi):
    L = 2 * xi * xi / np.pi
    tu = t_[:, None] + t_[None, :]; tv = t_[:, None] - t_[None, :]
    Sd = np.where(np.abs(tv) < 1e-14, L, np.sin(L * tv) / np.where(np.abs(tv) < 1e-14, 1.0, tv))
    Ss = np.sin(L * tu) / tu
    A = np.sqrt(wt_)[:, None] * np.sqrt(wt_)[None, :] / np.pi
    lp = np.linalg.slogdet(np.eye(240) - A * (Sd + Ss))[1]
    lm = np.linalg.slogdet(np.eye(240) - A * (Sd - Ss))[1]
    return np.log(2 * xi) + 2 * (lp - lm)

print("=== (i) N=400: exact composition vs HP-measured G (mpmath dps 220 table) ===")
meas400 = {20: 0.743, 30: 0.253, 40: 0.0372, 50: 0.00260, 60: 8.97e-5}
lng = composition_lng2(400, list(meas400))
print("   j   xi    G_comp(N=400)   G_measured(HP)")
for j, m in meas400.items():
    print(f"  {j:3d}  {j/20:.1f}   {20*np.exp(lng[j]):.5e}     {m:.3e}")

print("\n=== (ii) N-convergence of G(N) at fixed xi vs the closed-form limit ===")
print("   xi    N=400        N=1600       N=6400       closed limit   sqrtN*(closed-N6400)")
for xi in (1.0, 1.5, 2.0):
    row = []
    for N in (400, 1600, 6400):
        j = int(round(xi * np.sqrt(N)))
        lg = composition_lng2(N, [j])[j]
        row.append(np.sqrt(N) * np.exp(lg))
    gc = np.exp(logG_closed(xi))
    print(f"  {xi:4.1f}  {row[0]:.6f}   {row[1]:.6f}   {row[2]:.6f}   {gc:.6f}      {80*(gc-row[2]):+.4f}")
    d = [gc - r for r in row]
    print(f"        closed-minus-N: {d[0]:+.5f} / {d[1]:+.5f} / {d[2]:+.5f}   (ratios {d[1]/d[0]:.3f}, {d[2]/d[1]:.3f}; 1/sqrtN rate = 0.5)")
print("done.")
