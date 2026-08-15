# m1_horizon_exact_layer.py — M1-HORIZON, 2026-07-22 (evanescent session). Certify the
# EXACT layer of the horizon closed form.
#
# (E2) Christoffel-Wronskian identity, FINITE N, claimed EXACT:
#        g_j^2 = h~_j / ( 2N * |P~_j(0) * P~_{j+1}(0)| )
#      where g_j = <e0, w_j> (ODD-Lanczos overlaps, weights z_n^-4) and h~_j, P~_j(0)
#      are the monic norms / values at 0 of the EVEN system (weights 2 x_n = 2 z_n^-2),
#      both for the SAME truncation N. Proof: g_j = -sqrt(M2/N) q_j^ODD(0) (second-kind
#      function at 0); monic ODD Q_j = [P~_{j+1} - r_j P~_j]/x (Christoffel at 0),
#      r_j = P~_{j+1}(0)/P~_j(0); <Q_j,1/x>_nu = (1/(2M2))[q~_{j+1}(0) - r_j q~_j(0)]
#      = h~_j/(2M2 P~_j(0)) by the Casoratian P~_j q~_{j+1} - P~_{j+1} q~_j = h~_j;
#      and ||Q_j||^2_nu = -h~_j P~_{j+1}(0)/(2M2 P~_j(0)). All M2 cancel.
#      Verified here at N=100, dps=120, j <= 34 (identity => any N certifies it).
#
# (INF) Infinite-system identity, EXACT IN Q (equivalent to the PROVEN gem identity):
#        h~_j^inf = (4j+3) * |P~_j(0) P~_{j+1}(0)|^inf
#      via the Lambert/tan S-fraction alpha_k = 1/((2k-1)(2k+1)) (mu_L Jacobi
#      contraction b_0=a1, b_k=a_{2k}+a_{2k+1}, a_k^2=a_{2k-1}a_{2k}), all rational.
#      This makes the infinite-system part of G(xi) the naive law 2*xi exactly.
# Not RH/GRH.
import mpmath as mp
from fractions import Fraction

# ---------- (INF): exact rational check ----------
print("=== (INF) h_j = (4j+3)|P_j(0)P_{j+1}(0)| for the infinite EVEN (mu_L/tan) system, exact in Q ===")
JQ = 60
al = {m: Fraction(1, (2 * m - 1) * (2 * m + 1)) for m in range(1, 2 * JQ + 4)}
b = {0: al[1]}
a2 = {}
for k in range(1, JQ + 2):
    b[k] = al[2 * k] + al[2 * k + 1]
    a2[k] = al[2 * k - 1] * al[2 * k]
# monic P_j(0): P_{k+1}(0) = -b_k P_k(0) - a2_k P_{k-1}(0); h_j = h_0 * prod a2, h_0 = m_0 = 1
P = {0: Fraction(1), 1: -b[0]}
for k in range(1, JQ + 1):
    P[k + 1] = -b[k] * P[k] - a2[k] * P[k - 1]
h = {0: Fraction(1)}
for k in range(1, JQ + 1):
    h[k] = h[k - 1] * a2[k]
ok = True
for j in range(0, JQ):
    lhs = h[j]
    rhs = (4 * j + 3) * abs(P[j] * P[j + 1])
    if lhs != rhs:
        ok = False
        print(f"   j={j}: MISMATCH  h={lhs}  (4j+3)|PP|={rhs}")
print(f"   j=0..{JQ-1}: identity holds EXACTLY in Q: {ok}")
print(f"   (sanity: b_0={b[0]}, a_1^2={a2[1]}, P_1(0)={P[1]}, h_1={h[1]})")

# ---------- (E2): finite-N exact identity, high precision ----------
print("\n=== (E2) g_j^2 * 2N * |P~_j(0)P~_{j+1}(0)| / h~_j = 1  (N=100, dps=120) ===")
mp.mp.dps = 120
N = 100
z = [(mp.mpf(2 * n - 1) / 2) * mp.pi for n in range(1, N + 1)]
x = [zz ** -2 for zz in z]

def lanczos(wts, J):
    """Full-reorth Lanczos on diag(x) from the sqrt-weight vector.
    Returns orthonormal-Jacobi (alpha_j, beta_j) and the vectors."""
    mass = mp.fsum(wts)
    v = [mp.sqrt(w / mass) for w in wts]
    Vs = [v]
    alphas, betas = [], []
    wv = [x[n] * v[n] for n in range(N)]
    a0 = mp.fsum(v[n] * wv[n] for n in range(N))
    alphas.append(a0)
    wv = [wv[n] - a0 * v[n] for n in range(N)]
    for j in range(1, J + 1):
        bj = mp.sqrt(mp.fsum(t * t for t in wv))
        betas.append(bj)
        vn = [t / bj for t in wv]
        for u in Vs:
            c = mp.fsum(u[n] * vn[n] for n in range(N))
            vn = [vn[n] - c * u[n] for n in range(N)]
        nv = mp.sqrt(mp.fsum(t * t for t in vn))
        vn = [t / nv for t in vn]
        Vs.append(vn)
        wv = [x[n] * vn[n] - bj * Vs[-2][n] for n in range(N)]
        aj = mp.fsum(vn[n] * wv[n] for n in range(N))
        alphas.append(aj)
        wv = [wv[n] - aj * vn[n] for n in range(N)]
    return alphas, betas, Vs, mass

J = 34
# ODD system: weights z^-4 -> g_j
wodd = [zz ** -4 for zz in z]
_, _, Vs_o, _ = lanczos(wodd, J + 1)
e0 = mp.mpf(1) / mp.sqrt(N)
g = [mp.fsum(e0 * V[n] for n in range(N)) for V in Vs_o]
# EVEN system: weights 2x_n -> monic P~_j(0), h~_j
weven = [2 * xx for xx in x]
alE, beE, _, massE = lanczos(weven, J + 2)
Pm = {0: mp.mpf(1), 1: -alE[0]}
hm = {0: massE}
for k in range(1, J + 2):
    Pm[k + 1] = -alE[k] * Pm[k] - (beE[k - 1] ** 2) * Pm[k - 1]
    hm[k] = hm[k - 1] * beE[k - 1] ** 2
print("   j   xi      g_j              ratio-1 (should be ~1e-dps)")
worst = mp.mpf(0)
for j in range(0, J + 1):
    ratio = g[j] ** 2 * 2 * N * abs(Pm[j] * Pm[j + 1]) / hm[j]
    dev = abs(ratio - 1)
    worst = max(worst, dev)
    if j in (0, 1, 2, 5, 10, 15, 20, 25, 30, 34):
        print(f"  {j:3d} {float(j/mp.sqrt(N)):5.2f}  {mp.nstr(g[j], 8):>15}   {mp.nstr(dev, 3)}")
print(f"   WORST |ratio-1| over j=0..{J}: {mp.nstr(worst, 3)}")
print("done.")
