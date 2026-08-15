# interval_grids_arb.py -- INFINITE_SIDE item 5 (the last theorem-grade close).
# Interval-certify the operator grid quantities delta_pm, delta'_pm, eta_+, gap_pm on
# the even/odd sine kernels K^pm_L on (0,1), L = 2 xi^2/pi, at the standard cells
# xi = 1, 1.5, 2, 2.5, 3, 3.35.  Flips the sealed Nystrom float64 numerics
# (WRITEUP_XI_LIMIT_C2_rate_constants.md sec.1; WRITEUP_OPERATOR_EM_lemma.md) to
# rigorous ball enclosures at fixed GL order, with a GL-300/400/500 stability table.
#
# METHOD (all rigorous, python-flint 0.8.0 ball arithmetic):
#  * Certified GL nodes: Newton-Kantorovich.  Point-evaluations of P_n, P_n' at high
#    precision (stable; no interval wrapping) + the rigorous Markov bound
#    |P_n''| <= n^2(n^2-1)/3 on [-1,1] certify a unique root in a tiny ball.  The
#    forward recurrence is NEVER evaluated over a wide ball (that blows up ~2^n).
#  * delta_X  = (2/pi) <c, (I-K)^{-1} c>,   c^+ = sqrt(w) cos(Lt), c^- = sqrt(w) sin(Lt)
#    delta'_X = (4/pi) <cdot, (I-K)^{-1} c> + delta^2   (exact rank-one calculus)
#    eta_+    = <1, (I-K^+)^{-1} 1>          (even parity only; 1 -> sqrt(w))
#    all from ONE rigorous ball inverse A^{-1} = (I-K)^{-1} (arb_mat.inv).
#  * gap_X = 1 - lambda_max(K) = lambda_min(I-K), two-sided rigorous enclosure:
#       gap_hi = Rayleigh <v,(I-K)v>/<v,v>  (v = float bottom eigvec; upper bound)
#       gap_lo = largest g with (1-g)I - K > 0 verified by BALL CHOLESKY (Sylvester);
#                a positive-definiteness certificate, no diagonalization.
#
# HONEST SCOPE: this certifies the finite Nystrom objects at fixed GL order (the
# GL-300/400/500 agreement is the discretization evidence).  The Nystrom->operator
# analyticity envelope (fully displayed quadrature error) is a SEPARATE upgrade,
# flagged, NOT claimed here.  Not RH/GRH.
import sys, time, math
import numpy as np
from numpy.polynomial.legendre import leggauss
from flint import arb, acb, arb_mat, acb_mat, ctx

PREC_WORK = 424          # working bits for kernel / inverse / Cholesky
PREC_NODE = 2048         # bits for node certification only
PI = None                # set after ctx.prec

LOG = open(__file__.replace('.py', '_run.log'), 'w')
def log(s):
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

# ---------- certified Gauss-Legendre nodes on (0,1) --------------------------
def legendre_P_dP(n, x):
    p0 = arb(1); p1 = arb(x)
    for k in range(2, n+1):
        p0, p1 = p1, ((2*k-1)*x*p1 - (k-1)*p0)/k
    dp = n*(x*p1 - p0)/(x*x - 1)
    return p1, dp

_NODE_CACHE = {}
def certified_gl_nodes(n):
    if n in _NODE_CACHE:
        return _NODE_CACHE[n]
    save = ctx.prec; ctx.prec = PREC_NODE
    Mpp = arb(n)**2 * (arb(n)**2 - 1) / 3        # Markov: sup_{[-1,1]}|P_n''|
    x0, _ = leggauss(n)
    T = []; WT = []; maxrad = 0.0
    for xv in x0:
        x = arb(float(xv))
        for _ in range(14):
            p, dp = legendre_P_dP(n, x)
            x = (x - p/dp).mid()
        x = arb(x.mid())
        fP, fdP = legendre_P_dP(n, x)
        aP = abs(fP); adP = abs(fdP)
        h = aP * Mpp / (adP*adP)
        assert float(h.mid()) + float(h.rad()) < 0.5, f"NK fails at {xv}"
        rho = 2*aP / (adP * (1 + (1 - 2*h).sqrt()))
        rho_hi = float(rho.mid()) + float(rho.rad())
        maxrad = max(maxrad, rho_hi)
        xc = arb(x.mid(), rho_hi)                 # certified node ball on (-1,1)
        pc, dpc = legendre_P_dP(n, xc)
        w = 2/((1 - xc*xc)*dpc*dpc)
        T.append((xc + 1)/2); WT.append(w/2)      # map to (0,1)
    ctx.prec = save
    _NODE_CACHE[n] = (T, WT, maxrad)
    return _NODE_CACHE[n]

# ---------- kernel, objects --------------------------------------------------
def build(n, L, sign, T, WT):
    """Return (A = I-K as arb_mat real, SW list, c list, cd list)."""
    SW = [wt.sqrt() for wt in WT]
    La = arb(L)
    A = arb_mat(n, n)
    c = []; cd = []
    for i in range(n):
        ti = T[i]
        for j in range(n):
            tj = T[j]
            if i == j:
                Sd = La
            else:
                tv = ti - tj
                Sd = (La*tv).sin()/tv
            tu = ti + tj
            Ss = (La*tu).sin()/tu
            f = SW[i]*SW[j]/PI
            k = f*(Sd + Ss) if sign > 0 else f*(Sd - Ss)
            A[i, j] = -k
        A[i, i] = A[i, i] + arb(1)
        if sign > 0:
            c.append(SW[i]*(La*ti).cos()); cd.append(SW[i]*(-ti*(La*ti).sin()))
        else:
            c.append(SW[i]*(La*ti).sin()); cd.append(SW[i]*( ti*(La*ti).cos()))
    return A, SW, c, cd

def ball_pd(Kr, mu):
    """Rigorous: True if ball Cholesky proves mu*I - Kr > 0 (all pivots > 0)."""
    n = Kr.nrows()
    L = [[arb(0)]*(i+1) for i in range(n)]
    for i in range(n):
        for k in range(i+1):
            s = (arb(mu) - Kr[i, k]) if i == k else (-Kr[i, k])
            for j in range(k):
                s = s - L[i][j]*L[k][j]
            if i == k:
                if not (float(s.mid()) - float(s.rad()) > 0):
                    return False
                L[i][i] = s.sqrt()
            else:
                L[i][k] = s/L[k][k]
    return True

def float_kernel(n, L, sign, T, SW):
    tf = np.array([float(t.mid()) for t in T]); swf = np.array([float(x.mid()) for x in SW])
    TU = tf[:, None]+tf[None, :]; TV = tf[:, None]-tf[None, :]
    Sd = np.where(np.abs(TV) < 1e-14, L, np.sin(L*TV)/np.where(np.abs(TV) < 1e-14, 1, TV))
    Ss = np.sin(L*TU)/TU
    Af = swf[:, None]*swf[None, :]/math.pi
    return Af*(Sd + Ss) if sign > 0 else Af*(Sd - Ss)

def cell_objects(n, L, sign, T, WT):
    A, SW, c, cd = build(n, L, sign, T, WT)
    Ainv = A.inv()                                # rigorous ball inverse
    # y = A^{-1} c
    y = [sum((Ainv[i, j]*c[j] for j in range(n)), arb(0)) for i in range(n)]
    cc  = sum((c[i]*y[i] for i in range(n)), arb(0))
    cdc = sum((cd[i]*y[i] for i in range(n)), arb(0))
    delta  = (arb(2)/PI)*cc
    ddelta = (arb(4)/PI)*cdc + delta*delta
    eta = None
    if sign > 0:
        w = [sum((Ainv[i, j]*SW[j] for j in range(n)), arb(0)) for i in range(n)]
        eta = sum((SW[i]*w[i] for i in range(n)), arb(0))
    # ---- gap ----
    Kf = float_kernel(n, L, sign, T, SW)
    evf, V = np.linalg.eigh(Kf); gap_f = 1 - evf[-1]; v = V[:, -1]
    # Kr = K as arb_mat (= I - A off-diagonal; rebuild real K)
    Kr = arb_mat(n, n)
    for i in range(n):
        for j in range(n):
            Kr[i, j] = -A[i, j]
        Kr[i, i] = Kr[i, i] + arb(1)
    # gap_hi: Rayleigh on A=I-K with bottom eigvec v
    vv = [arb(float(x)) for x in v]
    Av = [sum((A[i, j]*vv[j] for j in range(n)), arb(0)) for i in range(n)]
    num = sum((vv[i]*Av[i] for i in range(n)), arb(0))
    den = sum((vv[i]*vv[i] for i in range(n)), arb(0))
    ray = num/den
    gap_hi = float(ray.mid()) + float(ray.rad())
    # gap_lo: verify (1-g)I - K > 0 by ball Cholesky; anchor at gap_f
    gap_lo = None
    for frac in (1 - 1e-3, 1 - 3e-3, 1 - 1e-2, 1 - 3e-2, 1 - 1e-1):
        g = gap_f*frac
        if ball_pd(Kr, 1 - g):
            gap_lo = g; break
    return dict(delta=delta, ddelta=ddelta, eta=eta, gap_lo=gap_lo, gap_hi=gap_hi,
                gap_f=gap_f, Ainv_ok=True)

# ---------- driver -----------------------------------------------------------
def enc(x):
    """compact mid +/- rad string"""
    return f"{float(x.mid()):+.9f} (r {float(x.rad()):.1e})"

def main():
    global PI
    ctx.prec = PREC_WORK
    PI = arb.pi()
    log(f"=== interval_grids_arb @ {time.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"prec_work={PREC_WORK} bits, prec_node={PREC_NODE} ===")
    cells = [1.0, 1.5, 2.0, 2.5, 3.0, 3.35]
    if '--probe' in sys.argv:
        cells = [2.0]
    orders = [400] if '--probe' in sys.argv else [300, 400, 500]
    for n in orders:
        t0 = time.time()
        T, WT, mr = certified_gl_nodes(n)
        log(f"\n#### GL-{n}: certified nodes ({time.time()-t0:.1f}s), max node ball radius {mr:.1e}")
        log(f"{'xi':>5} {'par':>3} | {'delta':>22} | {'delta_prime':>22} | "
            f"{'eta_+':>16} | {'gap_lo':>12} {'gap_hi':>12}")
        for xi in cells:
            L = 2*xi*xi/math.pi
            for sign, name in ((+1, 'E'), (-1, 'O')):
                tc = time.time()
                o = cell_objects(n, L, sign, T, WT)
                etastr = enc(o['eta']) if o['eta'] is not None else (' '*16)
                log(f"{xi:5.2f} {name:>3} | {enc(o['delta']):>22} | {enc(o['ddelta']):>22} | "
                    f"{etastr:>16} | {o['gap_lo']:.5e} {o['gap_hi']:.5e}  ({time.time()-tc:.0f}s)")
    log("\ndone.")
    LOG.close()

if __name__ == '__main__':
    main()
