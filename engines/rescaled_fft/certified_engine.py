# certified_engine.py — rigor-ladder steps 3+5(+6 core): ball-certified F(t) evaluation.
#
# Pipeline (Booker (5.18)-(5.19), certified):
#   S_m^(k) = sum_{n in bucket m} a_n n^{-1/2} (log(n/sqrtN) - m*h)^k
#     computed in float64 with RIGOROUS error radii:
#     - buckets are CONTIGUOUS n-slices (u = log(n/sqrtN) is monotone), so each bucket sum
#       is np.sum over a slice -> pairwise summation, proven bound (b + log2 p) u Sum|t|
#       (numpy pairwise block b = 128), NOT bincount's unbounded sequential accumulation;
#     - per-term evaluation errors (sqrt, div, log, d^k) bounded in ulps:
#         Delta(term) <= |c| [ (k+5) u_mach (h/2)^k + k (h/2)^{k-1} Dd ],
#       Dd = 1.6e-14 (abs error of d = log n - logsqN - m*h_exact, incl. h-representation);
#     - radius: Rad_m^(k) = A_m [ (h/2)^k ((k+5) + 128 + log2 p_m) u_mach
#                                 + k (h/2)^{k-1} Dd ],  A_m = pairwise Sum|c| (inflated).
#   Fhat(x_l) ball = N^{1/4} sum_k sum_m G^(k)((l+m)h) S_m^(k) / k!   -- G^(k) certified
#     (g_kernel_arb, exact grid u = l*pi/32, radii <= 1e-72), S as acb balls mid +- Rad.
#   F(t): by POISSON SUMMATION the computable sum
#         (2pi/B) [ Fhat_0 + 2 sum_{l>=1} Re( Fhat_l e^{i l h t} ) ]  =  sum_j F(t + jB)
#     EXACTLY. So a ball evaluation of the left side encloses F(t) up to the alias terms
#     F(t +- B), ... (step 4; bounded separately -- at B = 64 they are ~e^{-2pi(1+eta)(B-t)}).
#   Zero certification: at a bracket [lo, hi], ball F(lo), F(hi) both excluding 0 with
#     opposite signs => certified sign change (modulo the declared step-4 tail terms).
#
# STATUS: certified MODULO the step-4 analytic tails (coefficient truncation n > M,
# bucket-Taylor K-remainder, aliasing). Those are the next rung; this module measures and
# prints the enclosure radii that decide the production operating point (eta, M, precision).
import numpy as np
import mpmath as mp
import time
from flint import acb, arb, ctx

U_MACH = 2.0 ** -53


def certified_S(an_file, B, K=14, verbose=True):
    """Bucket sums S_m^(k) (K x nbuck float64) + rigorous radii (K x nbuck).
    Production: K = 16 (step-4 Taylor-remainder requirement)."""
    a = np.load(an_file)
    M = len(a) - 1
    h = 2 * np.pi / B                      # float64 h; exact-h correction inside Dd
    logsqN = float(mp.log(mp.mpf(21) ** 10) / 2)
    t0 = time.time()
    n = np.arange(1, M + 1, dtype=np.float64)
    c = a[1:].astype(np.float64) / np.sqrt(n)
    u = np.log(n) - logsqN
    midx = np.rint(u / h).astype(np.int64)
    d = u - midx * h
    # bucket boundaries: midx is nondecreasing (u monotone; rint ties can't invert order
    # beyond 1 ulp -- enforce and verify)
    assert np.all(np.diff(midx) >= 0), "bucket index not monotone"
    m_min, m_max = int(midx[0]), int(midx[-1])
    edges = np.searchsorted(midx, np.arange(m_min, m_max + 2))
    nbuck = m_max - m_min + 1
    S = np.zeros((K, nbuck)); R = np.zeros((K, nbuck))
    Dd = 1.6e-14
    for mi in range(nbuck):
        s0, s1 = edges[mi], edges[mi + 1]
        if s0 == s1:
            continue
        p = s1 - s0
        cs = c[s0:s1]; ds = d[s0:s1]
        A = float(np.sum(np.abs(cs))) * (1 + 1e-12)      # Sum|c|, inflated
        sumbound = (128 + np.log2(max(p, 2))) * U_MACH
        dk = np.ones(p)
        for k in range(K):
            S[k, mi] = np.sum(cs * dk)
            R[k, mi] = A * ((h / 2) ** k * ((k + 5) * U_MACH + sumbound)
                            + k * (h / 2) ** (k - 1) * Dd)
            dk = dk * ds
    if verbose:
        print(f"certified S: M={M}, {nbuck} buckets (m {m_min}..{m_max}), "
              f"max radius k=0: {R[0].max():.2e}, {time.time()-t0:.1f}s", flush=True)
    return S, R, m_min, m_max


def certified_fhat(S, R, m_min, m_max, B, eta_num, eta_den, prec=250, u_hi_l=None,
                   gcache=None, verbose=True):
    """Ball Fhat_l for l = 0..l_top, from certified G-table at eta = eta_num/eta_den.
    Requires B = 64 (the exact-grid G table u = l*pi/32)."""
    assert B % 2 == 0, "B even (grid u = l*pi/(B/2))"
    grid_den = B // 2
    ctx.prec = prec
    import g_kernel_arb as ga
    K = S.shape[0]
    # G needed at (l+m)h for l >= 0, m in [m_min, m_max]; table covers -157..134
    if u_hi_l is None:
        # kernel dead beyond 8*pi*cos(pi eta/2)*e^{u/4} > 170  (|G| < 1e-73)
        import math
        e = eta_num / eta_den
        u_hi = 4 * math.log(170 / (8 * math.pi * math.cos(math.pi * e / 2)))
        u_hi_l = int(np.ceil(u_hi / (2 * np.pi / B)))
    t0 = time.time()
    gtab = {}
    ctx.prec = ga.PREC        # G-table needs full precision: residue-series cancellation
    for l in range(m_min, u_hi_l + 1):
        gtab[l] = ga.g_all_k(l, eta_num=eta_num, eta_den=eta_den, grid_den=grid_den)
    ctx.prec = prec           # assembly precision (balls keep their certified radii)
    if verbose:
        print(f"certified G-table: l = {m_min}..{u_hi_l} (prec {ga.PREC}), "
              f"{time.time()-t0:.1f}s", flush=True)
    invfact = [1 / arb.fac_ui(k) for k in range(K)]
    N4 = (arb(21) ** 10).root(4)
    l_top = u_hi_l - m_min                     # beyond: every (l+m)h > u_hi, Fhat ~ 0
    t0 = time.time()
    Fhat = []
    for l in range(0, l_top + 1):
        acc = acb(0)
        for mi in range(0, m_max - m_min + 1):
            m = m_min + mi
            if l + m > u_hi_l:
                break
            g = gtab[l + m]
            for k in range(K):
                if S[k, mi] == 0 and R[k, mi] == 0:
                    continue
                acc += g[k] * (acb(arb(S[k, mi], R[k, mi])) * invfact[k])
        Fhat.append(acc * N4)
    if verbose:
        r = max(max(float(f.real.rad()), float(f.imag.rad())) for f in Fhat)
        print(f"certified Fhat: {len(Fhat)} entries, max radius {r:.2e}, "
              f"{time.time()-t0:.1f}s", flush=True)
    return Fhat


def F_ball(t, Fhat, B, prec=250):
    """Ball enclosure of sum_j F(t + jB) = (2pi/B)[Fhat_0 + 2 sum Re(Fhat_l e^{ilht})].
    t may be arb or float. Alias terms F(t±B),... are the declared step-4 remainder."""
    ctx.prec = prec
    pi = arb.pi()
    h_t = (2 * pi / B) * (t if isinstance(t, arb) else arb(t))
    acc = Fhat[0].real
    for l in range(1, len(Fhat)):
        acc += 2 * (Fhat[l] * acb(0, l * h_t).exp()).real
    return (2 * pi / B) * acc


def shakedown():
    """M = 1e8, eta = 3/10, B = 64: certify sign changes at the ball-certified low-zero
    brackets and print the radius profile that picks the production operating point."""
    import csv
    B = 64
    S, R, m_min, m_max = certified_S('an_chi8_100M.npy', B)
    Fhat = certified_fhat(S, R, m_min, m_max, B, 3, 10)

    # Lambda(1/2) sanity (float reference 5.8379065826347)
    lam = F_ball(0.0, Fhat, B)
    N = mp.mpf(21) ** 10
    gam_half = 16 * mp.e ** (mp.log(N) / 4 - 2 * mp.log(2 * mp.pi)) * mp.gamma(mp.mpf('0.5')) ** 4
    Lmid = float(lam.mid()) / float(gam_half)
    print(f"\nL(1/2) ball: {Lmid:.13f} +/- {float(lam.rad())/float(gam_half):.2e}"
          f"   (ref 5.8379065826347)")

    # radius profile of F on a t-grid (what precision the enclosure delivers vs height)
    print("\nt, |F| mid, radius, S/N:")
    for t in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        f = F_ball(t, Fhat, B)
        m_, r_ = abs(float(f.mid())), float(f.rad())
        print(f"  t={t:4.1f}  |F|={m_:.3e}  rad={r_:.3e}  S/N={m_/r_:8.1f}")

    # certified sign changes at the master-table brackets
    print("\ncertified-bracket sign changes (modulo step-4 tails):")
    rows = [r for r in csv.DictReader(open(
        r'..\chi8_certified_zeros_MASTER_2026-07-14\chi8_zeros_master.csv'))
        if r.get('bracket_lo') and r.get('bracket_hi')]
    ok = 0
    for row in rows[:10]:
        j = int(row['j']); lo = float(row['bracket_lo']); hi = float(row['bracket_hi'])
        flo, fhi = F_ball(lo, Fhat, B), F_ball(hi, Fhat, B)
        slo = flo > 0 if flo != 0 else None
        shi = fhi > 0 if fhi != 0 else None
        # arb comparisons: x > 0 is True only if the whole ball is positive
        def sgn(x):
            if x > 0: return +1
            if x < 0: return -1
            return 0     # ball straddles 0 -> cannot certify
        a, b = sgn(flo), sgn(fhi)
        verdict = 'CERTIFIED SIGN CHANGE' if a * b == -1 else \
                  ('ball straddles 0' if 0 in (a, b) else 'SAME SIGN **')
        if a * b == -1: ok += 1
        print(f"  gamma_{j}: [{lo},{hi}]  F(lo) sgn {a:+d} rad {float(flo.rad()):.1e}, "
              f"F(hi) sgn {b:+d} rad {float(fhi.rad()):.1e}  -> {verdict}")
    print(f"\n{ok}/10 brackets certified (modulo step-4 tails) at M=1e8, eta=0.3")


if __name__ == '__main__':
    shakedown()
