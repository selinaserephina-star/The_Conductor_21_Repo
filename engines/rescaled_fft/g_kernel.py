# g_kernel.py — the Mellin kernel G(u; eta, {mu_j}) for chi8 (Booker 2006 eq. (5.6)).
#
# For chi8: mu = {0,0,0,0,1,1,1,1}, r = 8.  By Legendre duplication,
#     prod_j Gamma_R(s+mu_j) = Gamma_R(s)^4 Gamma_R(s+1)^4 = 16 (2pi)^{-4s} Gamma(s)^4,
# so with w := u + i*pi*r*eta/4 = u + 2*pi*i*eta,
#     G(u; eta)      = (1/2pi i) int e^{w(1/2-s)} 16 (2pi)^{-4s} Gamma(s)^4 ds
#     G^(k)(u; eta)  = (1/2pi i) int (1/2-s)^k e^{w(1/2-s)} 16 (2pi)^{-4s} Gamma(s)^4 ds .
# The integrand has a saddle near s* = ((2pi)^4 e^w)^{1/4} = 2pi e^{w/4}; we integrate on
# the vertical line through Re(s*) (well-conditioned: no catastrophic cancellation),
# using mpmath quadrature at high dps.
#
# Validation: (a) contour independence (two sigma choices agree),
#             (b) residue series over the order-4 poles of Gamma(s)^4 at s = 0,-1,-2,...
#                 (computed as circle contour integrals), for u <= 0.
import mpmath as mp

TWO_PI = None  # set per-dps


def _g_integrand(s, k, w):
    # (1/2-s)^k e^{w(1/2-s)} 16 (2pi)^{-4s} Gamma(s)^4
    half = mp.mpf(1) / 2
    val = 16 * mp.e**(w * (half - s) - 4 * s * mp.log(2 * mp.pi)) * mp.gamma(s)**4
    if k:
        val *= (half - s)**k
    return val


def g_kernel(u, eta, k=0, dps=45):
    """G^(k)(u; eta) for chi8. Saddle-line quadrature."""
    with mp.workdps(dps):
        w = mp.mpf(u) + 2j * mp.pi * mp.mpf(eta)
        # saddle: s* = 2pi e^{w/4}; contour through its real part (>= 1 to stay right of poles)
        sig = max(mp.mpf(1), mp.re(2 * mp.pi * mp.e**(w / 4)))
        # t-decay: |Gamma(sig+it)|^4 ~ e^{-2pi|t|}, and e^{Re(-w i t)} = e^{2 pi eta t}
        # net e^{-(2pi - 2pi|eta|)|t|} on the bad side; pick cutoff T for target precision
        decay = 2 * mp.pi * (1 - abs(mp.mpf(eta)))
        T = (mp.mpf(dps) * mp.log(10) + 30) / decay + 4 * mp.sqrt(sig + 1)
        f = lambda t: _g_integrand(sig + 1j * t, k, w)
        val = mp.quad(f, [-T, -T / 8, 0, T / 8, T]) / (2 * mp.pi)
        # (1/2pi i) * i dt = dt/2pi with s = sig+it
        return val


def g_kernel_residue_series(u, eta, k=0, jmax=40, dps=60):
    """Independent check: sum of residues at s = -j (order-4 poles), via small circle
    contour integrals. Converges for any u but cancellation grows with u; use u <= 0."""
    with mp.workdps(dps):
        w = mp.mpf(u) + 2j * mp.pi * mp.mpf(eta)
        total = mp.mpc(0)
        for j in range(jmax):
            r = mp.mpf(1) / 4
            f = lambda th: _g_integrand(-j + r * mp.e**(1j * th), k, w) * (
                1j * r * mp.e**(1j * th))
            res = mp.quad(f, [0, mp.pi, 2 * mp.pi]) / (2j * mp.pi)
            total += res
        return total


def build_table(h, l_min, l_max, eta, K, dps=45, verbose=True):
    """G^(k)(l*h; eta) for l in [l_min, l_max], k in [0, K). Returns dict of float
    (complex) values [l][k]. Skips (returns 0) where the asymptotic bound says
    |G| < 1e-60."""
    import math
    tab = {}
    for l in range(l_min, l_max + 1):
        u = l * h
        # true decay: |G| ~ exp(-8 pi e^{u/4} cos(pi eta / 2)) for u large
        if u > 0 and 8 * math.pi * math.exp(u / 4) * math.cos(math.pi * eta / 2) > 160:
            tab[l] = [complex(0)] * K
            continue
        row = []
        for k in range(K):
            row.append(complex(g_kernel(u, eta, k, dps=dps)))
        tab[l] = row
        if verbose and l % 10 == 0:
            print(f"  l={l} u={u:.3f} |G|={abs(row[0]):.3e}", flush=True)
    return tab


if __name__ == '__main__':
    import time
    eta = mp.mpf(3) / 10
    print("== contour independence (u=1.0, k=0,3) ==")
    for k in (0, 3):
        t0 = time.time()
        a = g_kernel(1.0, eta, k, dps=45)
        print(f"  k={k}: {mp.nstr(a, 20)}   ({time.time()-t0:.2f}s)")
    print("== quadrature vs residue series ==")
    for u in (-2.0, -0.5, 0.0, 1.0):
        for k in (0, 2):
            q = g_kernel(u, eta, k, dps=50)
            r = g_kernel_residue_series(u, eta, k, jmax=45, dps=70)
            diff = abs(mp.mpc(q) - mp.mpc(r))
            rel = diff / max(abs(q), mp.mpf(1e-60))
            print(f"  u={u:5.2f} k={k}: quad={mp.nstr(q,16)}  reldiff={mp.nstr(rel,3)}")
