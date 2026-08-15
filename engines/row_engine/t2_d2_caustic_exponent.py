# t2_d2_caustic_exponent.py — decide the excursion exponent BEFORE proving anything:
# E_j := |min_m ps_m| (the negative excursion; kappa = 1 + E at large j) needs only
# n <= ~6j rows, so the ladder extends to j = 8192 where sqrt(j) vs j^(1/3) separate.
# Plus: Olver scaling collapse of c_n at s = (z_n - mu)/mu^(1/3), and mp row spots.
import numpy as np
import mpmath as mp
from scipy.special import jv, yv

NEED = lambda j: [2*j - 3, 2*j - 2, 2*j - 1, 2*j, 2*j + 1, 2*j + 2]

def _near_AB(z, j, ktop):
    """(mant, log) pairs for A_k (backward Miller, A_0 = 1 exact norm) and B_k (forward,
    rescaled) at the needed orders, for z below the turning/oscillatory boundary."""
    need = set(NEED(j))
    u_hi = np.zeros_like(z); u = np.ones_like(z); ls = np.zeros_like(z)
    Am = {}; Al = {}
    for k in range(ktop, 0, -1):
        u_lo = ((2 * k + 1) / z) * u - u_hi
        u_hi, u = u, u_lo
        m = np.abs(u)
        big = m > 1e150
        if np.any(big):
            u[big] /= m[big]; u_hi[big] /= m[big]; ls[big] += np.log(m[big])
        if (k - 1) in need:
            Am[k - 1] = u.copy(); Al[k - 1] = ls.copy()
    u0 = u.copy(); l0 = ls.copy()
    A = {k: (Am[k] / u0, Al[k] - l0) for k in Am}
    b_lo = np.zeros_like(z); b = np.ones_like(z); lb = np.zeros_like(z)
    Bm = {}; Bl = {}
    if 1 in need: Bm[1] = b.copy(); Bl[1] = lb.copy()
    for k in range(1, max(need) + 1):
        b_new = ((2 * k + 1) / z) * b - b_lo
        b_lo, b = b, b_new
        m = np.abs(b)
        big = m > 1e150
        if np.any(big):
            b[big] /= m[big]; b_lo[big] /= m[big]; lb[big] += np.log(m[big])
        if (k + 1) in need:
            Bm[k + 1] = b.copy(); Bl[k + 1] = lb.copy()
    B = {k: (Bm[k], Bl[k]) for k in Bm}
    return A, B

def _pair(pa, pb):
    """product of two (mant, log) pairs -> plain float array (exp of combined log)."""
    with np.errstate(all='ignore'):
        v = pa[0] * pb[0] * np.exp(pa[1] + pb[1])
    return v

def rows(j, N, ktop_extra=80):
    """K-row and c-row at atoms 1..N — pure rational, scipy/mp-free.
    far (z >= zstar): forward recurrence (verified stable);
    near (z < zstar): backward Miller for A (minimal), rescaled forward for B."""
    mu = 2 * j + 0.5
    kmax = 2 * j + 2
    zstar = 3.0 * kmax + 20.0
    n = np.arange(1, N + 1, dtype=float); z = (n - 0.5) * np.pi
    c = np.zeros(N); K = np.zeros(N)
    need = NEED(j)
    def assemble(A, B, zz):
        with np.errstate(all='ignore'):
            GJJ = ((2 * zz / np.pi) * (A[(2*j-1, 2*j+2)] / (mu + 1) - A[(2*j-3, 2*j)] / (mu - 1))
                   - (1 / np.pi) * ((2*mu - 1) * A[(2*j, 2*j+2)] / (mu + 1)
                                    - (2*mu - 5) * A[(2*j-2, 2*j)] / (mu - 1)))
            GY = (-(4 / np.pi) * B[(2*j+1, 2*j)] + (2 * zz / np.pi) * B[(2*j, 2*j)] / (mu + 1)
                  + (4 / np.pi) * B[(2*j-1, 2*j-2)] - (2 * zz / np.pi) * B[(2*j-2, 2*j-2)] / (mu - 1))
            cc = GJJ + GY
            KK = A[('proj',)] * (4 * mu / zz ** 2) - (np.pi / 2) * cc
        cc[~np.isfinite(cc)] = 0.0; KK[~np.isfinite(KK)] = 0.0
        return cc, KK
    # near block
    sel = z < zstar
    if np.any(sel):
        zn = z[sel]
        ktop = int(zstar) + ktop_extra
        An, Bn = _near_AB(zn, j, ktop)
        AA = {(a, b): _pair(An[a], An[b]) for (a, b) in
              [(2*j-1, 2*j+2), (2*j-3, 2*j), (2*j, 2*j+2), (2*j-2, 2*j)]}
        AA[('proj',)] = _pair(An[2*j], An[2*j])
        AB = {(a, b): _pair(An[a], Bn[b]) for (a, b) in
              [(2*j+1, 2*j), (2*j, 2*j), (2*j-1, 2*j-2), (2*j-2, 2*j-2)]}
        cc, KK = assemble(AA, AB, zn)
        c[sel] = cc; K[sel] = KK
    # far block
    self_far = ~sel
    if np.any(self_far):
        zf = z[self_far]
        need_set = set(need)
        a0, a1 = np.ones_like(zf), 1.0 / zf
        b0, b1 = np.zeros_like(zf), np.ones_like(zf)
        A = {}; B2 = {}
        if 1 in need_set: A[1] = a1.copy(); B2[1] = b1.copy()
        for k in range(1, kmax + 1):
            a0, a1 = a1, ((2 * k + 1) / zf) * a1 - a0
            b0, b1 = b1, ((2 * k + 1) / zf) * b1 - b0
            if (k + 1) in need_set:
                A[k + 1] = a1.copy(); B2[k + 1] = b1.copy()
        AA = {(a, b): A[a] * A[b] for (a, b) in
              [(2*j-1, 2*j+2), (2*j-3, 2*j), (2*j, 2*j+2), (2*j-2, 2*j)]}
        AA[('proj',)] = A[2*j] * A[2*j]
        AB = {(a, b): A[a] * B2[b] for (a, b) in
              [(2*j+1, 2*j), (2*j, 2*j), (2*j-1, 2*j-2), (2*j-2, 2*j-2)]}
        cc, KK = assemble(AA, AB, zf)
        c[self_far] = cc; K[self_far] = KK
    return z, c, K

def mp_spot(j, ns):
    mp.mp.dps = 30
    mu = 2 * j + 0.5
    out = []
    for n in ns:
        zz = mp.mpf(2 * n - 1) * mp.pi / 2
        J = lambda kk: mp.besselj(kk + mp.mpf(1)/2, zz); Y = lambda kk: mp.bessely(kk + mp.mpf(1)/2, zz)
        dW = ((J(2*j-1) - (2*mu-1)*J(2*j)/(2*zz) + Y(2*j)) * J(2*j+2) / (mu+1)
              - (J(2*j-3) - (2*mu-5)*J(2*j-2)/(2*zz) + Y(2*j-2)) * J(2*j) / (mu-1))
        out.append(float(zz**2 * dW))
    return out

if __name__ == "__main__":  # guard added next session so rows() is importable unchanged
    print("=== (E) excursion ladder: E_j = |min ps| on n <= 6j (Miller/forward rational rows) ===")
    print("   j       E_j      E/sqrt(j)   E/j^(1/3)   m*_E/j   verif")
    prev = None
    for j in [256, 512, 1024, 2048, 4096, 8192]:
        N = 6 * j
        z, c, K = rows(j, N)
        ps = np.cumsum(K)
        iE = int(np.argmax(-ps))
        E = float(-ps[iE])
        # verification: mp spots where mpmath converges (j <= 1024); ktop-stability all j
        spots = [int(round(x * j)) for x in [0.60, 0.637, 0.70, 1.0]]
        z2, c2, K2 = rows(j, max(spots) + 2, ktop_extra=330)
        ktop_dev = max(abs(c[n - 1] - c2[n - 1]) for n in spots)
        if j <= 1024:
            truth = mp_spot(j, spots)
            vdev = max(abs(c[n - 1] - t) for n, t in zip(spots, truth))
            ver = f"mp {vdev:.0e} / ktop {ktop_dev:.0e}"
        else:
            ver = f"ktop {ktop_dev:.0e}"
        line = f"{j:6d}  {E:9.4f}   {E/np.sqrt(j):8.4f}   {E/j**(1/3):8.4f}   {(iE+1)/j:6.3f}   {ver}"
        if prev is not None:
            line += f"   exp {np.log(E/prev)/np.log(2):.3f}"
        prev = E
        print(line)

    print("=== (S) Olver collapse: c_n at s = (z_n - mu)/mu^(1/3) ===")
    SVALS = [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    print("    j   " + "".join(f"  s={s:5.1f}" for s in SVALS))
    for j in [512, 1024, 2048, 4096, 8192]:
        mu = 2 * j + 0.5
        z, c, K = rows(j, int(3 * j))
        row = []
        for s in SVALS:
            zt = mu + s * mu ** (1.0 / 3.0)
            n0 = int(round(zt / np.pi + 0.5))
            row.append(c[n0 - 1])
        print(f"{j:6d} " + "".join(f" {v:+7.3f}" for v in row))
    print("done.")
