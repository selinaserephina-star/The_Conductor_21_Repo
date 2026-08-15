# t2_d2_kappa_full_ladder.py — STEP 4: the corrected FULL kappa ladder on certified
# Miller rows (protocol identical to t2_d2_kappa_protocol_resolution.py: kappa =
# max(sup_fwd, sup_rev), sup_rev = sup_m |1 - ps_{m-1}| init 1.0, cap = max(2mu^2/pi, 4000)),
# but with NO scipy jv/yv anywhere (float-trap #4): near block (z < zstar) from
# t2_d2_caustic_exponent.rows() (backward Miller + forward B, log-mantissa), far block
# the same needed-orders rational forward recurrence as rows()'s far branch (verified
# below for exact equality against rows() on an overlap window before use).
import numpy as np
from t2_d2_caustic_exponent import rows, NEED

CH = 200_000

def far_K(j, n_lo, n_hi):
    """K rows for atoms n_lo..n_hi, all with z >= zstar (forward recurrence, needed orders)."""
    mu = 2 * j + 0.5
    kmax = 2 * j + 2
    n = np.arange(n_lo, n_hi + 1, dtype=float)
    z = (n - 0.5) * np.pi
    need_set = set(NEED(j))
    a0, a1 = np.ones_like(z), 1.0 / z
    b0, b1 = np.zeros_like(z), np.ones_like(z)
    A = {}; B2 = {}
    if 1 in need_set: A[1] = a1.copy(); B2[1] = b1.copy()
    for k in range(1, kmax + 1):
        a0, a1 = a1, ((2 * k + 1) / z) * a1 - a0
        b0, b1 = b1, ((2 * k + 1) / z) * b1 - b0
        if (k + 1) in need_set:
            A[k + 1] = a1.copy(); B2[k + 1] = b1.copy()
    # products first, then the assemble expression — the EXACT operation ordering of
    # rows()'s far branch, so the chunked scan is bitwise-identical to the certified code
    AA = {(a, b): A[a] * A[b] for (a, b) in
          [(2*j-1, 2*j+2), (2*j-3, 2*j), (2*j, 2*j+2), (2*j-2, 2*j)]}
    AAproj = A[2*j] * A[2*j]
    AB = {(a, b): A[a] * B2[b] for (a, b) in
          [(2*j+1, 2*j), (2*j, 2*j), (2*j-1, 2*j-2), (2*j-2, 2*j-2)]}
    with np.errstate(all='ignore'):
        GJJ = ((2 * z / np.pi) * (AA[(2*j-1, 2*j+2)] / (mu + 1) - AA[(2*j-3, 2*j)] / (mu - 1))
               - (1 / np.pi) * ((2*mu - 1) * AA[(2*j, 2*j+2)] / (mu + 1)
                                - (2*mu - 5) * AA[(2*j-2, 2*j)] / (mu - 1)))
        GY = (-(4 / np.pi) * AB[(2*j+1, 2*j)] + (2 * z / np.pi) * AB[(2*j, 2*j)] / (mu + 1)
              + (4 / np.pi) * AB[(2*j-1, 2*j-2)] - (2 * z / np.pi) * AB[(2*j-2, 2*j-2)] / (mu - 1))
        cc = GJJ + GY
        K = AAproj * (4 * mu / z ** 2) - (np.pi / 2) * cc
    K[~np.isfinite(K)] = 0.0
    return K

# --- far-block equality check against rows() before any production use ---
for jt in [8, 64]:
    zstar = 3.0 * (2 * jt + 2) + 20.0
    n_star = int(np.ceil(zstar / np.pi + 0.5)) + 2
    Ntest = n_star + 5000
    _, _, Kref = rows(jt, Ntest)
    Kfar = far_K(jt, n_star + 1, Ntest)
    d = np.max(np.abs(Kfar - Kref[n_star:]))
    assert d == 0.0, f"far-block mismatch at j={jt}: {d}"
print("far-block chunk machinery == rows() far branch EXACTLY (j = 8, 64): OK")

def kappa_scan(j):
    mu = 2 * j + 0.5
    cap = int(max(2 * mu * mu / np.pi, 4000))
    zstar = 3.0 * (2 * j + 2) + 20.0
    n_star = min(int(np.ceil(zstar / np.pi + 0.5)) + 2, cap)
    _, _, Knear = rows(j, n_star)          # certified Miller near block (includes far tail bit)
    sup_fwd = 0.0; sup_rev = 1.0; mstar_f = 1; ps_run = 0.0
    lo = 1
    while lo <= cap:
        hi = min(lo + CH - 1, cap)
        if lo <= n_star:
            h2 = min(hi, n_star)
            K = Knear[lo - 1:h2]
            if hi > n_star:
                K = np.concatenate([K, far_K(j, n_star + 1, hi)])
        else:
            K = far_K(j, lo, hi)
        ps = ps_run + np.cumsum(K)
        i1 = int(np.argmax(np.abs(ps)))
        if abs(ps[i1]) > sup_fwd:
            sup_fwd = float(abs(ps[i1])); mstar_f = lo + i1
        rev = np.abs(1.0 - np.concatenate([[ps_run], ps[:-1]]))
        sup_rev = max(sup_rev, float(np.max(rev)))
        ps_run = float(ps[-1])
        lo = hi + 1
    return max(sup_fwd, sup_rev), sup_fwd, sup_rev, mstar_f, ps_run

CSV = {8: 1.1775044650866837, 16: 1.3359798836918013, 32: 1.53912320756528,
       64: 1.9088587014230043, 128: 2.2741835641142716, 192: 2.6403354907190546}
F7 = {256: 2.9103, 384: 3.2891, 512: 3.7008}   # scipy-hybrid values (contaminated at >=512)
A_INF = 0.346555372
print("   j    kappa      CSV/F7     fwd       rev      m*_f    rowsum     1 + a_inf mu^(1/3) - 0.85")
for j in [8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024]:
    kap, f, r, ms, rs = kappa_scan(j)
    ref = CSV.get(j, F7.get(j, float('nan')))
    mu = 2 * j + 0.5
    pred = 1 + A_INF * mu ** (1 / 3) - 0.85
    print(f"{j:5d}  {kap:8.4f}   {ref:8.4f}  {f:8.4f}  {r:8.4f}  {ms:7d}  {rs:9.5f}   {pred:8.4f}")
print("done.")
