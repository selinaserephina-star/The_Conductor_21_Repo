# gen_an_chi8.py — chi8 Dirichlet coefficients a_n, n <= M, pure Python (flint + numpy).
#
# Port of Step2_chi8_main_jets/gen_an_16M.gp / gen_an_160M.gp (PARI direuler recipe):
#   loc8(p) by Frobenius cycle type of x^7 - 7x + 3 mod p:
#     [1^7]       -> (1-X)^8            (class 1A, a_p = 8)
#     [1,1,1,2,2] -> (1-X)^4 (1+X)^4    (class 2A, a_p = 0)
#     [1,3,3]     -> (1-X)^2 (1+X+X^2)^3       (class 3A, a_p = -1)
#     [1,2,4]     -> (1-X)^2 (1+X)^2 (1+X^2)^2 (class 4A, a_p = 0)
#     [7]         -> (1-X)^2 (1+X+...+X^6)     (class 7A/7B, a_p = 1)
#   ramified: p=3 -> 1-X ;  p=7 -> 1
# a_n = coefficients of prod_p 1/loc8(p)(p^-s), expanded by an in-place multiplicative sieve.
#
# Usage: python gen_an_chi8.py M out.npy [--check blocks.csv]
import sys, time
import numpy as np
from flint import nmod_poly

# --- local factor polynomials (coeff lists, low to high), det(1 - Frob X) ---
def polymul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out

def polypow(a, k):
    out = [1]
    for _ in range(k):
        out = polymul(out, a)
    return out

LOC8 = {
    '1A': polypow([1, -1], 8),
    '2A': polymul(polypow([1, -1], 4), polypow([1, 1], 4)),
    '3A': polymul(polypow([1, -1], 2), polypow([1, 1, 1], 3)),
    '4A': polymul(polymul(polypow([1, -1], 2), polypow([1, 1], 2)), polypow([1, 0, 1], 2)),
    '7':  polymul(polypow([1, -1], 2), [1] * 7),
    'p3': [1, -1],
    'p7': [1],
}
CYCLE_TO_CLASS = {
    (1, 1, 1, 1, 1, 1, 1): '1A',
    (1, 1, 1, 2, 2): '2A',
    (1, 3, 3): '3A',
    (1, 2, 4): '4A',
    (7,): '7',
}
AP = {'1A': 8, '2A': 0, '3A': -1, '4A': 0, '7': 1, 'p3': 1, 'p7': 0}

def inv_series(l, K):
    """coefficients c_0..c_K of 1/sum(l_j X^j), l_0 = 1, exact ints"""
    c = [1] + [0] * K
    for k in range(1, K + 1):
        s = 0
        for j in range(1, min(k, len(l) - 1) + 1):
            s -= l[j] * c[k - j]
        c[k] = s
    return c

def frob_class(p):
    if p == 3:
        return 'p3'
    if p == 7:
        return 'p7'
    f = nmod_poly([3 % p, (-7) % p, 0, 0, 0, 0, 0, 1], p)
    degs = tuple(sorted(g.degree() for g, e in f.factor()[1] for _ in range(e)))
    cls = CYCLE_TO_CLASS.get(degs)
    assert cls is not None, f"unexpected cycle type {degs} at p={p}"
    return cls

def prime_sieve(M):
    isp = np.ones(M + 1, dtype=bool)
    isp[:2] = False
    for p in range(2, int(M ** 0.5) + 1):
        if isp[p]:
            isp[p * p::p] = False
    return np.nonzero(isp)[0]

def gen_an(M, progress=True, dtype=np.int64):
    # int32 mode: safe in practice (|a_n| stays ~1e3 empirically at 1e8; worst-case
    # intermediates c_k*b ~ 1e8 << 2^31); final max reported by callers.
    a = np.zeros(M + 1, dtype=dtype)
    a[1] = 1
    primes = prime_sieve(M)
    t0 = time.time()
    # a_p for large primes only needs the trace; full series for p^2 <= M
    for i, p in enumerate(primes):
        p = int(p)
        cls = frob_class(p)
        if p * p > M:
            ap = AP[cls]
            if ap:
                # a[p*m] += ap * a_old[m], m <= M//p ; a_old[m]=0 if p|m (p unprocessed)
                mmax = M // p
                a[p:p * mmax + 1:p] += ap * a[1:mmax + 1]
        else:
            K = 1
            while p ** (K + 1) <= M:
                K += 1
            c = inv_series(LOC8[cls], K)
            b = a[:M // p + 1].copy()
            for k in range(1, K + 1):
                pk = p ** k
                mmax = M // pk
                a[pk:pk * mmax + 1:pk] += c[k] * b[1:mmax + 1]
        if progress and (i & 0xFFFFF) == 0 and i:
            print(f"  {i}/{len(primes)} primes, {time.time() - t0:.1f}s", flush=True)
    if progress:
        print(f"  sieve done: {len(primes)} primes, {time.time() - t0:.1f}s", flush=True)
    return a

def main():
    M = int(float(sys.argv[1]))
    out = sys.argv[2]
    a = gen_an(M)
    print("a_n for n=1..20:", a[1:21].tolist())
    np.save(out, a)
    print(f"saved {out} ({a.nbytes / 1e6:.0f} MB)")
    if '--check' in sys.argv:
        csv = sys.argv[sys.argv.index('--check') + 1]
        import csv as csvmod
        nblocks = ok = 0
        with open(csv) as fh:
            for row in csvmod.DictReader(fh):
                lo, hi = int(row['block_start']), int(row['block_end'])
                if hi > M:
                    break
                s = int(np.abs(a[lo:hi + 1]).sum())
                mx = int(np.abs(a[lo:hi + 1]).max())
                nblocks += 1
                if s == int(row['sum_abs_an']) and mx == int(row['max_abs_an']):
                    ok += 1
                else:
                    print(f"MISMATCH block {lo}-{hi}: sum {s} vs {row['sum_abs_an']}, max {mx} vs {row['max_abs_an']}")
        print(f"block check vs PARI direuler: {ok}/{nblocks} blocks match")

if __name__ == '__main__':
    main()
