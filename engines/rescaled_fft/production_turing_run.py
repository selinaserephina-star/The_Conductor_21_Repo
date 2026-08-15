# production_turing_run.py — one-shot orchestrator for the chi8 Turing completeness count.
#
#   stage 1: a_n generation (gen_an_chi8.gen_an, chunk-streamed into bucket sums S_m^(k))
#   stage 2: load G-table (precomputed pickle for (h, eta, K) — ship gtab from the laptop
#            or build here with g_kernel.build_table)
#   stage 3: assemble Fhat, FFT -> F(m/A), sanity: L(1/2) vs 5.8379065826347
#   stage 4: zero scan on [0, t2_max], refinement; write zeros CSV
#   stage 5: turing_count condition -> N(T*) = 21 verdict
#
# Production (AWS r7i.4xlarge, 123 GB, Linux):  M = 7e9, eta = 0.82, A = 64, B = 64
#   python3 production_turing_run.py --M 7e9 --eta 0.82 --A 64 --B 64 --t2 26
# Local shakedown:
#   python  production_turing_run.py --M 1e8 --eta 0.82 --A 64 --B 64 --t2 14 \
#           --an an_chi8_100M.npy
# Two-channel practice: rerun with --eta 0.87; zero lists must agree.
import argparse, json, os, pickle, time
import numpy as np
import mpmath as mp

mp.mp.dps = 40
SQN = None  # set in main


def bucket_sums_from_array(a, h, K, use_longdouble=True):
    """S_m^(k) from an in-memory a_n array, chunked; longdouble accumulators on Linux."""
    M = len(a) - 1
    logsqN = float(10 * mp.log(21) / 2)
    acc_dtype = np.longdouble if use_longdouble else np.float64
    m_min = int(np.rint((0 - logsqN) / h))               # n=1 -> u=-logsqN (rint = bucket rule)
    m_max = int(np.rint((np.log(M) - logsqN) / h))
    nbuck = m_max - m_min + 1
    S = np.zeros((K, nbuck), dtype=acc_dtype)
    CH = 10**7
    t0 = time.time()
    for lo in range(1, M + 1, CH):
        hi = min(lo + CH - 1, M)
        n = np.arange(lo, hi + 1, dtype=np.float64)
        an = a[lo:hi + 1]
        nz = an != 0
        if not nz.any():
            continue
        n = n[nz]
        coef = an[nz].astype(np.float64) / np.sqrt(n)
        u = np.log(n) - logsqN
        midx = np.rint(u / h).astype(np.int64) - m_min
        d = u - (midx + m_min) * h
        dk = np.ones_like(d)
        for k in range(K):
            S[k] += np.bincount(midx, weights=coef * dk, minlength=nbuck).astype(acc_dtype)
            dk *= d
        if (lo // CH) % 50 == 0:
            print(f"  S-sums: n={hi:.3g}/{M:.3g} {time.time()-t0:.0f}s", flush=True)
    return S, m_min, m_max


def assemble_fhat(S, m_min, m_max, gtab, l_lo, l_hi, h, K, q):
    logN = float(10 * mp.log(21))
    N4 = np.exp(logN / 4)
    Garr = np.zeros((K, l_hi - l_lo + 1), dtype=np.complex128)
    for l in range(l_lo, l_hi + 1):
        for k in range(K):
            Garr[k, l - l_lo] = gtab[l][k]
    fact = np.array([float(mp.factorial(k)) for k in range(K)])
    Sf = np.asarray(S, dtype=np.float64)  # G is float64; downcast at the last moment
    Fhat = np.zeros(q // 2 + 1, dtype=np.complex128)
    for l in range(q // 2 + 1):
        lo = max(m_min, l_lo - l)
        hi = min(m_max, l_hi - l)
        if lo > hi:
            continue
        acc = 0j
        for k in range(K):
            acc += (Garr[k, l + lo - l_lo: l + hi - l_lo + 1]
                    * Sf[k, lo - m_min: hi - m_min + 1]).sum() / fact[k]
        Fhat[l] = acc
    return Fhat * N4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=float, required=True)
    ap.add_argument('--eta', type=float, default=0.82)
    ap.add_argument('--A', type=int, default=64)
    ap.add_argument('--B', type=int, default=64)
    ap.add_argument('--K', type=int, default=14)
    ap.add_argument('--t2', type=float, default=26.0)
    ap.add_argument('--an', default=None, help='precomputed a_n .npy (else generate)')
    ap.add_argument('--gtab', default=None, help='G-table pickle (else build, slow)')
    ap.add_argument('--out', default='turing_run')
    args = ap.parse_args()
    M = int(args.M)
    A, B, K, eta = args.A, args.B, args.K, args.eta
    q = A * B
    h = 2 * np.pi / B
    os.makedirs(args.out, exist_ok=True)

    # ---- stage 1: coefficients + bucket sums ----
    if args.an and os.path.exists(args.an):
        a = np.load(args.an)
        assert len(a) - 1 >= M, "an file shorter than M"
        a = a[:M + 1]
    else:
        from gen_an_chi8 import gen_an
        a = gen_an(M, dtype=np.int32)   # max |a_n| = 64 at 1e8; int32 ample
        print(f"max |a_n| = {np.abs(a).max()}", flush=True)
    S, m_min, m_max = bucket_sums_from_array(
        a, h, K, use_longdouble=(np.longdouble(1).itemsize > 8))
    del a
    np.save(f'{args.out}/S_sums.npy', np.asarray(S, dtype=np.float64))
    print(f"S_m^(k) done: m {m_min}..{m_max}", flush=True)

    # ---- stage 2: G table ----
    import math
    l_lo = m_min
    # G dead beyond exponent 180
    u_dead = 4 * math.log(180 / (8 * math.pi * math.cos(math.pi * eta / 2)))
    l_hi = int(np.ceil(u_dead / h)) + 2
    key = (round(h, 12), l_lo, l_hi, float(eta), K, 45)
    if args.gtab and os.path.exists(args.gtab):
        with open(args.gtab, 'rb') as f:
            ck, gtab = pickle.load(f)
        assert ck[0] == key[0] and ck[3] == key[3] and ck[4] >= K, f"gtab mismatch {ck} vs {key}"
        assert ck[1] <= l_lo, "gtab range too small at the low end"
        for l in range(ck[2] + 1, l_hi + 1):   # beyond table top: |G| < e^-100, treat as 0
            gtab[l] = [0j] * K
    else:
        from g_kernel import build_table
        print(f"building G table l={l_lo}..{l_hi} (SLOW; prefer shipping a pickle)", flush=True)
        gtab = build_table(h, l_lo, l_hi, eta, K, dps=45)
        with open(f'{args.out}/gtab.pkl', 'wb') as f:
            pickle.dump((key, gtab), f)

    # ---- stage 3: assemble + FFT ----
    Fhat = assemble_fhat(S, m_min, m_max, gtab, l_lo, l_hi, h, K, q)
    np.save(f'{args.out}/fhat.npy', Fhat)
    full = np.zeros(q, dtype=np.complex128)
    full[:len(Fhat)] = Fhat
    for l in range(1, len(Fhat)):
        if q - l >= len(Fhat):
            full[q - l] = np.conj(Fhat[l])
    F = ((2 * np.pi / B) * q * np.fft.ifft(full)).real
    np.save(f'{args.out}/F_grid.npy', F)
    logN = float(10 * mp.log(21))
    gam_half = 16 * np.exp(logN / 4 - 2 * np.log(2 * np.pi)) * np.pi ** 2
    L_half = F[0] / gam_half
    print(f"L(1/2) = {L_half:.13f}  (ref 5.8379065826347, diff {L_half-5.8379065826347:.2e})",
          flush=True)

    # ---- stage 4: zero scan ----
    from zero_scan import scan_zeros
    zeros, ts, vals, nrej = scan_zeros(Fhat, B, 0.05, args.t2, step=5e-4)
    np.savetxt(f'{args.out}/zeros_found.csv', zeros, header='gamma', comments='')
    print(f"{len(zeros)} sign changes accepted on [0.05, {args.t2}], "
          f"{nrej} rejected by the amplitude guard", flush=True)
    if nrej:
        print("WARNING: rejected crossings => noise floor reached inside the scan window;"
              " the tail list may be incomplete near t2 (verdict stays conservative).",
              flush=True)

    # ---- stage 5: Turing condition ----
    from turing_count import turing_condition, TSTAR, Phi
    tail = [z for z in zeros if z > TSTAR]
    below = [z for z in zeros if z <= TSTAR]
    r = turing_condition(tail, args.t2)
    r['n_below_Tstar_found'] = len(below)
    r['Phi_t2'] = Phi(args.t2)
    r['S_t2_implied'] = len(zeros) - r['Phi_t2']
    r['n_rejected_by_guard'] = int(nrej)
    print(json.dumps(r, indent=2, default=float))
    with open(f'{args.out}/turing_verdict.json', 'w') as f:
        json.dump(r, f, indent=2, default=float)
    if r['certified'] and len(below) == 21:
        print("\n*** N(T*) = 21 CERTIFIED (numerical grade; PZ Thm 2.3, one-sided) ***")
    else:
        print("\nnot certified at this t2 / parameter set (see verdict json)")


if __name__ == '__main__':
    main()
