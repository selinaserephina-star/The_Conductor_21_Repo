# build_fhat.py — assemble Fhat(2 pi n / B) for chi8 via Booker (5.18)-(5.19),
# then FFT to F(m/A) = Lambda(1/2 + i m/A) e^{2 pi eta m/A}.
#
#   Fhat(x) = N^{1/4} sum_{n<=M} a_n n^{-1/2} G(x + log(n/sqrt N); eta)
# bucketed:  u_m = m*h (h = 2pi/B, buckets aligned to the x-grid),
#   S_m^(k) = sum_{log(n/sqrtN) in [u_m-h/2, u_m+h/2)} a_n n^{-1/2} (log(n/sqrtN)-u_m)^k
#   Fhat(x_l) = N^{1/4} sum_m sum_k G^(k)((l+m)h)/k! S_m^(k)
#
# F real => Fhat(-x) = conj(Fhat(x)); FFT length q = A*B.
# chi8 entire => no residue/pole corrections anywhere.
import numpy as np
import mpmath as mp
import pickle, time, sys
from g_kernel import g_kernel

mp.mp.dps = 40


def build(M_file, A, B, eta, K=14, dps=42, gtab_cache=None):
    a = np.load(M_file)
    M = len(a) - 1
    q = A * B
    h = 2 * np.pi / B
    N = mp.mpf(21) ** 10
    logsqN = float(mp.log(N) / 2)          # 15.3467...
    N4 = mp.e ** (mp.log(N) / 4)           # N^{1/4}

    # ---- bucket sums S_m^(k) ----
    t0 = time.time()
    n = np.arange(1, M + 1, dtype=np.float64)
    coef = a[1:].astype(np.float64) / np.sqrt(n)
    u = np.log(n) - logsqN
    midx = np.rint(u / h).astype(np.int64)          # bucket index m
    d = u - midx * h                                # in [-h/2, h/2)
    m_min, m_max = int(midx.min()), int(midx.max())
    nbuck = m_max - m_min + 1
    S = np.zeros((K, nbuck))
    dk = np.ones_like(d)
    for k in range(K):
        S[k] = np.bincount(midx - m_min, weights=coef * dk, minlength=nbuck)
        dk *= d
    print(f"S_m^(k): {nbuck} buckets (m {m_min}..{m_max}), K={K}, {time.time()-t0:.1f}s", flush=True)

    # ---- G table on the l-grid ----
    # needed arguments: (l + m) h for x_l = l h >= 0 (l = 0..q/2) and m in [m_min, m_max];
    # G negligible for arg > u_hi; only args >= m_min*h occur.
    u_hi = 6.0
    l_lo, l_hi = m_min, int(np.ceil(u_hi / h))
    key = (round(h, 12), l_lo, l_hi, float(eta), K, dps)
    tab = None
    if gtab_cache:
        try:
            with open(gtab_cache, 'rb') as f:
                ck, tab = pickle.load(f)
            if ck != key:
                tab = None
        except FileNotFoundError:
            pass
    if tab is None:
        t0 = time.time()
        print(f"G table: l = {l_lo}..{l_hi} ({l_hi-l_lo+1} pts) x K={K} ...", flush=True)
        tab = {}
        import math
        for l in range(l_lo, l_hi + 1):
            uu = l * h
            if uu > 0 and 8 * math.pi * math.exp(uu / 4) * math.cos(math.pi * eta / 2) > 180:
                tab[l] = [0j] * K
                continue
            tab[l] = [complex(g_kernel(uu, eta, k, dps=dps)) for k in range(K)]
            if l % 20 == 0:
                print(f"  l={l} u={uu:+.2f} |G|={abs(tab[l][0]):.2e} ({time.time()-t0:.0f}s)", flush=True)
        if gtab_cache:
            with open(gtab_cache, 'wb') as f:
                pickle.dump((key, tab), f)
        print(f"G table done, {time.time()-t0:.0f}s", flush=True)

    # ---- assemble Fhat on x_l = l h, l = 0..q/2 ----
    t0 = time.time()
    Garr = np.zeros((K, l_hi - l_lo + 1), dtype=np.complex128)
    for l in range(l_lo, l_hi + 1):
        for k in range(K):
            Garr[k, l - l_lo] = tab[l][k]
    fact = np.array([float(mp.factorial(k)) for k in range(K)])
    l_max_x = q // 2
    Fhat = np.zeros(l_max_x + 1, dtype=np.complex128)
    # Fhat[l] = sum_k (1/k!) sum_m Garr[k, l+m] S[k, m]
    for l in range(0, l_max_x + 1):
        lo = max(m_min, l_lo - l)
        hi = min(m_max, l_hi - l)
        if lo > hi:
            continue
        acc = 0j
        for k in range(K):
            acc += (Garr[k, l + lo - l_lo: l + hi - l_lo + 1]
                    * S[k, lo - m_min: hi - m_min + 1]).sum() / fact[k]
        Fhat[l] = acc
    Fhat *= complex(N4)
    print(f"Fhat assembled ({l_max_x+1} pts, {time.time()-t0:.1f}s)", flush=True)
    return Fhat, dict(A=A, B=B, q=q, h=h, eta=float(eta), M=M, K=K)


def fft_to_F(Fhat, meta):
    """F(m/A) for m = 0..q-1 via inverse DFT (numpy, complex128)."""
    q, B = meta['q'], meta['B']
    full = np.zeros(q, dtype=np.complex128)
    full[:len(Fhat)] = Fhat
    for l in range(1, len(Fhat)):
        if q - l >= len(Fhat):
            full[q - l] = np.conj(Fhat[l])
    F = (2 * np.pi / B) * q * np.fft.ifft(full)
    return F.real  # F is real; imag part = numerical noise check


if __name__ == '__main__':
    M_file = sys.argv[1] if len(sys.argv) > 1 else 'an_chi8_2M.npy'
    eta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    A, B = 64, 40
    tag = f"eta{eta:.2f}".replace('.', '')
    Fhat, meta = build(M_file, A, B, eta, gtab_cache=f'gtab_B40_{tag}.pkl')
    np.save(f'fhat_{tag}.npy' if eta != 0.3 else 'fhat.npy', Fhat)
    F = fft_to_F(Fhat, meta)
    np.save('F_grid.npy', F)
    # Lambda(1/2) check
    lam_half = F[0]
    N = mp.mpf(21) ** 10
    gam_half = 16 * mp.e**(mp.log(N) / 4 - 2 * mp.log(2 * mp.pi)) * mp.gamma(mp.mpf('0.5'))**4
    L_half = lam_half / float(gam_half)
    print(f"\nLambda(1/2) = {lam_half:.6f}")
    print(f"L(1/2)      = {L_half:.13f}")
    print(f"reference     5.8379065826347   diff = {L_half - 5.8379065826347:.3e}")
