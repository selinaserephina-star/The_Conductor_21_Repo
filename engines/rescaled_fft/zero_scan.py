# zero_scan.py — from the FFT output F(m/A), reconstruct Z(t) and locate zeros;
# compare against the 21 certified zeros; run the counting bookkeeping.
#
#   F(t) = Lambda(1/2+it) e^{2 pi eta t}   (real, r=8 so pi*r*eta/4 = 2 pi eta)
#   Z(t) = Lambda(1/2+it) / |gamma(1/2+it)|   (Booker Fig 6.3 analogue)
#   |gamma(1/2+it)| = N^{1/4} * 16 (2pi)^{-2} |Gamma(1/2+it)|^4
#                   = N^{1/4} * 16 (2pi)^{-2} (pi / cosh(pi t))^2
#
# Refinement of zero locations: F(t) at arbitrary t from the same Fhat data,
#   F(t) = (2pi/B) * [ Fhat(0) + 2 Re sum_{l>=1} Fhat(l h) e^{i t l h} ]        (h = 2pi/B)
# (the Poisson-aliased form; aliases negligible for t in the usable window).
import numpy as np


def F_of_t(Fhat, B, t):
    """F(t) for scalar/array t directly from the Fhat samples (alias form)."""
    h = 2 * np.pi / B
    l = np.arange(len(Fhat))
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    ph = np.exp(1j * np.outer(t, l * h))
    vals = (2 * np.pi / B) * (ph @ Fhat + np.conj(ph[:, 1:] @ Fhat[1:]))
    # above adds conj-part for l>=1: F = (2pi/B) sum_{l in Z} Fhat(lh) e^{itlh}, Fhat(-x)=conj
    return vals.real - (2 * np.pi / B) * 0  # already both sides; Fhat[0] counted once


def gamma_abs_half(t, logN):
    """|gamma(1/2+it)| for chi8 (no N^{it/2} modulus)."""
    return (np.exp(logN / 4) * 16 / (2 * np.pi) ** 2
            * (np.pi / np.cosh(np.pi * np.asarray(t, dtype=np.float64))) ** 2)


def scan_zeros(Fhat, B, t_lo, t_hi, step=1e-3, refine_tol=1e-9, noise_kappa=30.0):
    """Sign changes of F(t) on [t_lo, t_hi], bisection-refined.

    Amplitude guard: a crossing counts only if BOTH adjacent lobes reach
    noise_kappa * (float64 FFT noise floor). Spurious noise crossings would
    inflate N_found and push the Turing inequality toward false certification,
    so dubious crossings are dropped (undercounting is the safe direction).
    Returns (zeros, ts, vals, n_rejected)."""
    ts = np.arange(t_lo, t_hi + step, step)
    vals = F_of_t(Fhat, B, ts)
    floor = noise_kappa * 1.1e-16 * np.abs(vals).max()
    sg = np.sign(vals)
    idx = np.nonzero(sg[:-1] * sg[1:] < 0)[0]
    zeros, rejected = [], 0
    for j, i in enumerate(idx):
        lo_edge = idx[j - 1] + 1 if j > 0 else 0
        hi_edge = idx[j + 1] if j + 1 < len(idx) else len(vals) - 1
        left_peak = np.abs(vals[lo_edge:i + 1]).max()
        right_peak = np.abs(vals[i + 1:hi_edge + 1]).max()
        if min(left_peak, right_peak) < floor:
            rejected += 1
            continue
        a, b = ts[i], ts[i + 1]
        fa, fb = vals[i], vals[i + 1]
        while b - a > refine_tol:
            m = 0.5 * (a + b)
            fm = F_of_t(Fhat, B, m)[0]
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        zeros.append(0.5 * (a + b))
    return np.array(zeros), ts, vals, rejected


def load_certified(path):
    import csv
    rows = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            g = float(row['gamma_or_midpoint'])
            lo = float(row['bracket_lo']) if row['bracket_lo'] else g - 5e-10
            hi = float(row['bracket_hi']) if row['bracket_hi'] else g + 5e-10
            rows.append((int(row['j']), g, lo, hi))
    return rows


if __name__ == '__main__':
    import mpmath as mp
    mp.mp.dps = 30
    Fhat = np.load('fhat.npy')
    B = 40
    logN = float(10 * mp.log(21))

    # usable ceiling: where does FFT noise floor overtake |gamma|-scale?
    zeros, ts, vals, nrej = scan_zeros(Fhat, B, 0.05, 6.6)
    print(f"found {len(zeros)} sign changes of Z(t) on [0.05, 6.6]:")
    cert = load_certified(r'..\chi8_certified_zeros_MASTER_2026-07-14\chi8_zeros_master.csv')
    used = set()
    for z in zeros:
        match = ''
        for j, g, lo, hi in cert:
            if lo - 2e-3 <= z <= hi + 2e-3:
                match = f"= gamma_{j} [{lo},{hi}] {'OK' if lo <= z <= hi else 'NEAR-EDGE'}"
                used.add(j)
                break
        print(f"  t = {z:.6f}   {match}")
    missing = [j for j, g, lo, hi in cert if j not in used]
    if missing:
        print("certified zeros NOT found:", missing)
    else:
        print("all 21 certified zeros matched.")
