# g_kernel_arb.py — ball-certified G-kernel table (rigor-ladder step 1, SCALING_MEMO §5).
#
# G^(k)(u; eta) = (1/2pi i) int (1/2-s)^k e^{w(1/2-s)} 16 (2pi)^{-4s} Gamma(s)^4 ds,
# w = u + 2*pi*i*eta  (chi8: mu = {0,0,0,0,1,1,1,1}; Legendre duplication as in g_kernel.py).
#
# Certified route = the power/log series of Booker 2006 (5.9)-(5.10): shift the contour to
# -infinity and sum residues at the order-4 poles s = -j of Gamma(s)^4.  Near s = -j, with
# eps = s + j and Gamma(s) = Gamma(1+eps) / (eps * prod_{m=1..j}(eps-m)):
#
#   integrand = A_j(eps) / eps^4,
#   A_j(eps)  = 16 (2pi)^{4j} e^{w(1/2+j)} * (j+1/2-eps)^k * e^{-beta*eps}
#               * Gamma(1+eps)^4 * prod_{m=1..j} (eps-m)^{-4},     beta = w + 4 log(2pi),
#   Res_{s=-j} = [eps^3] A_j(eps)     (order-4 pole => cubic Taylor coefficient).
#
# Every residue is computed EXACTLY in acb ball arithmetic with degree-3 truncated series:
#   Gamma(1+eps)   from exp(-euler_gamma*eps + zeta(2)/2 eps^2 - zeta(3)/3 eps^3),
#   prod (eps-m)^{-4} = (j!)^{-4} exp(4 H1 eps + 2 H2 eps^2 + (4/3) H3 eps^3),  Hi = sum m^-i.
#
# Tail bound — Lemma 5.1's geometric-domination logic, adapted to the Gamma(s)^4 form with
# the (1/2-s)^k factor present.  The EXACT ratio identity
#   A_{j+1}(eps) = A_j(eps) * (2pi)^4 e^w * ((j+3/2-eps)/(j+1/2-eps))^k / (eps-(j+1))^4
# gives on the disk |eps| <= 1/2 (where |j+1/2-eps| >= j, |j+3/2-eps| <= j+2,
# |eps-(j+1)| >= j+1/2, j >= 1):
#   max|A_{j+1}| <= q_j max|A_j|,   q_j = (2pi)^4 e^u ((j+2)/j)^k / (j+1/2)^4   (dec. in j).
# Cauchy on |eps| = 1/2:  |[eps^3] A| <= max|A| / (1/2)^3 = 8 max|A|.  Hence for q_J < 1:
#   sum_{j>J} |Res_j| <= 8 MA_J * q_J / (1 - q_J),
# with MA_J the per-factor upper bound of max_{|eps|<=1/2}|A_J|:
#   MA_J = 16 (2pi)^{4J} e^{u(J+1/2)} (J+1)^k e^{|beta|/2} MG4 / prod_{m=1..J}(m-1/2)^4,
# MG4 >= max|Gamma(1+eps)|^4 on the disk, enclosed once by acb box evaluation.
# The tail is added to the result as a centered ball => fully rigorous enclosure.
#
# Grid (production, locked in SCALING_MEMO §5b): u = l*(2pi/64) = l*pi/32 (EXACT in arb),
# l = -157..134, eta = 0.82 = 41/50 (exact), k = 0..13, precision 1500 bits
# (worst cancellation ~10^322 peak vs |G^(k)| ~ 10^-70 at u_max = 13.16; 1500 bits = 451
# digits leaves radii ~1e-100; spec requires < 1e-30).
#
# Output: gtab_arb_B64_eta082.csv (l, k, ball-notation string, mid re/im to 50 digits,
# radius upper bound).  Validation in __main__: midpoints vs the float table
# gtab_B64_eta082.pkl and spot checks vs the independent mpmath saddle-line quadrature.
import csv
import math
import time
from flint import acb, arb, ctx

PREC = 1500
L_MIN, L_MAX = -157, 134
K_DER = 14           # k = 0..13
ETA_NUM, ETA_DEN = 41, 50   # eta = 0.82 exactly
TAIL_TARGET = 1e-72  # absolute tail cutoff per entry (deep-tail |G| ~ 1e-57 at l=120
                     # must stay meaningfully enclosed, not just inside a coarse ball)
J_CAP = 3000         # safety cap (never near it on this grid)

# ---------- 4-term truncated series over acb: a(eps) = a0 + a1 eps + a2 eps^2 + a3 eps^3
def smul(a, b):
    return [a[0] * b[0],
            a[0] * b[1] + a[1] * b[0],
            a[0] * b[2] + a[1] * b[1] + a[2] * b[0],
            a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0]]

def sexp(a):
    """exp of a series with a[0] = 0 (truncated): 1 + P + P^2/2 + P^3/6."""
    assert a[0] == 0
    p2 = smul(a, a)
    p3 = smul(p2, a)
    return [acb(1) + p2[0] / 2 + p3[0] / 6,        # p2[0]=p3[0]=0 but keep the form exact
            a[1] + p2[1] / 2 + p3[1] / 6,
            a[2] + p2[2] / 2 + p3[2] / 6,
            a[3] + p2[3] / 2 + p3[3] / 6]

def coeff3(P, f1):
    """[eps^3] of P(eps)*f1(eps), both 4-term series."""
    return P[0] * f1[3] + P[1] * f1[2] + P[2] * f1[1] + P[3] * f1[0]

def ub(x):
    """float upper bound of a nonnegative arb, padded to cover double rounding."""
    v = float(x.mid() + x.rad())
    return v * (1 + 1e-9) + 1e-300

# ---------- certified G^(k)(u) for u = lu_num*pi/lu_den ----------
def g_all_k(l, verbose=False):
    """Certified balls [G^(0), ..., G^(13)](u; eta) at u = l*pi/32, eta = 41/50.
    Returns list of K_DER acb balls (residue sum + rigorous tail ball)."""
    pi = arb.pi()
    u = pi * l / 32
    eta2pi = pi * 2 * ETA_NUM / ETA_DEN
    w = acb(u, eta2pi)
    two_pi = 2 * pi
    log2pi = acb(two_pi).log()
    beta = w + 4 * log2pi
    e_u = u.exp()                       # |e^w|
    e_absbeta_half = (abs(beta) / 2).exp()

    # Gamma(1+eps)^4 series (constants at working precision)
    egam = arb.const_euler()
    z2 = arb(2).zeta()
    z3 = arb(3).zeta()
    g1 = sexp([acb(0), acb(-egam), acb(z2 / 2), acb(-z3 / 3)])   # Gamma(1+eps)
    G4 = smul(smul(g1, g1), smul(g1, g1))
    # e^{-beta eps} series; constant per u
    f2 = [acb(1), -beta, beta * beta / 2, -beta * beta * beta / 6]
    G4f2 = smul(G4, f2)

    # MG4: enclosure of max |Gamma(1+eps)|^4 over the disk |eps| <= 1/2 (box superset)
    box = acb(arb(1, 0.5), arb(0, 0.5))
    MG4 = abs(box.gamma()) ** 4

    acc = [acb(0) for _ in range(K_DER)]
    pref = 16 * (w / 2).exp()           # 16 (2pi)^{4j} e^{w(1/2+j)} at j = 0
    step = ((two_pi ** 4) * w.exp())    # multiply per j
    H1 = arb(0); H2 = arb(0); H3 = arb(0)
    fact4 = arb(1)                      # (j!)^4
    PPhalf = arb(1)                     # prod_{m=1..j} (m-1/2)^4
    j = 0
    while True:
        # f4 = (j!)^{-4} exp(4 H1 eps + 2 H2 eps^2 + (4/3) H3 eps^3)
        f4 = sexp([acb(0), acb(4 * H1), acb(2 * H2), acb(4 * H3 / 3)])
        inv = 1 / fact4
        P = smul(G4f2, [f4[0] * inv, f4[1] * inv, f4[2] * inv, f4[3] * inv])
        a = arb(2 * j + 1) / 2          # j + 1/2, exact
        f1 = [acb(1), acb(0), acb(0), acb(0)]
        for k in range(K_DER):
            acc[k] += pref * coeff3(P, f1)
            # f1 *= (a - eps)
            f1 = [f1[0] * a, f1[1] * a - f1[0], f1[2] * a - f1[1], f1[3] * a - f1[2]]

        # stopping test (valid for j >= 1; q_j decreasing in j, worst k = K_DER-1)
        if j >= 2:
            qj = (two_pi ** 4) * e_u * (arb(j + 2) / j) ** (K_DER - 1) / (arb(2 * j + 1) / 2) ** 4
            if qj < arb(1) / 2:
                MA = 16 * (two_pi ** (4 * j)) * ((arb(2 * j + 1) / 2) * u).exp() \
                     * arb(j + 1) ** (K_DER - 1) * e_absbeta_half * MG4 / PPhalf
                tail = 8 * MA * qj / (1 - qj)
                T = ub(tail)
                if T < TAIL_TARGET:
                    err = acb(arb(0, T), arb(0, T))
                    for k in range(K_DER):
                        acc[k] += err
                    if verbose:
                        print(f"    l={l}: J={j}, tail<{T:.2e}")
                    return acc
        j += 1
        if j > J_CAP:
            raise RuntimeError(f"J_CAP hit at l={l}")
        pref = pref * step
        H1 += arb(1) / j; H2 += arb(1) / j ** 2; H3 += arb(1) / j ** 3
        fact4 = fact4 * arb(j) ** 4
        PPhalf = PPhalf * (arb(2 * j - 1) / 2) ** 4

def build_table_arb(verbose=True):
    ctx.prec = PREC
    tab = {}
    t0 = time.time()
    for l in range(L_MIN, L_MAX + 1):
        tab[l] = g_all_k(l)
        if verbose and l % 20 == 0:
            r = max(max(float(v.real.rad()), float(v.imag.rad())) for v in tab[l])
            print(f"  l={l:>4}  u={l*math.pi/32:8.3f}  |G|={float(abs(tab[l][0]).mid()):.3e}"
                  f"  max_rad={r:.2e}  ({time.time()-t0:.0f}s)", flush=True)
    return tab

def save_table(tab, path='gtab_arb_B64_eta082.csv'):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['# G^(k)(l*pi/32; eta=41/50) certified balls, chi8 Gamma_C^4 kernel',
                    f'prec={PREC}', f'l={L_MIN}..{L_MAX}', f'K={K_DER}',
                    'residue series Booker (5.9)/(5.10) + Lemma 5.1-style tail'])
        w.writerow(['l', 'k', 'mid_re_50dig', 'mid_im_50dig', 'rad_ub'])
        for l in sorted(tab):
            for k in range(K_DER):
                v = tab[l][k]
                rad = max(float(v.real.rad()), float(v.imag.rad())) * (1 + 1e-9) + 1e-300
                w.writerow([l, k, v.real.mid().str(50, radius=False),
                            v.imag.mid().str(50, radius=False), f"{rad:.3e}"])

if __name__ == '__main__':
    import pickle
    ctx.prec = PREC
    print(f"== building certified G-table: l={L_MIN}..{L_MAX}, k=0..{K_DER-1}, "
          f"eta=41/50, prec={PREC} bits ==")
    tab = build_table_arb()
    save_table(tab)
    print("   saved gtab_arb_B64_eta082.csv")

    # ---- radii certificate ----
    worst = 0.0; worst_at = None
    for l in tab:
        for k in range(K_DER):
            v = tab[l][k]
            r = max(float(v.real.rad()), float(v.imag.rad()))
            if r > worst:
                worst, worst_at = r, (l, k)
    print(f"\n== radii: max = {worst:.3e} at (l,k)={worst_at}   "
          f"{'< 1e-30 OK' if worst < 1e-30 else '**FAIL**'} ==")

    # ---- midpoints vs float table ----
    # FINDING (2026-07-17, adjudicated by TWO independent methods — high-dps saddle-line
    # quadrature at dps 85-90 AND the residue-by-circle-contour series — which both agree
    # with the certified balls at every point tested): the float table (mpmath mp.quad at
    # dps 45) is RELIABLE for k <= 9 (row-relative < 1e-6 everywhere) but FAILS for
    # k = 10..13 at scattered |u| large (49/64/77/88 entries per k above 1e-6 row-relative,
    # worst O(1) row-relative at e.g. (-116,13), (127,13); none in l in [-60, 0]).
    # mp.quad under-resolves the |t|^k-weighted oscillatory integrand there.  Impact on the
    # float64 pipeline: k>=10 terms carry weight eps^k/k! <= 2e-20 in the Taylor assembly —
    # invisible at its validated 5e-14 level.  The certified table supersedes these entries.
    # Gate vs the float table is therefore row-relative over k <= 9 only; k >= 10 reported.
    meta, ftab = pickle.load(open('gtab_B64_eta082.pkl', 'rb'))
    print(f"\n== validation vs float table {meta} ==")
    wlow = 0.0; wlow_at = None; whi = 0.0; whi_at = None
    nhi = 0; ncmp = 0
    zmax = 0.0; zmax_at = None
    for l in sorted(ftab):
        rowscale = max(abs(x) for x in ftab[l])
        for k in range(K_DER):
            fv = ftab[l][k]
            av = complex(float(tab[l][k].real.mid()), float(tab[l][k].imag.mid()))
            if rowscale == 0:                # float-zeroed rows l >= 128 (k=0 skip rule)
                m = abs(av) / math.factorial(k)   # weight as used in the Taylor assembly
                if m > zmax: zmax, zmax_at = m, (l, k)
                continue
            ncmp += 1
            d = abs(av - fv) / rowscale
            if k <= 9:
                if d > wlow: wlow, wlow_at = d, (l, k)
            else:
                if d > 1e-6: nhi += 1
                if d > whi: whi, whi_at = d, (l, k)
    print(f"   {ncmp} entries; k<=9: max row-relative diff = {wlow:.3e} at (l,k)={wlow_at}"
          f"   {'OK (<1e-5)' if wlow < 1e-5 else '**CHECK**'}")
    print(f"   k>=10: {nhi} entries above 1e-6 row-relative, worst {whi:.3e} at {whi_at}"
          f"   (known float-table failure zone; certified table supersedes — see comment)")
    print(f"   float-zeroed rows: max |G^(k)|/k! = {zmax:.3e} at {zmax_at}"
          f"   {'OK (<1e-45, negligible in assembly)' if zmax < 1e-45 else '**CHECK**'}")

    # ---- independent spot check vs mpmath saddle-line quadrature (dps 85, converged) ----
    print("\n== spot check vs g_kernel.py quadrature (dps 85) ==")
    import mpmath as mp
    from g_kernel import g_kernel
    for l, k in ((-157, 0), (-145, 9), (-116, 13), (-64, 5), (0, 0), (0, 13),
                 (64, 3), (92, 0), (120, 7), (127, 13)):
        with mp.workdps(95):
            u_mp = mp.pi * l / 32
            q = g_kernel(u_mp, mp.mpf(ETA_NUM) / ETA_DEN, k, dps=85)
        av = tab[l][k]
        qa = acb(arb(str(mp.re(q))), arb(str(mp.im(q))))
        rel = float(abs(qa - av).mid()) / max(float(abs(av).mid()), 1e-300)
        print(f"   l={l:>4} k={k:>2}: |quad - ball|/|ball| = {rel:.3e}")
