# olver_ivl_37_osc_ladder_verify.py -- component F DECISIVE TEST: c2_true (fixed-anchor arb Olver
# ladder + Richardson, the _31 instrument verbatim) vs the CORRECTED EM closed form
#   c2_fix(w) = c2^EM(w) + (2^{1/3} pi/12) g0'(w),  g0 = 2 pi (AiBi)'   [olver_ivl_36: EM boundary
# coefficient is -h/12 (B2/2!), not _22's -h/6].  Prediction with NO free parameters:
#   Delta(w) = c2_true - c2^EM = (2^{1/3} pi^2/6) (AiBi)''(w).
# If c2_true - c2_fix ~ 0 across the anchors, component F's osc closed form IS c2_fix.  Not RH/GRH.
from flint import acb, arb, ctx
import mpmath as mp, numpy as np
ctx.prec = 200; mp.mp.dps = 40
PI = arb.pi()
def zeta_of(zc):
    z = arb(zc)
    if z < 1:
        s = (1 - z * z).sqrt(); return (arb(3) / 2 * (((1 + s) / z).log() - s)) ** (arb(2) / 3)
    s = (z * z - 1).sqrt(); return -((arb(3) / 2 * (s - (1 / z).acos())) ** (arb(2) / 3))
def B0_of(zt, zc):
    z = acb(zc); ztc = acb(zt); p = (1 - z * z) ** (acb(-1) / 2)
    return (-(ztc ** (acb(-1) / 2)) * ((3 * p - 5 * p ** 3) / 24 + acb(5) / 48 * ztc ** (acb(-3) / 2))).real
def factor(mu, z, kind):
    zc = z / mu; zt = zeta_of(zc); P = (4 * zt / (1 - zc * zc)) ** (arb(1) / 4); b0 = B0_of(zt, zc)
    x = arb(mu) ** (arb(2) / 3) * zt; ai, aip, bi, bip = [v.real for v in acb(x).airy()]
    if kind == 'J': return P * (ai * arb(mu) ** (-arb(1) / 3) + aip * arb(mu) ** (-arb(5) / 3) * b0)
    return -P * (bi * arb(mu) ** (-arb(1) / 3) + bip * arb(mu) ** (-arb(5) / 3) * b0)
def dsig(j, n):
    z = arb(2 * n - 1) * PI / 2
    def s(J):
        Jp = factor(2 * J + arb(3) / 2, z, 'J'); Jm = factor(2 * J - arb(1) / 2, z, 'J')
        Jmm = factor(2 * J - arb(3) / 2, z, 'J'); Ym = factor(2 * J - arb(1) / 2, z, 'Y')
        return (arb(2) / (4 * J + 1)) * (PI * z / 2) * Jp * (2 * (J - 1) * Jm - z * (Jmm + Ym))
    return s(j + 1) - s(j)
C2 = 2 ** (mp.mpf(1) / 3); C4 = 2 ** (mp.mpf(2) / 3); Sinf = mp.mpf(3) / 4 - 2 / mp.pi
def w_of(j, n):
    nu = 2 * j; z = (2 * n - 1) * mp.pi / 2; return C2 * mp.mpf(nu) ** (-mp.mpf(1) / 3) * (nu - z)
def n_at_w(j, wt):
    nu = 2 * j; return int(round(nu * (1 - mp.mpf(wt) / (C2 * mp.mpf(nu) ** (mp.mpf(2) / 3))) / mp.pi + 0.5))
def P_of(j, w):
    nu = 2 * j; A = mp.airyai(w); Ap = mp.airyai(w, 1); B = mp.airybi(w); Bp = mp.airybi(w, 1)
    Phi1 = C2 * (Ap * (A + B) - 1 / (2 * mp.pi)); Psi0 = -C4 * Phi1 + (mp.pi - 2) * (Ap * B + A * Bp)
    return -C4 * mp.mpf(nu) ** (mp.mpf(1) / 3) * A * B + Psi0 + Sinf
def c2EM(w):
    A = mp.airyai(w); Ap = mp.airyai(w, 1); B = mp.airybi(w); Bp = mp.airybi(w, 1); pi = mp.pi
    inner = (-30 * A**2 * w + 30 * pi * A**2 * w - 20 * pi**2 * A * B * w - 177 * A * B * w + 90 * pi * A * B * w
             - 150 * Ap**2 + 30 * pi * Ap**2 - 18 * Ap * B * w**2 - 20 * pi**2 * Ap * Bp - 75 * Ap * Bp + 90 * pi * Ap * Bp)
    return C2 * (-9 * w**2 + pi * inner) / (30 * pi)
def corr(w):   # (2^{1/3} pi/12) g0' = (2^{1/3} pi^2/3)(w AiBi + Ai'Bi')
    A = mp.airyai(w); Ap = mp.airyai(w, 1); B = mp.airybi(w); Bp = mp.airybi(w, 1)
    return C2 * mp.pi**2 / 3 * (w * A * B + Ap * Bp)
def c2fix(w): return c2EM(w) + corr(w)

ANCH = [-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -8.0]
JLAD = [4096, 8192, 16384, 32768, 65536]
raw = {a: [] for a in ANCH}
for j in JLAD:
    targ = {n_at_w(j, a): a for a in ANCH}
    nmax = max(targ); acc = arb(0)
    for n in range(1, nmax + 1):
        acc = acc + dsig(j, n)
        if n in targ:
            a = targ[n]; wv = float(w_of(j, n)); nu = 2 * j
            rho = float(mp.mpf(nu) ** (mp.mpf(1) / 3) * (mp.mpf(float(acc.mid())) - P_of(j, wv)))
            raw[a].append((float((2 * j) ** (-1.0 / 3)), wv, rho))
    print(f"  j={j} done", flush=True)

print("\n  anchor |  c2_true(fit)  resid |  c2^EM    |  c2_fix   | c2t-c2fix | Delta_meas vs Delta_pred")
mx = 0.0
for a in ANCH:
    eps = np.array([r[0] for r in raw[a]]); ws = np.array([r[1] for r in raw[a]]); rh = np.array([r[2] for r in raw[a]])
    w0 = ws.mean()
    M = np.vstack([np.ones_like(eps), eps, eps * eps]).T
    cf, *_ = np.linalg.lstsq(M, rh, rcond=None); resid = np.sqrt(np.mean((M @ cf - rh) ** 2))
    cem = float(c2EM(mp.mpf(w0))); cfx = float(c2fix(mp.mpf(w0))); cpr = float(corr(mp.mpf(w0)))
    d = cf[0] - cfx; mx = max(mx, abs(d))
    print(f"  {a:+5.1f}  |  {cf[0]:+8.4f}   {resid:6.3f} | {cem:+8.4f} | {cfx:+8.4f} | {d:+8.4f}  |  {cf[0]-cem:+8.4f} vs {cpr:+8.4f}")
print(f"\n  max |c2_true - c2_fix| over anchors = {mx:.4f}")
print("done.")
