# olver_ivl_36_em12_fix.py -- component F: the EM boundary-coefficient hypothesis.
# CLAIM: olver_ivl_22's EM operator used  Sum g = (1/h)Int + g/2 - (h/6) g'  but the correct
# trapezoidal Euler-Maclaurin boundary term for an atom-anchored half-line sum
# Sum_{p>=0} g(w+p h) is -(h/12) g'(w)  (B2/2! = 1/12).  If so:
#   c2_corr(w) = c2^EM(w) + (2^{1/3} pi/12) g0'(w),   g0 = 2 pi (AiBi)',
# i.e. Delta(w) = c2_true - c2^EM = (2^{1/3} pi^2/6) (AiBi)''(w)  -- NO free parameters,
# testable on the WINDOW (sec 2p Delta-kappa, where Poisson k=+-1 is exp-small -> cannot be the
# cause) and on the OSC side (c2_true(-2) ~ -0.1 vs c2^EM(-2) = -1.53).
# Parts: (A) toy numeric EM test of the coefficient; (B) sympy re-derivation (olver_ivl_22
# pipeline) with BOTH coefficients, symbolic checks; (C) window Delta-kappa + osc value table.
# Not RH/GRH.
import numpy as np, mpmath as mp
mp.mp.dps = 30

print("=" * 78)
print("PART A -- toy numeric test of the half-line EM boundary coefficient")
print("=" * 78)
# S(h) = Sum_{p>=0} f(w0+p h) for a smooth decaying-with-oscillation test function.
# EM:  S = (1/h) Int_w0^inf f + f(w0)/2 + C1 * h f'(w0) + O(h^3).   C1 = -1/12 (claim) vs -1/6 (_22).
def ftest(x):  # smooth, decaying, oscillatory -- generic
    return mp.e**(-x) * mp.cos(3 * x + mp.mpf(1) / 3)
def ftest_p(x):
    return -mp.e**(-x) * mp.cos(3 * x + mp.mpf(1) / 3) - 3 * mp.e**(-x) * mp.sin(3 * x + mp.mpf(1) / 3)
w0 = mp.mpf(1) / 5
I = mp.quad(ftest, [w0, mp.inf])
print("   h      | (S - I/h - f/2)/(h f')  -> should converge to C1")
for k in range(4, 10):
    h = mp.mpf(2) ** (-k)
    S = mp.nsum(lambda p: ftest(w0 + p * h), [0, mp.inf])
    C1 = (S - I / h - ftest(w0) / 2) / (h * ftest_p(w0))
    print(f"  2^-{k}   |  {mp.nstr(C1, 8)}")
print("  [-1/12 = -0.083333; -1/6 = -0.166667]")

print()
print("=" * 78)
print("PART B -- sympy: g0,g1,g2 extraction (olver_ivl_22 pipeline) + corrected c2")
print("=" * 78)
import sympy as sp
A, Ap, B, Bp, w, eps = sp.symbols('A Ap B Bp w eps')
c13 = sp.Integer(2) ** sp.Rational(1, 3); c23 = sp.Integer(2) ** sp.Rational(2, 3); pi = sp.pi

def ddw(e):
    e = sp.expand(e)
    return sp.expand(e.diff(w) + e.diff(A) * Ap + e.diff(Ap) * (w * A) + e.diff(B) * Bp + e.diff(Bp) * (w * B))

Phi0 = A * B
Phi1 = c13 * (2 * A * Ap - A * Bp + 3 * Ap * B) / 2
Phi2 = (c23 / 20) * (30 * Ap**2 - 15 * Ap * Bp - 10 * w * A * A + 9 * w * A * B + 3 * w * w * (Ap * B + A * Bp))
pref = pi / c13
nu = eps ** (-3); z = nu - (1 / c13) * w * eps ** (-1); nu2 = nu + 2
def ser(e, n): return sp.series(sp.expand(e), eps, 0, n).removeO()
epsp = ser(nu2 ** (sp.Rational(-1, 3)), 6)
w_new = ser(c13 * nu2 ** sp.Rational(2, 3) - c13 * nu2 ** sp.Rational(-1, 3) * z, 6)
Dw = ser(w_new - w, 6)
def sigma_of(e_eps, wshift):
    def shift(P):
        P1 = ddw(P); P2 = ddw(P1); P3 = ddw(P2); P4 = ddw(P3)
        return sp.expand(P + P1 * wshift + P2 * wshift**2 / 2 + P3 * wshift**3 / 6 + P4 * wshift**4 / 24)
    body = shift(Phi0) + e_eps * shift(Phi1) + e_eps**2 * shift(Phi2)
    return sp.expand(pref * e_eps ** (-1) * body)
sig0 = sigma_of(eps, 0)
sigS = sigma_of(epsp, Dw)
Dj = sp.expand(sigS - sig0)
Dj = sp.series(Dj, eps, 0, 3).removeO()
g0 = sp.expand(Dj.coeff(eps, 0)); g1 = sp.expand(Dj.coeff(eps, 1)); g2 = sp.expand(Dj.coeff(eps, 2))

def clean(e): return sp.simplify(sp.expand(sp.expand(e).subs(A * Bp, Ap * B + 1 / pi)))
# check: g0 == 2 pi (AiBi)' = 2 pi (Ap B + A Bp)
print(" g0 - 2*pi*(AiBi)'      =", clean(g0 - 2 * pi * (Ap * B + A * Bp)), "   [want 0]")
print(" g0 =", clean(g0))
# the correction term: (2^{1/3} pi/12) g0' = (2^{1/3} pi^2/6)(AiBi)''
corr = sp.expand(c13 * pi / 12 * ddw(g0))
print(" corr = (2^{1/3}pi/12) g0' =", clean(corr))
print(" corr - (2^{1/3}pi^2/6)*(2*w*A*B + 2*Ap*Bp + (1/pi)... ) check vs (AiBi)'':")
AiBi_pp = sp.expand(ddw(ddw(A * B)))   # = 2 w A B + 2 Ap Bp (+ Wronskian pieces via ABp)
print("   (AiBi)'' =", clean(AiBi_pp))
print("   corr - (2^{1/3}pi^2/6)(AiBi)'' =", clean(corr - c13 * pi**2 / 6 * AiBi_pp), "   [want 0]")

# ---- antiderivative machinery (verbatim from _22) ----
MON = ['one', 'A2', 'AAp', 'Ap2', 'AB', 'ApB', 'ApBp']
monexpr = {'one': sp.Integer(1), 'A2': A * A, 'AAp': A * Ap, 'Ap2': Ap * Ap, 'AB': A * B, 'ApB': Ap * B, 'ApBp': Ap * Bp}
def to_dict(e):
    e = sp.expand(e).subs(A * Bp, Ap * B + 1 / pi)
    e = sp.expand(e)
    d = {m: sp.Integer(0) for m in MON}
    for m in ['A2', 'AAp', 'Ap2', 'AB', 'ApB', 'ApBp']:
        d[m] = sp.expand(e.coeff(monexpr[m]))
    rebuilt = sum(d[m] * monexpr[m] for m in MON[1:])
    d['one'] = sp.expand(e - rebuilt)
    return d
Dbasis = {
    'one': {'one': sp.Integer(0)},
    'A2': {'AAp': sp.Integer(2)},
    'AAp': {'Ap2': sp.Integer(1), 'A2': w},
    'Ap2': {'AAp': 2 * w},
    'AB': {'ApB': sp.Integer(2), 'one': 1 / pi},
    'ApB': {'AB': w, 'ApBp': sp.Integer(1)},
    'ApBp': {'ApB': 2 * w, 'one': w / pi},
}
def antideriv(gdict):
    D = 6
    qs = {m: [sp.Symbol(f'q_{m}_{k}') for k in range(D + 1)] for m in MON}
    qpoly = {m: sum(qs[m][k] * w**k for k in range(D + 1)) for m in MON}
    Fp = {m: sp.Integer(0) for m in MON}
    for m in MON:
        Fp[m] += sp.diff(qpoly[m], w)
        for tgt, coef in Dbasis[m].items():
            Fp[tgt] = Fp.get(tgt, 0) + qpoly[m] * coef
    eqs = []
    for m in MON:
        diff = sp.expand(Fp[m] - gdict.get(m, 0))
        p = sp.Poly(diff, w)
        eqs += [c for c in p.all_coeffs()]
    allq = [qs[m][k] for m in MON for k in range(D + 1)]
    sol = sp.solve(eqs, allq, dict=True)
    if not sol: return None
    s = sol[0]
    F = {m: sp.expand(qpoly[m].subs(s)) for m in MON}
    F = {m: sp.expand(v.subs({sym: 0 for sym in v.free_symbols if str(sym).startswith('q_')})) for m, v in F.items()}
    return F
def eval_dict_to_inf(gdict):
    F = antideriv(gdict)
    if F is None: return None
    return sp.expand(-sum(F[m] * monexpr[m] for m in MON))

g0d = to_dict(g0); g1d = to_dict(g1); g2d = to_dict(g2)
K = 1 / (c13 * pi)
I0 = eval_dict_to_inf(g0d); I1 = eval_dict_to_inf(g1d); I2 = eval_dict_to_inf(g2d)
c0 = sp.expand(K * I0)
c1 = sp.expand(K * I1 + g0 / 2)
c2_bug = sp.expand(K * I2 + g1 / 2 - (c13 * pi / 6) * ddw(g0))    # _22's operator (h/6)
c2_fix = sp.expand(K * I2 + g1 / 2 - (c13 * pi / 12) * ddw(g0))   # corrected     (h/12)
print(" c0 (want -2^{2/3} AiBi):", clean(c0))
print(" c2_bug (reproduces _22):")
print("   ", clean(c2_bug))
print(" c2_fix (corrected EM):")
print("   ", clean(c2_fix))
print(" c2_fix - c2_bug - corr =", clean(c2_fix - c2_bug - corr), "   [want 0]")

print()
print("=" * 78)
print("PART C -- numeric: window Delta-kappa (sec 2p) + osc-side values")
print("=" * 78)
def airy_vals(x):
    return (mp.airyai(x), mp.airyai(x, 1), mp.airybi(x), mp.airybi(x, 1))
c2_bug_f = sp.lambdify((w, A, Ap, B, Bp), clean(c2_bug), 'mpmath')
c2_fix_f = sp.lambdify((w, A, Ap, B, Bp), clean(c2_fix), 'mpmath')
corr_f   = sp.lambdify((w, A, Ap, B, Bp), clean(corr),   'mpmath')
MEAS = {5: 0.005, 4: 0.009, 3: 0.023, 2: 0.042, 1: 0.064}   # sec 2p measured Delta-kappa(w)
print(" WINDOW: Delta-kappa_pred = (2^{1/3}pi/12) g0'(w)  vs  measured (sec 2p)")
print("   w  | pred    | measured")
for wv in [5, 4, 3, 2, 1]:
    a, ap, b, bp = airy_vals(wv)
    print(f"   {wv}  | {float(corr_f(wv,a,ap,b,bp)):+.4f} | {MEAS[wv]:+.4f}")
print()
print(" OSC side: c2_bug (=c2^EM), correction, c2_fix   [c2_true(-2) ~ -0.1 from _28/_31]")
print("    w    |  c2^EM    |  corr     |  c2_fix   | c2_fix/(1+|w|)^{7/4}")
for wv in [-0.55, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0, -8.0]:
    a, ap, b, bp = airy_vals(wv)
    cb = float(c2_bug_f(wv, a, ap, b, bp)); cf = float(c2_fix_f(wv, a, ap, b, bp))
    cc = float(corr_f(wv, a, ap, b, bp))
    print(f"  {wv:+6.2f} | {cb:+9.4f} | {cc:+9.4f} | {cf:+9.4f} | {abs(cf)/(1+abs(wv))**1.75:6.4f}")
print("done.")
