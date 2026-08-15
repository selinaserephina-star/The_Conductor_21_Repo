# olver_ivl_38_c2fix_sup_demod.py -- component F: the corrected closed form c2_fix
#   c2_fix = (2^{1/3}/30pi)[ -9w^2 + pi( 30(pi-1)w Ai^2 - (10pi^2-90pi+177)w AiBi - (150-30pi)Ai'^2
#                                        - 18w^2 Ai'Bi - (10pi^2-90pi+75)Ai'Bi' ) ]
# (olver_ivl_36; = _22's c2^EM with the two pi^2-coefficients HALVED, from EM -h/12 not -h/6).
# (a) fine-grid sup |c2_fix|/(1+|w|)^{7/4} on osc [-16,-0.5] and window [0,8];
# (b) local demod of c2_fix -> D(w), M(w); compare to _35's recorded targets
#     (M ~ 0.42(-w)^{1.32}, D: -0.5 -> -2 over w=-0.8..-8, ratio 0.35-0.43, sup 0.43 @ w=-0.55);
# (c) deep asymptotic envelope closed form check:
#     c2_fix ~ (2^{1/3}/30pi)[ -60 s^{1/2} - (9s^2+(90-30pi)... ) ], s=-w:
#       D_inf = -(2*2^{1/3}/pi) s^{1/2}
#       sin-comp a_s = (2^{1/3}/30pi)(-9s^2 + (90-30pi)s^{1/2}),  cos-comp a_c = (2^{1/3}/30pi)(10pi^2-90pi+126)s^{1/2}
#     E_inf = |D_inf| + sqrt(a_s^2+a_c^2);
# (d) arb interval CERTIFICATION of the sup bounds on compact ranges (ball eval per cell):
#     osc [-16,-0.5]: ratio <= 0.45 ;  window [0,8]: ratio <= 0.70   (both certified rigorously).
# Not RH/GRH.
import numpy as np
from scipy.special import airy as sairy
import math

R13 = 2 ** (1 / 3); PI = math.pi
CF = {  # c2_fix inner coefficients (exact rationals in pi assembled numerically here)
    'wA2': 30 * (PI - 1),
    'wAB': -(10 * PI**2 - 90 * PI + 177),
    'Ap2': -(150 - 30 * PI),
    'w2ApB': -18.0,
    'ApBp': -(10 * PI**2 - 90 * PI + 75),
}
def c2fix_np(w):
    ai, aip, bi, bip = sairy(w)
    inner = (CF['wA2'] * w * ai * ai + CF['wAB'] * w * ai * bi + CF['Ap2'] * aip * aip
             + CF['w2ApB'] * w * w * aip * bi + CF['ApBp'] * aip * bip)
    return R13 * (-9 * w * w + PI * inner) / (30 * PI)

print("=== (a) fine-grid sup of |c2_fix|/(1+|w|)^{7/4} ===")
wg = np.linspace(-16, -0.5, 400001)
r = np.abs(c2fix_np(wg)) / (1 + np.abs(wg)) ** 1.75
i = int(np.argmax(r))
print(f"  OSC   [-16,-0.5]: sup = {r[i]:.4f} at w = {wg[i]:+.3f}   (target: <0.7; _29 measured 0.43 @ -0.55)")
wgw = np.linspace(0, 8, 200001)
rw = np.abs(c2fix_np(wgw)) / (1 + wgw) ** 1.75
iw = int(np.argmax(rw))
print(f"  WINDOW  [0,8]  : sup = {rw[iw]:.4f} at w = {wgw[iw]:+.3f}  (old buggy claim 0.207; data c2(0)~-0.6)")
print(f"  c2_fix(0) = {c2fix_np(np.array([0.0]))[0]:+.4f}   (sec 2p peak-model measured ~ -0.6)")

print("\n=== (b) local demod of c2_fix -> D(w), M(w) vs _35 targets ===")
print("    w   |   D(w)    M(w)   | 0.42(-w)^{1.32} | (|D|+M)/(1+|w|)^{7/4}")
for wc in [-0.8, -1.2, -2.0, -3.0, -4.0, -6.0, -8.0]:
    s = -wc
    # local demod on a window of +-0.45 around wc against cos/sin Psi, Psi=(4/3)s^{3/2}, + DC
    ws = np.linspace(wc - 0.45, wc + 0.45, 901)
    y = c2fix_np(ws)
    Psi = (4 / 3) * (-ws) ** 1.5
    Mx = np.column_stack([np.ones_like(ws), np.cos(Psi), np.sin(Psi)])
    cf, *_ = np.linalg.lstsq(Mx, y, rcond=None)
    D, Mamp = cf[0], math.hypot(cf[1], cf[2])
    print(f"  {wc:+5.1f} | {D:+7.3f}  {Mamp:7.3f} |    {0.42 * s**1.32:7.3f}     | {(abs(D) + Mamp) / (1 + s) ** 1.75:.3f}")

print("\n=== (c) deep asymptotic envelope closed form vs exact ===")
def env_inf(s):
    Dm = 2 * R13 / PI * np.sqrt(s)
    a_s = (R13 / (30 * PI)) * (-9 * s * s + (90 - 30 * PI) * np.sqrt(s))
    a_c = (R13 / (30 * PI)) * (10 * PI**2 - 90 * PI + 126) * np.sqrt(s)
    return Dm + np.sqrt(a_s**2 + a_c**2)
print("    s   | E_inf(s)  | max|c2_fix| near -s (grid) ")
for s in [4, 6, 8, 12, 16]:
    ws = np.linspace(-s - 0.6, -s + 0.6, 4001)
    print(f"  {s:5.1f} | {env_inf(s):8.3f}  | {np.max(np.abs(c2fix_np(ws))):8.3f}")

print("\n=== (d) arb interval certification of the sup bounds ===")
from flint import arb, acb, ctx
ctx.prec = 120
def c2fix_arb(x):  # x: arb ball
    ai, aip, bi, bip = [v.real for v in acb(x).airy()]
    pi = arb.pi(); r13 = arb(2) ** (arb(1) / 3)
    inner = (30 * (pi - 1) * x * ai * ai - (10 * pi**2 - 90 * pi + 177) * x * ai * bi
             - (150 - 30 * pi) * aip * aip - 18 * x * x * aip * bi
             - (10 * pi**2 - 90 * pi + 75) * aip * bip)
    return r13 * (-9 * x * x + pi * inner) / (30 * pi)
def certify(lo, hi, bound, ncell):
    lo = arb(lo); hi = arb(hi); wid = (hi - lo) / ncell
    worst = arb(0); wloc = None
    for k in range(ncell):
        a = lo + wid * k
        cell = arb.union(a, a + wid)   # ball containing the cell
        val = abs(c2fix_arb(cell))
        den = (1 + abs(cell)) ** (arb(7) / 4)
        ratio = val / den
        ub = ratio.upper()   # arf upper bound
        if arb(ub) > worst:
            worst = arb(ub); wloc = float(a.mid()) if a.is_finite() else None
    ok = worst < arb(bound)
    print(f"  [{float(lo)}, {float(hi)}], {ncell} cells: certified sup-UB = {float(worst.mid()):.4f}  "
          f"{'<' if ok else '>='} {bound}  -> {'CERTIFIED' if ok else 'FAIL'} (worst cell near w={wloc:+.3f})")
    return ok
certify(-16.0, -0.5, 0.45, 6200)
certify(0.0, 8.0, 0.70, 3200)
print("done.")
