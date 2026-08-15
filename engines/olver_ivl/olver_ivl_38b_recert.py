# olver_ivl_38b_recert.py -- re-run of _38 part (d) osc certification with an honest bound (0.46):
# _38's assert 0.45 was set below the actual sup 0.4515 (cell-UB 0.4535); the theorem target is 0.7.
# Certify: sup_{w in [-16,-0.5]} |c2_fix(w)|/(1+|w|)^{7/4} <= 0.46 (< 0.7 with 1.5x margin).  Not RH/GRH.
from flint import arb, acb, ctx
ctx.prec = 120
def c2fix_arb(x):
    ai, aip, bi, bip = [v.real for v in acb(x).airy()]
    pi = arb.pi(); r13 = arb(2) ** (arb(1) / 3)
    inner = (30 * (pi - 1) * x * ai * ai - (10 * pi**2 - 90 * pi + 177) * x * ai * bi
             - (150 - 30 * pi) * aip * aip - 18 * x * x * aip * bi
             - (10 * pi**2 - 90 * pi + 75) * aip * bip)
    return r13 * (-9 * x * x + pi * inner) / (30 * pi)
lo = arb(-16); hi = arb("-0.5"); ncell = 6200; wid = (hi - lo) / ncell
worst = arb(0); wloc = None
for k in range(ncell):
    a = lo + wid * k
    cell = arb.union(a, a + wid)
    ratio = abs(c2fix_arb(cell)) / (1 + abs(cell)) ** (arb(7) / 4)
    ub = arb(ratio.upper())
    if ub > worst: worst = ub; wloc = float(a.mid())
ok46 = bool(worst < arb("0.46")); ok70 = bool(worst < arb("0.7"))
print(f"osc [-16,-0.5], {ncell} cells, prec 120: certified sup-UB = {float(worst.mid()):.4f} "
      f"(worst cell near w={wloc:+.3f})")
print(f"  <= 0.46 : {'CERTIFIED' if ok46 else 'FAIL'}")
print(f"  <  0.7  : {'CERTIFIED' if ok70 else 'FAIL'}   (theorem target, margin {0.7/float(worst.mid()):.2f}x)")
print("done.")
