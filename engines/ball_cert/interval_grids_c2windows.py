# interval_grids_c2windows.py -- INFINITE_SIDE item 5, consumer restatement.
# Re-state the C2 rate-constant write-up's window-ingredient table
# (WRITEUP_XI_LIMIT_C2_rate_constants.md sec.1) with rigorous ball enclosures.
# That table defines its ingredients as GRID quantities:
#   "grid over W with 0.12 margin, step 0.02; certified-numeric"
#     delta_hat_X = max_grid delta_X,  Lambda_X = max_grid |delta'_X|,
#     gap_hat_X   = min_grid gap_X.
# We reproduce those grid extrema as enclosures (delta/delta' from one ball inverse
# per grid node; gap via the interval_grids_arb gap method at the extremal node).
# GL-300 (stability already shown on the standard cells).  Not RH/GRH.
import time, math
import numpy as np
from flint import arb, ctx
import importlib.util
spec = importlib.util.spec_from_file_location('ig', 'interval_grids_arb.py')
ig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ig)

ctx.prec = ig.PREC_WORK
ig.PI = arb.pi()
LOG = open('interval_grids_c2windows_run.log', 'w')
def log(s):
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

N = 300
T, WT, mr = ig.certified_gl_nodes(N)
log(f"=== C2 window ingredients, GL-{N}, prec {ig.PREC_WORK} bits, "
    f"@ {time.strftime('%Y-%m-%d %H:%M:%S')} (node ball rad {mr:.1e}) ===")

def grid_row(x1, x2):
    xis = [round(x, 2) for x in np.arange(x1 - 0.12, x2 + 0.12 + 1e-9, 0.02)]
    # delta / delta' at every node (inv is cheap); track enclosed extrema
    best = {}
    for sign, par, dkey in ((+1, 'E', '+'), (-1, 'O', '-')):
        dmax = None; dmax_xi = None; lmax = None; lmax_xi = None
        gmin = None; gmin_xi = None
        for xi in xis:
            L = 2*xi*xi/math.pi
            A, SW, c, cd = ig.build(N, L, sign, T, WT)
            Ainv = A.inv()
            y = [sum((Ainv[i, j]*c[j] for j in range(N)), arb(0)) for i in range(N)]
            cc = sum((c[i]*y[i] for i in range(N)), arb(0))
            cdc = sum((cd[i]*y[i] for i in range(N)), arb(0))
            d = (arb(2)/ig.PI)*cc
            dp = (arb(4)/ig.PI)*cdc + d*d
            du = float(d.mid()) + float(d.rad())
            lu = abs(float(dp.mid())) + float(dp.rad())
            if dmax is None or du > dmax:
                dmax = du; dmax_xi = xi; dmax_enc = d
            if lmax is None or lu > lmax:
                lmax = lu; lmax_xi = xi; lmax_enc = dp
            # gap_hi (Rayleigh, cheap) to locate the min
            Kf = ig.float_kernel(N, L, sign, T, SW)
            evf, Vv = np.linalg.eigh(Kf); gf = 1 - evf[-1]
            if gmin is None or gf < gmin:
                gmin = gf; gmin_xi = xi
        # rigorous gap enclosure at the argmin node
        L = 2*gmin_xi*gmin_xi/math.pi
        o = ig.cell_objects(N, L, sign, T, WT)
        best[dkey] = dict(dhat=dmax_enc, dhat_xi=dmax_xi, Lam=lmax_enc, Lam_xi=lmax_xi,
                          gap_lo=o['gap_lo'], gap_hi=o['gap_hi'], gap_xi=gmin_xi)
    return best

for (x1, x2, ref) in [(1.0, 2.0, dict(dp=1.9674, dm=0.9847, Lp=0.4894, Lm=0.4763, gp=3.06e-2, gm=3.32e-1)),
                      (1.5, 2.5, dict(dp=2.7106, dm=1.7175, Lp=0.4948, Lm=0.4914, gp=2.09e-3, gm=5.23e-2))]:
    t = time.time()
    b = grid_row(x1, x2)
    log(f"\n#### window [{x1}, {x2}]  ({time.time()-t:.0f}s)   (sealed float refs in parens)")
    for k, kn in (('+', 'E/+'), ('-', 'O/-')):
        r = b[k]
        log(f"  {kn}: delta_hat = {float(r['dhat'].mid()):+.6f} (r {float(r['dhat'].rad()):.1e}) "
            f"@xi={r['dhat_xi']}  [ref {ref['dp' if k=='+' else 'dm']}]")
        log(f"       Lambda   = {abs(float(r['Lam'].mid())):.6f} (r {float(r['Lam'].rad()):.1e}) "
            f"@xi={r['Lam_xi']}  [ref {ref['Lp' if k=='+' else 'Lm']}]")
        log(f"       gap_hat  in [{r['gap_lo']:.5e}, {r['gap_hi']:.5e}] @xi={r['gap_xi']}  "
            f"[ref {ref['gp' if k=='+' else 'gm']}]")
log("\ndone.")
LOG.close()
