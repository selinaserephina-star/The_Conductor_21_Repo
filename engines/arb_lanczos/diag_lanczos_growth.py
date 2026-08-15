r"""Diagnose arb no-reorth Lanczos radius growth vs step, at several precisions,
on the certified D atoms.  Find where it explodes and whether precision rescues m=300."""
import json, mpmath as mp
from flint import arb, ctx

D = json.load(open(r"quantiles_certified.json"))
ROWS = D["rows"]
mp.mp.dps = 60

def ball(mid_str, rad):
    m = arb(mid_str); r = arb(mp.nstr(mp.mpf(rad), 20))
    return arb.union(m - r, m + r)

def dot(u, v):
    s = arb(0)
    for x, y in zip(u, v): s += x * y
    return s

def run(atoms, log_at):
    m = len(atoms)
    v = [arb(m).rsqrt()] * m
    w = [atoms[i] * v[i] for i in range(m)]
    a0 = dot(v, w); w = [w[i] - a0 * v[i] for i in range(m)]
    vprev = v
    out = {}
    for j in range(1, m):
        be = dot(w, w).sqrt()
        if float(be.rad()) == float('inf') or be.contains(0):
            out[j] = ('BROKE', float(be.mid()) if float(be.rad())!=float('inf') else 'inf')
            break
        vn = [w[i] / be for i in range(m)]
        w = [atoms[i] * vn[i] - be * vprev[i] for i in range(m)]
        al = dot(vn, w); w = [w[i] - al * vn[i] for i in range(m)]
        vprev = vn
        if j in log_at:
            out[j] = (float(be.mid()), float(be.rad()), float(al.rad()))
    return out

for prec, mode in ((1200,'ball'),(1200,'point'),(3000,'point'),(6000,'point')):
    ctx.prec = prec
    if mode == 'ball':
        ATOM_D = [arb(1) / ball(r["gD_mid"], r["gD_rad"])**2 for r in ROWS]
    else:
        ATOM_D = [arb(1) / arb(r["gD_mid"])**2 for r in ROWS]   # point atoms (rounding-only)
    res = run(ATOM_D, log_at=set(range(20, 300, 20)) | {50,100,150,200,250,290,298})
    print(f"\nprec={prec} mode={mode}")
    for j in sorted(res):
        v = res[j]
        if v[0] == 'BROKE':
            print(f"  step {j}: BROKE (beta={v[1]})"); break
        print(f"  step {j:3d}: beta={v[0]:.3e} rad(beta)={v[1]:.1e} rad(alpha)={v[2]:.1e}")
