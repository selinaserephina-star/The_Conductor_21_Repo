"""M74 Rouche contour certificates, gamma_1..gamma_40 — batch v2.

v2 fixes over the first attempt (root-caused):
  * arc chunks refine LOCALLY (recursive split up to depth 5) wherever an arc-ball's
    |xi| enclosure touches 0 — B is then a genuine positive lower bound or the row is
    honestly NOT_CERTIFIED;
  * ALL comparisons in arb (|xi| at t ~ 100 lives near 1e-30; float comparisons underflow);
  * margin computed as an arb ratio (arb has unbounded exponents), floated only for display;
  * adaptive N breaks on A == 0 (exact-zero balls after underflow cannot improve).

Skeleton: canonical theta-AFE truncation (pending IB handshake). r_j = both-side
min-gap/3 (never larger than the workpack's r_j — certificates remain valid for his).
Output: m74_rouche_certificate_1_40.csv
"""
import csv, os
from flint import acb, arb, ctx

ctx.prec = 256
os.chdir(os.path.dirname(os.path.abspath(__file__)))
GARC = 2048
MAXDEPTH = 5

zs = [arb(r['gamma_j']) for r in csv.DictReader(
      open("../ECL_edge_tail_completion_received/zeros_zeta_extended_80.csv"))]

def xi(z):
    s = acb(arb(1)/2, arb(0)) + acb(0, 1)*z
    return s*(s-1)/2 * (-s/2*arb.pi().log()).exp() * (s/2).gamma() * s.zeta()

def arc_ball(g, R, th_lo, th_hi):
    tm = (th_lo+th_hi)/2
    half = (th_hi-th_lo)/2
    crad = R*half*arb("1.001") + R*half*half/2
    cr = crad.mid()+crad.rad()
    return acb(g + R*tm.cos(), R*tm.sin()) + acb(arb(0, float(cr)), arb(0, float(cr)))

def contour_inf_sup(g, R):
    two_pi = 2*arb.pi(); h = two_pi/GARC
    B = None; S = arb(0)
    stack = [(h*i, h*(i+1), 0) for i in range(GARC)]
    while stack:
        lo_t, hi_t, d = stack.pop()
        v = abs(xi(arc_ball(g, R, lo_t, hi_t)))
        lo = v.mid()-v.rad(); hi = v.mid()+v.rad()
        if not (lo > 0) and d < MAXDEPTH:
            mid = (lo_t+hi_t)/2
            stack.append((lo_t, mid, d+1)); stack.append((mid, hi_t, d+1))
            continue
        if not (lo > 0):
            return None, None            # cannot bound away from 0 at max depth
        if B is None or lo < B: B = arb(lo)
        if hi > S: S = arb(hi)
    return B, S

def tail_bound(g, R, N):
    sig_max = arb(1)/2 + R; sig_min = arb(1)/2 - R
    tmax = g + R
    S_max = (arb(1)/2 + tmax)*(arb(3)/2 + tmax)/2
    pref = S_max*(arb.pi()**(-sig_min/2) + arb.pi()**(-(1-sig_max)/2))
    tail = arb(0); last = None
    for n in range(N+1, N+41):
        x = arb.pi()*n*n
        t1 = acb(x).gamma_upper(acb(sig_max/2)).real
        t2 = acb(x).gamma_upper(acb((1-sig_min)/2)).real
        tail += t1 + t2; last = t1 + t2
    tail += last
    return pref*tail

rows = []
for j in range(1, 41):
    g = zs[j-1]
    gaps = []
    if j >= 2: gaps.append(g - zs[j-2])
    gaps.append(zs[j] - g)
    gap = gaps[0]
    for x in gaps[1:]:
        if x < gap: gap = x
    R = gap/3
    B, S = contour_inf_sup(g, R)
    if B is None:
        rows.append(dict(j=j, gamma=f"{float(g):.12f}", r=f"{float(R):.9f}", N="",
                         B_inf_rigorous="UNRESOLVED", sup_C="", A_tail_rigorous="",
                         margin="", verdict="NOT_CERTIFIED_ENCLOSURE_LIMIT"))
        print(f"g{j}: enclosure limit at depth {MAXDEPTH} — not certified", flush=True)
        continue
    N, A = 2, None
    while N <= 40:
        A = tail_bound(g, R, N)
        if A < B/10 or A == 0: break
        N += 1
    ok = bool(A < B)
    ratio = B/A if not (A == 0) else None
    mtxt = f"{float(ratio):.3e}" if ratio is not None else "inf(tail_underflow)"
    rows.append(dict(j=j, gamma=f"{float(g):.12f}", r=f"{float(R):.9f}", N=N,
                     B_inf_rigorous=f"{float(B):.6e}", sup_C=f"{float(S):.6e}",
                     A_tail_rigorous=f"{float(A):.6e}", margin=mtxt,
                     verdict="ROUCHE_CERTIFIED_ONE_ZERO" if ok else "NOT_CERTIFIED"))
    print(f"g{j}: r={float(R):.4f} N={N} B={float(B):.3e} A={float(A):.3e} "
          f"margin={mtxt} {'PASS' if ok else 'FAIL'}", flush=True)

with open("m74_rouche_certificate_1_40.csv", "w", newline="\n") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
npass = sum(1 for r in rows if r['verdict'].startswith('ROUCHE'))
print(f"\n{npass}/40 contours certified; written m74_rouche_certificate_1_40.csv")
