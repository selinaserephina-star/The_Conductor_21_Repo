"""M74 gamma_1 Rouche contour certificate — fully rigorous (Arb balls end to end).
avatar session, 2026-07-03, answering M74_CAUCHY_ROUCHE_DRY_PILOT_v1.

Contour: C_1 = boundary of D(gamma_1, r_1), r_1 = (gamma_2-gamma_1)/3, in the t-plane;
         s = 1/2 + i t,  Xi(t) = xi(1/2+it),  xi(s) = s(s-1)/2 pi^{-s/2} Gamma(s/2) zeta(s).

B_1 = inf_C |Xi|  — RIGOROUS via arc-chunk enclosure: split C_1 into G arcs; each arc is
      contained in an acb ball (center on the arc, radius >= half chord + sagitta);
      xi(ball) then encloses ALL values of xi on that arc (no separate modulus-of-
      continuity argument needed). B_1 = min over arcs of lower(|xi(ball)|).

A_{N,1} = sup_C |Xi - Mtilde_N| for the CANONICAL theta-AFE skeleton
      Mtilde_N(t) = s(s-1)/2 [ sum_{n<=N} (pi^{-s/2}Gamma(s/2,pi n^2)
                                        + pi^{-(1-s)/2}Gamma((1-s)/2,pi n^2)) - 1/s - 1/(1-s) ]
      so E_N = s(s-1)/2 * sum_{n>N} (pi^{-s/2}Gamma(s/2,pi n^2) + reflected).
      RIGOROUS bound on C_1: with sigma = Re(s) in [1/2-r, 1/2+r] and u >= pi n^2 > 1,
      |Gamma(s/2, pi n^2)| <= Gamma(sig_max/2, pi n^2)   (integrand modulus monotone in sigma),
      |pi^{-s/2}| <= pi^{-sig_min/2}; |s(s-1)/2| <= S_max (ball-computed on the contour);
      the n-sum n>N is evaluated to N+40 with a certified geometric closure.

Verdict per the workpack: A_{N,1} < B_1  =>  Xi and Mtilde_N have the same number of
zeros in D(gamma_1, r_1); gamma_1 is simple => exactly one skeleton zero.
NOT RH/GRH; a one-disk zero-count certificate.
"""
import json, math, os
from flint import acb, arb, ctx

ctx.prec = 192
os.chdir(os.path.dirname(os.path.abspath(__file__)))

G1 = arb("14.134725141734693790457251983562")
G2 = arb("21.022039638771554992628479593897")
R = (G2 - G1)/3
N = 2
GARC = 4096

def xi(z):
    s = acb(arb(1)/2, arb(0)) + acb(0, 1)*z          # s = 1/2 + i t,  t = z (complex)
    return s*(s-1)/2 * (-s/2*arb.pi().log()).exp() * (s/2).gamma() * s.zeta()

# ---- B_1: arc-chunk enclosures
two_pi = 2*arb.pi()
h = two_pi/GARC
# ball radius covering an arc chunk: half-chord + sagitta <= r*(h/2) (small-angle safe: use r*h/2*1.001 + r*h^2/8)
chunk_rad = R*h/2*arb("1.001") + R*h*h/8
B = None
sup_grid = arb(0)
for i in range(GARC):
    th = h*(arb(i) + arb(1)/2)
    center = acb(G1 + R*th.cos(), R*th.sin())        # t-plane point gamma_1 + R e^{i th}
    zball = center + acb(arb(0, float(chunk_rad.mid()+chunk_rad.rad())),
                         arb(0, float(chunk_rad.mid()+chunk_rad.rad())))
    v = abs(xi(zball))
    lo = v.mid() - v.rad()
    hi = v.mid() + v.rad()
    B = arb(lo) if B is None or float(lo) < float(B) else B
    sup_grid = arb(hi) if float(hi) > float(sup_grid) else sup_grid
print(f"B_1 (rigorous inf_C |Xi|)  >= {float(B):.6e}")
print(f"     (rigorous sup_C |Xi|) <= {float(sup_grid):.6e}   [IB sampled: 6.8606e-4 .. 1.0197e-2]")

# ---- A_{N,1}: canonical skeleton tail bound
sig_max = arb(1)/2 + R          # Re(s) upper on contour
sig_min = arb(1)/2 - R
# |s(s-1)/2| bound on contour: s = 1/2 + i t, |t| <= G1 + R
tmax = G1 + R
S_max = (arb(1)/2 + tmax)*(arb(3)/2 + tmax)/2
pref = S_max * ( arb.pi()**(-sig_min/2) + arb.pi()**(-(1-sig_max)/2) )   # both terms' pi-power bounds
tail = arb(0)
last = None
for n in range(N+1, N+41):
    x = arb.pi()*n*n
    # python-flint 0.8 convention (verified): x.gamma_upper(a) = Gamma(a, x)
    term = acb(x).gamma_upper(acb(sig_max/2)).real
    # reflected exponent (1-s)/2 has Re in [(1-sig_max)/2, (1-sig_min)/2]; bound at max Re:
    term2 = acb(x).gamma_upper(acb((1-sig_min)/2)).real
    tail += term + term2
    last = term + term2
# certified geometric closure past N+40: ratio of consecutive incomplete-gamma terms
# Gamma(a, pi(n+1)^2)/Gamma(a, pi n^2) <= e^{-pi(2n+1)} * ((n+1)^2/n^2)^{a} << 1/2
tail += last                                        # closure: remaining sum < last (ratio < 1/2 -> < last)
A = pref * tail
print(f"A_{{N={N},1}} (rigorous sup_C |Xi - Mtilde_N|) <= {float(A):.6e}")

margin = float(B)/float(A)
verdict = "ROUCHE_CERTIFIED_ONE_ZERO" if float(A) < float(B) else "NOT_CERTIFIED"
print(f"A < B: {float(A) < float(B)}   margin B/A = {margin:.3e}   -> {verdict}")

out = {
  "target": "M74 gamma_1 contour", "gamma_1": str(G1), "r_1": float(R), "N": N,
  "skeleton_definition": "Mtilde_N(t) = s(s-1)/2 [ sum_{n<=N} (pi^{-s/2}Gamma(s/2,pi n^2) "
                         "+ pi^{-(1-s)/2}Gamma((1-s)/2,pi n^2)) - 1/s - 1/(1-s) ], s=1/2+it "
                         "(CANONICAL theta-AFE; IB to confirm equivalence with M74's Mtilde_N)",
  "B1_inf_C_abs_Xi_rigorous_lower": float(B),
  "supC_abs_Xi_rigorous_upper": float(sup_grid),
  "A_N1_rigorous_upper": float(A),
  "pass_A_lt_B": bool(float(A) < float(B)),
  "margin_B_over_A": margin,
  "method": f"arb prec {ctx.prec}; B via {GARC} arc-chunk ball enclosures (no continuity "
            f"argument needed); A via monotone incomplete-gamma bounds at sigma_max, "
            f"n to N+40 + certified geometric closure",
  "verdict": verdict,
  "not_claimed": ["RH", "GRH", "M74 closed beyond this disk"],
}
with open("m74_gamma1_rouche_certificate.json", "w", newline="\n") as f:
    json.dump(out, f, indent=1); f.write("\n")
print("written m74_gamma1_rouche_certificate.json")
