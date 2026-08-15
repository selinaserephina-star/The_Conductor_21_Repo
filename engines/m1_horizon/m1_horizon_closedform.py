# m1_horizon_closedform.py — M1-HORIZON, 2026-07-22. The caustic-limit profile IN CLOSED
# FORM: S(xi) = 1/2 - Si(2 xi^2/pi), where Si = sine integral. Derivation: Poisson rep
# j_{2i+1}(z)=(-1)^i int_0^1 sin(zt)P_{2i+1}(t)dt + the alternating-sin closed form
# sum(-1)^{n+1}sin(z_n t)=(-1)^{N+1}sin(N pi t)/(2cos(pi t/2)); the 1/cos(pi t/2) singularity
# at t=1 (Mehler-Heine P_nu(1-u)~J_0((2i+1)sqrt(2u))) gives, with i=xi sqrt(N),
#   S(xi) = (1/pi) int_0^inf [sin(pi v)/v] J_0(2 xi sqrt(2v)) dv = 1/2 - Si(2 xi^2/pi)/pi.
# Then G(xi)=8 xi S(xi)^2, s^2_inf(xi)=int_xi^inf G, int_0^inf G = 1 (normalization check),
# and the horizon xi_eps solves s^2_inf(xi_eps)=eps. Compare ALL to the measured scaling
# profile. Not RH/GRH.
import numpy as np
from scipy.special import sici
from scipy.integrate import quad
from scipy.optimize import brentq

def S(xi):
    si, _ = sici(2 * xi ** 2 / np.pi)
    return 0.5 - si / np.pi

def G(xi):
    return 8 * xi * S(xi) ** 2

print("=== S(xi) closed form vs measured caustic-limit profile ===")
meas = {0.25: 0.4864, 0.5: 0.4475, 0.75: 0.3840, 1.0: 0.2983, 1.25: 0.1963, 1.5: 0.0890, 2.0: 0.0712}
print("   xi   S_closed=1/2-Si(2xi^2/pi)/pi   measured(N=6400)")
for xi, m in meas.items():
    print(f"   {xi:4}   {abs(S(xi)):.4f}                        {m:.4f}")

print("\n=== normalization: int_0^inf G(xi) dxi  (must be 1) ===")
mass, err = quad(G, 0, 50, limit=400)
print(f"   int_0^inf 8 xi S(xi)^2 dxi = {mass:.6f}  (target 1; err {err:.1e})")

print("\n=== s^2_inf(xi) = int_xi^inf G  vs measured s^2(xi) ===")
meas_s2 = {1.0: 3.28e-1, 2.0: 8.17e-3, 3.0: 1.41e-5, 3.2: 2.90e-6}
for xi, m in meas_s2.items():
    val, _ = quad(G, xi, 50, limit=400)
    print(f"   xi={xi}:  s^2_inf(closed)={val:.3e}   measured(N=6400)={m:.2e}")

print("\n=== horizon xi_eps: solve s^2_inf(xi)=eps  vs measured ===")
meas_xi = {1e-3: 2.39, 1e-4: 2.74, 1e-6: 3.34, 1e-9: 4.06}
def s2inf(xi):
    v, _ = quad(G, xi, 60, limit=400); return v
for eps, m in meas_xi.items():
    xe = brentq(lambda xi: s2inf(xi) - eps, 0.5, 30)
    print(f"   eps={eps:.0e}: xi_eps(closed)={xe:.3f}   measured={m:.2f}")
print("\ndone. (S(xi)=1/2-Si(2xi^2/pi)/pi is the closed-form caustic profile; the horizon,")
print(" mass, and s^2 tail all follow — no Bessel sums / Lanczos needed.)")
