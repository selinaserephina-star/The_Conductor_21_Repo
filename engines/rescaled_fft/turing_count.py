# turing_count.py — the Turing completeness count for chi8.
#
# N(t) = Phi(t) + S(t),  Phi(t) = theta(t)/pi  (A = 0: Lambda entire, L(1/2) > 0).
# PZ Thm 2.3 gives  pi * int_{t1}^{t2} S(t) dt <= B  (B = 57.89 at t1 = T*, eps = 1/2).
#
# One-sided contradiction argument (rule out N(T*) >= 22):
#   if a zero <= T* were missed, then N(t) >= N_found(t) + 1 for all t >= T*, so
#     int_{T*}^{t2} N dt >= int N_found dt + (t2 - T*)
#   but int_{T*}^{t2} N dt = int Phi dt + int S dt <= int Phi dt + B/pi.
#   Contradiction iff   (t2 - T*) - D > B/pi,   D := int_{T*}^{t2} (Phi - N_found) dt.
#   (D = -int S_found dt; oscillates O(1) if the tail list is complete.)
#
# Usage: feed the certified sign-change list (gamma > T*) and it evaluates the
# exact condition; also reports, per t2, the largest B that would certify.
import numpy as np
import mpmath as mp

mp.mp.dps = 30
LOGQ = 10 * mp.log(21)
MU_OURS = [0, 0, 0, 0, 1, 1, 1, 1]
TSTAR = 5.969
B_PZ = 57.89          # from turing_constant_chi8.py (PZ Thm 2.3, eps=1/2, t1=T*)


def theta(t):
    s = mp.mpf('0.5') + 1j * mp.mpf(t)
    v = (s / 2) * LOGQ
    for m in MU_OURS:
        v += -((s + m) / 2) * mp.log(mp.pi) + mp.loggamma((s + m) / 2)
    return mp.im(v)


def Phi(t):
    return float(theta(t) / mp.pi)


def int_phi(t1, t2):
    return float(mp.quad(lambda t: theta(t) / mp.pi, [t1, t2]))


def turing_condition(zeros_tail, t2, t1=TSTAR, B=B_PZ, n_below=21):
    """zeros_tail: certified zeros in (t1, t2]. Returns dict with the condition."""
    zs = np.sort(np.asarray([z for z in zeros_tail if t1 < z <= t2], dtype=np.float64))
    # int_{t1}^{t2} N_found(t) dt with N_found(t) = n_below + #{z <= t}
    int_Nf = n_below * (t2 - t1) + float(np.sum(t2 - zs))
    D = int_phi(t1, t2) - int_Nf
    lhs = (t2 - t1) - D
    need = B / np.pi
    return dict(t2=t2, n_tail=len(zs), D=D, lhs=lhs, need=need,
                certified=bool(lhs > need),
                B_max_certifiable=float(np.pi * lhs))


if __name__ == '__main__':
    # current local reach: the 21 certified + newly found tail zeros (float64 grade)
    tail = [6.106223, 6.299157, 6.464523]
    print(f"Phi(T*) = {Phi(TSTAR):.4f}  (smooth count at T* = {TSTAR})")
    print(f"B (PZ Thm 2.3) = {B_PZ}, need lhs > B/pi = {B_PZ/np.pi:.2f}\n")
    for t2 in (6.6, 8.0, 10.0):
        r = turing_condition([z for z in tail if z <= t2], t2)
        print(f"t2 = {r['t2']:5.1f}: tail zeros = {r['n_tail']}, D = {r['D']:+.3f}, "
              f"lhs = {r['lhs']:6.2f} vs need {r['need']:.2f} -> "
              f"{'CERTIFIED' if r['certified'] else 'not yet'} "
              f"(B_max certifiable at this t2: {r['B_max_certifiable']:.1f})")
    print("\nproduction target: solve (t2 - T*) - D = B/pi + margin;"
          " with D ~ O(1), t2 ~ T* + 18.4 + D ~ 24-26.")
