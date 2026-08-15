"""K-modulus void-pressure Hankel + Li ledger, extended to m=17 with gamma_13..17.
Atoms 1/gamma^2; S_k = sum 1/gamma^{2k}; H_m = det[S_{i+j-1}]_{i,j=1..m} (Stieltjes).
Corner-scanned over the bracketed zeros (gamma_9..17); density-tail-corrected column;
Li zero-side partials. dps 60. Cross-checks vs the m=11/m=12 notes first.
"""
from mpmath import mp, mpf, matrix, det, log, pi, mpc, re as mre
import itertools
mp.dps = 60

EXACT = [mpf(s) for s in ["0.98638643304834","1.2687768809146","1.5621795538760",
    "1.9606305172848","2.1156185905806","2.3684771504231","2.6068937362515","2.9139773460924"]]
# bracketed certified zeros (lo,hi): gamma_9..17
BR = [("3.126","3.128"),("3.338","3.342"),("3.700","3.702"),("3.952","3.954"),
      ("4.081","4.087"),("4.357","4.379"),("4.548","4.550"),("4.692","4.696"),("4.886","4.890")]
BR = [(mpf(a),mpf(b)) for a,b in BR]

N = mpf(21)**10

def powersums(zeros, kmax):
    return [sum(mpf(1)/z**(2*k) for z in zeros) for k in range(1, kmax+1)]  # S[k-1]=S_k

def Hm(S, m):  # det[S_{i+j-1}], needs S_1..S_{2m-1}
    M = matrix(m)
    for i in range(m):
        for j in range(m):
            M[i,j] = S[i+j]   # 0-indexed: entry (i,j)=S_{i+j+1}=S[i+j]
    return det(M)

def tail_moment(k, T0):
    # int_{T0}^inf rho(t)/t^{2k} dt, rho(t)=(1/2pi)(log N + 8 log(t/2pi)); exact antiderivative
    a = log(N); c = mpf(8); p = 2*k
    # int t^{-p} dt = t^{1-p}/(1-p); int t^{-p} log(t/2pi) dt (by parts)
    # I1 = int_{T0}^inf t^{-p} dt = T0^{1-p}/(p-1)
    I1 = T0**(1-p)/(p-1)
    # I2 = int_{T0}^inf t^{-p} log(t/2pi) dt = [T0^{1-p}/(p-1)]*(log(T0/2pi) + 1/(p-1))
    I2 = (T0**(1-p)/(p-1))*(log(T0/(2*pi)) + mpf(1)/(p-1))
    return (a*I1 + c*I2)/(2*pi)

def hankel_scan(nz, mmax, with_tail=False):
    """nz = number of zeros to use (8..17). Corner-scan the bracketed ones among them."""
    nbr = max(0, nz-8)
    fixed = EXACT[:min(nz,8)]
    brs = BR[:nbr]
    kmax = 2*mmax-1
    mins = [None]*(mmax+1); maxs = [None]*(mmax+1)
    for corner in itertools.product(*[(lo,hi) for (lo,hi) in brs]) if brs else [()]:
        zeros = fixed + list(corner)
        S = powersums(zeros, kmax)
        if with_tail:
            T0 = max(zeros)  # tail past the top zero
            S = [S[k-1] + tail_moment(k, T0) for k in range(1, kmax+1)]
        for m in range(1, mmax+1):
            h = Hm(S, m)
            if mins[m] is None or h < mins[m]: mins[m] = h
            if maxs[m] is None or h > maxs[m]: maxs[m] = h
    return mins, maxs

def li_partial(nz, nmax):
    zeros = EXACT[:min(nz,8)] + [ (lo+hi)/2 for (lo,hi) in BR[:max(0,nz-8)] ]  # midpoints
    out = []
    for n in range(1, nmax+1):
        s = mpf(0)
        for g in zeros:
            rho = mpc(mpf(1)/2, g)
            s += 2 - 2*mre((1 - 1/rho)**n)
        out.append(s)
    return out

def fx(x):
    return mp.nstr(x, 4)

# ---- cross-checks vs the notes ----
print("=== CROSS-CHECK vs m=11 note (11 zeros): H_7~6.9e-27, collapse -0.77@m11 ===")
mn,mx = hankel_scan(11, 11)
print(f"  H_7 (11z): {fx(mn[7])} .. {fx(mx[7])}   (note: 6.884..6.935e-27)")
print(f"  H_11(11z): {fx(mn[11])} .. {fx(mx[11])}   (note: 5.574..5.955e-94)")
import mpmath
print(f"  collapse log10(H_11)/121 = {fx(mpmath.log10(mx[11])/121)}   (note ~ -0.77)")
li11 = li_partial(11, 10)
print(f"  Li(11z): l1={fx(li11[0])} l5={fx(li11[4])} l9={fx(li11[8])}   (note l1=2.869,l5=29.92,l9=32.04)")

print("\n=== m=17 LEDGER (17 certified zeros, corner-scanned over gamma_9..17) ===")
mn,mx = hankel_scan(17, 17)
mnT,mxT = hankel_scan(17, 17, with_tail=True)
print(f"{'m':>3} {'H_m min..max (17 atoms)':>34} {'spread%':>8} {'with density tail':>20} {'log10(Hm)/m^2':>14}")
for m in range(1, 18):
    spread = float((mx[m]-mn[m])/mx[m]*100) if mx[m] != 0 else 0
    cp = mpmath.log10(mx[m])/m**2
    print(f"{m:>3} {fx(mn[m])+' .. '+fx(mx[m]):>34} {spread:8.2f} {fx(mxT[m]):>20} {fx(cp):>14}")

print("\n=== Li zero-side partials (17 zeros) ===")
li17 = li_partial(17, 18)
li12 = li_partial(12, 18)
print(f"{'n':>3} {'12 zeros':>12} {'17 zeros':>12}")
for n in [1,5,9,12,14,16,18]:
    print(f"{n:>3} {fx(li12[n-1]):>12} {fx(li17[n-1]):>12}")
# turnover: largest n where partial still rising (17 zeros)
turn = 1
for n in range(2, 19):
    if li17[n-1] > li17[n-2]: turn = n
print(f"turnover (17 zeros): partials rise through n ~ {turn}")
