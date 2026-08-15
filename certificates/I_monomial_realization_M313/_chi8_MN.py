import numpy as np, mpmath as mp
from sympy.polys.galoistools import gf_pow_mod, gf_gcd, gf_sub
from sympy.polys.domains import ZZ
from sympy import primerange
mp.mp.dps=25
Ncond=3**10*7**10
fc=[1,0,0,0,0,0,-7,3]

def frob_class(p):
    if p==3: return '3ram'
    if p==7: return '7ram'
    fp=[c%p for c in fc]
    xp=gf_pow_mod([1,0],p,fp,p,ZZ)
    g=gf_gcd(gf_sub(xp,[1,0],p,ZZ),fp,p,ZZ); r=len(g)-1
    if r==7: return '1A'
    if r==0: return '7'
    if r==3: return '2A'
    if r==1:
        xp2=gf_pow_mod(xp,p,fp,p,ZZ); g2=gf_gcd(gf_sub(xp2,[1,0],p,ZZ),fp,p,ZZ)
        return '3A' if (len(g2)-1)==1 else '4A'
    return '2A'

def series_inv(poly,K):           # power series of 1/poly up to X^K ; poly=list low->high
    a=np.zeros(K+1); a[0]=1.0/poly[0]
    for k in range(1,K+1):
        s=0.0
        for j in range(1,min(k,len(poly)-1)+1): s+=poly[j]*a[k-j]
        a[k]=-s/poly[0]
    return a
# local inverse Euler factors per class (coeffs of det(1-rho X), low->high in X)
def local(cls,K):
    if cls=='1A': inv=np.poly1d([1]); base=[1,-1]; p=base; 
    # build via convolution of factors:
    if cls=='1A': fac=[1,-1]; full=[1.0]; 
    # easier: define inverse polynomial then series_inv
    invpoly={
     '1A':np.polynomial.polynomial.polypow([1,-1],8),         # (1-X)^8
     '2A':np.polynomial.polynomial.polypow([1,0,-1],4),       # (1-X^2)^4
     '3A':np.polynomial.polynomial.polymul(np.polynomial.polynomial.polypow([1,-1],2),
                                            np.polynomial.polynomial.polypow([1,1,1],3)),
     '4A':np.polynomial.polynomial.polypow([1,0,0,0,-1],2),   # (1-X^4)^2
     '7' :np.polynomial.polynomial.polymul([1,-1],[1,0,0,0,0,0,0,-1]), # (1-X)(1-X^7)
     '3ram':[1,-1],            # (1-X)
     '7ram':[1],               # 1
    }[cls]
    return series_inv(list(np.atleast_1d(invpoly)),K)

# build a_n for n=1..M
M=400
Kmax=9
primes=list(primerange(2,M+1))
cls={p:frob_class(p) for p in primes}
locseries={p:local(cls[p],Kmax) for p in primes}
a=np.zeros(M+1); a[1]=1.0
# multiplicative fill via smallest prime factor
spf=np.zeros(M+1,dtype=int)
for p in primes: 
    for m in range(p,M+1,p):
        if spf[m]==0: spf[m]=p
for n in range(2,M+1):
    p=spf[n]; k=0; m=n
    while m%p==0: m//=p; k+=1
    a[n]=locseries[p][k]*a[m]
print("a_n n=1..12:",[round(a[n],2) for n in range(1,13)],"(a_2=8? totally-split 2 => yes if 1A)")
print("class of 2,5,11,13:",cls[2],cls[5],cls[11],cls[13])

def GR(s): return mp.power(mp.pi,-s/2)*mp.gamma(s/2)
def theta(t):
    s=mp.mpf(1)/2+1j*mp.mpf(t)
    return float((t/2)*mp.log(Ncond)+mp.im(4*mp.log(GR(s))+4*mp.log(GR(s+1))))
logn=np.log(np.arange(1,M+1)); inv_sqrt=1.0/np.sqrt(np.arange(1,M+1)); an=a[1:M+1]
def Z(t,Mt):
    th=theta(t)
    return 2*np.sum(an[:Mt]*inv_sqrt[:Mt]*np.cos(th - t*logn[:Mt]))

known=[0.98638643304834,1.2687768809146,1.5621795538760,1.9606305172848,
       2.1156185905806,2.3684771504231,2.6068937362515,2.9139773460924]
for Mt in [50,150,400]:
    ts=np.arange(0.3,3.6,0.002); zs=np.array([Z(t,Mt) for t in ts])
    sc=[]
    for i in range(len(ts)-1):
        if zs[i]*zs[i+1]<0:
            t0=ts[i]-zs[i]*(ts[i+1]-ts[i])/(zs[i+1]-zs[i]); sc.append(t0)
    matched=[min(known,key=lambda k:abs(k-z)) for z in sc]
    err=[abs(z-m) for z,m in zip(sc,matched)]
    print(f"\nM_terms={Mt}: {len(sc)} skeleton zeros in (0.3,3.6):")
    print("  ",[f"{z:.4f}" for z in sc])
    print(f"   max err vs known (first 8): {max([e for z,e in zip(sc,err) if z<3.0], default=0):.4f}")
