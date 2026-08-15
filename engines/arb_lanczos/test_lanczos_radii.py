r"""Probe: does arb Lanczos (multiplication operator on m atoms, start (1..1)/sqrt m)
keep ball radii tiny?  Compare plain 3-term vs full reorth, at a few precisions, and
against the float Jacobi params.  Decides the precision/reorth for the certificate."""
import numpy as np, mpmath as mp
from flint import arb, ctx

# float reference Jacobi (reorth), same as ladder_extend_300.py
def jacobi_float(atoms):
    m=len(atoms); v=np.ones(m)/np.sqrt(m); a=[];b=[];Vs=[v.copy()]
    w=atoms*v; a0=v@w; a.append(a0); w=w-a0*v
    for j in range(1,m):
        be=np.linalg.norm(w)
        if be<1e-13: break
        b.append(be); vn=w/be
        for u in Vs: vn=vn-(u@vn)*u
        vn/=np.linalg.norm(vn); Vs.append(vn); w=atoms*vn-be*Vs[-2]; al=vn@w; a.append(al); w=w-al*vn
    return np.array(a),np.array(b)

def dot(u,v):
    s=arb(0)
    for x,y in zip(u,v): s+=x*y
    return s

def jacobi_arb(atoms, reorth):
    m=len(atoms)
    inv_sq=arb(m).rsqrt()
    v=[inv_sq]*m
    a=[]; b=[]; Vs=[list(v)]
    w=[atoms[i]*v[i] for i in range(m)]
    a0=dot(v,w); a.append(a0)
    w=[w[i]-a0*v[i] for i in range(m)]
    maxrad=arb(0)
    for j in range(1,m):
        be=dot(w,w).sqrt()
        b.append(be)
        vn=[w[i]/be for i in range(m)]
        if reorth:
            for u in Vs:
                c=dot(u,vn)
                vn=[vn[i]-c*u[i] for i in range(m)]
            nn=dot(vn,vn).sqrt()
            vn=[vn[i]/nn for i in range(m)]
        Vs.append(vn)
        prev=Vs[-2]
        w=[atoms[i]*vn[i]-be*prev[i] for i in range(m)]
        al=dot(vn,w); a.append(al)
        w=[w[i]-al*vn[i] for i in range(m)]
        for x in (al,be):
            r=arb(x.rad());
            if r>maxrad: maxrad=r
    return a,b,maxrad

mp.mp.dps=40
# use the frozen CSV D-atoms for the test (first 40)
import csv
rD=[]
with open(r"quantiles_production.csv") as f:
    for r in csv.DictReader(f): rD.append(float(r["gamma_D"]))
m=40
atoms_f=np.array([1.0/g**2 for g in rD[:m]])
af,bf=jacobi_float(atoms_f)

for prec in (200,400,800):
    ctx.prec=prec
    atoms_a=[arb(1)/arb(mp.nstr(mp.mpf(g),38))**2 for g in rD[:m]]
    for reorth in (False,True):
        a,b,mr=jacobi_arb(atoms_a,reorth)
        # compare last a,b to float
        da=abs(float(a[-1].mid())-af[len(a)-1]); db=abs(float(b[-1].mid())-bf[len(b)-1])
        print(f"prec={prec:4d} reorth={reorth!s:5}  maxrad={float(mr):.2e}  "
              f"|da_last|={da:.2e} |db_last|={db:.2e}  (n_a={len(a)})")
