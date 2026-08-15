# olver_ivl_22_c2_EM.py — Psi_2 project FINAL: j-difference + midpoint-EM assembly of the summand
# profile (Phi0,Phi1,Phi2) -> c2(w) (the eps^2 cumulative profile).  Self-validates by reproducing
# c0 = -2^{2/3}AiBi and c1 = Psi0, then reads c2.  Airy-ring antiderivatives by undetermined coeffs.
# Not RH/GRH.
import sympy as sp
A,Ap,B,Bp,w,eps = sp.symbols('A Ap B Bp w eps')
c13=sp.Integer(2)**sp.Rational(1,3); c23=sp.Integer(2)**sp.Rational(2,3); pi=sp.pi

# w-derivative in the Airy ring: A'=Ap, Ap'=w A, B'=Bp, Bp'=w B
def ddw(e):
    e=sp.expand(e)
    return sp.expand(e.diff(w)+e.diff(A)*Ap+e.diff(Ap)*(w*A)+e.diff(B)*Bp+e.diff(Bp)*(w*B))

Phi0=A*B
Phi1=c13*(2*A*Ap-A*Bp+3*Ap*B)/2                       # verified (=2^{1/3}[Ai'(Ai+Bi)-1/(2pi)])
Phi2=(c23/20)*(30*Ap**2-15*Ap*Bp-10*w*A*A+9*w*A*B+3*w*w*(Ap*B+A*Bp))   # verified

# sigma(eps,w) = pi 2^{-1/3} eps^{-1}(Phi0+eps Phi1+eps^2 Phi2)
pref=pi/c13
# j-step: nu=eps^{-3}, z=nu-2^{-1/3} w nu^{1/3}; nu2=nu+2; eps'=nu2^{-1/3}; w_new=2^{1/3}nu2^{2/3}-2^{1/3}nu2^{-1/3}z
nu=eps**(-3); z=nu-(1/c13)*w*eps**(-1); nu2=nu+2
def ser(e,n): return sp.series(sp.expand(e),eps,0,n).removeO()
epsp=ser(nu2**(sp.Rational(-1,3)),6)
w_new=ser(c13*nu2**sp.Rational(2,3)-c13*nu2**sp.Rational(-1,3)*z,6)
Dw=ser(w_new-w,6)                                     # Delta w as eps-series

# sigma(eps', w+Dw): substitute eps->epsp in the eps-coefficients, Taylor in Dw for the w-shift
def sigma_of(e_eps, wshift):
    # Phi_k(w+wshift) via Airy Taylor to needed order (wshift starts O(eps))
    def shift(P):
        P1=ddw(P); P2=ddw(P1); P3=ddw(P2); P4=ddw(P3)
        return sp.expand(P+P1*wshift+P2*wshift**2/2+P3*wshift**3/6+P4*wshift**4/24)
    body=shift(Phi0)+e_eps*shift(Phi1)+e_eps**2*shift(Phi2)
    return sp.expand(pref*e_eps**(-1)*body)

sig0=sigma_of(eps,0)
sigS=sigma_of(epsp,Dw)
Dj=sp.expand(sigS-sig0)
Dj=sp.series(Dj,eps,0,3).removeO()                    # Delta_j sigma to eps^2
g0=sp.expand(Dj.coeff(eps,0)); g1=sp.expand(Dj.coeff(eps,1)); g2=sp.expand(Dj.coeff(eps,2))

# ---- Airy-ring antiderivative: given g (poly in A,Ap,B,Bp with w-poly coeffs, degree-2, no B^2),
#      find F with F'=g in basis {1,A2,AAp,Ap2,AB,ApB,ApBp}; ABp reduced via ABp=ApB+1/pi.
MON=['one','A2','AAp','Ap2','AB','ApB','ApBp']
monexpr={'one':sp.Integer(1),'A2':A*A,'AAp':A*Ap,'Ap2':Ap*Ap,'AB':A*B,'ApB':Ap*B,'ApBp':Ap*Bp}
def to_dict(e):
    e=sp.expand(e).subs(A*Bp, Ap*B+1/pi)              # ABp = ApB + 1/pi
    e=sp.expand(e)
    d={m:sp.Integer(0) for m in MON}
    # collect coefficients of each monomial (treat A,Ap,B,Bp as symbols)
    for m in ['A2','AAp','Ap2','AB','ApB','ApBp']:
        c=e.coeff(monexpr[m])
        # remove cross-contamination: coeff picks terms with that product; subtract
        d[m]=sp.expand(c)
    # rebuild and get remainder as 'one'
    rebuilt=sum(d[m]*monexpr[m] for m in MON[1:])
    d['one']=sp.expand(e-rebuilt)
    # sanity: 'one' should have no A,Ap,B,Bp
    return d
# derivative of each basis monomial as a dict (w-poly coeffs), using ABp->ApB+1/pi
Dbasis={
 'one':{'one':sp.Integer(0)},
 'A2':{'AAp':sp.Integer(2)},
 'AAp':{'Ap2':sp.Integer(1),'A2':w},
 'Ap2':{'AAp':2*w},
 'AB':{'ApB':sp.Integer(2),'one':1/pi},               # (AB)'=ApB+ABp=2ApB+1/pi
 'ApB':{'AB':w,'ApBp':sp.Integer(1)},
 'ApBp':{'ApB':2*w,'one':w/pi},                        # (ApBp)'=w(ABp+ApB)=2w ApB+w/pi
}
def antideriv(gdict):
    # F = sum q_m(w) mon; solve F' = g.  q_m polynomials in w up to degree D.
    D=6
    qs={m:[sp.Symbol(f'q_{m}_{k}') for k in range(D+1)] for m in MON}
    qpoly={m:sum(qs[m][k]*w**k for k in range(D+1)) for m in MON}
    # F' dict
    Fp={m:sp.Integer(0) for m in MON}
    for m in MON:
        # d/dw(qpoly[m]*mon_m) = qpoly[m]' * mon_m + qpoly[m]* d(mon_m)
        Fp[m]+=sp.diff(qpoly[m],w)                     # q' * mon_m (same monomial)
        for tgt,coef in Dbasis[m].items():
            Fp[tgt]=Fp.get(tgt,0)+qpoly[m]*coef
    # equations: Fp[m] == gdict[m] for each monomial, matched by powers of w
    eqs=[]
    for m in MON:
        diff=sp.expand(Fp[m]-gdict.get(m,0))
        p=sp.Poly(diff,w)
        eqs+=[c for c in p.all_coeffs()]
    allq=[qs[m][k] for m in MON for k in range(D+1)]
    sol=sp.solve(eqs,allq,dict=True)
    if not sol: return None
    s=sol[0]
    F={m:sp.expand(qpoly[m].subs(s)) for m in MON}
    # free symbols (integration const) -> set to 0
    F={m:sp.expand(v.subs({sym:0 for sym in v.free_symbols if str(sym).startswith('q_')})) for m,v in F.items()}
    return F
def eval_dict_to_inf(gdict):
    F=antideriv(gdict)
    if F is None: return None
    # int_w^inf g = F(inf)-F(w) = -F(w)  (Airy monomials ->0; 'one' w-poly part must ->0: check)
    val=-sum(F[m]*monexpr[m] for m in MON)
    return sp.expand(val)

# assemble c0,c1,c2
g0d=to_dict(g0); g1d=to_dict(g1); g2d=to_dict(g2)
K=1/(c13*pi)
I0=eval_dict_to_inf(g0d); I1=eval_dict_to_inf(g1d); I2=eval_dict_to_inf(g2d)
c0=sp.expand(K*I0)
c1=sp.expand(K*I1 + g0/2)
c2=sp.expand(K*I2 + g1/2 - (c13*pi/6)*ddw(g0))

def clean(e): return sp.simplify(sp.expand(e).subs(A*Bp,Ap*B+1/pi))
print("=== VALIDATION ===")
print(" c0 (want -2^{2/3} A B):", clean(c0))
print(" c1 (want Psi0=-2^{2/3}Phi1+(pi-2)(AiBi)'):")
Psi0=sp.expand(-c23*Phi1+(pi-2)*(Ap*B+A*Bp))
print("    c1        =", clean(c1))
print("    Psi0(expc)=", clean(Psi0))
print("    c1 - Psi0 =", clean(c1-Psi0))
print("=== RESULT c2(w) ===")
print("   c2 =", clean(c2))
print("done.")
