# Proof that chi_8 (Steinberg of PSL(2,7)) is MONOMIAL: chi_8 = Ind_H^G(psi),
# H = 7:3 Frobenius (order 21 = conductor), [H,H]=Z7 (Singer), H^ab=Z3 (trit), psi cubic.
# Consequence (Artin): L(s,chi_8) = L_{K_H}(s,psi) is a HECKE L-function over the octic field K_H=L^H.
# => RH(chi_8) = GRH for the cubic Hecke character psi ; Levinson-Conrey positive-proportion applies.
import itertools, numpy as np, cmath
def matmul(A,B): return tuple((np.array(A).reshape(3,3)@np.array(B).reshape(3,3)%2).flatten())
I=tuple(np.eye(3,dtype=int).flatten())
def det2(M): return int(round(np.linalg.det(np.array(M).reshape(3,3))))%2
G=[tuple(int(x) for x in t) for t in itertools.product([0,1],repeat=9) if det2(t)==1]
def order(g):
    x=g;k=1
    while x!=I: x=matmul(x,g);k+=1
    return k
order_={g:order(g) for g in G}
inv={g:next(h for h in G if matmul(g,h)==I) for g in G}
s=next(g for g in G if order_[g]==7)
S7=set();y=I
for _ in range(7): S7.add(y); y=matmul(y,s)
conj=lambda g,a: matmul(matmul(g,a),inv[g])
H=[g for g in G if set(conj(g,a) for a in S7)==S7]          # normalizer of <s>, order 21
comm={matmul(matmul(matmul(a,b),inv[a]),inv[b]) for a in H for b in H}
Hd=set(comm);ch=True
while ch:
    ch=False
    for a in list(Hd):
        for b in list(Hd):
            p=matmul(a,b)
            if p not in Hd: Hd.add(p);ch=True                # [H,H] = Z7
w=cmath.exp(2j*cmath.pi/3)
cosetkey=lambda h: frozenset(matmul(h,d) for d in Hd)
keys=[];
for h in H:
    k=cosetkey(h)
    if k not in keys: keys.append(k)
idx={h:keys.index(cosetkey(h)) for h in H}; id0=idx[I]
psi=lambda h: w**((idx[h]-id0)%3)
Hset=set(H); psio=lambda y: psi(y) if y in Hset else 0
Ind=lambda g: sum(psio(matmul(matmul(inv[x],g),x)) for x in G)/len(H)
reps={'1':I,'2':next(g for g in G if order_[g]==2),'3':next(g for g in G if order_[g]==3),
      '4':next(g for g in G if order_[g]==4)}
sev=[g for g in G if order_[g]==7]; reps['7A']=sev[0]
cl7A={conj(g,sev[0]) for g in G}; reps['7B']=next(g for g in sev if g not in cl7A)
chi8={'1':8,'2':0,'3':-1,'4':0,'7A':1,'7B':1}
print("H order",len(H),"[H,H] order",len(Hd),"H^ab order",len(H)//len(Hd))
ok=all(abs(Ind(g).real-chi8[n])<1e-6 and abs(Ind(g).imag)<1e-6 for n,g in reps.items())
for n,g in reps.items(): print(f"  Ind(psi)[{n}]={Ind(g).real:+.3f}  chi_8={chi8[n]:+d}")
print("chi_8 = Ind_H^G(psi) (psi cubic 1-dim):",ok,"  => MONOMIAL => L(s,chi_8) is a Hecke L-function")
