import numpy as np
from math import pi,cos,sin,sqrt,exp
import random
from Potts_func_inhomogeneous import *
import sys

scriptname,Lx,Ly,p,lA,Nstep=sys.argv
Lx=int(Lx)
Ly=int(Ly)
lA=int(lA)
p=float(p)
Nstep=int(Nstep)

lambd=0.2#0.13#0.2

folder='/home/gchiriac/NonMarkov/StatMech/Montecarlo/Inhom/Results'#L'+str(L)
fileConfig = folder + '/lambd'+str(round(lambd,2))+'/Config_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'_p'+str(p)+'_lA'+str(lA)+'_lambd'+str(lambd)+'.dat'

n=2
m=1
Q = n*m+1
d=10
with open(fileConfig,'rb') as f:
    config = np.loadtxt(f)
spin1 = np.array(config,dtype=np.uint8)

q=6#factorial(Q)
#Nstep = 100000
Ntherm = 1
Ninterval = 50

def Rate(t,lam,p):
    return p*(lam**2+np.exp(-lam*t)*lam*(- lam*np.cos(t)+np.sin(t)))/(lam**2+lam*np.exp(-np.pi*lam/2))

pM = np.array([Rate(ti,lambd,p)*np.ones(Lx) for ti in 0.5*np.arange(Ly)])

AA = Coupling_matr(Q,pM,d)
g = gx(n,m)
contour = boundary(Lx,lA,g)

fileEn = folder + '/lambd'+str(round(lambd,2))+'/Energy_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'_p'+str(p)+'_lA'+str(lA)+'_lambd'+str(lambd)+'.dat'
filesp = folder + '/lambd'+str(round(lambd,2))+'/Spin_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'_p'+str(p)+'_lA'+str(lA)+'_lambd'+str(lambd)+'.dat'
filesp2 = folder + '/lambd'+str(round(lambd,2))+'/SpinQ_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'_p'+str(p)+'_lA'+str(lA)+'_lambd'+str(lambd)+'.dat'

EBu,ECo,sp,sp2 = Montecarlo(Lx,Ly,spin1,q,AA,Nstep,boundary=contour,Ntherm=Ntherm,prnt=0,Ninterval=Ninterval)
with open(fileEn,'ab') as f:
    np.savetxt(f,np.transpose(np.array([EBu,ECo])))
with open(filesp,'ab') as f:
    np.savetxt(f,sp)
with open(filesp2,'ab') as f:
    np.savetxt(f,sp2)