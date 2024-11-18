from math import sin,cos,sqrt,exp,log,pi,factorial
import numpy as np
import random
from itertools import * 

### The following are function from the more_itertools package which may not be installed on your cluster ###

def permutation_index(element, iterable):
    """Given a permutation element and an iterable sequence (e.g. the range of the first Q integers) returns the index of that permutation in the standard itertools mapping."""
    index = 0
    pool = list(iterable)
    for i, x in zip(range(len(pool), -1, -1), element):
        r = pool.index(x)
        index = index * i + r
        del pool[r]
    return index

def nth_permutation(iterable, r, index):
    """Given an iterable sequence iterable, a range r and an index the permutation of iterable associated to that index in the standard itertools mapping."""
    pool = list(iterable)
    n = len(pool)
    if r is None or r == n:
        r, c = n, factorial(n)
    elif not 0 <= r < n:
        raise ValueError
    else:
        c = factorial(n) // factorial(n - r)
    if index < 0:
        index += c
    if not 0 <= index < c:
        raise IndexError
    if c == 0:
        return tuple()
    result = [0] * r
    q = index * factorial(n) // c if r < n else index
    for d in range(1, n + 1):
        q, i = divmod(q, d)
        if 0 <= n - d < r:
            result[n - d] = i
        if q == 0:
            break
    return tuple(map(pool.pop, result))

def perm_mult(g1,g2):
    "Multiplies two permutations g1 and g2"
    gv=np.array(g1)
    return gv[np.array(g2[:])]

def perm_inv(g1):
    """constructs the inverse of a permutations g1"""
    return np.array([np.where(np.array(g1)==i)[0][0] for i in range(len(g1))])

def Coupling_matr(Q,p,d,tol=1e-16):
    """Constructs coupling matrix in the limit of large d for arbitrary Q and p"""
    q=factorial(Q)
    Perm_m = np.array([[perm_mult(perm_inv(nth_permutation(range(Q),Q,j)),nth_permutation(range(Q),Q,i)) for j in range(q)] for i in range(q)])
    AA=np.zeros((q,q))
    for i in range(q):
        for j in range(q):
            if np.count_nonzero(Perm_m[i,j]-np.arange(Q)!=0)==2:
                AA[i,j]+=1
    if len(np.shape(p))==0:
        return max(p,tol)+np.identity(q)*(1-p)+(1-p)*AA/d
    else:
        return np.array([[max(pr,tol)+np.identity(q)*(1-pr)+(1-pr)*AA/d for pr in row] for row in p])
    
def gx(n,m):
    """Permutation associated to the calculation of the n-th Renyi entropy in a Q=n*m+1 replica space.
    The permutation is given by m copies of the cyclic permutation (2,3,...,n,1), with the Q-th element left untouched."""
    out = []
    Q=n*m+1
    for x in range(m):
        for j in range(n):
            out.append(x*n+(j+1)%n)
    out.append(n*m)
    return np.uint8(permutation_index(np.array(out),range(Q)))

def boundary(Lx,lA,g):
    """Construction of boundary with the permutation gx on the central lA sites and identity (0) on the remaining Lx-lA"""
    out = np.zeros(Lx,dtype=np.uint8)
    lL = int((Lx-lA)/2)
    out[lL:lL+lA] = g
    return out

def next_neighbors(ix,iy,Lx,Ly):
    if iy==0:
        out = [[(ix-1)%Lx,iy+1],[ix,iy+1]]
    elif iy%2==0:
        out = [[(ix-1)%Lx,iy+1],[ix,iy+1],[(ix-1)%Lx,iy-1],[ix,iy-1]]
    else:
        if iy==Ly-1:
            out = [[(ix+1)%Lx,iy-1],[ix,iy-1]]
        else:
            out = [[(ix+1)%Lx,iy+1],[ix,iy+1],[(ix+1)%Lx,iy-1],[ix,iy-1]]
    return out

#@nb.njit()
def cycles(q):
    """Creates a random operator acting on the space {1,...q} of order 2.
    I.e. it is a permutation from Sq made of only transpositions.
    Generates a random permutation of q elements.
    Then takes them pairwise and creates int(q/2) cycles composed of the number of each of these pairs."""
    L = int(q/2)
    out = np.zeros(q,dtype=np.uint8)
    v = np.random.permutation(q)
    out[v[::2]] = v[1::2]
    out[v[1::2]] = v[::2]
    return out
    
def Energy(spin1,spin2,coupling):
    """Energy of two Potts spins given the coupling matrix (dependent on d and p)."""
    weight = coupling[spin1,spin2]
    return np.float(-np.log(weight))

def EnergyArray(spin,boundary,coupling):
    Ly,Lx = np.shape(spin)
    EnA = 0.5*np.array([[np.sum([Energy(spin[jy,jx],spin[ky,kx],coupling[jy,jx]) for kx,ky in next_neighbors(jx,jy,Lx,Ly)]) for jx in range(Lx)] for jy in range(Ly)])
    for i in range(Lx):
        EnA[-1,i] += Energy(spin[-1,i],boundary[i],coupling[-1,i])
    return EnA

def Energy1D(spin1,spin2,coupling):
    """Energy of two 1D arrays of Potts spins given the coupling matrix."""
    if len(np.shape(coupling))==2:
        weight = coupling[spin1[:],spin2[:]]
    else:
        weight = np.array([coupling[i,spin1[i],spin2[i]] for i in range(len(spin1))])
    return np.sum(-np.log(weight))

def Energy2D(spin1,spin2,coupling):
    """Energy of two 2D arrays of Potts spins given the coupling matrix."""
    Ly,Lx = np.shape(spin1)
    if len(np.shape(coupling))==2: 
        weight = coupling[spin1[:,:],spin2[:,:]]
    else:
        weight = np.array([[coupling[i,j,spin1[i,j],spin2[i,j]] for j in range(Lx)] for i in range(Ly)])
    return np.sum(-np.log(weight))

def EnergyBulk(spin,coupling):
    """Bulk energy of a Potts spin lattice, given the coupling matrix and the configuration of next neighbors"""
    En=0
    En += Energy2D(spin[:-1:2,1:],spin[1::2,:-1],coupling)#distance between odd and even rows next neighbors
    En += Energy2D(spin[:-1:2,1:],spin[1::2,1:],coupling)#distance between odd and even rows next neighbors
    En += Energy1D(spin[:-1:2,0],spin[1::2,-1],coupling[:,0,:,:])
    En += Energy2D(spin[1:-1:2,:-1],spin[2::2,1:],coupling)#distance between odd and even rows next neighbors
    En += Energy2D(spin[1:-1:2,:-1],spin[2::2,:-1],coupling)#distance between odd and even rows next neighbors
    En += Energy1D(spin[1:-1:2,0],spin[2::2,-1],coupling[:,0,:,:])
    return En#np.float128(En)

def EnergyBulkHalf(spin0,coupling,l0):
    """Bulk energy of a Potts spin lattice, given the coupling matrix and the configuration of next neighbors"""
    spin = spin0[-l0:,:]
    En=0
    En += Energy2D(spin[:-1:2,1:],spin[1::2,:-1],coupling)#distance between odd and even rows next neighbors
    En += Energy2D(spin[:-1:2,1:],spin[1::2,1:],coupling)#distance between odd and even rows next neighbors
    En += Energy1D(spin[:-1:2,0],spin[1::2,-1],coupling[:,0,:,:])
    En += Energy2D(spin[1:-1:2,:-1],spin[2::2,1:],coupling)#distance between odd and even rows next neighbors
    En += Energy2D(spin[1:-1:2,:-1],spin[2::2,:-1],coupling)#distance between odd and even rows next neighbors
    En += Energy1D(spin[1:-1:2,0],spin[2::2,-1],coupling[:,0,:,:])
    return En#np.float128(En)

def EnergyBC(spin,boundary,coupling):
    """Boundary energy of a Potts spin lattice, given the boundary"""
    return Energy1D(spin[-1,:],boundary,coupling)

def EnergyTOT(spin,boundary,coupling):
    """Energy of two permutations on neighboring sites given the dimension of the local Hilbert space
    and the measurement probability."""
    return EnergyBulk(spin,coupling)+EnergyBC(spin,boundary,coupling)


def Wolff_step(spin,Lx,Ly,r,coupling,boundary=None):
    """One step in the Wolff algorithm."""
    m = [np.random.randint(Ly),np.random.randint(Lx)] #choose random site on physical lattice
    cluster = [] #stack=[] #initialize stack of sites in the cluster and add m to it
    cluster.append(m)
    spin_c = spin.copy() #create copy of lattice spin configuration to not mess with the original one
    log_rflip = -np.log(np.random.rand())#+0.000000000000001)
    DeltaEboundary = 0.0
    for elem in cluster: #loop over the length of the stack until it's empty
        my,mx = elem #get coordinates of cluster element.
        sm = spin[my,mx] #get the spin value on that site
        rsm = r[sm]#rs(r,sm,Q) #get transformed spin
        new_elements = [] #list with the new sites to add to the cluster
        for jx,jy in next_neighbors(mx,my,Lx,Ly): #visit next neighbors sites
        #Works for square lattice. However our lattice has coupling along the diagonals, so we may need to modify that
            coupl = coupling[min(my,jy),mx,:,:]
            DeltaE = Energy(rsm,spin[jy,jx],coupl)-Energy(sm,spin[jy,jx],coupl) #calculate difference in energy
            p_add = max(0,1-exp(-DeltaE)) #get probability to add it
            if [jy,jx] not in cluster and np.random.uniform(0.,1.)<p_add: #if successfull, and the site is not in cluster, add it      
                new_elements.append([jy,jx])
        if my==Ly-1 and boundary.any()!=None: #if boundary is present and the site is next to it, add the energy difference to E_b
            DeltaEboundary += Energy(rsm,boundary[mx],coupling[-1,mx,:,:])-Energy(sm,boundary[mx],coupling[-1,mx,:,:])
        cluster += new_elements #add new elements to cluster
        spin_c[my,mx] = rsm #transform the spin on the original site
    if DeltaEboundary>log_rflip: #if energy cost for boundary is too high, do not flip the cluster
        spin_out = spin
    else:  #otherwise flip it
        spin_out = spin_c
    return spin_out #return spin configuration

def Montecarlo(Lx,Ly,spin,q,coupling,Nstep,boundary=None,Ntherm=1000,prnt=0,Ninterval=20,config=0):
    EBu_out = []
    ECo_out = []
    counter = 0
    
    for i in range(Ntherm):
        r = cycles(q)
        spin = Wolff_step(spin,Lx,Ly,r,coupling,boundary=boundary)
        if i%Ninterval==0:
            if prnt!=0:
                print(spin)
                
    spin_av = spin.copy()
    spin2_av = (spin.copy())**2
    counter += 1
    for i in range(Nstep):
        r = cycles(q)
        spin = Wolff_step(spin,Lx,Ly,r,coupling,boundary=boundary)
        if i%Ninterval==0:
            if prnt!=0:
                print(spin)
            EBu_out.append(EnergyBulk(spin,coupling))
            ECo_out.append(EnergyBC(spin,boundary,coupling[-1]))
            spin_av += spin.copy()
            spin2_av += (spin.copy())**2
            counter += 1
    if config==0:
        return np.array(EBu_out),np.array(ECo_out),spin_av/counter,spin2_av/counter
    else:
        return np.array(EBu_out),np.array(ECo_out),spin_av/counter,spin2_av/counter,spin

def MontecarloEn(Lx,Ly,spin,q,coupling,Nstep,boundary=None,Ntherm=1000,prnt=0,Ninterval=20):
    counter = 0

    for i in range(Ntherm):
        r = cycles(q)
        spin = Wolff_step(spin,Lx,Ly,r,coupling,boundary=boundary)

    En_av = EnergyArray(spin,boundary,coupling)
    En2_av = (En_av.copy())**2
    counter += 1
    for i in range(Nstep):
        r = cycles(q)
        spin = Wolff_step(spin,Lx,Ly,r,coupling,boundary=boundary)
        if i%Ninterval==0:
            Etmp = EnergyArray(spin,boundary,coupling)
            En_av += Etmp.copy()
            En2_av += (Etmp.copy())**2
            counter += 1
    return En_av/counter,En2_av/counter

def MontecarloHalf(Lx,Ly,l0,spin,q,coupling,Nstep,boundary=None,Ntherm=1000,prnt=0,Ninterval=20,config=0):
    EBu_out = []
    ECo_out = []
    EBh_out = []

    for i in range(Ntherm):
        r = cycles(q)
        spin = Wolff_step(spin,Lx,Ly,r,coupling,boundary=boundary)

    for i in range(Nstep):
        r = cycles(q)
        spin = Wolff_step(spin,Lx,Ly,r,coupling,boundary=boundary)
        if i%Ninterval==0:
            EBu_out.append(EnergyBulk(spin,coupling))
            EBh_out.append(EnergyBulkHalf(spin,coupling,l0))
            ECo_out.append(EnergyBC(spin,boundary,coupling[-1]))
    if config==0:
        return np.array(EBu_out),np.array(EBh_out),np.array(ECo_out)
    else:
        return np.array(EBu_out),np.array(EBh_out),np.array(ECo_out),spin
