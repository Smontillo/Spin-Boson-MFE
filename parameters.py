import numpy as np
import numba as nb
# ==================================

# FUNCTIONS
# ==================================
def model(M):

    #        |  M0 |  M1 |  M2  |  M3 |  M4 |  M5 |
    ε   = np.array([  0.0,   0.0,  1.0,  1.0,  0.0,  5.0  ])
    ξ   = np.array([ 0.09,  0.09,  0.1,  0.1,  2.0,  4.0  ])
    β   = np.array([  0.1,   5.0, 0.25,  5.0,  1.0,  0.1  ])
    ωc  = np.array([  2.5,   2.5,  1.0,  2.5,  1.0,  2.0  ])
    Δ   = np.array([  1.0,   1.0,  1.0,  1.0,  1.0,  1.0  ])
    N   = np.array([  100,   100,  100,  100,  400,  400  ])

    return ε[M], ξ[M], β[M], ωc[M], Δ[M], N[M]  

# DRUDE - LORENTZ SPECTRAL DENSITY
# @nb.jit(nopython=True, fastmath=True)
# def J_DrudeL(λ, γ, ω):
#     return (2 * γ * λ * ω) / (ω**2 + γ**2)

# BATH PARAMETERS
@nb.jit(nopython=True, fastmath=True)
def BathParam(ξ, ωc, ndof):   
    ωm = 4.0
    ω0 = ωc * ( 1-np.exp(-ωm) ) / ndof
    cj = np.zeros(( ndof ), dtype = np.complex128)
    ωj = np.zeros(( ndof ))
    for d in range(ndof):
        ωj[d] =  -ωc * np.log(1 - (d+1)*ω0/ωc)
        cj[d] =  np.sqrt(ξ * ω0) * ωj[d]  
    # for d in range(N):
    #     ωj[d] = γD[0] * np.tan(np.pi/2 * (d - 1/2)/N)
    #     cj[d] = 2 *(λD[0]/(2 * N))**0.5 * ωj[d]
    # for k in range(len(λD)):
    #     J = J_DrudeL(λD[k], γD[k], ω)    # Drude-Lorentz part
    
    #     Fω = np.zeros(len(ω))
    #     for i in range(len(ω)):
    #         Fω[i] = (4/np.pi) * np.sum(J[:i]/ω[:i]) * dω

    #     λs =  Fω[-1]
    #     for i in range(N):
    #         costfunc = np.abs(Fω-(((float(i)+0.5)/float(N))*λs))
    #         m = np.argmin((costfunc))
    #         ωj[k * N + i] = ω[m]
    #     cj[k * N: (k+1) * N] = ωj[k * N: (k+1) * N] * ((λs/(2*float(N)))**0.5)
    return cj , ωj

# CREATION OPERATOR
@nb.jit(nopython=True, fastmath=True)
def creation(n):
    a = np.zeros((n,n), dtype = np.complex128)
    b = np.array([(x+1)**0.5 for x in range(n)], dtype = np.complex128)
    np.fill_diagonal(a[1:], b)
    return a

@nb.jit(nopython=True, fastmath=True)
def dHij_cons():
    dHij = np.zeros((ndof ,2, 2), dtype = np.complex128)
    dHij[:,0,0] = cj
    dHij[:,1,1] = -cj
    return dHij

@nb.jit(nopython=True, fastmath=True)
def Hel_cons():
    Vij = np.zeros((2,2), dtype = np.complex128)

    Vij[0,0] = ε
    Vij[1,1] = - ε

    Vij[0,1], Vij[1,0] = Δ, Δ 
    return Vij

'''
    SIMULATION PARAMETERS 
'''

# PHYSICAL CONSTANTS
# ==================================
fstoau = 41.341                           # 1 fs = 41.341 a.u.
cmtoau = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK  = 3.1577464e+05 


# SYSTEM PARAMETERS ==================================
M = 1.0
ε, ξ, β, ωc, Δ, ndof = model(5)
ρ0 = np.zeros((2,2), dtype = np.complex128)
ρ0[0,0] = 1
# SIMULATION PARAMETERS ==============================
NTraj    = 1000
tf       = 4.9 * fstoau
dtN      = 0.1
NSteps   = int(tf/dtN)
Sim_time = np.array([(x * dtN) for x in range(NSteps)])
Estep    = 20                           # MUST BE EVEN!!!!
dtE      = dtN/Estep
nskip    = 1 

# BATH PARAMETERS ==============================
cj, ωj = BathParam(ξ, ωc, ndof)
# print(cj[-1], ωj[-1]/cmtoau)
# TIME INDEPENDENT FUNCTIONS ==============================
Hel   = Hel_cons()
dHij  = dHij_cons()
