import numpy as np
import numba as nb
import parameters as par
# ===========================

# MOLECULAR - BATH COUPLING HAMILTONIAN
@nb.jit(nopython=True, fastmath=True)
def H_BC(data):
    cj  = par.cj
    H0  = np.zeros((2,2), dtype= np.complex128)
    H0[0,0] += np.sum(cj * data.x) 
    H0[1,1] -= np.sum(cj * data.x) 
    data.H_bc = H0 * 1.0

# INITIAL DENSITY MATRIX# INITIALIZE THE BATH DOF
@nb.jit(nopython=True, fastmath=True)
def initR(data):
    β  = par.β
    ωj = par.ωj

    σP = np.sqrt(ωj / (2 * np.tanh(0.5*β*ωj)))
    σx = σP/ωj

    # data.x = np.array(np.loadtxt('/scratch/smontill/Simpkins/SemiClassical-NAMD/R.txt'))
    # data.P = np.loadtxt('/scratch/smontill/Simpkins/SemiClassical-NAMD/P.txt')
    data.x = np.random.normal(loc=0.0, scale=1.0, size= len(ωj)) * σx
    data.P = np.random.normal(loc=0.0, scale=1.0, size= len(ωj)) * σP
    
    