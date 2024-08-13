import numpy as np
import numba as nb
import model
import parameters as par
# ==================================

# FORCE MATRIX
@nb.jit(nopython=True, fastmath=True)
def Force(data, F):
    F[:]   = 0
    F[:]  -= par.ωj**2 * data.x 
    par_sum = np.sum(par.dHij * data.ρt * 1.0, axis= 1).real
    F[:] -= np.sum(par_sum, axis = 1).real

# PROPAGATION
@nb.jit(nopython=True, fastmath=True)
def RK4(data):
    H = par.Hel * 1.0 + data.H_bc * 1.0
    ρ = data.ρt * 1.0
    dt = par.dtE * 1.0
    for k in range(int(par.Estep/2)):
        k1 = von_Newman(ρ.copy(), H.copy())
        k2 = von_Newman(ρ.copy() + 0.5 * dt * k1, H.copy())
        k3 = von_Newman(ρ.copy() + 0.5 * dt * k2, H.copy())
        k4 = von_Newman(ρ.copy() + dt * k3, H.copy())
        ρ  = ρ.copy() + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    data.ρt = 1.0 * ρ

# VON - NEWMAN EQUATION
@nb.jit(nopython=True, fastmath=True)
def von_Newman(ρf, H):
    return -1j * (H @ ρf - ρf @ H)

#  VELOCITY VERLET PROPAGATOR
@nb.jit(nopython=True, fastmath=True)
def VelVer(data) : 
    data.v[:] = data.P[:]/par.M 
    # half electronic evolution
    RK4(data)  # THE RK4 DOES HALF OF THE ELECTRONIC PROPAGATION!!!!
    # Force(data, data.F1)
    # ======= Nuclear Block ==================================
    data.x[:] += data.v[:] * par.dtN + 0.5 * data.F1[:] * par.dtN** 2 / par.M
    model.H_BC(data)
    #-----------------------------
    Force(data, data.F2) # force at t2
    data.v[:] += 0.5 * (data.F1[:] + data.F2[:]) * par.dtN / par.M
    data.P[:] = data.v[:] * par.M
    data.F1[:] = data.F2[:] * 1.0
    # ======================================================
    RK4(data)  # THE RK4 DOES HALF OF THE ELECTRONIC PROPAGATION!!!!

# RUN TRAJECTORIES
@nb.jit(nopython=True, fastmath=True)
def run_traj(data):
    model.initR(data)
    data.ρt = par.ρ0
    Force(data, data.F1)
    iskip = 0
    for st in range(data.nsteps):
        if (st % par.nskip == 0):
            ρ = data.ρt
            ρ = ρ.copy()
            data.ρw[iskip,:]  = ρ.reshape(1,2**2)
            # data.pos[iskip] = data.x[0]
            # data.forc[iskip] = data.F1[0]
            iskip += 1
        VelVer(data)
        