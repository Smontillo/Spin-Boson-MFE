import numpy as np
from numba import int32, float64, complex128
from numba.experimental import jitclass
from numba import jit
import parameters as par
# ==================================

spec = [
    ('nsteps',                  int32),
    ('ndof',                    int32),
    ('x',                  float64[:]),
    ('pos',                  float64[:]),
    ('forc',                  float64[:]),
    ('P',                  float64[:]),
    ('v',                  float64[:]),
    ('F1',                 float64[:]),
    ('F2',                 float64[:]),
    ('ρt',            complex128[:,:]),
    ('H_bc',          complex128[:,:]),
    ('ρw',            complex128[:,:]),
]

@jitclass(spec)
class trajData(object):
    def __init__(self, ndof, nsteps):
        self.nsteps = nsteps
        self.ndof   = ndof
        self.x      = np.zeros(self.ndof, dtype = np.float64)
        self.pos    = np.zeros((nsteps), dtype = np.float64)
        self.forc    = np.zeros((nsteps), dtype = np.float64)
        self.P      = np.zeros(self.ndof, dtype = np.float64)
        self.v      = np.zeros(self.ndof, dtype = np.float64)
        self.F1     = np.zeros(self.ndof, dtype = np.float64)
        self.F2     = np.zeros(self.ndof, dtype = np.float64)
        self.ρt     = np.zeros((2, 2), dtype = np.complex128)
        self.H_bc   = np.zeros((2, 2), dtype = np.complex128)
        self.ρw     = np.zeros((nsteps, 2**2) , dtype = np.complex128)
        