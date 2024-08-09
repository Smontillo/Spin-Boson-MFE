#!/software/anaconda3/2020.11/bin/python
#SBATCH -p debug
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH --output=qjob.out
#SBATCH --error=qjob.err
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

import numpy as np
import time as tm
import sys, os
import MFE as method
import parameters as par
import TrajClass as tc
import model
# =================================
sys.path.append(os.popen("pwd").read().split("/tmpdir")[0]) # include parent directory which has method and model files
JOBID = str(os.environ["SLURM_ARRAY_JOB_ID"]) # get ID of this job
TASKID = str(os.environ["SLURM_ARRAY_TASK_ID"]) # get ID of this task within the array
# =========================
# Parallelization
# =========================
nrank = int(TASKID)        # kind of JOD ID for a job
size = 20                  # total number of processors available

# =================================
# COMPILATION 
# =================================
com_ti = tm.time()
ndof_dummy = par.ndof
nsteps_dummy = 2
data_dummy = tc.trajData(ndof_dummy, nsteps_dummy)
data_dummy.ρt = par.ρ0

# MODEL FUNCTIONS =================
model.initR(data_dummy)
model.H_BC(data_dummy)

# METHOD FUNCTIONS ================
method.Force(data_dummy, data_dummy.F1)
method.RK4(data_dummy)
method.VelVer(data_dummy)
method.run_traj(data_dummy)

com_tf = tm.time()
print(f'Compilation time --> {np.round(com_tf - com_ti,2)} s or {np.round((com_tf - com_ti)/60,2)} min')
print(' ================================================================================================= ')
# =================================
# SIMULATION
# =================================

tot_Tasks = par.NTraj
NTasks = tot_Tasks//size
NRem = tot_Tasks - (NTasks*size)
TaskArray = [i for i in range(nrank * NTasks , (nrank+1) * NTasks)]
for i in range(NRem):
    if i == nrank: 
        TaskArray.append((NTasks*size)+i)
TaskArray = np.array(TaskArray)

ρw = np.zeros((par.NSteps, 2**2), dtype = np.complex128)
pos = np.zeros((par.NSteps), dtype = np.float64)
forc = np.zeros((par.NSteps), dtype = np.float64)
trajData = tc.trajData(par.ndof, par.NSteps)

sim_ti = tm.time()
for i in range(len(TaskArray)):
    method.run_traj(trajData)
    ρw += np.real(trajData.ρw)
    pos += np.real(trajData.pos)
    forc += np.real(trajData.forc)
sim_tf = tm.time()
print(f'Simulation time --> {np.round(sim_tf - sim_ti,2)} s or {np.round((sim_tf - sim_ti)/60,2)} min')

try:
    np.savetxt(f'../data/rho_{nrank}.txt', ρw/len(TaskArray))
except:
    np.savetxt(f'./data/rho_{nrank}.txt', ρw/len(TaskArray))
    np.savetxt(f'./data/pos_{nrank}.txt', pos/len(TaskArray))
    np.savetxt(f'./data/forc_{nrank}.txt', forc/len(TaskArray))