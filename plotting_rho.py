import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import time as tm
import parameters as par
# ===================================

data = np.zeros((par.NSteps, 2), dtype = np.complex128)

for i in range(20):
    data[:,0] += np.loadtxt(f'data/rho_{i}.txt', dtype = np.complex128)[:,0]
    data[:,1] += np.loadtxt(f'data/rho_{i}.txt', dtype = np.complex128)[:,3]

time_A = np.loadtxt('../SemiClassical-NAMD/output/mfe-spinBoson-0.txt')[:,0]
# dataA = np.loadtxt('../SemiClassical-NAMD/output/mfe-spinBoson.txt')
dataA = np.zeros((len(time_A), 3))
for a in range(20):
    dataA += np.loadtxt(f'../SemiClassical-NAMD/output/mfe-spinBoson-{a}.txt')
# ========================================
fig, ax = plt.subplots( figsize = (4.5,4.5))

    # ax[i].plot(HEOM[:,0]/par.fstoau ,HEOM[:,i+1],  ls = '-', lw = 2, color = 'black', alpha = 1) #r'$\rho_{\nu 6}$'
ax.plot(par.Sim_time[:]/par.fstoau, data[:,0]/20,  ls = '-', lw = 2) 
ax.plot(par.Sim_time[:]/par.fstoau, data[:,1]/20,  ls = '-', lw = 2) 
ax.plot(time_A/par.fstoau, dataA[:,1]/20,  ls = '-', lw = 1, c = 'black')
ax.plot(time_A/par.fstoau, dataA[:,2]/20,  ls = '-', lw = 1, c = 'black')
# ax.set_xlim(0,0.5)
plt.savefig('images/pop.png', dpi = 300, bbox_inches='tight')
plt.close()

# ===================================
# data = np.loadtxt('data/pos_0.txt')
# dataA = np.loadtxt('../SemiClassical-NAMD/pos.txt')
# # dataA = np.zeros((len(time_A), 3))
# # for a in range(10):
# #     dataA += np.loadtxt(f'../SemiClassical-NAMD/output/mfe-spinBoson-{a}.txt')
# # ========================================

# fig, ax = plt.subplots( figsize = (4.5,4.5))

#     # ax[i].plot(HEOM[:,0]/par.fstoau ,HEOM[:,i+1],  ls = '-', lw = 2, color = 'black', alpha = 1) #r'$\rho_{\nu 6}$'
# ax.plot(par.Sim_time[:]/par.fstoau, data[:],  ls = '-', lw = 3 , marker = 'o', markersize = 5) 
# ax.plot(time_A/par.fstoau, dataA[:],  ls = '-', lw = 2, c = 'black', marker = 'o', markersize = 3)
# # ax.set_xlim(0,0.5)
# plt.savefig('images/pos.png', dpi = 300, bbox_inches='tight')
# plt.close()

# data = np.loadtxt('data/forc_0.txt')
# dataA = np.loadtxt('../SemiClassical-NAMD/forc.txt')

# fig, ax = plt.subplots( figsize = (4.5,4.5))

#     # ax[i].plot(HEOM[:,0]/par.fstoau ,HEOM[:,i+1],  ls = '-', lw = 2, color = 'black', alpha = 1) #r'$\rho_{\nu 6}$'
# ax.plot(par.Sim_time[:]/par.fstoau, data[:],  ls = '-', lw = 3 , marker = 'o', markersize = 5) 
# ax.plot(time_A/par.fstoau, dataA[:],  ls = '-', lw = 2, c = 'black', marker = 'o', markersize = 3)
# # ax.set_xlim(0,0.5)
# plt.savefig('images/forc.png', dpi = 300, bbox_inches='tight')
# plt.close()