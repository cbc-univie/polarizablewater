import MDAnalysis
from newanalysis.correl import correlateParallel
from newanalysis.functions import atomsPerResidue, residueFirstAtom 
from newanalysis.functions import dipoleMomentByResidue
import numpy as np
import time
from datetime import timedelta
import gc

base = "/site/raid5/marion/thz/spce/from_spce_4000/timeseries/"

mu_tot = np.loadtxt(f"{base}/short_tot.ser")

n = 10000
dt = 0.002 #yees
time = np.linspace(0.0,n*dt,n)

mu_sum_tot = np.zeros((n,3))
for i in range(len(mu_tot[0])//3):
    mu_sum_tot[:,0] += mu_tot[:,3*i]
    mu_sum_tot[:,1] += mu_tot[:,3*i+1]
    mu_sum_tot[:,2] += mu_tot[:,3*i+2]


out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_tot.T),
                     np.ascontiguousarray(mu_sum_tot.T),
                  out, ltc = 0)

f1 = open('MU_MU_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt, out[j]))
f1.close()

mu_perm = np.loadtxt(f"{base}/short_rot.ser")

mu_sum_perm = np.zeros((n,3))
for i in range(len(mu_perm[0])//3):
    mu_sum_perm[:,0] += mu_perm[:,3*i]
    mu_sum_perm[:,1] += mu_perm[:,3*i+1]
    mu_sum_perm[:,2] += mu_perm[:,3*i+2]


out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_perm.T),
                     np.ascontiguousarray(mu_sum_perm.T),
                  out, ltc = 0)

f1 = open('MU_MU_perm.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt, out[j]))
f1.close()

out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_perm.T),
                     np.ascontiguousarray(mu_sum_tot.T),
                  out, ltc = 0)

f1 = open('MU_perm_MU_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt, out[j]))
f1.close()


mu_ind = np.loadtxt(f"{base}/short_trans.ser")

mu_sum_ind = np.zeros((n,3))
for i in range(len(mu_ind[0])//3):
    mu_sum_ind[:,0] += mu_ind[:,3*i]
    mu_sum_ind[:,1] += mu_ind[:,3*i+1]
    mu_sum_ind[:,2] += mu_ind[:,3*i+2]


out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_ind.T),
                     np.ascontiguousarray(mu_sum_ind.T),
                  out, ltc = 0)

f1 = open('MU_MU_ind.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt, out[j]))
f1.close()

out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_ind.T),
                     np.ascontiguousarray(mu_sum_tot.T),
                  out, ltc = 0)

f1 = open('MU_ind_MU_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt, out[j]))
f1.close()

out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_ind.T),
                     np.ascontiguousarray(mu_sum_perm.T),
                  out, ltc = 0)

f1 = open('MU_ind_MU_perm.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt, out[j]))
f1.close()

out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_perm.T),
                     np.ascontiguousarray(mu_sum_ind.T),
                  out, ltc = 0)

f1 = open('MU_perm_MU_ind.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt, out[j]))
f1.close()
