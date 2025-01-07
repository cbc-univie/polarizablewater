import MDAnalysis
from newanalysis.correl import correlateParallel
from newanalysis.functions import atomsPerResidue, residueFirstAtom 
from newanalysis.functions import dipoleMomentByResidue
import numpy as np
import time
from datetime import timedelta
import gc

skip = 1

file_first = 1
file_last = 10

base = "/base_directory/"
base_directory = base

u = MDAnalysis.Universe(f"{base}hoh_500.psf",
            ["%s%d.dcd" % (base + "/traj/nvt_fine_", j) for j in range(file_first, file_last + 1)])


tot_mu = np.load(f"{base_directory}/analysis/tot_dipole_fine_matrix.npy")


boxlength = round(u.coord.dimensions[0], 4)
dt = round(u.trajectory.dt, 4)

n = int(u.trajectory.n_frames/skip)
if u.trajectory.n_frames%skip != 0:
    n+=1

tmp = u.select_atoms("all")

tmp_number = tmp.n_residues
mass = tmp.masses
charge = tmp.charges
apr = atomsPerResidue(tmp) # needed as kwargs for centerOfMassByresidue, dipoleMomentByResidue
rfa = residueFirstAtom(tmp) # needed as kwargs for centerOfMassByresidue, dipoleMomentByResidue
mu = np.zeros((n,tmp_number*3))

counter = 0
start=time.time()
print ("")

for ts in u.trajectory[::skip]:
    print("\033[1AFrame %d of %d (%4.1f%%)" % (ts.frame,u.trajectory.n_frames,ts.frame/u.trajectory.n_frames*100),
        "      Elapsed time: %s" % str(timedelta(seconds=(time.time()-start)))[:7],
        "      Est. time left: %s" % (str(timedelta(seconds=(time.time()-start)/(ts.frame+1) * (u.trajectory.n_frames-ts.frame))))[:7])
    
    coor = np.ascontiguousarray(tmp.positions, dtype='double') #array shape needed for C
    com = tmp.center_of_mass(compound='residues')
    mu[counter] = dipoleMomentByResidue(tmp, coor=coor,charges=charge,masses=mass,com=com, apr=apr, rfa=rfa, axis=0).flatten()

    counter += 1

mu_tot = np.add(mu, tot_mu)

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

del mu_tot
del tot_mu
gc.collect()


perm_mu = np.load(f"{base_directory}/analysis/perm_dipole_fine_matrix.npy")
mu_perm = np.add(mu, perm_mu)

del mu
del perm_mu
gc.collect()

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


mu_ind = np.load(f"{base_directory}/analysis/ind_dipole_fine_matrix.npy")

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
