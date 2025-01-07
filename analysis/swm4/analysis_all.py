import MDAnalysis
from newanalysis.correl import correlateParallel
from newanalysis.functions import atomsPerResidue, residueFirstAtom 
from newanalysis.functions import dipoleMomentByResidue
import numpy as np
import time
from datetime import timedelta

skip = 2

file_first = 1
file_last = 1

base = "/base_directory/"

u = MDAnalysis.Universe(f"{base}/swm4_4000.psf",
            ["%s%d.dcd" % (base + "/traj/nvt_fine_", j) for j in range(file_first, file_last + 1)])

boxlength = round(u.coord.dimensions[0], 4)
dt = round(u.trajectory.dt, 5)

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


sel_wat_o = u.select_atoms("name OH2")
sel_wat_d = u.select_atoms("name DOH2")
d_charge  = sel_wat_d.charges
n_mols    = sel_wat_o.n_residues
print(n_mols)
print(tmp_number)

mu_ind = np.zeros((n,n_mols*3))
mu_perm = np.zeros((n,n_mols*3))

counter = 0
start=time.time()
print ("")

for ts in u.trajectory[::skip]:
    print("\033[1AFrame %d of %d (%4.1f%%)" % (ts.frame,u.trajectory.n_frames,ts.frame/u.trajectory.n_frames*100),
        "      Elapsed time: %s" % str(timedelta(seconds=(time.time()-start)))[:7],
        "      Est. time left: %s" % (str(timedelta(seconds=(time.time()-start)/(ts.frame+1) * (u.trajectory.n_frames-ts.frame))))[:7])

    coor_wat = np.ascontiguousarray(sel_wat_d.positions-sel_wat_o.positions, dtype='double')
    coor = np.ascontiguousarray(tmp.positions, dtype='double') #array shape needed for C
    com = tmp.center_of_mass(compound='residues')
    mu[counter] = dipoleMomentByResidue(tmp, coor=coor,charges=charge,masses=mass,com=com, apr=apr, rfa=rfa, axis=0).flatten()
    mu_ind[counter]   = (coor_wat * d_charge[:, None]).flatten()
    mu_perm[counter] = mu[counter] - mu_ind[counter]

    counter += 1


time = np.linspace(0.0,n*dt*skip,n)
np.savetxt("mu_ind.ts", np.c_[time, mu_ind])
np.savetxt("mu_perm.ts", np.c_[time, mu_perm])
np.savetxt("mu.ts", np.c_[time, mu])

out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu.T),
                     np.ascontiguousarray(mu.T),
                  out, ltc = 0)

f1 = open('mu_mu_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out[j]))
f1.close()

mu_sum = np.zeros((n,3))
for i in range(len(mu[0])//3):
    mu_sum[:,0] += mu[:,3*i]
    mu_sum[:,1] += mu[:,3*i+1]
    mu_sum[:,2] += mu[:,3*i+2]


out2 = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum.T),
                     np.ascontiguousarray(mu_sum.T),
                  out2, ltc = 0)

f1 = open('MU_MU_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out2[j]))
f1.close()

out3 = out2-out

f1 = open('cross_mu_mu_tot.dat', "w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out3[j]))
f1.close()


out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_ind.T),
                     np.ascontiguousarray(mu_ind.T),
                  out, ltc = 0)

f1 = open('mu_mu_ind.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out[j]))
f1.close()

mu_sum_ind = np.zeros((n,3))
for i in range(len(mu_ind[0])//3):
    mu_sum_ind[:,0] += mu_ind[:,3*i]
    mu_sum_ind[:,1] += mu_ind[:,3*i+1]
    mu_sum_ind[:,2] += mu_ind[:,3*i+2]


out2 = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_ind.T),
                     np.ascontiguousarray(mu_sum_ind.T),
                  out2, ltc = 0)

f1 = open('MU_MU_ind.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out2[j]))
f1.close()

out3=out2-out

f1 = open('cross_mu_mu_ind.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out3[j]))
f1.close()


out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_perm.T),
                     np.ascontiguousarray(mu_perm.T),
                  out, ltc = 0)

f1 = open('mu_mu_perm.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out[j]))
f1.close()

mu_sum_perm = np.zeros((n,3))
for i in range(len(mu_ind[0])//3):
    mu_sum_perm[:,0] += mu_perm[:,3*i]
    mu_sum_perm[:,1] += mu_perm[:,3*i+1]
    mu_sum_perm[:,2] += mu_perm[:,3*i+2]


out2 = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_perm.T),
                     np.ascontiguousarray(mu_sum_perm.T),
                  out2, ltc = 0)

f1 = open('MU_MU_perm.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out2[j]))
f1.close()

out3=out2-out

f1 = open('cross_mu_mu_perm.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out3[j]))
f1.close()



out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_perm.T),
                     np.ascontiguousarray(mu_ind.T),
                  out, ltc = 0)

f1 = open('mu_perm_mu_ind.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out[j]))
f1.close()

out2 = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_perm.T),
                     np.ascontiguousarray(mu_sum_ind.T),
                  out2, ltc = 0)

f1 = open('MU_perm_MU_ind.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out2[j]))
f1.close()

out3=out2-out

f1 = open('cross_mu_perm_mu_ind.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out3[j]))
f1.close()


out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_ind.T),
                     np.ascontiguousarray(mu_perm.T),
                  out, ltc = 0)

f1 = open('mu_ind_mu_perm.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out[j]))
f1.close()

out2 = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_ind.T),
                     np.ascontiguousarray(mu_sum_perm.T),
                  out2, ltc = 0)

f1 = open('MU_ind_MU_perm.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out2[j]))
f1.close()

out3=out2-out

f1 = open('cross_mu_ind_mu_perm.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out3[j]))
f1.close()



out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_ind.T),
                     np.ascontiguousarray(mu.T),
                  out, ltc = 0)

f1 = open('mu_ind_mu_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out[j]))
f1.close()

out2 = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_ind.T),
                     np.ascontiguousarray(mu_sum.T),
                  out2, ltc = 0)

f1 = open('MU_ind_MU_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out2[j]))
f1.close()

out3=out2-out

f1 = open('cross_mu_ind_mu_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out3[j]))
f1.close()


out = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_perm.T),
                     np.ascontiguousarray(mu.T),
                  out, ltc = 0)

f1 = open('mu_perm_mu_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out[j]))
f1.close()

out2 = np.zeros((n,))
correlateParallel(np.ascontiguousarray(mu_sum_perm.T),
                     np.ascontiguousarray(mu_sum.T),
                  out2, ltc = 0)

f1 = open('MU_perm_MU_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out2[j]))
f1.close()

out3=out2-out

f1 = open('cross_mu_perm_mu_tot.dat',"w")
for j in range(n):
    f1.write("%10.5f %10.5f\n" % (j*dt*skip, out3[j]))
f1.close()
