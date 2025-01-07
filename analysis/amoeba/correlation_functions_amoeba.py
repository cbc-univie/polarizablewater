import MDAnalysis
from newanalysis.correl import correlateParallel
from newanalysis.functions import atomsPerResidue, residueFirstAtom # "apr", "rfa", needed for faster calculation of centerOfMassByResidue, dipoleMomentByResidue
from newanalysis.functions import centerOfMassByResidue, dipoleMomentByResidue
import numpy as np
import time
import sys
import json
import glob
import gc

############################################################################################################################################################
# Trajectory
############################################################################################################################################################
base_directory = "/base_directory/"
psf_file = base_directory + "hoh_500.psf"
dcd_file = base_directory + "traj/nvt_fine_"

perm_mu = np.load(f"{base_directory}/analysis/perm_dipole_fine_matrix.npy")
ind_mu = np.load(f"{base_directory}/analysis/ind_dipole_fine_matrix.npy")
tot_mu = np.load(f"{base_directory}/analysis/tot_dipole_fine_matrix.npy")

file_first = 1
file_last  = 6 #must be the same number of files which were used for the calculation of the dipole matrices
skip = 1 #if one it already fits the data from the dipole matrices, otherwise the matrices must be adjusted to take every nth line from the .npy


#take every nth line from the dipole matrices
#perm_mu = perm_mu[0::skip]
#ind_mu = ind_mu[0::skip]
#tot_mu = tot_mu[0::skip]
#note: loading the entire input data and then just using parts of it takes a lot of unnecessary RAM 
#so if the skip is very high it might be more useful to convert the data to smaller data files first to be able to compute longer time scales


print("psf: %s"%psf_file)
for j in range(file_first, file_last + 1):
    print("dcd: %s%d.dcd"% (dcd_file, j) )

print("\t skip: %s"%skip)

u=MDAnalysis.Universe(psf_file,
                      ["%s%d.dcd" % (dcd_file, j) for j in range(file_first, file_last + 1)]
                      )

boxl = np.float64(round(u.coord.dimensions[0],4))
dt=round(u.trajectory.dt,4) * skip

n = int(u.trajectory.n_frames/skip)
if u.trajectory.n_frames%skip != 0:
    n+=1


############################################################################################################################################################
# molecular species
############################################################################################################################################################
print('\n> Defining selections ...')

tmp = u.select_atoms("all")

tmp_number = tmp.n_residues
mass = tmp.masses
charge = tmp.charges
apr = atomsPerResidue(tmp)  # needed as kwargs for centerOfMassByresidue, dipoleMomentByResidue
rfa = residueFirstAtom(tmp) # needed as kwargs for centerOfMassByresidue, dipoleMomentByResidue
mu = np.zeros((n,tmp_number*3))

mu_ind = ind_mu
del ind_mu
gc.collect()

############################################################################################################################################################
# Reading trajectory
############################################################################################################################################################

ctr=0
start=time.time()
com = np.zeros((tmp_number,3))
print('\n')
for ts in u.trajectory[::skip]:
    print("\033[1A> Frame %d of %d" % (ts.frame,u.trajectory.n_frames), "\tElapsed time: %.2f hours" % ((time.time()-start)/3600))

    coor = np.ascontiguousarray(tmp.positions, dtype='double') #array shape needed for C
    com = tmp.center_of_mass(compound='residues')
    mu[ctr] = dipoleMomentByResidue(tmp, coor=coor,charges=charge,masses=mass,com=com, apr=apr, rfa=rfa, axis=0).flatten()
    #mu[i][ctr] = tmp.dipoleMomentByResidue(coor=coor,charges=charge[i],masses=mass[i],com=com,axis=0)
    ctr+=1

mu_tot = np.add(mu, tot_mu)

del tot_mu
gc.collect()

mu_perm = np.add(mu,perm_mu)
del perm_mu

del mu
gc.collect()

############################################################################################################################################################
# Calculating correlation functions
############################################################################################################################################################

def timeseries(n,filename,f):
    print("\tWriting timeseries: %s"%filename)
    nres = len(f[0]) // 3
    print("\tNumber of residues: %s"%nres)
    
    f1 = open(filename,"w")
    for j in range(n):
        f1.write("%.5e"%(j*dt))
        for ires in range(nres):
            for xyz in range(3):
                index = 3*ires + xyz
                f1.write(" %.5e"%f[j,index])
        f1.write("\n")    
    f1.close()

    
def correlation(n,filename,f):
    print("\tWriting correlation function: %s"%filename)
    out = np.zeros((n,))
    correlateParallel(f.T,f.T,out,ltc=0)
    
    f1 = open(filename,"w")
    for j in range(n):
        f1.write("%12.7f %12.7f\n" % (j*dt, out[j]))
    f1.close()

def crosscorrelation(n,filename,f1,f2):
    print("\tWriting crosscorrelation function: %s"%filename)
    out = np.zeros((n,))
    correlateParallel(f1.T,f2.T,out,ltc=0)

    f3 = open(filename,"w")
    for j in range(n):
        f3.write("%12.7f %12.7f\n" % (j*dt, out[j]))
    f3.close()


############################################################################################################################################################
# Post processing
############################################################################################################################################################
print("\nPost-processing ...")

# mu are the total dipole moments (perm + ind)
timeseries(n,"mu_tot.dat",mu_tot)
correlation(n,"mu_mu_tot.dat",mu_tot)

#mu_ind are the induced dipole moments
timeseries(n,"mu_ind.dat",mu_ind)
correlation(n,"mu_mu_ind.dat",mu_ind)

crosscorrelation(n,"mu_ind_mu_tot.dat",mu_ind, mu_tot)
crosscorrelation(n,"mu_ind_mu_perm.dat",mu_ind, mu_perm)

#mu_perm are the permanent dipole moments
timeseries(n,"mu_perm.dat",mu_perm)
correlation(n,"mu_mu_perm.dat",mu_perm)

crosscorrelation(n,"mu_perm_mu_tot.dat",mu_perm, mu_tot)
crosscorrelation(n,"mu_perm_mu_ind.dat",mu_perm, mu_ind)
