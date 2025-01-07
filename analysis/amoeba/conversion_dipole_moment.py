import numpy as np
import gc

file_first = 1
file_last = 6

n_res = 500 #number_of_residues
apr= 3  #atoms_per_residue

#convolute the dipole moment arrays: maybe there is an easier way to convolute the dipole moments arrays?
#takes way too much data 

perm_dipole_list = []

for j in range(file_first, file_last + 1):
    perm_dipole_j = np.loadtxt(f"permanent_dipoles_fine_{j}.dat")
    perm_dipole_list.append(perm_dipole_j)
    
perm_dipole = np.vstack(perm_dipole_list)

timesteps_per_file = int(len(perm_dipole)/(file_last-file_first+1)/apr/n_res)

#actually builds a 3D array taking 3 elements in a row from the 2D array to build a 3rd axis and then takes the sum of this axis to get a 2D array again
#this step only works for pure substances or if the number of atoms per residue is the same. Otherwise some different calculations must be considered

perm_dipole = perm_dipole.reshape(-1,apr,perm_dipole.shape[-1]).sum(1)

#flattens the 2D array to a 1D array

perm_dipole = perm_dipole.flatten()

#reshape 1D array to a 2D array in the form needed for mdanalysis
#replace 2000 with number of timesteps and 4000 with n_molecules
'''
shape needed:
+--+--+--+--+--+--+--+---+--+--+--+
|  |x1|y1|z1|x2|y2|z2|...|xn|yn|zn|
+==+==+==+==+==+==+==+===+==+==+==+
|t1|  |  |  |  |  |  |   |  |  |  |
+--+--+--+--+--+--+--+---+--+--+--+
|t2|  |  |  |  |  |  |   |  |  |  |
+--+--+--+--+--+--+--+---+--+--+--+
|. |  |  |  |  |  |  |   |  |  |  | 
+--+--+--+--+--+--+--+---+--+--+--+
|. |  |  |  |  |  |  |   |  |  |  |
+--+--+--+--+--+--+--+---+--+--+--+
|. |  |  |  |  |  |  |   |  |  |  |
+--+--+--+--+--+--+--+---+--+--+--+
|tm|  |  |  |  |  |  |   |  |  |  |
+--+--+--+--+--+--+--+---+--+--+--+
 where x,y,z are coordinate axes, n is n_molecules, m is n_timesteps
'''
perm_dipole = np.reshape(perm_dipole, (timesteps_per_file*(file_last-file_first+1),n_res*3))

#openmm uses units qe * nm, mdaanlysis qe * angstrom

perm_dipole = perm_dipole*10

#save as binary file to increase loading speed for next step.

np.save(f"perm_dipole_fine_matrix.npy", perm_dipole)

del perm_dipole
del perm_dipole_list

gc.collect()

#do the same for induced dipole moments

ind_dipole_list = []

for j in range(file_first, file_last + 1):
    ind_dipole_j = np.loadtxt(f"induced_dipoles_fine_{j}.dat")
    ind_dipole_list.append(ind_dipole_j)
    
ind_dipole = np.vstack(ind_dipole_list)

timesteps_per_file = int(len(ind_dipole)/(file_last-file_first+1)/apr/n_res)

ind_dipole = ind_dipole.reshape(-1,apr,ind_dipole.shape[-1]).sum(1) 

ind_dipole = ind_dipole.flatten()

ind_dipole = np.reshape(ind_dipole, (timesteps_per_file*(file_last-file_first+1),n_res*3))

ind_dipole = ind_dipole*10

np.save(f"ind_dipole_fine_matrix.npy", ind_dipole)

del ind_dipole
del ind_dipole_list

gc.collect()

tot_dipole_list = []

for j in range(file_first, file_last + 1):
    tot_dipole_j = np.loadtxt(f"total_dipoles_fine_{j}.dat")
    tot_dipole_list.append(tot_dipole_j)

tot_dipole = np.vstack(tot_dipole_list)

timesteps_per_file = int(len(tot_dipole)/(file_last-file_first+1)/apr/n_res)

tot_dipole = tot_dipole.reshape(-1,apr,tot_dipole.shape[-1]).sum(1)

tot_dipole = tot_dipole.flatten()

tot_dipole = np.reshape(tot_dipole, (timesteps_per_file*(file_last-file_first+1),n_res*3))

tot_dipole = tot_dipole*10

np.save(f"tot_dipole_fine_matrix.npy", tot_dipole)

quit()
