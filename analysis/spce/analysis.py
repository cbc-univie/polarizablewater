import MDAnalysis
from newanalysis.helpers import dipoleMomentNeutralSelection
from newanalysis.correl import correlateParallel
from newanalysis.miscellaneous import atomicIndDip, deriveIndDip
import numpy as np
import copy
import time
from datetime import timedelta

skip = 1
cutoff = 10000 #time at which trajectory is cutoff

file_first = 2
file_last = 2

n_max = cutoff
n_min = cutoff
n     = cutoff

result_thz_tot                  = np.zeros((n_max,))
result_thz_translation          = np.zeros((n_max,))
result_thz_rotation             = np.zeros((n_max,))
result_thz_translation_rotation = np.zeros((n_max,))

base = "/base_directory/"

u = MDAnalysis.Universe(f"{base}/spce_4000.psf",
            ["%s%d.dcd" % (base + "/traj/nvt_fine_", j) for j in range(file_first, file_last + 1)])
v = MDAnalysis.Universe(f"{base}/spce_4000.psf",
            ["%s%d.dcd" % (base + "/traj/nvt_fine_vel_", j) for j in range(file_first, file_last + 1)])

boxlength = round(u.coord.dimensions[0], 4)
deltatime = round(u.trajectory.dt, 4)

sel_wat_u = u.select_atoms("resname SPCE")
sel_wat_v = v.select_atoms("resname SPCE")

sel_wat_ind_u = u.select_atoms("resname SPCE and name O*")
sel_wat_ind_v = v.select_atoms("resname SPCE and name O*")

n_wat        = sel_wat_u.n_residues
nat_wat      = sel_wat_u.n_atoms
nat_wat_ind  = sel_wat_ind_u.n_atoms
charge_wat   = sel_wat_u.charges
alpha_wat    = np.ones(nat_wat_ind) 
alpha_wat[:] = 1.14 

timeseries_tot = np.zeros((n, 3))
timeseries_inddip = np.zeros((n, nat_wat_ind, 3))
timeseries_inddip_derived = np.zeros((n, nat_wat_ind, 3))
timeseries_dip = np.zeros((n, 3))

counter = 0
start=time.time()
print ("")

for ts in u.trajectory[:cutoff:skip]:
    print("\033[1AFrame %d of %d (%4.1f%%)" % (ts.frame,u.trajectory.n_frames,ts.frame/u.trajectory.n_frames*100),
        "      Elapsed time: %s" % str(timedelta(seconds=(time.time()-start)))[:7],
        "      Est. time left: %s" % (str(timedelta(seconds=(time.time()-start)/(ts.frame+1) * (u.trajectory.n_frames-ts.frame))))[:7])

    coor_wat = np.ascontiguousarray(sel_wat_u.positions, dtype='double')
    timeseries_dip[counter] = dipoleMomentNeutralSelection(coor_wat, charge_wat)

    v.trajectory[ts.frame-1]

    coor_wat = np.ascontiguousarray(sel_wat_u.positions, dtype='double')
    vel_wat  = np.ascontiguousarray(sel_wat_v.positions, dtype='double')

    coor_wat_ind = np.ascontiguousarray(sel_wat_ind_u.positions, dtype='double')
    vel_wat_ind  = np.ascontiguousarray(sel_wat_ind_v.positions, dtype='double')

    prev_ind_dipoles = copy.deepcopy(timeseries_inddip[counter-1,:,:])

    atomicIndDip(coor_wat, charge_wat, alpha_wat, coor_wat_ind,
          prev_ind_dipoles, timeseries_inddip[counter], boxlength)
    deriveIndDip(coor_wat_ind, vel_wat_ind, timeseries_inddip[counter],
          timeseries_inddip_derived[counter], boxlength)

    counter += 1

timeseries_sum_derived = np.sum(timeseries_inddip_derived, axis = 1)
timeseries_tot         = timeseries_dip + timeseries_sum_derived

np.savetxt("results/short_tot.ser",   timeseries_tot)
np.savetxt("results/short_trans.ser", timeseries_sum_derived)
np.savetxt("results/short_rot.ser",   timeseries_dip)
