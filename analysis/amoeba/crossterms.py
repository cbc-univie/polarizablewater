import numpy as np
import itertools

correlations = ["MU_MU_tot", "MU_MU_perm", "MU_MU_ind", "MU_perm_MU_tot", "MU_ind_MU_tot", "MU_perm_MU_ind", "MU_ind_MU_perm"]
autocorrelations = ["mu_mu_tot", "mu_mu_perm", "mu_mu_ind", "mu_perm_mu_tot", "mu_ind_mu_tot", "mu_perm_mu_ind", "mu_ind_mu_perm"]

for n_corr,corr in enumerate(correlations):
    w, c = np.genfromtxt(f"{corr}.dat", unpack = True)
    for n_auto, auto in enumerate(autocorrelations):
        if n_auto == n_corr:
            w_a, c_a = np.genfromtxt(f"{auto}.dat", unpack = True)
            c_c = c - c_a
            cross = np.vstack((w, c_c))
            np.savetxt(f"cross_{auto}.dat",cross.T)
