import numpy as np
import scipy

'''
prefactor
    1         4 pi
-------- * ---------
4 pi e_0    3 V k T
'''
k = 8.61733326 #10^(-5) eV/K
e_0 = 8.85     #10^(-12) As/Vm
elementary_charge = 1.6022 #10^(-19) As
c = 0.0299792 # speed of light in cm / ps
T = 300 #K
V = 120256.89915812826 #in angstrom^3
beta = 1/(k*T)
prefactor =  elementary_charge * 1.E8 / (3 * V * k * T * e_0)
print(prefactor/c)

input_files = [ 
                #"gradient_cut_mu_mu_tot.IR",
                #"gradient_cut_mu_mu_ind.IR",
                #"gradient_cut_mu_mu_perm.IR",
                #"gradient_cut_mu_ind_mu_tot.IR",
                #"gradient_cut_mu_perm_mu_tot.IR",
                #"gradient_cut_mu_ind_mu_perm.IR",
                #"gradient_cut_mu_perm_mu_ind.IR",
                #"gradient_cut_cross_mu_mu_tot.IR",
                #"gradient_cut_cross_mu_mu_ind.IR",
                #"gradient_cut_cross_mu_mu_perm.IR",
                #"gradient_cut_cross_mu_ind_mu_tot.IR",
                #"gradient_cut_cross_mu_perm_mu_tot.IR",
                #"gradient_cut_cross_mu_ind_mu_perm.IR",
                #"gradient_cut_cross_mu_perm_mu_ind.IR",
                "gradient_cut_MU_MU_tot.IR",
                "gradient_cut_MU_MU_ind.IR",
                "gradient_cut_MU_MU_perm.IR",
                "gradient_cut_MU_ind_MU_tot.IR",
                "gradient_cut_MU_perm_MU_tot.IR",
                "gradient_cut_MU_ind_MU_perm.IR",
                "gradient_cut_MU_perm_MU_ind.IR",
                ]

output_files = [
                #"gradient_cut_mu_mu.thz",
                #"gradient_cut_mu_mu_ind.thz",
                #"gradient_cut_mu_mu_perm.thz",
                #"gradient_cut_mu_ind_mu_tot.thz",
                #"gradient_cut_mu_perm_mu_tot.thz",
                #"gradient_cut_mu_ind_mu_perm.thz",
                #"gradient_cut_mu_perm_mu_ind.thz",
                #"gradient_cut_cross_mu_mu.thz",
                #"gradient_cut_cross_mu_mu_ind.thz",
                #"gradient_cut_cross_mu_mu_perm.thz",
                #"gradient_cut_cross_mu_ind_mu_tot.thz",
                #"gradient_cut_cross_mu_perm_mu_tot.thz",
                #"gradient_cut_cross_mu_ind_mu_perm.thz",
                #"gradient_cut_cross_mu_perm_mu_ind.thz",
                "gradient_cut_MU_MU.thz",
                "gradient_cut_MU_MU_ind.thz",
                "gradient_cut_MU_MU_perm.thz",
                "gradient_cut_MU_ind_MU_tot.thz",
                "gradient_cut_MU_perm_MU_tot.thz",
                "gradient_cut_MU_ind_MU_perm.thz",
                "gradient_cut_MU_perm_MU_ind.thz",
                ]

n_files = len(input_files)

for f in range(n_files):
    omega, spectrum = np.loadtxt(input_files[f], unpack = True)
    spectrum *= prefactor/c 
    spectrum_filter = scipy.signal.savgol_filter(spectrum, 100, 1)
    print(len(spectrum))

    print(len(spectrum_filter))
    
    np.savetxt(output_files[f], np.c_[omega,spectrum])
    np.savetxt("filter_"+output_files[f],np.c_[omega,spectrum_filter])
