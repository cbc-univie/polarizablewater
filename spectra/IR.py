import numpy as np
import sys

print("~"*120)
print("Laplace transform of < dmd/dt(0) * dmd/dt(t) >")
print("~"*120)


############################################################################################################################################################
# Laplace transform
############################################################################################################################################################
# zero padding does not change
def laplacetransform(time,corfun):
    print('>\tLaplace transform')
    xlen = len(time)
    #x = time
    #y = corfun
    x = np.zeros(2*xlen-1)
    y = np.zeros(2*xlen-1)
    x[0:xlen]  = -time[::-1]
    x[xlen-1:] = time
    y[xlen-1:] = corfun
    
    # simple Fourier transform    
    dt = x[1]-x[0]
    freq = 1./dt
    shifty = np.fft.fftshift(y)
    fourier = np.fft.fft(shifty)/freq
    
    N = int(len(fourier)/2)+1
    
    frequencies = np.linspace(0,freq/2,N,endpoint=True)
    
    return frequencies, fourier


############################################################################################################################################################
# Apodization = Smoothing
############################################################################################################################################################
def apodization(alpha,x,y):
    # J. Non.-Cryst. Solids (1992), 140, 350
    # Phys. Rev. Lett. (1996), 77, 4023
    print('>\tApodization of correlation function with alpha = %10.5f'%alpha)
    
    xlen = len(x)
    ysmooth = np.zeros(xlen)
    
    for i in range(xlen):
        # apodization
        ysmooth[i] = y[i] * np.exp(-alpha*x[i]*x[i])
    return ysmooth


############################################################################################################################################################
# maxperiod, e.g. first 100 ps of the trajectory corresponds to 0.2 cm^-1 IR resolution
############################################################################################################################################################
def maxperiod(x,y,xmax):
    print('>\tMaximum simulation period = %8.1f ps'%xmax)
    new_x = []
    new_y = []
    xlen = len(x)

    for i in range(xlen):
        new_x.append(x)
        new_y.append(y)
        if x[i] >xmax:
            break
    return new_x, new_y


############################################################################################################################################################
# Prefactor
############################################################################################################################################################
def quantum_correction(frequencies,spectrum):
    print("\tApplying quantum correction")

    # beta = 1. / (1.3806 10^-23 J/K * 300 K )
    # hbar = 1.0545718*10^-34 J/s
    # factor = beta * hbar ( 10^12 ps/s)
    beta_hbar = 0.0254617
    for i in range(len(frequencies)):
        qc = beta_hbar * frequencies[i] / (1.0 - np.exp(-beta_hbar*frequencies[i]))
        spectrum[i] *= qc
    return spectrum

############################################################################################################################################################
# Baseline correction
############################################################################################################################################################
def baseline(fourier):
    print('>\tBaseline correction')
    min = np.min(spectrum)
    return spectrum -min


############################################################################################################################################################
def find_first_index(array, threshold):
    mask = array > threshold
    indices = np.where(mask)[0]
    if len(indices) > 0:
        return indices[0]
    return -1  # If no value is larger than the threshold


############################################################################################################################################################
# Main program
############################################################################################################################################################

# correlation function
mudot_mudot = np.loadtxt(sys.argv[1])
time   = mudot_mudot[:,0]
corfun = mudot_mudot[:,1]

# apodization = gaussian smoothing
alpha = 0.1
corfun = apodization(alpha,time,corfun)


# Laplace transform
frequencies, fourier = laplacetransform(time,corfun)


THz2cm = 33.356

minfreq = find_first_index(frequencies,0./THz2cm)
maxfreq = find_first_index(frequencies,1600./THz2cm)
frequencies = frequencies[minfreq:maxfreq]
spectrum = np.real(fourier[minfreq:maxfreq])
#spectrum = spectrum/frequencies brauchen wir nicht da später mit omega/c multipliziert wird
max_spectrum = np.max(spectrum)
#spectrum /= max_spectrum


# Baseline correction
#spectrum = baseline(spectrum)


# Quantum corrections
#spectrum = quantum_correction(frequencies,spectrum)


output = sys.argv[1].split(".")[0]+".IR"
print("\n\tWriting %s"%output) 
f1 = open(output,'w')
for j in range(len(spectrum)):
    current_frequency = frequencies[j]*THz2cm
    f1.write("%8.1f %12.8f\n" %(current_frequency,spectrum[j]))
f1.close()
