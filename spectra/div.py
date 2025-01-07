import numpy as np
import sys

file = sys.argv[1]

t, C = np.loadtxt(f"{file}", unpack =True)

timescale = t[1]-t[0]
J = -np.gradient(np.gradient(C)/timescale)/timescale

np.savetxt(f"gradient_{file}", np.c_[t,J])
