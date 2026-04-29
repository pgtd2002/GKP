import numpy as np
from sympy import Matrix
from scipy.io import loadmat
from functions.concatenation import *
from functions.binary_stabilizer_generators import *
from decoders.surface_decoder import *
import pandas as pd

#------------------------------
#this part imports the Stabilizer matrix
#------------------------------



# Load the .mat file
data = loadmat("g4_d7.mat")

# Extract parity-check matrices
Hx = data["Hx"] % 2
Hz = data["Hz"] % 2

# Number of physical qubits
n = Hx.shape[1]

# Zero blocks
Zx = np.zeros((Hx.shape[0], n), dtype=int)
Zz = np.zeros((Hz.shape[0], n), dtype=int)

# Direct sum to form stabilizer generator matrix
S = np.block([
    [Hx, Zx],
    [Zz, Hz]
]) % 2





####################
#------------------------------
#this part rearranges the stab matrix in X and Z block (Stabilizer Standard Form)
#------------------------------
G,r,info=stabilizer_standard_form(S)



