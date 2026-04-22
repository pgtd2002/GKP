from functions.lattice import *
import numpy as np


M_hex=hex_lattice_generator_n_modes(1)
M_sq=square_lattice_qpqp(1)
S_T= np.linalg.inv(M_sq) @ M_hex

hex_lattice = M_hex @ S_T

