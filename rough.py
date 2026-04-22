from functions.concatenation import build_concatenated_gkp_generator_qqpp
import numpy as np
from functions.binary_stabilizer_generators import shor_code_913
from functions.binary_stabilizer_generators import *
from functions.lattice import *
from functions.concatenation import *
S=steane_code_713()
G,j,col_perm=stabilizer_standard_form(S)
M,info=build_concatenated_gkp_generator_qqpp(S)

A= M @ Omega_qqpp(7) @ M.T
inverse_perm = np.argsort(col_perm)

G_original_order = G[:, inverse_perm]
print((A+A.T))