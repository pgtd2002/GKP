from functions.concatenation import build_concatenated_gkp_generator_qqpp
import numpy as np
from functions.binary_stabilizer_generators import shor_code_913
from functions.binary_stabilizer_generators import *
from functions.lattice import *

x=repetition_code_XXZZ(2,'X')
y=[x,[0,0,0,0]]
z,l=build_concatenated_gkp_generator_qqpp(x)

print(x)

print(l)

print(z*np.sqrt(2))