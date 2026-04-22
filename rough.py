from functions.concatenation import build_concatenated_gkp_generator_qqpp
import numpy as np
from functions.binary_stabilizer_generators import shor_code_913
from functions.binary_stabilizer_generators import repetition_code_XXZZ
from functions.lattice import *

x=repetition_code_XXZZ(2, 'X')
y=[x,[0,0,0,0]]
z,l=build_concatenated_gkp_generator_qqpp(x)
m=qpqp_to_qqpp_permutation(2)
print(l)
print(x)
print(z)
