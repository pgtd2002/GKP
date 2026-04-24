from functions.binary_stabilizer_generators import *
from functions.lattice import *
from functions.concatenation import *

S=repetition_code_XXZZ(3, 'X')
M,info=build_concatenated_gkp_generator_qqpp(S)

N=square_lattice_qpqp(3)
pc=qpqp_to_qqpp_permutation(3) @ N
print(abs(int((np.linalg.det(pc)))))

print(M*np.sqrt(2))
print(np.linalg.det(M))
print(info)