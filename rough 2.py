import numpy as np
import sympy as sp
from functions.lattice import *
from functions.concatenation import *
from functions.binary_stabilizer_generators import *

shor=shor_code_913

M,info= build_concatenated_gkp_generator_qqpp(shor)
print(info)
# N=info["N"]
# k=info["k"]
# r=info["r"]
#
#
# I_rr=M[:r,:r]
# A1=M[:r,r:N-k]

M=np.sqrt(2)*M

def extract_concatenated_blocks(M, info):

    N = info["N"]
    k = info["k"]
    r = info["r"]

    n2 = N - k - r

    # -------------------------
    # Column boundaries
    # -------------------------

    c1 = 0
    c2 = r
    c3 = r + n2
    c4 = N
    c5 = N + r
    c6 = N + r + n2
    c7 = 2 * N

    # -------------------------
    # Row boundaries
    # -------------------------

    r1 = 0
    r2 = r
    r3 = r + n2
    r4 = r + n2 + k
    r5 = r + n2 + k + r
    r6 = r + n2 + k + r + n2
    r7 = 2 * N

    # =====================================================
    # First row block
    # [ I  A1  A2 | B  0  C ]
    # =====================================================

    I_rr = M[r1:r2, c1:c2]

    A1 = M[r1:r2, c2:c3]

    A2 = M[r1:r2, c3:c4]

    B = M[r1:r2, c4:c5]

    Zero_r_n2 = M[r1:r2, c5:c6]

    C = M[r1:r2, c6:c7]

    # =====================================================
    # Second row block
    # [ 0  0  0 | D  I  E ]
    # =====================================================

    Zero_n2_r = M[r2:r3, c1:c2]

    Zero_n2_n2 = M[r2:r3, c2:c3]

    Zero_n2_k = M[r2:r3, c3:c4]

    D = M[r2:r3, c4:c5]

    I_n2 = M[r2:r3, c5:c6]

    E = M[r2:r3, c6:c7]

    # =====================================================
    # 2I blocks
    # =====================================================

    TwoI_k_q = M[r3:r4, c3:c4]

    TwoI_r_p = M[r4:r5, c4:c5]

    TwoI_n2_q = M[r5:r6, c2:c3]

    TwoI_k_p = M[r6:r7, c6:c7]

    return dict(

        I_rr=I_rr,
        A1=A1,
        A2=A2,
        B=B,
        C=C,

        D=D,
        E=E,

        I_n2=I_n2,

        TwoI_k_q=TwoI_k_q,
        TwoI_r_p=TwoI_r_p,
        TwoI_n2_q=TwoI_n2_q,
        TwoI_k_p=TwoI_k_p,

    )

mat=extract_concatenated_blocks(M,info)
print(mat["I_rr"])
print(mat["A1"])
print(mat["A2"])
print(mat["B"])
print(mat["C"])
print(mat["D"])
print(mat["E"])
print(mat["I_n2"])

