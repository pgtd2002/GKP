import numpy as np
from sympy import Matrix
from scipy.io import loadmat
from functions.concatenation import *
from functions.binary_stabilizer_generators import *

# Load the .mat file
data = loadmat("g3_d3.mat")

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



S=steane_code_713()
# S=shor_code_913
M,info=build_concatenated_gkp_generator_qqpp(S)
det=np.linalg.det(M)
enc_dim=np.log2(det)

print(info)
print(np.sqrt(2)*M)


import numpy as np
import matplotlib.pyplot as plt

def show_binary_matrix(M, title="Binary Matrix"):
    """
    Visualize a binary matrix using black/white colors.

    Parameters
    ----------
    M : ndarray
        Binary matrix (0/1)
    title : str
        Plot title
    """

    M = np.array(M) % 2  # ensure binary

    plt.figure(figsize=(8, 6))
    plt.imshow(M, cmap="gray_r", aspect="auto")

    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.show()


show_binary_matrix(np.sqrt(2)*M)

