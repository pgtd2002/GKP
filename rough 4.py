import numpy as np
from sympy import Matrix
from scipy.io import loadmat
from functions.concatenation import *
from functions.binary_stabilizer_generators import *
import pandas as pd

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



#S=steane_code_713()
# S=shor_code_913
M,info=build_concatenated_gkp_generator_qqpp(S)
det=np.linalg.det(M)
enc_dim=np.log2(det)

from openpyxl import Workbook
import numpy as np

def export_matrix_to_excel(M, filename="matrix.xlsx", sheet_name="Matrix"):
    """
    Export a matrix to an Excel (.xlsx) file.

    Parameters
    ----------
    M : ndarray or list
        Matrix to export
    filename : str
        Output file name
    sheet_name : str
        Excel sheet name
    """

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    M = np.array(M)

    for row in M:
        ws.append(row.tolist())

    wb.save(filename)

    print(f"Matrix exported to {filename}")

print(info)



