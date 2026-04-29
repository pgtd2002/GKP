import numpy as np

def repetition_code_XXZZ(number_of_qubit,pauli: str):
    stabs = []
    n=number_of_qubit
    for i in range(n - 1):

        x = [0] * n
        z = [0] * n

        if pauli == 'X':

            x[i] = 1
            x[i + 1] = 1

        elif pauli == 'Z':

            z[i] = 1
            z[i + 1] = 1

        elif pauli == 'Y':

            x[i] = 1
            x[i + 1] = 1
            z[i] = 1
            z[i + 1] = 1

        stabs.append(x + z)

    return np.array(stabs)



shor_code_913=np.array([
        # Phase-flip stabilizers (X-type)
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X1 X2 X3 X4 X5 X6
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X4 X5 X6 X7 X8 X9

        # Bit-flip stabilizers (Z-type)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Z1 Z2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # Z2 Z3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Z4 Z5
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Z5 Z6
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # Z7 Z8
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # Z8 Z9
    ],
        dtype=int,
    )

import numpy as np

import numpy as np

def steane_code_713():

    G = np.array([

        # X-type stabilizers
        [0,0,0,1,0,1,1,  0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,  0,0,0,0,0,0,0],
        [0,0,0,1,1,0,1,  0,0,0,0,0,0,0],

        # Z-type stabilizers
        [0,0,0,0,0,0,0,  1,0,0,1,0,1,1],
        [0,0,0,0,0,0,0,  0,1,0,1,1,0,1],
        [0,0,0,0,0,0,0,  0,0,1,0,1,1,1],

    ], dtype=int)

    return G


import numpy as np


def rotated_surface_code_stabilizers(d):
    """
    Generate rotated surface code stabilizers for distance d.

    Returns
    -------
    Hx : ndarray
        X stabilizer matrix (binary)
    Hz : ndarray
        Z stabilizer matrix (binary)
    """

    if d % 2 == 0:
        raise ValueError("Distance d must be odd for rotated surface code")

    n = d * d

    Hx_rows = []
    Hz_rows = []

    # map (i,j) -> qubit index
    def q(i, j):
        return i * d + j

    for i in range(d):
        for j in range(d):

            # checkerboard pattern
            if (i + j) % 2 == 0:

                qubits = []

                # neighbors (up/down/left/right)
                if i > 0:
                    qubits.append(q(i - 1, j))
                if i < d - 1:
                    qubits.append(q(i + 1, j))
                if j > 0:
                    qubits.append(q(i, j - 1))
                if j < d - 1:
                    qubits.append(q(i, j + 1))

                if len(qubits) >= 2:

                    row = np.zeros(n, dtype=int)

                    for qubit in qubits:
                        row[qubit] = 1

                    Hx_rows.append(row)

            else:

                qubits = []

                if i > 0:
                    qubits.append(q(i - 1, j))
                if i < d - 1:
                    qubits.append(q(i + 1, j))
                if j > 0:
                    qubits.append(q(i, j - 1))
                if j < d - 1:
                    qubits.append(q(i, j + 1))

                if len(qubits) >= 2:

                    row = np.zeros(n, dtype=int)

                    for qubit in qubits:
                        row[qubit] = 1

                    Hz_rows.append(row)

    Hx = np.array(Hx_rows, dtype=int)
    Hz = np.array(Hz_rows, dtype=int)

    ###Stacking
    mx, n = Hx.shape
    mz, _ = Hz.shape

    top = np.hstack([Hx, np.zeros((mx, n), dtype=int)])
    bottom = np.hstack([np.zeros((mz, n), dtype=int), Hz])

    return np.vstack([top, bottom])