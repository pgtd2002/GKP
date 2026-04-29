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





def get_rotated_surface_code_matrix(distance: int) -> np.ndarray:
    """
    Generates the binary parity-check matrix for a rotated surface code
    as the direct sum of the X-stabilizer and Z-stabilizer matrices.

    Args:
        distance (int): The distance of the surface code (must be an odd integer).

    Returns:
        np.ndarray: A binary matrix of shape ((d^2 - 1), 2 * d^2) representing
                    the direct sum [ [H_X, 0], [0, H_Z] ].
    """
    if distance % 2 == 0 or distance < 3:
        raise ValueError("Rotated surface codes typically require an odd distance >= 3.")

    d = distance
    n = d * d  # Total number of data qubits

    # Map (row, col) coordinates to an integer qubit ID (0 to d^2 - 1)
    def get_id(r, c):
        return r * d + c

    x_stabilizers = []
    z_stabilizers = []

    # 1. Bulk Stabilizers (Weight-4)
    for r in range(d - 1):
        for c in range(d - 1):
            qubits = [get_id(r, c), get_id(r, c + 1), get_id(r + 1, c), get_id(r + 1, c + 1)]
            if (r + c) % 2 == 0:
                x_stabilizers.append(qubits)
            else:
                z_stabilizers.append(qubits)

    # 2. Boundary Stabilizers (Weight-2)
    for i in range(d - 1):
        # Left boundary X (column 0)
        if i % 2 != 0:
            x_stabilizers.append([get_id(i, 0), get_id(i + 1, 0)])

        # Right boundary X (column d-1)
        if i % 2 == 0:
            x_stabilizers.append([get_id(i, d - 1), get_id(i + 1, d - 1)])

        # Top boundary Z (row 0)
        if i % 2 == 0:
            z_stabilizers.append([get_id(0, i), get_id(0, i + 1)])

        # Bottom boundary Z (row d-1)
        if i % 2 != 0:
            z_stabilizers.append([get_id(d - 1, i), get_id(d - 1, i + 1)])

    # 3. Construct Binary Sub-Matrices (H_X and H_Z)
    num_x = len(x_stabilizers)
    num_z = len(z_stabilizers)

    H_X = np.zeros((num_x, n), dtype=int)
    H_Z = np.zeros((num_z, n), dtype=int)

    for i, stab in enumerate(x_stabilizers):
        for q in stab:
            H_X[i, q] = 1

    for i, stab in enumerate(z_stabilizers):
        for q in stab:
            H_Z[i, q] = 1

    # 4. Construct the Direct Sum: H = H_X ⊕ H_Z
    # Structure: [ H_X   0  ]
    #            [  0   H_Z ]
    top_block = np.hstack((H_X, np.zeros((num_x, n), dtype=int)))
    bottom_block = np.hstack((np.zeros((num_z, n), dtype=int), H_Z))

    H = np.vstack((top_block, bottom_block))

    return H






