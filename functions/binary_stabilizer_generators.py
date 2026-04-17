import numpy as np

def repetition_code_XXZZ(number_of_qubit,pauli: str):
    stabs = np.array([])
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

    return stabs



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

