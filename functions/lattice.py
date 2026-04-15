import numpy as np



def square_lattice_qpqp(n):
    """
    Generate generator matrix for square GKP lattice
    ordering: (qpqp)
    """

    I = np.eye(n)

    M = np.block([
        [np.sqrt(2)*I, np.zeros((n,n))],
        [np.zeros((n,n)), np.sqrt(2)*I]
    ])

    return M

def Omega_qpqp(n):
    """
    Omega = I_N ⊗ [[0,1],[-1,0]]

    Ordering:
    (q1,p1,q2,p2,...)
    """

    omega = np.array([
        [0, 1],
        [-1, 0]
    ])

    return np.kron(np.eye(n), omega)





def Omega_qqpp(n):
    """
    Omega for ordering:
    (q1,...,qn,p1,...,pn)
    """

    I = np.eye(n)
    Z = np.zeros((n, n))

    Omega = np.block([
        [Z, I],
        [-I, Z]
    ])

    return Omega



def qqpp_to_qpqp_permutation(n):
    """
    Permutation matrix converting:

    (q1,...,qn,p1,...,pn)
        →
    (q1,p1,q2,p2,...)
    """

    P = np.zeros((2*n, 2*n))

    for i in range(n):

        P[2*i, i] = 1
        P[2*i+1, n+i] = 1

    return P

def qpqp_to_qqpp_permutation(n):
    return np.linalg.inv(qqpp_to_qpqp_permutation(n))



def dual_lattice(M, ordering="qqpp"):

    n = M.shape[0] // 2

    if ordering == "qqpp":
        Om = Omega_qqpp(n)
    else:
        Om = Omega_qpqp(n)

    A = M @ Om @ M.T

    return Om @ np.linalg.inv(A) @ M


def symplectic_checker(S, ordering="qpqp", tol=1e-9):
    """
    Check if matrix S is symplectic.

    ordering:
        "qpqp" or "qqpp"
    """

    n = S.shape[0] // 2

    if ordering == "qpqp":
        Om = Omega_qpqp(n)

    elif ordering == "qqpp":
        Om = Omega_qqpp(n)

    else:
        raise ValueError("ordering must be 'qpqp' or 'qqpp'")

    return np.allclose(S @ Om @ S.T, Om, atol=tol)




def symplectic_error(S, ordering="qpqp"):
    """
    Return Frobenius norm of symplectic violation.
    """

    n = S.shape[0] // 2

    if ordering == "qpqp":
        Om = Omega_qpqp(n)
    else:
        Om = Omega_qqpp(n)

    diff = S @ Om @ S.T - Om

    return np.linalg.norm(diff)

