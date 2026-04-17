import numpy as np

"""
Full construction of concatenated square GKP lattice generator matrix
from a binary stabilizer matrix G.

Implements Appendix A procedure:

1) Take stabilizer matrix G (binary, size (N-k) x 2N)
2) Bring G into standard form using Gaussian elimination over GF(2)
3) Construct the concatenated lattice generator matrix
   M_conc^(sq) in qqpp ordering

Coordinates are assumed to be ordered as:

    (q1, ..., qN, p1, ..., pN)

Arithmetic for stabilizer manipulation is over GF(2).
Lattice matrix is real-valued.
"""


# ============================================================
# GF(2) utilities
# ============================================================

def gf2_swap_rows(M, i, j):
    M[[i, j]] = M[[j, i]]


def gf2_swap_cols(M, i, j):
    M[:, [i, j]] = M[:, [j, i]]


def gf2_row_add(M, src, dst):
    M[dst] = (M[dst] + M[src]) % 2


# ============================================================
# Standard form conversion
# ============================================================

def stabilizer_standard_form(G):
    """
    Convert stabilizer matrix G into standard form

        [ I  A1  A2 | B  0  C ]
        [ 0   0   0 | D  I  E ]

    using Gaussian elimination over GF(2).

    Parameters
    ----------
    G : ndarray (binary)
        Shape (N-k, 2N)

    Returns
    -------
    G_std : ndarray
        Stabilizer matrix in standard form

    r : int
        Rank of X block

    col_perm : list
        Column permutation applied
    """

    G = G.copy() % 2

    m, total_cols = G.shape
    N = total_cols // 2

    col_perm = list(range(total_cols))

    # -----------------------------
    # Step 1: eliminate X block
    # -----------------------------

    r = 0

    for col in range(N):
        pivot_row = None

        for row in range(r, m):
            if G[row, col] == 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        if pivot_row != r:
            gf2_swap_rows(G, pivot_row, r)

        for row in range(m):
            if row != r and G[row, col] == 1:
                gf2_row_add(G, r, row)

        if col != r:
            gf2_swap_cols(G, col, r)
            col_perm[col], col_perm[r] = col_perm[r], col_perm[col]

        r += 1

        if r == m:
            break

    # -----------------------------
    # Step 2: eliminate Z block
    # -----------------------------

    z_start = N
    pivot_row = r

    for col in range(z_start, 2 * N):
        if pivot_row >= m:
            break

        row_found = None

        for row in range(pivot_row, m):
            if G[row, col] == 1:
                row_found = row
                break

        if row_found is None:
            continue

        if row_found != pivot_row:
            gf2_swap_rows(G, row_found, pivot_row)

        for row in range(m):
            if row != pivot_row and G[row, col] == 1:
                gf2_row_add(G, pivot_row, row)

        target_col = z_start + (pivot_row - r)

        if col != target_col:
            gf2_swap_cols(G, col, target_col)
            col_perm[col], col_perm[target_col] = (
                col_perm[target_col],
                col_perm[col],
            )

        pivot_row += 1

    return G, r, col_perm


# ============================================================
# Concatenated GKP lattice construction
# ============================================================

def build_concatenated_gkp_generator(G_binary):
    """
    Construct M_conc^(sq) from stabilizer matrix G.

    Parameters
    ----------
    G_binary : ndarray
        Binary stabilizer matrix (N-k x 2N)

    Returns
    -------
    M : ndarray (float)
        Lattice generator matrix

    info : dict
        Contains useful metadata
    """

    G_std, r, col_perm = stabilizer_standard_form(G_binary)

    m, total_cols = G_std.shape
    N = total_cols // 2

    k = N - m

    remaining = m - r

    # Column block sizes

    n1 = r
    n2 = remaining
    n3 = k

    total_rows = 2 * N

    M = np.zeros((total_rows, 2 * N), dtype=float)

    row = 0

    # ----------------------------------
    # Stabilizer rows
    # ----------------------------------

    for i in range(m):
        M[row] = G_std[i]
        row += 1

    # ----------------------------------
    # Remaining square GKP generators
    # ----------------------------------

    # logical q block

    for i in range(n3):
        col = n1 + n2 + i
        M[row, col] = 2
        row += 1

    # stabilizer p partners

    for i in range(n1):
        col = N + i
        M[row, col] = 2
        row += 1

    # remaining p partners

    for i in range(n2):
        col = N + n1 + i
        M[row, col] = 2
        row += 1

    # logical p block

    for i in range(n3):
        col = N + n1 + n2 + i
        M[row, col] = 2
        row += 1

    # Global normalization

    M = M / np.sqrt(2)

    info = {
        "N": N,
        "k": k,
        "num_stabilizers": m,
        "r": r,
        "column_permutation": col_perm,
    }

    return M, info
