import numpy as np

def gf2_swap_rows(M, i, j):
    M[[i, j]] = M[[j, i]]

def gf2_swap_cols(M, i, j):
    M[:, [i, j]] = M[:, [j, i]]

def gf2_row_add(M, src, dst):
    M[dst] = (M[dst] + M[src]) % 2

def stabilizer_standard_form(G):
    G = G.copy() % 2
    m, total_cols = G.shape
    N = total_cols // 2
    col_perm = list(range(total_cols))

    # Eliminate X block
    r = 0
    for col in range(N):
        pivot = None
        for row in range(r, m):
            if G[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != r:
            gf2_swap_rows(G, pivot, r)
        for row in range(m):
            if row != r and G[row, col] == 1:
                gf2_row_add(G, r, row)
        if col != r:
            gf2_swap_cols(G, col, r)
            col_perm[col], col_perm[r] = col_perm[r], col_perm[col]
        r += 1
        if r == m:
            break

    # Eliminate Z block
    pivot_row = r
    for col in range(N, 2 * N):
        if pivot_row >= m:
            break
        found = None
        for row in range(pivot_row, m):
            if G[row, col] == 1:
                found = row
                break
        if found is None:
            continue
        if found != pivot_row:
            gf2_swap_rows(G, found, pivot_row)
        for row in range(m):
            if row != pivot_row and G[row, col] == 1:
                gf2_row_add(G, pivot_row, row)
        target_col = N + (pivot_row - r)
        if col != target_col:
            gf2_swap_cols(G, col, target_col)
            col_perm[col], col_perm[target_col] = col_perm[target_col], col_perm[col]
        pivot_row += 1

    return G, r, col_perm


def build_concatenated_gkp_generator_qqpp(G_binary):
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
