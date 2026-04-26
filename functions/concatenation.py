import numpy as np


# =========================
# GF(2) operations
# =========================

def gf2_swap_rows(M, i, j):
    M[[i, j]] = M[[j, i]]


def gf2_swap_cols(M, i, j):
    M[:, [i, j]] = M[:, [j, i]]


def gf2_row_add(M, src, dst):
    M[dst] = (M[dst] + M[src]) % 2


def gf2_independent_row_basis(M):
    """
    Return an independent row basis of M over GF(2).
    Redundant rows are removed so downstream rank assumptions hold.
    """
    A = M.copy() % 2
    m, n = A.shape
    r = 0

    for col in range(n):
        pivot = None
        for row in range(r, m):
            if A[row, col] == 1:
                pivot = row
                break

        if pivot is None:
            continue

        if pivot != r:
            gf2_swap_rows(A, pivot, r)

        for row in range(m):
            if row != r and A[row, col] == 1:
                gf2_row_add(A, r, row)

        r += 1
        if r == m:
            break

    return A[:r].copy(), r



# ==========================================================
# Correct stabilizer standard form
# ==========================================================

def stabilizer_standard_form(G):

    G = G.copy() % 2
    G, _ = gf2_independent_row_basis(G)

    m, total_cols = G.shape
    N = total_cols // 2

    col_perm = list(range(total_cols))

    # ======================================================
    # STEP 1 — X elimination
    # ======================================================

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
            col_perm[col], col_perm[r] = (
                col_perm[r],
                col_perm[col],
            )

        r += 1

        if r == m:
            break

    # parameters

    n2 = m - r
    pivot_row = r

    # ======================================================
    # STEP 2 — Z elimination with column placement
    # ======================================================

    for i in range(n2):

        target_col = N + r + i

        pivot_col = None
        pivot_row_found = None

        # search pivot anywhere in Z

        for col in range(N, 2 * N):

            for row in range(pivot_row, m):

                if G[row, col] == 1:

                    pivot_col = col
                    pivot_row_found = row
                    break

            if pivot_col is not None:
                break

        if pivot_col is None:
            raise ValueError("Z block rank deficiency")

        # move pivot row

        if pivot_row_found != pivot_row:
            gf2_swap_rows(G, pivot_row_found, pivot_row)

        # clear column

        for row in range(m):
            if row != pivot_row and G[row, pivot_col] == 1:
                gf2_row_add(G, pivot_row, row)

        # move pivot column into canonical position

        if pivot_col != target_col:

            gf2_swap_cols(G, pivot_col, target_col)

            col_perm[pivot_col], col_perm[target_col] = (
                col_perm[target_col],
                col_perm[pivot_col],
            )

        pivot_row += 1

    info = dict(
        N=N,
        m=m,
        r=r,
        n2=n2,
        k=N - m,
        column_permutation=col_perm,
    )

    return G, r, col_perm



def build_concatenated_gkp_generator_qqpp(G_binary):

    G_std, r, col_perm = stabilizer_standard_form(G_binary)

    m, total_cols = G_std.shape
    N = total_cols // 2

    k  = N - m
    n2 = m - r   # = N - k - r

    n1 = r
    n3 = k

    # Column block offsets (in permuted basis)
    b1 = 0          # q: I    width r
    b2 = r          # q: A₁   width n2   ← Block 5 goes here
    b3 = r + n2     # q: A₂   width k    ← Block 3 goes here
    b4 = N          # p: B/D  width r    ← Block 4 goes here
    b5 = N + r      # p: 0/I  width n2   (already in stab rows)
    b6 = N + r + n2 # p: C/E  width k    ← Block 6 goes here

    M = np.zeros((2 * N, 2 * N), dtype=float)
    row = 0

    # Block 1: r rows — [I | A₁ | A₂ | B | 0 | C]
    for i in range(r):
        M[row] = G_std[i]
        row += 1

    # Block 2: n2 rows — [0 | 0 | 0 | D | I | E]
    for i in range(n2):
        M[row] = G_std[r + i]
        row += 1

    # Block 3: k rows — [0 | 0 | 2I | 0 | 0 | 0]  (q-side at b3)
    for i in range(n3):
        M[row, b3 + i] = 2
        row += 1

    # Block 4: r rows — [0 | 0 | 0 | 2I | 0 | 0]  (p-side at b4)
    for i in range(n1):
        M[row, b4 + i] = 2
        row += 1

    # Block 5: n2 rows — [0 | 2I | 0 | 0 | 0 | 0]  (q-side at b2)  ← YOUR BUG WAS HERE
    for i in range(n2):
        M[row, b2 + i] = 2
        row += 1

    # Block 6: k rows — [0 | 0 | 0 | 0 | 0 | 2I]  (p-side at b6)
    for i in range(n3):
        M[row, b6 + i] = 2
        row += 1

    assert row == 2 * N, f"Row count mismatch: {row} != {2*N}"

    M = M / np.sqrt(2)



    info = {
        "N": N,
        "k": k,
        "num_stabilizers": m,
        "r": r,
        "n2": n2,
        "column_permutation": col_perm,
    }

    return M, info
