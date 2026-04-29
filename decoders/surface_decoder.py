import numpy as np
import networkx as nx
from math import sqrt, pi


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def nearest_integer(x):
    return np.floor(x + 0.5)


def second_nearest_integer(x):
    n = nearest_integer(x)
    if x >= n:
        return n - 1
    else:
        return n + 1


def rounding_cost_difference(tk):
    """
    Implements Eq. (118):

    (w(tk) - tk)^2 - ([tk] - tk)^2
    """
    n = nearest_integer(tk)
    w = second_nearest_integer(tk)

    return (w - tk) ** 2 - (n - tk) ** 2


# ------------------------------------------------------------
# Build MWPM graph (Eqs. 116–118)
# ------------------------------------------------------------

def build_matching_graph(G_matrix, t_sub):
    """
    G_matrix: stabilizer matrix (binary)
    t_sub: vector (either q or p)

    returns:
        graph
        shared index sets
    """

    m, n = G_matrix.shape

    graph = nx.Graph()
    shared_sets = {}

    for i in range(m):
        graph.add_node(i)

    for i in range(m):
        for j in range(i + 1, m):

            shared = np.where(
                (G_matrix[i] == 1) &
                (G_matrix[j] == 1)
            )[0]

            if len(shared) == 0:
                continue

            shared_sets[(i, j)] = shared

            weight = min(
                rounding_cost_difference(t_sub[k])
                for k in shared
            )

            graph.add_edge(i, j, weight=weight)

    return graph, shared_sets


# ------------------------------------------------------------
# Find violated stabilizers (Eq. 121)
# ------------------------------------------------------------

def find_highlighted_vertices(G_matrix, x_other):
    """
    Computes:

    mod(g_i^T x'', 2) != 0
    """

    syndrome = (G_matrix @ x_other) % 2

    return [
        i for i, s in enumerate(syndrome)
        if s == 1
    ]


# ------------------------------------------------------------
# Minimum-weight perfect matching
# ------------------------------------------------------------

def mwpm_select_edges(graph, highlighted):

    if len(highlighted) % 2 == 1:
        boundary = max(graph.nodes) + 1
        graph.add_node(boundary)

        for v in highlighted:
            graph.add_edge(boundary, v, weight=0)

        highlighted = highlighted + [boundary]

    subgraph = graph.subgraph(highlighted)

    matching = nx.algorithms.matching.min_weight_matching(
        subgraph,
        weight="weight"
    )

    return matching


# ------------------------------------------------------------
# Main decoder (Algorithm 6)
# ------------------------------------------------------------

def decode_surface_gkp(
        M_surf_qqpp,
        t):
    """
    Parameters
    ----------
    M_surf_qqpp : ndarray
        Binary stabilizer matrix

    t : ndarray
        Noise / syndrome vector (length 2N)

    Returns
    -------
    chi : ndarray
        Closest lattice vector
    """
    t=pow(np.pi,-0.5) * t
    t = np.asarray(t)

    N2 = len(t)
    assert N2 % 2 == 0

    # --------------------------------------------------------
    # Split q / p
    # --------------------------------------------------------

    t_q = t[0::2]
    t_p = t[1::2]

    # nearest integer vectors

    x_q = nearest_integer(t_q)
    x_p = nearest_integer(t_p)

    # --------------------------------------------------------
    # Split stabilizer matrix
    # --------------------------------------------------------

    rows = M_surf_qqpp.shape[0] // 2

    G_q = M_surf_qqpp[:rows, 0::2]
    G_p = M_surf_qqpp[rows:, 1::2]

    # --------------------------------------------------------
    # Build graphs
    # --------------------------------------------------------

    graph_q, shared_q = build_matching_graph(
        G_q,
        t_q
    )

    graph_p, shared_p = build_matching_graph(
        G_p,
        t_p
    )

    # --------------------------------------------------------
    # Find violated stabilizers
    # --------------------------------------------------------

    H_q = find_highlighted_vertices(
        G_q,
        x_p
    )

    H_p = find_highlighted_vertices(
        G_p,
        x_q
    )

    # --------------------------------------------------------
    # Solve MWPM
    # --------------------------------------------------------

    match_q = mwpm_select_edges(
        graph_q,
        H_q
    )

    match_p = mwpm_select_edges(
        graph_p,
        H_p
    )

    # --------------------------------------------------------
    # Decide which coordinates to flip
    # --------------------------------------------------------

    flip_q = np.zeros_like(x_q)
    flip_p = np.zeros_like(x_p)

    for i, j in match_q:

        if (i, j) in shared_q:
            indices = shared_q[(i, j)]
        elif (j, i) in shared_q:
            indices = shared_q[(j, i)]
        else:
            continue

        k = min(
            indices,
            key=lambda idx:
            rounding_cost_difference(t_q[idx])
        )

        flip_q[k] = 1

    for i, j in match_p:

        if (i, j) in shared_p:
            indices = shared_p[(i, j)]
        elif (j, i) in shared_p:
            indices = shared_p[(j, i)]
        else:
            continue

        k = min(
            indices,
            key=lambda idx:
            rounding_cost_difference(t_p[idx])
        )

        flip_p[k] = 1

    # --------------------------------------------------------
    # Apply flips
    # --------------------------------------------------------

    for i in range(len(x_q)):
        if flip_q[i]:
            x_q[i] = second_nearest_integer(
                t_q[i]
            )

    for i in range(len(x_p)):
        if flip_p[i]:
            x_p[i] = second_nearest_integer(
                t_p[i]
            )

    # --------------------------------------------------------
    # Combine q and p
    # --------------------------------------------------------

    x = np.zeros(N2)

    x[0::2] = x_q
    x[1::2] = x_p

    # final scaling

    chi = sqrt(pi) * x

    return chi