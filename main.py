import numpy as np
from sympy import Matrix
from scipy.io import loadmat
from functions.concatenation import *
from functions.binary_stabilizer_generators import *
from decoders.surface_decoder import *
import pandas as pd
import plotly.express as px
from functions.noise_simulator import *
from functions.lattice import *
from decoders.surface_decoder import *


def run_single_trial(
    dim,
    sigma,
    M,
    G,
    rng,
):
    """
    One Monte Carlo decoding trial.
    """

    # 1. Sample noise
    xi = sample_gaussian_noise(dim, sigma, rng)

    # 2. Decode
    correction = decode_surface_gkp(G,xi)
    # 3. Residual
    residual = (xi - correction)/np.sqrt(2*np.pi)

    # 4. Success check
    success = is_in_stabilizer_lattice_square_fast(residual,M)

    return success





def monte_carlo_success_rate(sigma,M,G,num_trials=4,seed=0):
    """
    Run Monte Carlo simulation.

    Returns
    -------
    success_rate
    logical_error_rate
    """

    rng = np.random.default_rng(seed)
    dim = M.shape[0]
    success_count = 0

    for _ in range(num_trials):

        success = run_single_trial(
            dim,
            sigma,
            M,
            G,
            rng

        )

        if success:
            success_count += 1

    success_rate = success_count / num_trials

    logical_error_rate = 1 - success_rate

    return success_rate, logical_error_rate



G=get_rotated_surface_code_matrix(3)
G,_,_=stabilizer_standard_form(G)
sigma=0.6
M,_=build_concatenated_gkp_generator_qqpp(G)

print(monte_carlo_success_rate(sigma,M,G))