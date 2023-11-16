"""
Implementation of the Hierarchical Correlation Block Model (HCBM) matrix.
"Clustering financial time series: How long is enough?" by Marti, G., Andler, S., Nielsen, F. and Donnat, P.
https://www.ijcai.org/Proceedings/16/Papers/367.pdf
"""
import numpy as np
import pandas as pd
from statsmodels.sandbox.distributions.multivariate import multivariate_t_rvs

def _hcbm_mat_helper(mat, n_low=0, n_high=214, rho_low=0.1, rho_high=0.9, blocks=4, depth=4):
    if False:
        i = 10
        return i + 15
    '\n    Helper function for `generate_hcmb_mat` that recursively places rho values to HCBM matrix\n    given as an input.\n\n    By using a uniform distribution we select the start and end locations of the blocks in the\n    matrix. For each block, we recurse depth times and repeat splitting up the sub-matrix into\n    blocks. Each depth level has a unique correlation (rho) values generated from a uniform\n    distributions, and bounded by `rho_low` and `rho_high`. This function works as a\n    side-effect to the `mat` parameter.\n\n    It is reproduced with modifications from the following paper:\n    `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.\n    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.\n    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_\n\n    :param mat: (np.array) Parent HCBM matrix.\n    :param n_low: (int) Start location of HCMB matrix to work on.\n    :param n_high: (int) End location of HCMB matrix to work on.\n    :param rho_low: (float) Lower correlation bound of the matrix. Must be greater or equal\n    to 0.\n    :param rho_high: (float) Upper correlation bound of the matrix. Must be less or equal to 1.\n    :param blocks: (int) Maximum number of blocks to generate per level of depth.\n    :param depth: (int) Depth of recursion for generating new blocks.\n    '
    pass

def generate_hcmb_mat(t_samples, n_size, rho_low=0.1, rho_high=0.9, blocks=4, depth=4, permute=False):
    if False:
        i = 10
        return i + 15
    '\n    Generates a Hierarchical Correlation Block Model (HCBM) matrix  of correlation values.\n\n    By using a uniform distribution we select the start and end locations of the blocks in the\n    matrix. For each block, we recurse depth times and repeat splitting up the sub-matrix into\n    blocks. Each depth level has a unique correlation (rho) values generated from a uniform\n    distributions, and bounded by `rho_low` and `rho_high`.\n\n    It is reproduced with modifications from the following paper:\n    `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.\n    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.\n    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_\n\n    :param t_samples: (int) Number of HCBM matrices to generate.\n    :param n_size: (int) Size of HCBM matrix.\n    :param rho_low: (float) Lower correlation bound of the matrix. Must be greater or equal to 0.\n    :param rho_high: (float) Upper correlation bound of the matrix. Must be less or equal to 1.\n    :param blocks: (int) Number of blocks to generate per level of depth.\n    :param depth: (int) Depth of recursion for generating new blocks.\n    :param permute: (bool) Whether to permute the final HCBM matrix.\n    :return: (np.array) Generated HCBM matrix of shape (t_samples, n_size, n_size).\n    '
    pass

def time_series_from_dist(corr, t_samples=1000, dist='normal', deg_free=3):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates a time series from a given correlation matrix.\n\n    It uses multivariate sampling from distributions to create the time series. It supports\n    normal and student-t distributions. This method relies and acts as a wrapper for the\n    `np.random.multivariate_normal` and\n    `statsmodels.sandbox.distributions.multivariate.multivariate_t_rvs` modules.\n    `<https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html>`_\n    `<https://www.statsmodels.org/stable/sandbox.html?highlight=sandbox#module-statsmodels.sandbox>`_\n\n    It is reproduced with modifications from the following paper:\n    `Marti, G., Andler, S., Nielsen, F. and Donnat, P., 2016.\n    Clustering financial time series: How long is enough?. arXiv preprint arXiv:1603.04017.\n    <https://www.ijcai.org/Proceedings/16/Papers/367.pdf>`_\n\n    :param corr: (np.array) Correlation matrix.\n    :param t_samples: (int) Number of samples in the time series.\n    :param dist: (str) Type of distributions to use.\n        Can take the values ["normal", "student"].\n    :param deg_free: (int) Degrees of freedom. Only used for student-t distribution.\n    :return: (pd.DataFrame) The resulting time series of shape (len(corr), t_samples).\n    '
    pass