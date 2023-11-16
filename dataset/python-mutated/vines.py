"""
Implementation of generating financial correlation matrices from
"Generating random correlation matrices based on vines and extended onion method"
by Daniel Lewandowski, Dorota Kurowicka, Harry Joe.
https://www.sciencedirect.com/science/article/pii/S0047259X09000876
and "Generating random correlation matrices based partial correlations" by Harry Joe.
https://www.sciencedirect.com/science/article/pii/S0047259X05000886
"""
import numpy as np

def _correlation_from_partial_dvine(partial_correlations, a_beta, b_beta, row, col):
    if False:
        print('Hello World!')
    "\n    Calculates a correlation based on partical correlations using the D-vine method.\n\n    It samples from a beta distribution, adjusts it to the range [-1, 1]. Sets this value\n    as the starting partial correlation, and follows the D-vine to calculate the final\n    correlation.\n\n    :param partial_correlations: (np.array) Matrix of current partial correlations. It is\n        modified during this function's execution.\n    :param a_beta: (float) Alpha parameter of the beta distribution to sample from.\n    :param b_beta: (float) Beta parameter of the beta distribution to sample from.\n    :param row: (int) Starting row of the partial correlation matrix.\n    :param col: (int) Starting column of the partial correlation matrix.\n    :return: (float) Calculated correlation.\n    "
    pass

def _correlation_from_partial_cvine(partial_correlations, a_beta, b_beta, row, col):
    if False:
        print('Hello World!')
    "\n    Calculates a correlation based on partical correlations using the C-vine method.\n\n    It samples from a beta distribution, adjusts it to the range [-1, 1]. Sets this value\n    as the starting partial correlation, and follows the C-vine to calculate the final\n    correlation.\n\n    :param partial_correlations: (np.array) Matrix of current partial correlations. It is\n        modified during this function's execution.\n    :param a_beta: (float) Alpha parameter of the beta distribution to sample from.\n    :param b_beta: (float) Beta parameter of the beta distribution to sample from.\n    :param row: (int) Starting row of the partial correlation matrix.\n    :param col: (int) Starting column of the partial correlation matrix.\n    :return: (float) Calculated correlation.\n    "
    pass

def _q_vector_correlations(corr_mat, r_factor, dim):
    if False:
        while True:
            i = 10
    '\n    Sample from unit vector uniformly on the surface of the k_loc-dimensional hypersphere and\n    obtains the q vector of correlations.\n\n    :param corr_mat (np.array) Correlation matrix.\n    :param r_factor (np.array) R factor vector based on correlation matrix.\n    :param dim: (int) Dimension of the hypersphere to sample from.\n    :return: (np.array) Q vector of correlations.\n    '
    pass

def sample_from_dvine(dim=10, n_samples=1, beta_dist_fixed=None):
    if False:
        return 10
    '\n    Generates uniform correlation matrices using the D-vine method.\n\n    It is reproduced with modifications from the following paper:\n    `Joe, H., 2006. Generating random correlation matrices based on partial correlations.\n    Journal of Multivariate Analysis, 97(10), pp.2177-2189.\n    <https://www.sciencedirect.com/science/article/pii/S0047259X05000886>`_\n\n    It uses the partial correlation D-vine to generate partial correlations. The partial\n    correlations\n    are sampled from a uniform beta distribution and adjusted to thr range [-1, 1]. Then these\n    partial correlations are converted into raw correlations by using a recursive formula based\n    on its location on the vine.\n\n    :param dim: (int) Dimension of correlation matrix to generate.\n    :param n_samples: (int) Number of samples to generate.\n    :param beta_dist_fixed: (tuple) Overrides the beta distribution parameters. The input is\n        two float parameters (alpha, beta), used in the distribution. (None by default)\n    :return: (np.array) Generated correlation matrices of shape (n_samples, dim, dim).\n    '
    pass

def sample_from_cvine(dim=10, eta=2, n_samples=1, beta_dist_fixed=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates uniform correlation matrices using the C-vine method.\n\n    It is reproduced with modifications from the following paper:\n    `Lewandowski, D., Kurowicka, D. and Joe, H., 2009. Generating random correlation matrices based\n    on vines and extended onion method. Journal of multivariate analysis, 100(9), pp.1989-2001.\n    <https://www.sciencedirect.com/science/article/pii/S0047259X09000876>`_\n\n    It uses the partial correlation C-vine to generate partial correlations. The partial\n    correlations\n    are sampled from a uniform beta distribution proportional to its determinant and the factor\n    eta.\n    and adjusted to thr range [-1, 1]. Then these partial correlations are converted into raw\n    correlations by using a recursive formula based on its location on the vine.\n\n    :param dim: (int) Dimension of correlation matrix to generate.\n    :param eta: (int) Corresponds to uniform distribution of beta.\n        Correlation matrix `S` has a distribution proportional to [det C]^(eta - 1)\n    :param n_samples: (int) Number of samples to generate.\n    :param beta_dist_fixed: (tuple) Overrides the beta distribution parameters. The input is\n        two float parameters (alpha, beta), used in the distribution. (None by default)\n    :return: (np.array) Generated correlation matrices of shape (n_samples, dim, dim).\n    '
    pass

def sample_from_ext_onion(dim=10, eta=2, n_samples=1):
    if False:
        while True:
            i = 10
    '\n    Generates uniform correlation matrices using extended onion method.\n\n    It is reproduced with modifications from the following paper:\n    `Lewandowski, D., Kurowicka, D. and Joe, H., 2009. Generating random correlation matrices based\n    on vines and extended onion method. Journal of multivariate analysis, 100(9), pp.1989-2001.\n    <https://www.sciencedirect.com/science/article/pii/S0047259X09000876>`_\n\n    It uses the extended onion to generate correlations sampled from a uniform beta distribution.\n    It starts with a one-dimensional matrix, and it iteratively grows the matrix by adding extra\n    rows and columns by sampling from the convex, closed, compact and full-dimensional set on the\n    surface of a k-dimensional hypersphere.\n\n    :param dim: (int) Dimension of correlation matrix to generate.\n    :param eta: (int) Corresponds to uniform distribution of beta.\n        Correlation matrix `S` has a distribution proportional to [det C]^(eta - 1)\n    :param n_samples: (int) Number of samples to generate.\n    :return: (np.array) Generated correlation matrices of shape (n_samples, dim, dim).\n    '
    pass