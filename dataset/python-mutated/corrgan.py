"""
Implementation of sampling realistic financial correlation matrices from
"CorrGAN: Sampling Realistic Financial Correlation Matrices using
Generative Adversarial Networks" by Gautier Marti.
https://arxiv.org/pdf/1910.09504.pdf
"""
from os import listdir, path
import numpy as np
from scipy.cluster import hierarchy
from statsmodels.stats.correlation_tools import corr_nearest

def sample_from_corrgan(model_loc, dim=10, n_samples=1):
    if False:
        print('Hello World!')
    '\n    Samples correlation matrices from the pre-trained CorrGAN network.\n\n    It is reproduced with modifications from the following paper:\n    `Marti, G., 2020, May. CorrGAN: Sampling Realistic Financial Correlation Matrices Using\n    Generative Adversarial Networks. In ICASSP 2020-2020 IEEE International Conference on\n    Acoustics, Speech and Signal Processing (ICASSP) (pp. 8459-8463). IEEE.\n    <https://arxiv.org/pdf/1910.09504.pdf>`_\n\n    It loads the appropriate CorrGAN model for the required dimension. Generates a matrix output\n    from this network. Symmetries this matrix and finds the nearest correlation matrix\n    that is positive semi-definite. Finally, it maximizes the sum of the similarities between\n    adjacent leaves to arrange it with hierarchical clustering.\n\n    The CorrGAN network was trained on the correlation profiles of the S&P 500 stocks. Therefore\n    the output retains these properties. In addition, the final output retains the following\n    6 stylized facts:\n\n    1. Distribution of pairwise correlations is significantly shifted to the positive.\n\n    2. Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first\n    eigenvalue (the market).\n\n    3. Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other\n    large eigenvalues (industries).\n\n    4. Perron-Frobenius property (first eigenvector has positive entries).\n\n    5. Hierarchical structure of correlations.\n\n    6. Scale-free property of the corresponding Minimum Spanning Tree (MST).\n\n    :param model_loc: (str) Location of folder containing CorrGAN models.\n    :param dim: (int) Dimension of correlation matrix to sample.\n        In the range [2, 200].\n    :param n_samples: (int) Number of samples to generate.\n    :return: (np.array) Sampled correlation matrices of shape (n_samples, dim, dim).\n    '
    pass