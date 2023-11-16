"""
Implementations of mutual information (I) and variation of information (VI) codependence measures from Cornell
lecture slides: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes
"""
import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score

def get_optimal_number_of_bins(num_obs: int, corr_coef: float=None) -> int:
    if False:
        return 10
    '\n    Calculates optimal number of bins for discretization based on number of observations\n    and correlation coefficient (univariate case).\n\n    Algorithms used in this function were originally proposed in the works of Hacine-Gharbi et al. (2012)\n    and Hacine-Gharbi and Ravier (2018). They are described in the Cornell lecture notes:\n    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes (p.26)\n\n    :param num_obs: (int) Number of observations.\n    :param corr_coef: (float) Correlation coefficient, used to estimate the number of bins for univariate case.\n    :return: (int) Optimal number of bins.\n    '
    pass

def get_mutual_info(x: np.array, y: np.array, n_bins: int=None, normalize: bool=False, estimator: str='standard') -> float:
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns mutual information (MI) between two vectors.\n\n    This function uses the discretization with the optimal bins algorithm proposed in the works of\n    Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).\n\n    Read Cornell lecture notes for more information about the mutual information:\n    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.\n\n    This function supports multiple ways the mutual information can be estimated:\n\n    1. ``standard`` - the standard way of estimation - binning observations according to a given\n       number of bins and applying the MI formula.\n    2. ``standard_copula`` - estimating the copula (as a normalized ranking of the observations) and\n       applying the standard mutual information estimator on it.\n    3. ``copula_entropy`` - estimating the copula (as a normalized ranking of the observations) and\n       calculating its entropy. Then MI estimator = (-1) * copula entropy.\n\n    The last two estimators' implementation is taken from the blog post by Dr. Gautier Marti.\n    Read this blog post for more information about the differences in the estimators:\n    https://gmarti.gitlab.io/qfin/2020/07/01/mutual-information-is-copula-entropy.html\n\n    :param x: (np.array) X vector.\n    :param y: (np.array) Y vector.\n    :param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.\n                         (None by default)\n    :param normalize: (bool) Flag used to normalize the result to [0, 1]. (False by default)\n    :param estimator: (str) Estimator to be used for calculation. [``standard``, ``standard_copula``, ``copula_entropy``]\n                            (``standard`` by default)\n    :return: (float) Mutual information score.\n    "
    pass

def variation_of_information_score(x: np.array, y: np.array, n_bins: int=None, normalize: bool=False) -> float:
    if False:
        print('Hello World!')
    '\n    Returns variantion of information (VI) between two vectors.\n\n    This function uses the discretization using optimal bins algorithm proposed in the works of\n    Hacine-Gharbi et al. (2012) and Hacine-Gharbi and Ravier (2018).\n\n    Read Cornell lecture notes for more information about the variation of information:\n    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes.\n\n    :param x: (np.array) X vector.\n    :param y: (np.array) Y vector.\n    :param n_bins: (int) Number of bins for discretization, if None the optimal number will be calculated.\n                         (None by default)\n    :param normalize: (bool) True to normalize the result to [0, 1]. (False by default)\n    :return: (float) Variation of information score.\n    '
    pass