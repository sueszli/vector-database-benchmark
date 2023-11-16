import numpy as np
import scipy.stats

def proportion_difference(proportion_1, proportion_2, n_1, n_2=None):
    if False:
        while True:
            i = 10
    '\n    Computes the test statistic and p-value for a difference of\n    proportions test.\n\n    Parameters\n    -----------\n    proportion_1 : float\n        The first proportion\n    proportion_2 : float\n        The second proportion\n    n_1 : int\n        The sample size of the first test sample\n    n_2 : int or None (default=None)\n        The sample size of the second test sample.\n        If `None`, `n_1`=`n_2`.\n\n    Returns\n    -----------\n\n    z, p : float or None, float\n        Returns the z-score and the p-value\n\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/evaluate/proportion_difference/\n\n    '
    if n_2 is None:
        n_2 = n_1
    var_1 = proportion_1 * (1.0 - proportion_1) / n_1
    var_2 = proportion_2 * (1.0 - proportion_2) / n_2
    z = (proportion_1 - proportion_2) / np.sqrt(var_1 + var_2)
    p = scipy.stats.norm.cdf(z)
    return (z, p)