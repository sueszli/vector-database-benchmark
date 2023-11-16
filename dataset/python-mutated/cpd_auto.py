import numpy as np
from .cpd_nonlin import cpd_nonlin

def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Detect change points automatically selecting their number\n\n    :param K: Kernel between each pair of frames in video\n    :param ncp: Maximum number of change points\n    :param vmax: Special parameter\n    :param desc_rate: Rate of descriptor sampling, vmax always corresponds to 1x\n    :param kwargs: Extra parameters for ``cpd_nonlin``\n    :return: Tuple (cps, costs)\n        - cps - best selected change-points\n        - costs - costs for 0,1,2,...,m change-points\n    '
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False, **kwargs)
    N = K.shape[0]
    N2 = N * desc_rate
    penalties = np.zeros(m + 1)
    ncp = np.arange(1, m + 1)
    penalties[1:] = vmax * ncp / (2.0 * N2) * (np.log(float(N2) / ncp) + 1)
    costs = scores / float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin(K, m_best, **kwargs)
    return (cps, scores2)