"""
Implementation of distance using the Generic Non-Parametric Representation approach from "Some contributions to the
clustering of financial time series and applications to credit default swaps" by Gautier Marti
https://www.researchgate.net/publication/322714557
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import ot

def spearmans_rho(x: np.array, y: np.array) -> float:
    if False:
        return 10
    "\n    Calculates a statistical estimate of Spearman's rho - a copula-based dependence measure.\n\n    Formula for calculation:\n    rho = 1 - (6)/(T*(T^2-1)) * Sum((X_t-Y_t)^2)\n\n    It is more robust to noise and can be defined if the variables have an infinite second moment.\n    This statistic is described in more detail in the work by Gautier Marti\n    https://www.researchgate.net/publication/322714557 (p.54)\n\n    This method is a wrapper for the scipy spearmanr function. For more details about the function and its parameters,\n    please visit scipy documentation\n    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html\n\n    :param x: (np.array/pd.Series) X vector\n    :param y: (np.array/pd.Series) Y vector (same number of observations as X)\n    :return: (float) Spearman's rho statistical estimate\n    "
    pass

def gpr_distance(x: np.array, y: np.array, theta: float) -> float:
    if False:
        print('Hello World!')
    '\n    Calculates the distance between two Gaussians under the Generic Parametric Representation (GPR) approach.\n\n    According to the original work https://www.researchgate.net/publication/322714557 (p.70):\n    "This is a fast and good proxy for distance d_theta when the first two moments ... predominate". But it\'s not\n    a good metric for heavy-tailed distributions.\n\n    Parameter theta defines what type of information dependency is being tested:\n    - for theta = 0 the distribution information is tested\n    - for theta = 1 the dependence information is tested\n    - for theta = 0.5 a mix of both information types is tested\n\n    With theta in [0, 1] the distance lies in range [0, 1] and is a metric. (See original work for proof, p.71)\n\n    :param x: (np.array/pd.Series) X vector.\n    :param y: (np.array/pd.Series) Y vector (same number of observations as X).\n    :param theta: (float) Type of information being tested. Falls in range [0, 1].\n    :return: (float) Distance under GPR approach.\n    '
    pass

def gnpr_distance(x: np.array, y: np.array, theta: float, n_bins: int=50) -> float:
    if False:
        print('Hello World!')
    '\n    Calculates the empirical distance between two random variables under the Generic Non-Parametric Representation\n    (GNPR) approach.\n\n    Formula for the distance is taken from https://www.researchgate.net/publication/322714557 (p.72).\n\n    Parameter theta defines what type of information dependency is being tested:\n    - for theta = 0 the distribution information is tested\n    - for theta = 1 the dependence information is tested\n    - for theta = 0.5 a mix of both information types is tested\n\n    With theta in [0, 1] the distance lies in the range [0, 1] and is a metric.\n    (See original work for proof, p.71)\n\n    This method is modified as it uses 1D Optimal Transport Distance to measure\n    distribution distance. This solves the issue of defining support and choosing\n    a number of bins. The number of bins can be given as an input to speed up calculations.\n    Big numbers of bins can take a long time to calculate.\n\n    :param x: (np.array/pd.Series) X vector.\n    :param y: (np.array/pd.Series) Y vector (same number of observations as X).\n    :param theta: (float) Type of information being tested. Falls in range [0, 1].\n    :param n_bins: (int) Number of bins to use to split the X and Y vector observations.\n        (100 by default)\n    :return: (float) Distance under GNPR approach.\n    '
    pass