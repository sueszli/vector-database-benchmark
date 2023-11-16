"""
An implementation of the Exact Fit of the first 3 Moments (EF3M) of finding the parameters that make up the mixture
of 2 Gaussian distributions. Based on the work by Lopez de Prado and Foreman (2014) "A mixture of two Gaussians
approach to mathematical portfolio oversight: The EF3M algorithm." Quantitative Finance, Vol. 14, No. 5, pp. 913-930.
"""
import sys
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import gaussian_kde
from numba import njit, objmode

class M2N:
    """
    M2N - A Mixture of 2 Normal distributions
    This class is used to contain parameters and equations for the EF3M algorithm, when fitting parameters to a mixture
    of 2 Gaussian distributions.

    :param moments: (list) The first five (1... 5) raw moments of the mixture distribution.
    :param epsilon: (float) Fitting tolerance
    :param factor: (float) Lambda factor from equations
    :param n_runs: (int) Number of times to execute 'singleLoop'
    :param variant: (int) The EF3M variant to execute, options are 1: EF3M using first 4 moments, 2: EF3M using
     first 5 moments
    :param max_iter: (int) Maximum number of iterations to perform in the 'fit' method
    :param num_workers: (int) Number of CPU cores to use for multiprocessing execution. Default is -1 which sets
     num_workers to all cores.

    """

    def __init__(self, moments, epsilon=10 ** (-5), factor=5, n_runs=1, variant=1, max_iter=100000, num_workers=-1):
        if False:
            i = 10
            return i + 15
        "\n        Constructor\n\n        :param moments: (list) The first five (1... 5) raw moments of the mixture distribution.\n        :param epsilon: (float) Fitting tolerance\n        :param factor: (float) Lambda factor from equations\n        :param n_runs: (int) Number of times to execute 'singleLoop'\n        :param variant: (int) The EF3M variant to execute, options are 1: EF3M using first 4 moments, 2: EF3M using\n         first 5 moments\n        :param max_iter: (int) Maximum number of iterations to perform in the 'fit' method\n        :param num_workers: (int) Number of CPU cores to use for multiprocessing execution. Default is -1 which sets\n         num_workers to all cores.\n\n        The parameters of the mixture are defined by a list, where:\n            parameters = [mu_1, mu_2, sigma_1, sigma_2, p_1]\n        "
        pass

    def fit(self, mu_2):
        if False:
            i = 10
            return i + 15
        '\n        Fits and the parameters that describe the mixture of the 2 Normal distributions for a given set of initial\n        parameter guesses.\n\n        :param mu_2: (float) An initial estimate for the mean of the second distribution.\n        '
        pass

    def get_moments(self, parameters, return_result=False):
        if False:
            print('Hello World!')
        "\n        Calculates and returns the first five (1...5) raw moments corresponding to the newly estimated parameters.\n\n        :param parameters: (list) List of parameters if the specific order [mu_1, mu_2, sigma_1, sigma_2, p_1]\n        :param return_result: (bool) If True, method returns a result instead of setting the 'self.new_moments'\n         attribute.\n        :return: (list) List of the first five moments\n        "
        pass

    def iter_4(self, mu_2, p_1):
        if False:
            i = 10
            return i + 15
        '\n        Evaluation of the set of equations that make up variant #1 of the EF3M algorithm (fitting using the first\n        four moments).\n\n        :param mu_2: (float) Initial parameter value for mu_2\n        :param p_1: (float) Probability defining the mixture; p_1, 1 - p_1\n        :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values,\n         divide-by-zero), otherwise an empty list is returned.\n        '
        pass

    def iter_5(self, mu_2, p_1):
        if False:
            i = 10
            return i + 15
        '\n        Evaluation of the set of equations that make up variant #2 of the EF3M algorithm (fitting using the first five\n        moments).\n\n        :param mu_2: (float) Initial parameter value for mu_2\n        :param p_1: (float) Probability defining the mixture; p_1, 1-p_1\n        :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values,\n         divide-by-zero), otherwise an empty list is returned.\n        '
        pass

    def single_fit_loop(self, epsilon=0):
        if False:
            print('Hello World!')
        '\n        A single scan through the list of mu_2 values, cataloging the successful fittings in a DataFrame.\n\n        :param epsilon: (float) Fitting tolerance.\n        :return: (pd.DataFrame) Fitted parameters and error\n        '
        pass

    def mp_fit(self):
        if False:
            i = 10
            return i + 15
        "\n        Parallelized implementation of the 'single_fit_loop' method. Makes use of dask.delayed to execute multiple\n        calls of 'single_fit_loop' in parallel.\n\n        :return: (pd.DataFrame) Fitted parameters and error\n        "
        pass

def centered_moment(moments, order):
    if False:
        while True:
            i = 10
    "\n    Compute a single moment of a specific order about the mean (centered) given moments about the origin (raw).\n\n    :param moments: (list) First 'order' raw moments\n    :param order: (int) The order of the moment to calculate\n    :return: (float) The central moment of specified order.\n    "
    pass

def raw_moment(central_moments, dist_mean):
    if False:
        print('Hello World!')
    '\n    Calculates a list of raw moments given a list of central moments.\n\n    :param central_moments: (list) The first n (1...n) central moments as a list.\n    :param dist_mean: (float) The mean of the distribution.\n    :return: (list) The first n+1 (0...n) raw moments.\n    '
    pass

def most_likely_parameters(data, ignore_columns='error', res=10000):
    if False:
        print('Hello World!')
    '\n    Determines the most likely parameter estimate using a KDE from the DataFrame of the results of the fit from the\n    M2N object.\n\n    :param data: (pandas.DataFrame) Contains parameter estimates from all runs.\n    :param ignore_columns: (string, list) Column or columns to exclude from analysis.\n    :param res: (int) Resolution of the kernel density estimate.\n    :return: (dict) Labels and most likely estimates for parameters.\n    '
    pass

@njit()
def iter_4_jit(mu_2, p_1, m_1, m_2, m_3, m_4):
    if False:
        return 10
    '\n    "Numbarized" evaluation of the set of equations that make up variant #1 of the EF3M algorithm (fitting using the\n    first four moments).\n\n    :param mu_2: (float) Initial parameter value for mu_2\n    :param p_1: (float) Probability defining the mixture; p_1, 1 - p_1\n    :param m_1, m_2, m_3, m_4: (float) The first four (1... 4) raw moments of the mixture distribution.\n    :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values,\n        divide-by-zero), otherwise an empty list is returned.\n    '
    pass

@njit()
def iter_5_jit(mu_2, p_1, m_1, m_2, m_3, m_4, m_5):
    if False:
        for i in range(10):
            print('nop')
    '\n    "Numbarized" evaluation of the set of equations that make up variant #2 of the EF3M algorithm (fitting using the\n     first five moments).\n\n    :param mu_2: (float) Initial parameter value for mu_2\n    :param p_1: (float) Probability defining the mixture; p_1, 1-p_1\n    :param m_1, m_2, m_3, m_4, m_5: (float) The first five (1... 5) raw moments of the mixture distribution.\n    :return: (list) List of estimated parameter if no invalid values are encountered (e.g. complex values,\n        divide-by-zero), otherwise an empty list is returned.\n    '
    pass