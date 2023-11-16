"""
This implementation lets user generate dependence and distance matrix based on the various methods of Information
Codependence  described in Cornell lecture notes on Codependence:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3512994&download=yes
"""
import numpy as np
import pandas as pd
from mlfinlab.codependence.information import variation_of_information_score, get_mutual_info
from mlfinlab.codependence.correlation import distance_correlation
from mlfinlab.codependence.gnpr_distance import spearmans_rho, gpr_distance, gnpr_distance
from mlfinlab.codependence.optimal_transport import optimal_transport_dependence

def get_dependence_matrix(df: pd.DataFrame, dependence_method: str, theta: float=0.5, n_bins: int=None, normalize: bool=True, estimator: str='standard', target_dependence: str='comonotonicity', gaussian_corr: float=0.7, var_threshold: float=0.2) -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    '\n    This function returns a dependence matrix for elements given in the dataframe using the chosen dependence method.\n\n    List of supported algorithms to use for generating the dependence matrix: ``information_variation``,\n    ``mutual_information``, ``distance_correlation``, ``spearmans_rho``, ``gpr_distance``, ``gnpr_distance``,\n    ``optimal_transport``.\n\n    :param df: (pd.DataFrame) Features.\n    :param dependence_method: (str) Algorithm to be use for generating dependence_matrix.\n    :param theta: (float) Type of information being tested in the GPR and GNPR distances. Falls in range [0, 1].\n                          (0.5 by default)\n    :param n_bins: (int) Number of bins for discretization in ``information_variation`` and ``mutual_information``,\n                         if None the optimal number will be calculated. (None by default)\n    :param normalize: (bool) Flag used to normalize the result to [0, 1] in ``information_variation`` and\n                             ``mutual_information``. (True by default)\n    :param estimator: (str) Estimator to be used for calculation in ``mutual_information``.\n                            [``standard``, ``standard_copula``, ``copula_entropy``] (``standard`` by default)\n    :param target_dependence: (str) Type of target dependence to use in ``optimal_transport``.\n                                    [``comonotonicity``, ``countermonotonicity``, ``gaussian``,\n                                    ``positive_negative``, ``different_variations``, ``small_variations``]\n                                    (``comonotonicity`` by default)\n    :param gaussian_corr: (float) Correlation coefficient to use when creating ``gaussian`` and\n                                  ``small_variations`` copulas. [from 0 to 1] (0.7 by default)\n    :param var_threshold: (float) Variation threshold to use for coefficient to use in ``small_variations``.\n                                  Sets the relative area of correlation in a copula. [from 0 to 1] (0.2 by default)\n    :return: (pd.DataFrame) Dependence matrix.\n    '
    pass

def get_distance_matrix(X: pd.DataFrame, distance_metric: str='angular') -> pd.DataFrame:
    if False:
        return 10
    '\n    Applies distance operator to a dependence matrix.\n\n    This allows to turn a correlation matrix into a distance matrix. Distances used are true metrics.\n\n    List of supported distance metrics to use for generating the distance matrix: ``angular``, ``squared_angular``,\n    and ``absolute_angular``.\n\n    :param X: (pd.DataFrame) Dataframe to which distance operator to be applied.\n    :param distance_metric: (str) The distance metric to be used for generating the distance matrix.\n    :return: (pd.DataFrame) Distance matrix.\n    '
    pass