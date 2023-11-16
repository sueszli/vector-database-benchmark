"""Module containing methods for calculating correlation between variables."""
import math
from collections import Counter
from typing import List, Union
import numpy as np
import pandas as pd
from scipy.stats import entropy
from deepchecks.utils.distribution.preprocessing import value_frequency

def conditional_entropy(x: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> float:
    if False:
        print('Hello World!')
    '\n    Calculate the conditional entropy of x given y: S(x|y).\n\n    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy\n\n    Parameters\n    ----------\n    x: Union[List, np.ndarray, pd.Series]\n        A sequence of numerical_variable without nulls\n    y: Union[List, np.ndarray, pd.Series]\n        A sequence of numerical_variable without nulls\n\n    Returns\n    -------\n    float\n        Representing the conditional entropy\n    '
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    s_xy = 0.0
    for xy in xy_counter:
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        s_xy += p_xy * math.log(p_y / p_xy, math.e)
    return s_xy

def theil_u_correlation(x: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> float:
    if False:
        i = 10
        return i + 15
    "\n    Calculate the Theil's U correlation of y to x.\n\n    Theil's U is an asymmetric measure ranges [0,1] based on entropy which answers the question: how well does\n    variable y explains variable x? For more information see https://en.wikipedia.org/wiki/Uncertainty_coefficient\n\n    Parameters\n    ----------\n    x: Union[List, np.ndarray, pd.Series]\n        A sequence of a categorical variable values without nulls\n    y: Union[List, np.ndarray, pd.Series]\n        A sequence of a categorical variable values without nulls\n\n    Returns\n    -------\n    float\n        Representing the Theil U correlation between y and x\n    "
    s_xy = conditional_entropy(x, y)
    values_probabilities = value_frequency(x)
    s_x = entropy(values_probabilities)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def symmetric_theil_u_correlation(x: Union[List, np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]) -> float:
    if False:
        print('Hello World!')
    "\n    Calculate the symmetric Theil's U correlation of y to x.\n\n    Parameters\n    ----------\n    x: Union[List, np.ndarray, pd.Series]\n        A sequence of a categorical variable values without nulls\n    y: Union[List, np.ndarray, pd.Series]\n        A sequence of a categorical variable values without nulls\n\n    Returns\n    -------\n    float\n        Representing the symmetric Theil U correlation between y and x\n    "
    h_x = entropy(value_frequency(x))
    h_y = entropy(value_frequency(y))
    u_xy = theil_u_correlation(x, y)
    u_yx = theil_u_correlation(y, x)
    u_sym = (h_x * u_xy + h_y * u_yx) / (h_x + h_y)
    return u_sym

def correlation_ratio(categorical_data: Union[List, np.ndarray, pd.Series], numerical_data: Union[List, np.ndarray, pd.Series], ignore_mask: Union[List[bool], np.ndarray]=None) -> float:
    if False:
        return 10
    '\n    Calculate the correlation ratio of numerical_variable to categorical_variable.\n\n    Correlation ratio is a symmetric grouping based method that describe the level of correlation between\n    a numeric variable and a categorical variable. returns a value in [0,1].\n    For more information see https://en.wikipedia.org/wiki/Correlation_ratio\n\n    Parameters\n    ----------\n    categorical_data: Union[List, np.ndarray, pd.Series]\n        A sequence of categorical values encoded as class indices without nulls except possibly at ignored elements\n    numerical_data: Union[List, np.ndarray, pd.Series]\n        A sequence of numerical values without nulls except possibly at ignored elements\n    ignore_mask: Union[List[bool], np.ndarray[bool]] default: None\n        A sequence of boolean values indicating which elements to ignore. If None, includes all indexes.\n\n    Returns\n    -------\n    float\n        Representing the correlation ratio between the variables.\n    '
    if ignore_mask:
        numerical_data = numerical_data[~np.asarray(ignore_mask)]
        categorical_data = categorical_data[~np.asarray(ignore_mask)]
    cat_num = int(np.max(categorical_data) + 1)
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = numerical_data[categorical_data == i]
        n_array[i] = cat_measures.shape[0]
        y_avg_array[i] = np.average(cat_measures.astype(float))
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(numerical_data, y_total_avg), 2))
    if denominator == 0:
        eta = 0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta