"""Module containing all outliers algorithms used in the library."""
from typing import Sequence, Tuple, Union
import numpy as np
from deepchecks.core.errors import DeepchecksValueError
EPS = 0.001

def iqr_outliers_range(data: np.ndarray, iqr_range: Tuple[int, int], scale: float, sharp_drop_ratio: float=0.9) -> Tuple[float, float]:
    if False:
        return 10
    "Calculate outliers range on the data given using IQR.\n\n    Parameters\n    ----------\n    data: np.ndarray\n        Data to calculate outliers range for.\n    iqr_range: Tuple[int, int]\n        Two percentiles which define the IQR range\n    scale: float\n        The scale to multiply the IQR range for the outliers' detection. When the percentiles values are the same\n        (When many samples have the same value),\n        the scale will be modified based on the closest element to the percentiles values and\n        the `sharp_drop_ratio` parameter.\n    sharp_drop_ratio: float, default : 0.9\n        A threshold for the sharp drop outliers detection. When more than `sharp_drop_ratio` of the data\n        contain the same value the rest will be considered as outliers. Also used to normalize the scale in case\n        the percentiles values are the same.\n    Returns\n    -------\n    Tuple[float, float]\n        Tuple of lower limit and upper limit of outliers range\n    "
    if len(iqr_range) != 2 or any((x < 0 or x > 100 for x in iqr_range)) or all((x < 1 for x in iqr_range)):
        raise DeepchecksValueError('IQR range must contain two numbers between 0 to 100')
    if scale < 1:
        raise DeepchecksValueError('IQR scale must be greater than 1')
    (q1, q3) = np.percentile(data, sorted(iqr_range))
    if q1 == q3:
        common_percent_in_total = np.sum(data == q1) / len(data)
        if common_percent_in_total > sharp_drop_ratio:
            return (q1 - EPS, q1 + EPS)
        else:
            closest_dist_to_common = min(np.abs(data[data != q1] - q1))
            scale = sharp_drop_ratio + (scale - 1) * (1 - common_percent_in_total)
            return (q1 - closest_dist_to_common * scale, q1 + closest_dist_to_common * scale)
    else:
        iqr = q3 - q1
        return (q1 - scale * iqr, q3 + scale * iqr)

def sharp_drop_outliers_range(data_percents: Sequence, sharp_drop_ratio: float=0.9, max_outlier_percentage: float=0.05) -> Union[float, None]:
    if False:
        while True:
            i = 10
    'Calculate outliers range on the data given using sharp drop.\n\n    Parameters\n    ----------\n    data_percents : np.ndarray\n        Counts of data to calculate outliers range for. The data is assumed to be sorted from the most common to the\n        least common.\n    sharp_drop_ratio : float , default 0.9\n        The sharp drop threshold to use for the outliers detection.\n    max_outlier_percentage : float , default 0.05\n        The maximum percentage of data that can be considered as "outliers".\n    '
    if not 1 - EPS < sum(data_percents) < 1 + EPS:
        raise DeepchecksValueError('Data percents must sum to 1')
    for i in range(len(data_percents) - 1):
        if sum(data_percents[:i + 1]) < 1 - max_outlier_percentage:
            continue
        if 1 - data_percents[i + 1] / data_percents[i] >= sharp_drop_ratio:
            return data_percents[i + 1]
    else:
        return None