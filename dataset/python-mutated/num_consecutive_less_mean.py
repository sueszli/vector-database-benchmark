import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable
from featuretools.primitives.base import AggregationPrimitive

class NumConsecutiveLessMean(AggregationPrimitive):
    """Determines the length of the longest subsequence below the mean.

    Description:
        Given a list of numbers, find the longest subsequence of numbers
        smaller than the mean of the entire sequence. Return the length
        of the longest subsequence.

    Args:
        skipna (bool): If this is False and any value in x is `NaN`, then
            the result will be `NaN`. If True, `NaN` values are skipped.
            Default is True.

    Examples:
        >>> num_consecutive_less_mean = NumConsecutiveLessMean()
        >>> num_consecutive_less_mean([1, 2, 3, 4, 5, 6])
        3.0

        We can also control the way `NaN` values are handled.

        >>> num_consecutive_less_mean = NumConsecutiveLessMean(skipna=False)
        >>> num_consecutive_less_mean([1, 2, 3, 4, 5, 6, None])
        nan
    """
    name = 'num_consecutive_less_mean'
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={'numeric'})
    stack_on_self = False
    default_value = 0

    def __init__(self, skipna=True):
        if False:
            for i in range(10):
                print('nop')
        self.skipna = skipna

    def get_function(self):
        if False:
            i = 10
            return i + 15

        def num_consecutive_less_mean(x):
            if False:
                while True:
                    i = 10
            if x.isnull().all():
                return np.nan
            if not self.skipna and x.isnull().values.any():
                return np.nan
            x_mean = x.mean()
            if np.isnan(x.mean()):
                return np.nan
            x = x.dropna().reset_index(drop=True)
            above_mean_indices = x[x >= x_mean].index.to_series()
            if above_mean_indices.empty:
                return len(x)
            above_mean_indices[-1] = -1
            above_mean_indices[len(x)] = len(x)
            above_mean_indices = above_mean_indices.sort_index()
            above_mean_indices_shifted = above_mean_indices.shift(1)
            diffs = above_mean_indices - above_mean_indices_shifted
            max_gap = diffs.max() - 1
            return max_gap
        return num_consecutive_less_mean