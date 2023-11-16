import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable
from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive

class CountAboveMean(AggregationPrimitive):
    """Calculates the number of values that are above the mean.

    Args:
        skipna (bool): Determines if to use NA/null values. Defaults to
            True to skip NA/null.

    Examples:
        >>> count_above_mean = CountAboveMean()
        >>> count_above_mean([1, 2, 3, 4, 5])
        2

        The way NaNs are treated can be controlled.

        >>> count_above_mean_skipna = CountAboveMean(skipna=False)
        >>> count_above_mean_skipna([1, 2, 3, 4, 5, None])
        nan
    """
    name = 'count_above_mean'
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={'numeric'})
    stack_on_self = False

    def __init__(self, skipna=True):
        if False:
            return 10
        self.skipna = skipna

    def get_function(self):
        if False:
            print('Hello World!')

        def count_above_mean(x):
            if False:
                i = 10
                return i + 15
            mean = x.mean(skipna=self.skipna)
            if np.isnan(mean):
                return np.nan
            return len(x[x > mean])
        return count_above_mean