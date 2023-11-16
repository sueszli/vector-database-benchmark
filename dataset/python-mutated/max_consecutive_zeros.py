from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, Integer
from featuretools.primitives.base import AggregationPrimitive

class MaxConsecutiveZeros(AggregationPrimitive):
    """Determines the maximum number of consecutive zero values in the input

    Args:
        skipna (bool): Ignore any `NaN` values in the input. Default is True.

    Examples:
        >>> max_consecutive_zeros = MaxConsecutiveZeros()
        >>> max_consecutive_zeros([1.0, -1.4, 0, 0.0, 0, -4.3])
        3

        `NaN` values can be ignored with the `skipna` parameter

        >>> max_consecutive_zeros_skipna = MaxConsecutiveZeros(skipna=False)
        >>> max_consecutive_zeros_skipna([1.0, -1.4, 0, None, 0.0, -4.3])
        1
    """
    name = 'max_consecutive_zeros'
    input_types = [[ColumnSchema(logical_type=Integer)], [ColumnSchema(logical_type=Double)]]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={'numeric'})
    stack_on_self = False
    default_value = 0

    def __init__(self, skipna=True):
        if False:
            return 10
        self.skipna = skipna

    def get_function(self):
        if False:
            return 10

        def max_consecutive_zeros(x):
            if False:
                print('Hello World!')
            if self.skipna:
                x = x.dropna()
            x[x.notnull()] = x[x.notnull()].eq(0)
            not_equal = x != x.shift()
            not_equal_sum = not_equal.cumsum()
            consecutive = x.groupby(not_equal_sum).cumcount() + 1
            consecutive_zero = consecutive * x
            return consecutive_zero.max()
        return max_consecutive_zeros