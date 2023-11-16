from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, Integer
from featuretools.primitives.base import AggregationPrimitive

class MaxConsecutiveNegatives(AggregationPrimitive):
    """Determines the maximum number of consecutive negative values in the input

    Args:
        skipna (bool): Ignore any `NaN` values in the input. Default is True.

    Examples:
        >>> max_consecutive_negatives = MaxConsecutiveNegatives()
        >>> max_consecutive_negatives([1.0, -1.4, -2.4, -5.4, 2.9, -4.3])
        3

        `NaN` values can be ignored with the `skipna` parameter

        >>> max_consecutive_negatives_skipna = MaxConsecutiveNegatives(skipna=False)
        >>> max_consecutive_negatives_skipna([1.0, 1.4, -2.4, None, -2.9, -4.3])
        2
    """
    name = 'max_consecutive_negatives'
    input_types = [[ColumnSchema(logical_type=Integer)], [ColumnSchema(logical_type=Double)]]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={'numeric'})
    stack_on_self = False
    default_value = 0

    def __init__(self, skipna=True):
        if False:
            i = 10
            return i + 15
        self.skipna = skipna

    def get_function(self):
        if False:
            print('Hello World!')

        def max_consecutive_negatives(x):
            if False:
                for i in range(10):
                    print('nop')
            if self.skipna:
                x = x.dropna()
            x[x.notnull()] = x[x.notnull()].lt(0)
            not_equal = x != x.shift()
            not_equal_sum = not_equal.cumsum()
            consecutive = x.groupby(not_equal_sum).cumcount() + 1
            consecutive_neg = consecutive * x
            return consecutive_neg.max()
        return max_consecutive_negatives