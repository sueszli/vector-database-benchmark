from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Double, Integer
from featuretools.primitives.base import AggregationPrimitive

class MaxConsecutivePositives(AggregationPrimitive):
    """Determines the maximum number of consecutive positive values in the input

    Args:
        skipna (bool): Ignore any `NaN` values in the input. Default is True.

    Examples:
        >>> max_consecutive_positives = MaxConsecutivePositives()
        >>> max_consecutive_positives([1.0, -1.4, 2.4, 5.4, 2.9, -4.3])
        3

        `NaN` values can be ignored with the `skipna` parameter

        >>> max_consecutive_positives_skipna = MaxConsecutivePositives(skipna=False)
        >>> max_consecutive_positives_skipna([1.0, -1.4, 2.4, None, 2.9, 4.3])
        2
    """
    name = 'max_consecutive_positives'
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
            i = 10
            return i + 15

        def max_consecutive_positives(x):
            if False:
                while True:
                    i = 10
            if self.skipna:
                x = x.dropna()
            x[x.notnull()] = x[x.notnull()].gt(0)
            not_equal = x != x.shift()
            not_equal_sum = not_equal.cumsum()
            consecutive = x.groupby(not_equal_sum).cumcount() + 1
            consecutive_pos = consecutive * x
            return consecutive_pos.max()
        return max_consecutive_positives