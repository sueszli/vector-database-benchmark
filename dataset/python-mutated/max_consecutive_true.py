from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Integer
from featuretools.primitives.base import AggregationPrimitive

class MaxConsecutiveTrue(AggregationPrimitive):
    """Determines the maximum number of consecutive True values in the input

    Examples:
        >>> max_consecutive_true = MaxConsecutiveTrue()
        >>> max_consecutive_true([True, False, True, True, True, False])
        3
    """
    name = 'max_consecutive_true'
    input_types = [ColumnSchema(logical_type=Boolean)]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={'numeric'})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        if False:
            for i in range(10):
                print('nop')

        def max_consecutive_true(x):
            if False:
                i = 10
                return i + 15
            not_equal = x != x.shift()
            not_equal_sum = not_equal.cumsum()
            consecutive = x.groupby(not_equal_sum).cumcount() + 1
            consecutive_true = consecutive * x
            return consecutive_true.max()
        return max_consecutive_true