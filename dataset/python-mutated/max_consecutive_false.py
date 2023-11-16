from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Integer
from featuretools.primitives.base import AggregationPrimitive

class MaxConsecutiveFalse(AggregationPrimitive):
    """Determines the maximum number of consecutive False values in the input

    Examples:
        >>> max_consecutive_false = MaxConsecutiveFalse()
        >>> max_consecutive_false([True, False, False, True, True, False])
        2
    """
    name = 'max_consecutive_false'
    input_types = [ColumnSchema(logical_type=Boolean)]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={'numeric'})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        if False:
            while True:
                i = 10

        def max_consecutive_false(x):
            if False:
                while True:
                    i = 10
            x[x.notnull()] = ~x[x.notnull()].astype(bool)
            not_equal = x != x.shift()
            not_equal_sum = not_equal.cumsum()
            consecutive = x.groupby(not_equal_sum).cumcount() + 1
            consecutive_false = consecutive * x
            return consecutive_false.max()
        return max_consecutive_false