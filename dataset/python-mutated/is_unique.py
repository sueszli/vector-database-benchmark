from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable
from featuretools.primitives.base import AggregationPrimitive

class IsUnique(AggregationPrimitive):
    """Determines whether or not a series of discrete is all unique.

    Description:
        Given a series of discrete values, return True if each
        value in the series is unique. If any value is repeated,
        return False.

    Examples:
        >>> is_unique = IsUnique()
        >>> is_unique(['red', 'blue', 'green', 'yellow'])
        True

        If the series is not unique, return False

        >>> is_unique = IsUnique()
        >>> is_unique(['red', 'blue', 'green', 'blue'])
        False
    """
    name = 'is_unique'
    input_types = [ColumnSchema(semantic_tags={'category'})]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    stack_on_self = False
    default_value = False

    def get_function(self):
        if False:
            while True:
                i = 10

        def is_unique(x):
            if False:
                i = 10
                return i + 15
            return x.is_unique
        return is_unique