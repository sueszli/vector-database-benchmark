import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable, NaturalLanguage
from featuretools.primitives.base import TransformPrimitive
NATURAL_LANGUAGE_SEPARATORS = [' ', '.', ',', '!', '?', ';', '\n']

class NumUniqueSeparators(TransformPrimitive):
    """Calculates the number of unique separators.

    Description:
        Given a string and a list of separators, determine
        the number of unique separators in each string. If a string
        is null determined by pd.isnull return pd.NA.

    Args:
        separators (list, optional): a list of separator characters to count.
            ``[" ", ".", ",", "!", "?", ";", "\\n"]`` is used by default.

    Examples:
        >>> x = ["First. Line.", "This. is the second, line!", "notinlist@#$%^%&"]
        >>> num_unique_separators = NumUniqueSeparators([".", ",", "!"])
        >>> num_unique_separators(x).tolist()
        [1, 3, 0]
    """
    name = 'num_unique_separators'
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={'numeric'})

    def __init__(self, separators=NATURAL_LANGUAGE_SEPARATORS):
        if False:
            return 10
        assert separators is not None, 'separators needs to be defined'
        self.separators = separators

    def get_function(self):
        if False:
            return 10

        def count_unique_separator(s):
            if False:
                print('Hello World!')
            if pd.isnull(s):
                return pd.NA
            return len(set(self.separators).intersection(set(s)))

        def get_separator_count(column):
            if False:
                while True:
                    i = 10
            return column.apply(count_unique_separator)
        return get_separator_count