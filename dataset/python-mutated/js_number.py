import numbers
from typing import Optional, Union

class JSNumberBoundsException(Exception):
    pass

class JSNumber(object):
    """Utility class for exposing JavaScript Number constants."""
    MAX_SAFE_INTEGER = (1 << 53) - 1
    MIN_SAFE_INTEGER = -((1 << 53) - 1)
    MAX_VALUE = 1.7976931348623157e+308
    MIN_VALUE = 5e-324
    MIN_NEGATIVE_VALUE = -MAX_VALUE

    @classmethod
    def validate_int_bounds(cls, value: int, value_name: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Validate that an int value can be represented with perfect precision\n        by a JavaScript Number.\n\n        Parameters\n        ----------\n        value : int\n        value_name : str or None\n            The name of the value parameter. If specified, this will be used\n            in any exception that is thrown.\n\n        Raises\n        ------\n        JSNumberBoundsException\n            Raised with a human-readable explanation if the value falls outside\n            JavaScript int bounds.\n\n        '
        if value_name is None:
            value_name = 'value'
        if value < cls.MIN_SAFE_INTEGER:
            raise JSNumberBoundsException('%s (%s) must be >= -((1 << 53) - 1)' % (value_name, value))
        elif value > cls.MAX_SAFE_INTEGER:
            raise JSNumberBoundsException('%s (%s) must be <= (1 << 53) - 1' % (value_name, value))

    @classmethod
    def validate_float_bounds(cls, value: Union[int, float], value_name: Optional[str]) -> None:
        if False:
            while True:
                i = 10
        'Validate that a float value can be represented by a JavaScript Number.\n\n        Parameters\n        ----------\n        value : float\n        value_name : str or None\n            The name of the value parameter. If specified, this will be used\n            in any exception that is thrown.\n\n        Raises\n        ------\n        JSNumberBoundsException\n            Raised with a human-readable explanation if the value falls outside\n            JavaScript float bounds.\n\n        '
        if value_name is None:
            value_name = 'value'
        if not isinstance(value, (numbers.Integral, float)):
            raise JSNumberBoundsException('%s (%s) is not a float' % (value_name, value))
        elif value < cls.MIN_NEGATIVE_VALUE:
            raise JSNumberBoundsException('%s (%s) must be >= -1.797e+308' % (value_name, value))
        elif value > cls.MAX_VALUE:
            raise JSNumberBoundsException('%s (%s) must be <= 1.797e+308' % (value_name, value))