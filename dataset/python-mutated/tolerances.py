"""
Tolerances mixin class.
"""
from abc import ABCMeta
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT

class TolerancesMeta(ABCMeta):
    """Metaclass to handle tolerances"""

    def __init__(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        cls._ATOL_DEFAULT = ATOL_DEFAULT
        cls._RTOL_DEFAULT = RTOL_DEFAULT
        cls._MAX_TOL = 0.0001
        super().__init__(cls, args, kwargs)

    def _check_value(cls, value, value_name):
        if False:
            return 10
        'Check if value is within valid ranges'
        if value < 0:
            raise QiskitError(f'Invalid {value_name} ({value}) must be non-negative.')
        if value > cls._MAX_TOL:
            raise QiskitError(f'Invalid {value_name} ({value}) must be less than {cls._MAX_TOL}.')

    @property
    def atol(cls):
        if False:
            while True:
                i = 10
        'Default absolute tolerance parameter for float comparisons.'
        return cls._ATOL_DEFAULT

    @atol.setter
    def atol(cls, value):
        if False:
            for i in range(10):
                print('nop')
        'Set default absolute tolerance parameter for float comparisons.'
        cls._check_value(value, 'atol')
        cls._ATOL_DEFAULT = value

    @property
    def rtol(cls):
        if False:
            while True:
                i = 10
        'Default relative tolerance parameter for float comparisons.'
        return cls._RTOL_DEFAULT

    @rtol.setter
    def rtol(cls, value):
        if False:
            i = 10
            return i + 15
        'Set default relative tolerance parameter for float comparisons.'
        cls._check_value(value, 'rtol')
        cls._RTOL_DEFAULT = value

class TolerancesMixin(metaclass=TolerancesMeta):
    """Mixin Class for tolerances"""

    @property
    def atol(self):
        if False:
            while True:
                i = 10
        'Default absolute tolerance parameter for float comparisons.'
        return self.__class__.atol

    @property
    def rtol(self):
        if False:
            return 10
        'Default relative tolerance parameter for float comparisons.'
        return self.__class__.rtol