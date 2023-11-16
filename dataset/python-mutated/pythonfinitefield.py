"""Implementation of :class:`PythonFiniteField` class. """
from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.domains.pythonintegerring import PythonIntegerRing
from sympy.utilities import public

@public
class PythonFiniteField(FiniteField):
    """Finite field based on Python's integers. """
    alias = 'FF_python'

    def __init__(self, mod, symmetric=True):
        if False:
            while True:
                i = 10
        return super().__init__(mod, PythonIntegerRing(), symmetric)