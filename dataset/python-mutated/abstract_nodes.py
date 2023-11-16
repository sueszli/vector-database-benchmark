"""This module provides containers for python objects that are valid
printing targets but are not a subclass of SymPy's Printable.
"""
from sympy.core.containers import Tuple

class List(Tuple):
    """Represents a (frozen) (Python) list (for code printing purposes)."""

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, list):
            return self == List(*other)
        else:
            return self.args == other

    def __hash__(self):
        if False:
            return 10
        return super().__hash__()