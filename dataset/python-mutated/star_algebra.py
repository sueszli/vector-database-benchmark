"""The star algebra mixin abstract base class."""
from abc import ABC, abstractmethod
from numbers import Integral
from qiskit.quantum_info.operators.mixins import MultiplyMixin
from qiskit.utils.deprecation import deprecate_func

class StarAlgebraMixin(MultiplyMixin, ABC):
    """Deprecated: The star algebra mixin class.
    Star algebra is an algebra with an adjoint.

    This class overrides:
        - ``*``, ``__mul__``, `__rmul__`,  -> :meth:`mul`
        - ``/``, ``__truediv__``,  -> :meth:`mul`
        - ``__neg__`` -> :meth:``mul`
        - ``+``, ``__add__``, ``__radd__`` -> :meth:`add`
        - ``-``, ``__sub__``, `__rsub__`,  -> :meth:a`add`
        - ``@``, ``__matmul__`` -> :meth:`compose`
        - ``**``, ``__pow__`` -> :meth:`power`
        - ``~``, ``__invert__`` -> :meth:`adjoint`

    The following abstract methods must be implemented by subclasses:
        - :meth:`mul(self, other)`
        - :meth:`add(self, other)`
        - :meth:`compose(self, other)`
        - :meth:`adjoint(self)`
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def mul(self, other: complex):
        if False:
            i = 10
            return i + 15
        'Return scalar multiplication of self and other, overloaded by `*`.'

    def __mul__(self, other: complex):
        if False:
            for i in range(10):
                print('nop')
        return self.mul(other)

    def _multiply(self, other: complex):
        if False:
            return 10
        return self.mul(other)

    @abstractmethod
    def add(self, other):
        if False:
            while True:
                i = 10
        'Return Operator addition of self and other, overloaded by `+`.'

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other == 0:
            return self
        return self.add(other)

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other == 0:
            return self
        return self.add(other)

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        return self.add(-other)

    def __rsub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.neg().add(other)

    @abstractmethod
    def compose(self, other):
        if False:
            while True:
                i = 10
        'Overloads the matrix multiplication operator `@` for self and other.\n        `Compose` computes operator composition between self and other (linear algebra-style:\n        A@B(x) = A(B(x))).\n        '

    def power(self, exponent: int):
        if False:
            print('Hello World!')
        'Return Operator composed with self multiple times, overloaded by ``**``.'
        if not isinstance(exponent, Integral):
            raise TypeError(f"Unsupported operand type(s) for **: '{type(self).__name__}' and '{type(exponent).__name__}'")
        if exponent < 1:
            raise ValueError('The input `exponent` must be a positive integer.')
        res = self
        for _ in range(1, exponent):
            res = res.compose(self)
        return res

    def __matmul__(self, other):
        if False:
            return 10
        return self.compose(other)

    def __pow__(self, exponent: int):
        if False:
            print('Hello World!')
        return self.power(exponent)

    @abstractmethod
    def adjoint(self):
        if False:
            while True:
                i = 10
        "Returns the complex conjugate transpose (dagger) of self.adjoint\n\n        Returns:\n            An operator equivalent to self's adjoint.\n        "

    def __invert__(self):
        if False:
            i = 10
            return i + 15
        'Overload unary `~` to return Operator adjoint.'
        return self.adjoint()