import sympy
from sympy.multipledispatch import dispatch
__all__ = ['SingletonInt']

class SingletonInt(sympy.AtomicExpr):
    _op_priority = 99999

    def __new__(cls, *args, coeff=None, **kwargs):
        if False:
            print('Hello World!')
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    def __init__(self, val, *, coeff=1):
        if False:
            print('Hello World!')
        self._val = val
        self._coeff = coeff
        super().__init__()

    def _eval_Eq(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, SingletonInt) and other._val == self._val and (self._coeff == other._coeff):
            return sympy.true
        else:
            return sympy.false

    @property
    def free_symbols(self):
        if False:
            return 10
        return set()

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, SingletonInt):
            raise ValueError('SingletonInt cannot be multiplied by another SingletonInt')
        return SingletonInt(self._val, coeff=self._coeff * other)

    def __rmul__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, SingletonInt):
            raise ValueError('SingletonInt cannot be multiplied by another SingletonInt')
        return SingletonInt(self._val, coeff=self._coeff * other)

    def __add__(self, other):
        if False:
            return 10
        raise NotImplementedError('NYI')

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('NYI')

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        raise NotImplementedError('NYI')

    def __floordiv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('NYI')

    def __mod__(self, other):
        if False:
            print('Hello World!')
        raise NotImplementedError('NYI')

@dispatch(sympy.Integer, SingletonInt)
def _eval_is_ge(a, b):
    if False:
        return 10
    if a < 2:
        return sympy.false
    raise ValueError('Symbolic SingletonInt: Relation is indeterminate')

@dispatch(SingletonInt, sympy.Integer)
def _eval_is_ge(a, b):
    if False:
        i = 10
        return i + 15
    if b <= 2:
        return sympy.true
    raise ValueError('Symbolic SingletonInt: Relation is indeterminate')

@dispatch(SingletonInt, SingletonInt)
def _eval_is_ge(a, b):
    if False:
        i = 10
        return i + 15
    if a._val == b._val:
        if a._coeff >= b._coeff:
            return sympy.true
        else:
            return sympy.false
    raise ValueError('Symbolic SingletonInt: Relation is indeterminate')