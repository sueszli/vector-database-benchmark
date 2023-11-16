from functools import reduce
from ...exceptions import SmtlibError
import uuid
import re
import copy
from typing import Union, Optional, Dict, Tuple

class ExpressionException(SmtlibError):
    """
    Expression exception
    """
    pass

class ExpressionEvalError(SmtlibError):
    """Exception raised when an expression can't be concretized, typically
    when calling __bool__() fails to produce a True/False boolean result
    """
    pass

class XSlotted(type):
    """
    Metaclass that will propagate slots on multi-inheritance classes
    Every class should define __xslots__ (instead of __slots__)

    class Base(object, metaclass=XSlotted, abstract=True):
        pass

    class A(Base, abstract=True):
        __xslots__ = ('a',)
        pass

    class B(Base, abstract=True):
        __xslots__ = ('b',)
        pass

    class C(A, B):
        pass

    # Normal case / baseline
    class X(object):
        __slots__ = ('a', 'b')

    c = C()
    c.a = 1
    c.b = 2

    x = X()
    x.a = 1
    x.b = 2

    import sys
    print (sys.getsizeof(c),sys.getsizeof(x)) #same value
    """

    def __new__(cls, clsname, bases, attrs, abstract=False):
        if False:
            print('Hello World!')
        xslots = frozenset(attrs.get('__xslots__', ()))
        for base in bases:
            xslots = xslots.union(getattr(base, '__xslots__', ()))
        attrs['__xslots__'] = tuple(xslots)
        if abstract:
            attrs['__slots__'] = tuple()
        else:
            attrs['__slots__'] = attrs['__xslots__']
        attrs['__hash__'] = object.__hash__
        return super().__new__(cls, clsname, bases, attrs)

class Expression(object, metaclass=XSlotted, abstract=True):
    """Abstract taintable Expression."""
    __xslots__: Tuple[str, ...] = ('_taint',)

    def __init__(self, *, taint: Union[tuple, frozenset]=(), **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self.__class__ is Expression:
            raise TypeError
        super().__init__()
        self._taint = frozenset(taint)

    def __repr__(self):
        if False:
            return 10
        return '<{:s} at {:x}{:s}>'.format(type(self).__name__, id(self), self.taint and '-T' or '')

    @property
    def is_tainted(self):
        if False:
            print('Hello World!')
        return len(self._taint) != 0

    @property
    def taint(self):
        if False:
            while True:
                i = 10
        return self._taint

def issymbolic(value) -> bool:
    if False:
        while True:
            i = 10
    '\n    Helper to determine whether an object is symbolic (e.g checking\n    if data read from memory is symbolic)\n\n    :param object value: object to check\n    :return: whether `value` is symbolic\n    :rtype: bool\n    '
    return isinstance(value, Expression)

def istainted(arg, taint=None):
    if False:
        while True:
            i = 10
    "\n    Helper to determine whether an object if tainted.\n    :param arg: a value or Expression\n    :param taint: a regular expression matching a taint value (eg. 'IMPORTANT.*'). If None, this function checks for any taint value.\n    "
    if not issymbolic(arg):
        return False
    if taint is None:
        return len(arg.taint) != 0
    for arg_taint in arg.taint:
        m = re.match(taint, arg_taint, re.DOTALL | re.IGNORECASE)
        if m:
            return True
    return False

def get_taints(arg, taint=None):
    if False:
        print('Hello World!')
    "\n    Helper to list an object taints.\n    :param arg: a value or Expression\n    :param taint: a regular expression matching a taint value (eg. 'IMPORTANT.*'). If None, this function checks for any taint value.\n    "
    if not issymbolic(arg):
        return
    for arg_taint in arg.taint:
        if taint is not None:
            m = re.match(taint, arg_taint, re.DOTALL | re.IGNORECASE)
            if m:
                yield arg_taint
        else:
            yield arg_taint
    return

def taint_with(arg, *taints, value_bits=256, index_bits=256):
    if False:
        print('Hello World!')
    "\n    Helper to taint a value.\n    :param arg: a value or Expression\n    :param taint: a regular expression matching a taint value (eg. 'IMPORTANT.*'). If None, this function checks for any taint value.\n    "
    tainted_fset = frozenset(tuple(taints))
    if not issymbolic(arg):
        if isinstance(arg, int):
            arg = BitVecConstant(size=value_bits, value=arg)
            arg._taint = tainted_fset
        else:
            raise ValueError('type not supported')
    elif isinstance(arg, BitVecVariable):
        arg = arg + BitVecConstant(size=value_bits, value=0, taint=tainted_fset)
    else:
        arg = copy.copy(arg)
        arg._taint |= tainted_fset
    return arg

class Bool(Expression, abstract=True):
    """Bool expressions represent symbolic value of truth"""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)

    def cast(self, value: Union['Bool', int, bool], **kwargs) -> Union['BoolConstant', 'Bool']:
        if False:
            while True:
                i = 10
        if isinstance(value, Bool):
            return value
        return BoolConstant(value=bool(value), **kwargs)

    def __cmp__(self, *args):
        if False:
            while True:
                i = 10
        raise NotImplementedError('CMP for Bool')

    def __invert__(self):
        if False:
            i = 10
            return i + 15
        return BoolNot(value=self)

    def __eq__(self, other):
        if False:
            return 10
        return BoolEqual(a=self, b=self.cast(other))

    def __hash__(self):
        if False:
            while True:
                i = 10
        return object.__hash__(self)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return BoolNot(value=self == self.cast(other))

    def __and__(self, other):
        if False:
            i = 10
            return i + 15
        return BoolAnd(a=self, b=self.cast(other))

    def __or__(self, other):
        if False:
            i = 10
            return i + 15
        return BoolOr(a=self, b=self.cast(other))

    def __xor__(self, other):
        if False:
            i = 10
            return i + 15
        return BoolXor(a=self, b=self.cast(other))

    def __rand__(self, other):
        if False:
            print('Hello World!')
        return BoolAnd(a=self.cast(other), b=self)

    def __ror__(self, other):
        if False:
            return 10
        return BoolOr(a=self.cast(other), b=self)

    def __rxor__(self, other):
        if False:
            return 10
        return BoolXor(a=self.cast(other), b=self)

    def __bool__(self):
        if False:
            while True:
                i = 10
        from .visitors import simplify
        x = simplify(self)
        if isinstance(x, Constant):
            return x.value
        raise ExpressionEvalError('__bool__ for Bool')

class BoolVariable(Bool):
    __xslots__: Tuple[str, ...] = ('_name',)

    def __init__(self, *, name: str, **kwargs):
        if False:
            return 10
        assert ' ' not in name
        super().__init__(**kwargs)
        self._name = name

    @property
    def name(self):
        if False:
            while True:
                i = 10
        return self._name

    def __copy__(self, memo=''):
        if False:
            return 10
        raise ExpressionException('Copying of Variables is not allowed.')

    def __deepcopy__(self, memo=''):
        if False:
            return 10
        raise ExpressionException('Copying of Variables is not allowed.')

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<{:s}({:s}) at {:x}>'.format(type(self).__name__, self.name, id(self))

    @property
    def declaration(self):
        if False:
            i = 10
            return i + 15
        return f'(declare-fun {self.name} () Bool)'

class BoolConstant(Bool):
    __xslots__: Tuple[str, ...] = ('_value',)

    def __init__(self, *, value: bool, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self._value = value

    def __bool__(self):
        if False:
            return 10
        return self.value

    @property
    def value(self):
        if False:
            print('Hello World!')
        return self._value

class BoolOperation(Bool, abstract=True):
    """An operation that results in a Bool"""
    __xslots__: Tuple[str, ...] = ('_operands',)

    def __init__(self, *, operands: Tuple, **kwargs):
        if False:
            print('Hello World!')
        self._operands = operands
        kwargs.setdefault('taint', reduce(lambda x, y: x.union(y.taint), operands, frozenset()))
        super().__init__(**kwargs)

    @property
    def operands(self):
        if False:
            return 10
        return self._operands

class BoolNot(BoolOperation):

    def __init__(self, *, value, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(operands=(value,), **kwargs)

class BoolAnd(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(operands=(a, b), **kwargs)

class BoolOr(BoolOperation):

    def __init__(self, *, a: 'Bool', b: 'Bool', **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(operands=(a, b), **kwargs)

class BoolXor(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            return 10
        super().__init__(operands=(a, b), **kwargs)

class BoolITE(BoolOperation):

    def __init__(self, *, cond: 'Bool', true: 'Bool', false: 'Bool', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(operands=(cond, true, false), **kwargs)

class BitVec(Expression, abstract=True):
    """BitVector expressions have a fixed bit size"""
    __xslots__: Tuple[str, ...] = ('size',)

    def __init__(self, *, size, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.size = size

    @property
    def mask(self):
        if False:
            i = 10
            return i + 15
        return (1 << self.size) - 1

    @property
    def signmask(self):
        if False:
            print('Hello World!')
        return 1 << self.size - 1

    def cast(self, value: Union['BitVec', str, int, bytes], **kwargs) -> Union['BitVecConstant', 'BitVec']:
        if False:
            while True:
                i = 10
        if isinstance(value, BitVec):
            assert value.size == self.size
            return value
        if isinstance(value, (str, bytes)) and len(value) == 1:
            value = ord(value)
        if not isinstance(value, int):
            value = int(value)
        return BitVecConstant(size=self.size, value=value, **kwargs)

    def __add__(self, other):
        if False:
            print('Hello World!')
        return BitVecAdd(a=self, b=self.cast(other))

    def __sub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return BitVecSub(a=self, b=self.cast(other))

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        return BitVecMul(a=self, b=self.cast(other))

    def __mod__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return BitVecMod(a=self, b=self.cast(other))

    def __lshift__(self, other):
        if False:
            return 10
        return BitVecShiftLeft(a=self, b=self.cast(other))

    def __rshift__(self, other):
        if False:
            return 10
        return BitVecShiftRight(a=self, b=self.cast(other))

    def __and__(self, other):
        if False:
            print('Hello World!')
        return BitVecAnd(a=self, b=self.cast(other))

    def __xor__(self, other):
        if False:
            i = 10
            return i + 15
        return BitVecXor(a=self, b=self.cast(other))

    def __or__(self, other):
        if False:
            print('Hello World!')
        return BitVecOr(a=self, b=self.cast(other))

    def __div__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return BitVecDiv(a=self, b=self.cast(other))

    def __truediv__(self, other):
        if False:
            return 10
        return BitVecDiv(a=self, b=self.cast(other))

    def __floordiv__(self, other):
        if False:
            i = 10
            return i + 15
        return self / other

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        return BitVecAdd(a=self.cast(other), b=self)

    def __rsub__(self, other):
        if False:
            print('Hello World!')
        return BitVecSub(a=self.cast(other), b=self)

    def __rmul__(self, other):
        if False:
            print('Hello World!')
        return BitVecMul(a=self.cast(other), b=self)

    def __rmod__(self, other):
        if False:
            while True:
                i = 10
        return BitVecMod(a=self.cast(other), b=self)

    def __rtruediv__(self, other):
        if False:
            while True:
                i = 10
        return BitVecDiv(a=self.cast(other), b=self)

    def __rdiv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return BitVecDiv(a=self.cast(other), b=self)

    def __rlshift__(self, other):
        if False:
            i = 10
            return i + 15
        return BitVecShiftLeft(a=self.cast(other), b=self)

    def __rrshift__(self, other):
        if False:
            return 10
        return BitVecShiftRight(a=self.cast(other), b=self)

    def __rand__(self, other):
        if False:
            return 10
        return BitVecAnd(a=self.cast(other), b=self)

    def __rxor__(self, other):
        if False:
            i = 10
            return i + 15
        return BitVecXor(a=self.cast(other), b=self)

    def __ror__(self, other):
        if False:
            print('Hello World!')
        return BitVecOr(a=self.cast(other), b=self)

    def __invert__(self):
        if False:
            return 10
        return BitVecXor(a=self, b=self.cast(self.mask))

    def __lt__(self, other):
        if False:
            return 10
        return LessThan(a=self, b=self.cast(other))

    def __le__(self, other):
        if False:
            print('Hello World!')
        return LessOrEqual(a=self, b=self.cast(other))

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return BoolEqual(a=self, b=self.cast(other))

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return object.__hash__(self)

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return BoolNot(value=BoolEqual(a=self, b=self.cast(other)))

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return GreaterThan(a=self, b=self.cast(other))

    def __ge__(self, other):
        if False:
            return 10
        return GreaterOrEqual(a=self, b=self.cast(other))

    def __neg__(self):
        if False:
            print('Hello World!')
        return BitVecNeg(a=self)

    def ugt(self, other):
        if False:
            for i in range(10):
                print('nop')
        return UnsignedGreaterThan(a=self, b=self.cast(other))

    def uge(self, other):
        if False:
            for i in range(10):
                print('nop')
        return UnsignedGreaterOrEqual(a=self, b=self.cast(other))

    def ult(self, other):
        if False:
            return 10
        return UnsignedLessThan(a=self, b=self.cast(other))

    def ule(self, other):
        if False:
            print('Hello World!')
        return UnsignedLessOrEqual(a=self, b=self.cast(other))

    def udiv(self, other):
        if False:
            while True:
                i = 10
        return BitVecUnsignedDiv(a=self, b=self.cast(other))

    def rudiv(self, other):
        if False:
            for i in range(10):
                print('nop')
        return BitVecUnsignedDiv(a=self.cast(other), b=self)

    def sdiv(self, other):
        if False:
            print('Hello World!')
        return BitVecDiv(a=self, b=self.cast(other))

    def rsdiv(self, other):
        if False:
            for i in range(10):
                print('nop')
        return BitVecDiv(a=self.cast(other), b=self)

    def srem(self, other):
        if False:
            while True:
                i = 10
        return BitVecRem(a=self, b=self.cast(other))

    def rsrem(self, other):
        if False:
            i = 10
            return i + 15
        return BitVecRem(a=self.cast(other), b=self)

    def urem(self, other):
        if False:
            while True:
                i = 10
        return BitVecUnsignedRem(a=self, b=self.cast(other))

    def rurem(self, other):
        if False:
            print('Hello World!')
        return BitVecUnsignedRem(a=self.cast(other), b=self)

    def sar(self, other):
        if False:
            while True:
                i = 10
        return BitVecArithmeticShiftRight(a=self, b=self.cast(other))

    def sal(self, other):
        if False:
            while True:
                i = 10
        return BitVecArithmeticShiftLeft(a=self, b=self.cast(other))

    def Bool(self):
        if False:
            i = 10
            return i + 15
        return self != 0

class BitVecVariable(BitVec):
    __xslots__: Tuple[str, ...] = ('_name',)

    def __init__(self, *, size: int, name: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert ' ' not in name
        super().__init__(size=size, **kwargs)
        self._name = name

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    def __copy__(self, memo=''):
        if False:
            return 10
        raise ExpressionException('Copying of Variables is not allowed.')

    def __deepcopy__(self, memo=''):
        if False:
            print('Hello World!')
        raise ExpressionException('Copying of Variables is not allowed.')

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<{:s}({:s}) at {:x}>'.format(type(self).__name__, self.name, id(self))

    @property
    def declaration(self):
        if False:
            for i in range(10):
                print('nop')
        return f'(declare-fun {self.name} () (_ BitVec {self.size}))'

class BitVecConstant(BitVec):
    __xslots__: Tuple[str, ...] = ('_value',)

    def __init__(self, *, size: int, value: int, **kwargs):
        if False:
            while True:
                i = 10
        MASK = (1 << size) - 1
        self._value = value & MASK
        super().__init__(size=size, **kwargs)

    def __bool__(self):
        if False:
            print('Hello World!')
        return self.value != 0

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if self.taint:
            return super().__eq__(other)
        return self.value == other

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return super().__hash__()

    @property
    def value(self):
        if False:
            print('Hello World!')
        return self._value

    @property
    def signed_value(self):
        if False:
            while True:
                i = 10
        if self._value & self.signmask:
            return self._value - (1 << self.size)
        else:
            return self._value

class BitVecOperation(BitVec, abstract=True):
    """An operation that results in a BitVec"""
    __xslots__: Tuple[str, ...] = ('_operands',)

    def __init__(self, *, size, operands: Tuple, **kwargs):
        if False:
            return 10
        self._operands = operands
        kwargs.setdefault('taint', reduce(lambda x, y: x.union(y.taint), operands, frozenset()))
        super().__init__(size=size, **kwargs)

    @property
    def operands(self):
        if False:
            print('Hello World!')
        return self._operands

class BitVecAdd(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            return 10
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecSub(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            return 10
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecMul(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecDiv(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecUnsignedDiv(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecMod(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecRem(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecUnsignedRem(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecShiftLeft(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecShiftRight(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecArithmeticShiftLeft(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecArithmeticShiftRight(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            return 10
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecAnd(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecOr(BitVecOperation):

    def __init__(self, *, a: BitVec, b: BitVec, **kwargs):
        if False:
            print('Hello World!')
        assert a.size == b.size
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecXor(BitVecOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(size=a.size, operands=(a, b), **kwargs)

class BitVecNot(BitVecOperation):

    def __init__(self, *, a, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(size=a.size, operands=(a,), **kwargs)

class BitVecNeg(BitVecOperation):

    def __init__(self, *, a, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(size=a.size, operands=(a,), **kwargs)

class LessThan(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(operands=(a, b), **kwargs)

class LessOrEqual(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(operands=(a, b), **kwargs)

class BoolEqual(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            return 10
        if isinstance(a, BitVec) or isinstance(b, BitVec):
            assert a.size == b.size
        super().__init__(operands=(a, b), **kwargs)

class GreaterThan(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            return 10
        assert a.size == b.size
        super().__init__(operands=(a, b), **kwargs)

class GreaterOrEqual(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert a.size == b.size
        super().__init__(operands=(a, b), **kwargs)

class UnsignedLessThan(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert a.size == b.size
        super().__init__(operands=(a, b), **kwargs)

class UnsignedLessOrEqual(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            print('Hello World!')
        assert a.size == b.size
        super().__init__(operands=(a, b), **kwargs)

class UnsignedGreaterThan(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            while True:
                i = 10
        assert a.size == b.size
        super().__init__(operands=(a, b), **kwargs)

class UnsignedGreaterOrEqual(BoolOperation):

    def __init__(self, *, a, b, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert a.size == b.size
        super(UnsignedGreaterOrEqual, self).__init__(operands=(a, b), **kwargs)

class Array(Expression, abstract=True):
    """An Array expression is an unmutable mapping from bitvector to bitvector

    array.index_bits is the number of bits used for addressing a value
    array.value_bits is the number of bits used in the values
    array.index_max counts the valid indexes starting at 0. Accessing outside the bound is undefined
    """
    __xslots__: Tuple[str, ...] = ('_index_bits', '_index_max', '_value_bits')

    def __init__(self, *, index_bits: int, index_max: Optional[int], value_bits: int, **kwargs):
        if False:
            while True:
                i = 10
        assert index_bits in (32, 64, 256)
        assert value_bits in (8, 16, 32, 64, 256)
        assert index_max is None or (index_max >= 0 and index_max < 2 ** index_bits)
        self._index_bits = index_bits
        self._index_max = index_max
        self._value_bits = value_bits
        super().__init__(**kwargs)
        assert type(self) is not Array, 'Abstract class'

    def _get_size(self, index):
        if False:
            for i in range(10):
                print('nop')
        (start, stop) = self._fix_index(index)
        size = stop - start
        if isinstance(size, BitVec):
            from .visitors import simplify
            size = simplify(size)
        else:
            size = BitVecConstant(size=self.index_bits, value=size)
        assert isinstance(size, BitVecConstant)
        return size.value

    def _fix_index(self, index):
        if False:
            return 10
        '\n        :param slice index:\n        '
        (stop, start) = (index.stop, index.start)
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        return (start, stop)

    def cast(self, possible_array):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(possible_array, bytearray):
            arr = ArrayVariable(index_bits=self.index_bits, index_max=len(possible_array), value_bits=8, name='cast{}'.format(uuid.uuid1()))
            for (pos, byte) in enumerate(possible_array):
                arr = arr.store(pos, byte)
            return arr
        raise ValueError

    def cast_index(self, index: Union[int, 'BitVec']) -> Union['BitVecConstant', 'BitVec']:
        if False:
            return 10
        if isinstance(index, int):
            return BitVecConstant(size=self.index_bits, value=index)
        assert index.size == self.index_bits
        return index

    def cast_value(self, value: Union['BitVec', str, bytes, int]) -> Union['BitVecConstant', 'BitVec']:
        if False:
            while True:
                i = 10
        if isinstance(value, BitVec):
            assert value.size == self.value_bits
            return value
        if isinstance(value, (str, bytes)) and len(value) == 1:
            value = ord(value)
        if not isinstance(value, int):
            value = int(value)
        return BitVecConstant(size=self.value_bits, value=value)

    def __len__(self):
        if False:
            while True:
                i = 10
        if self.index_max is None:
            raise ExpressionException('Array max index not set')
        return self.index_max

    @property
    def index_bits(self):
        if False:
            i = 10
            return i + 15
        return self._index_bits

    @property
    def value_bits(self):
        if False:
            i = 10
            return i + 15
        return self._value_bits

    @property
    def index_max(self):
        if False:
            for i in range(10):
                print('nop')
        return self._index_max

    def select(self, index):
        if False:
            print('Hello World!')
        index = self.cast_index(index)
        return ArraySelect(array=self, index=index)

    def store(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        return ArrayStore(array=self, index=self.cast_index(index), value=self.cast_value(value))

    def write(self, offset, buf):
        if False:
            i = 10
            return i + 15
        if not isinstance(buf, (Array, bytearray)):
            raise TypeError(f'Array or bytearray expected got {type(buf)}')
        arr = self
        for (i, val) in enumerate(buf):
            arr = arr.store(offset + i, val)
        return arr

    def read(self, offset, size):
        if False:
            for i in range(10):
                print('nop')
        return ArraySlice(array=self, offset=offset, size=size)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(index, slice):
            (start, stop) = self._fix_index(index)
            size = self._get_size(index)
            return ArraySlice(array=self, offset=start, size=size)
        elif self.index_max is not None:
            if not isinstance(index, Expression) and index >= self.index_max:
                raise IndexError
        return self.select(self.cast_index(index))

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15

        def compare_buffers(a, b):
            if False:
                for i in range(10):
                    print('nop')
            if len(a) != len(b):
                return BoolConstant(value=False)
            cond = BoolConstant(value=True)
            for i in range(len(a)):
                cond = BoolAnd(a=cond.cast(a[i] == b[i]), b=cond)
                if cond is BoolConstant(value=False):
                    return BoolConstant(value=False)
            return cond
        return compare_buffers(self, other)

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return BoolNot(value=self == other)

    def __hash__(self):
        if False:
            print('Hello World!')
        return super().__hash__()

    @property
    def underlying_variable(self):
        if False:
            for i in range(10):
                print('nop')
        array = self
        while not isinstance(array, ArrayVariable):
            array = array.array
        return array

    def read_BE(self, address, size):
        if False:
            for i in range(10):
                print('nop')
        address = self.cast_index(address)
        bytes = []
        for offset in range(size):
            bytes.append(self.get(address + offset, 0))
        return BitVecConcat(size_dest=size * self.value_bits, operands=tuple(bytes))

    def read_LE(self, address, size):
        if False:
            i = 10
            return i + 15
        address = self.cast_index(address)
        bytes = []
        for offset in range(size):
            bytes.append(self.get(address + offset, 0))
        return BitVecConcat(size_dest=size * self.value_bits, operands=tuple(reversed(bytes)))

    def write_BE(self, address, value, size):
        if False:
            i = 10
            return i + 15
        address = self.cast_index(address)
        value = BitVecConstant(size=size * self.value_bits, value=0).cast(value)
        array = self
        for offset in range(size):
            array = array.store(address + offset, BitVecExtract(operand=value, offset=(size - 1 - offset) * self.value_bits, size=self.value_bits))
        return array

    def write_LE(self, address, value, size):
        if False:
            i = 10
            return i + 15
        address = self.cast_index(address)
        value = BitVecConstant(size=size * self.value_bits, value=0).cast(value)
        array = self
        for offset in reversed(range(size)):
            array = array.store(address + offset, BitVecExtract(operand=value, offset=(size - 1 - offset) * self.value_bits, size=self.value_bits))
        return array

    def __add__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, (Array, bytearray)):
            raise TypeError("can't concat Array to {}".format(type(other)))
        if isinstance(other, Array):
            if self.index_bits != other.index_bits or self.value_bits != other.value_bits:
                raise ValueError('Array sizes do not match for concatenation')
        from .visitors import simplify
        new_arr = ArrayProxy(array=ArrayVariable(index_bits=self.index_bits, index_max=self.index_max + len(other), value_bits=self.value_bits, name='concatenation{}'.format(uuid.uuid1())))
        for index in range(self.index_max):
            new_arr[index] = simplify(self[index])
        for index in range(len(other)):
            new_arr[index + self.index_max] = simplify(other[index])
        return new_arr

    def __radd__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, (Array, bytearray, bytes)):
            raise TypeError("can't concat Array to {}".format(type(other)))
        if isinstance(other, Array):
            if self.index_bits != other.index_bits or self.value_bits != other.value_bits:
                raise ValueError('Array sizes do not match for concatenation')
        from .visitors import simplify
        new_arr = ArrayProxy(array=ArrayVariable(index_bits=self.index_bits, index_max=self.index_max + len(other), value_bits=self.value_bits, name='concatenation{}'.format(uuid.uuid1())))
        for index in range(len(other)):
            new_arr[index] = simplify(other[index])
        _concrete_cache = new_arr._concrete_cache
        for index in range(self.index_max):
            new_arr[index + len(other)] = simplify(self[index])
        new_arr._concrete_cache.update(_concrete_cache)
        return new_arr

class ArrayVariable(Array):
    __xslots__: Tuple[str, ...] = ('_name',)

    def __init__(self, *, index_bits, index_max, value_bits, name, **kwargs):
        if False:
            return 10
        assert ' ' not in name
        super().__init__(index_bits=index_bits, index_max=index_max, value_bits=value_bits, **kwargs)
        self._name = name

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self._name

    def __copy__(self, memo=''):
        if False:
            return 10
        raise ExpressionException('Copying of Variables is not allowed.')

    def __deepcopy__(self, memo=''):
        if False:
            i = 10
            return i + 15
        raise ExpressionException('Copying of Variables is not allowed.')

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<{:s}({:s}) at {:x}>'.format(type(self).__name__, self.name, id(self))

    @property
    def declaration(self):
        if False:
            while True:
                i = 10
        return f'(declare-fun {self.name} () (Array (_ BitVec {self.index_bits}) (_ BitVec {self.value_bits})))'

class ArrayOperation(Array):
    """An operation that result in an Array"""
    __xslots__: Tuple[str, ...] = ('_operands',)

    def __init__(self, *, array: Array, operands: Tuple, **kwargs):
        if False:
            i = 10
            return i + 15
        self._operands = (array, *operands)
        kwargs.setdefault('taint', reduce(lambda x, y: x.union(y.taint), operands, frozenset()))
        super().__init__(index_bits=array.index_bits, index_max=array.index_max, value_bits=array.value_bits, **kwargs)

    @property
    def operands(self):
        if False:
            return 10
        return self._operands

class ArrayStore(ArrayOperation):

    def __init__(self, *, array: 'Array', index: 'BitVec', value: 'BitVec', **kwargs):
        if False:
            print('Hello World!')
        assert index.size == array.index_bits
        assert value.size == array.value_bits
        super().__init__(array=array, operands=(index, value), **kwargs)

    @property
    def array(self):
        if False:
            i = 10
            return i + 15
        return self.operands[0]

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self.operands[0].name

    @property
    def index(self):
        if False:
            print('Hello World!')
        return self.operands[1]

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        return self.operands[2]

    def __getstate__(self):
        if False:
            return 10
        state = {}
        array = self
        items = []
        while isinstance(array, ArrayStore):
            items.append((array.index, array.value))
            array = array.array
        state['_array'] = array
        state['_items'] = items
        state['_taint'] = self.taint
        return state

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        array = state['_array']
        for (index, value) in reversed(state['_items'][0:]):
            array = array.store(index, value)
        self._index_bits = array.index_bits
        self._index_max = array.index_max
        self._value_bits = array.value_bits
        self._taint = state['_taint']
        (index, value) = state['_items'][0]
        self._operands = (array, index, value)

class ArraySlice(ArrayOperation):
    __xslots__: Tuple[str, ...] = ('_slice_offset', '_slice_size')

    def __init__(self, *, array: Union['Array', 'ArrayProxy'], offset: int, size: int, **kwargs):
        if False:
            i = 10
            return i + 15
        if not isinstance(array, Array):
            raise ValueError('Array expected')
        if isinstance(array, ArrayProxy):
            array = array._array
        self._operands = (array,)
        super().__init__(array=array, operands=(self.cast_index(offset), self.cast_index(size)), **kwargs)
        self._slice_offset = offset
        self._slice_size = size

    @property
    def array(self):
        if False:
            i = 10
            return i + 15
        return self._operands[0]

    @property
    def underlying_variable(self):
        if False:
            while True:
                i = 10
        return self.array.underlying_variable

    @property
    def index_bits(self):
        if False:
            while True:
                i = 10
        return self.array.index_bits

    @property
    def index_max(self):
        if False:
            print('Hello World!')
        return self._slice_size

    @property
    def value_bits(self):
        if False:
            print('Hello World!')
        return self.array.value_bits

    def select(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self.array.select(index + self._slice_offset)

    def store(self, index, value):
        if False:
            return 10
        return ArraySlice(array=self.array.store(index + self._slice_offset, value), offset=self._slice_offset, size=self._slice_size)

class ArrayProxy(Array):
    __xslots__: Tuple[str, ...] = ('constraints', '_default', '_concrete_cache', '_written', '_array', '_name')

    def __init__(self, *, array: Array, default: Optional[int]=None, **kwargs):
        if False:
            return 10
        self._default = default
        self._concrete_cache: Dict[int, int] = {}
        self._written = None
        if isinstance(array, ArrayProxy):
            super().__init__(index_bits=array.index_bits, index_max=array.index_max, value_bits=array.value_bits, **kwargs)
            self._array: Array = array._array
            self._name: str = array._name
            if default is None:
                self._default = array._default
            self._concrete_cache = dict(array._concrete_cache)
            self._written = set(array.written)
        elif isinstance(array, ArrayVariable):
            super().__init__(index_bits=array.index_bits, index_max=array.index_max, value_bits=array.value_bits, **kwargs)
            self._array = array
            self._name = array.name
        else:
            super().__init__(index_bits=array.index_bits, index_max=array.index_max, value_bits=array.value_bits, **kwargs)
            self._name = array.underlying_variable.name
            self._array = array

    @property
    def underlying_variable(self):
        if False:
            return 10
        return self._array.underlying_variable

    @property
    def array(self):
        if False:
            return 10
        return self._array

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    @property
    def operands(self):
        if False:
            print('Hello World!')
        return (self._array,)

    @property
    def index_bits(self):
        if False:
            return 10
        return self._array.index_bits

    @property
    def index_max(self):
        if False:
            for i in range(10):
                print('nop')
        return self._array.index_max

    @property
    def value_bits(self):
        if False:
            i = 10
            return i + 15
        return self._array.value_bits

    @property
    def taint(self):
        if False:
            for i in range(10):
                print('nop')
        return self._array.taint

    def select(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self.get(index)

    def store(self, index, value):
        if False:
            while True:
                i = 10
        if not isinstance(index, Expression):
            index = self.cast_index(index)
        if not isinstance(value, Expression):
            value = self.cast_value(value)
        from .visitors import simplify
        index = simplify(index)
        if isinstance(index, Constant):
            self._concrete_cache[index.value] = value
        else:
            self._concrete_cache = {}
        self.written.add(index)
        self._array = self._array.store(index, value)
        return self

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(index, slice):
            (start, stop) = self._fix_index(index)
            size = self._get_size(index)
            array_proxy_slice = ArrayProxy(array=ArraySlice(array=self, offset=start, size=size), default=self._default)
            array_proxy_slice._concrete_cache = {}
            for (k, v) in self._concrete_cache.items():
                if k >= start and k < start + size:
                    array_proxy_slice._concrete_cache[k - start] = v
            for i in self.written:
                array_proxy_slice.written.add(i - start)
            return array_proxy_slice
        else:
            if self.index_max is not None:
                if not isinstance(index, Expression) and index >= self.index_max:
                    raise IndexError
            return self.get(index, self._default)

    def __setitem__(self, index, value):
        if False:
            while True:
                i = 10
        if isinstance(index, slice):
            (start, stop) = self._fix_index(index)
            size = self._get_size(index)
            assert len(value) == size
            for i in range(size):
                self.store(start + i, value[i])
        else:
            self.store(index, value)

    def __getstate__(self):
        if False:
            while True:
                i = 10
        state = {}
        state['_default'] = self._default
        state['_array'] = self._array
        state['name'] = self.name
        state['_concrete_cache'] = self._concrete_cache
        return state

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self._default = state['_default']
        self._array = state['_array']
        self._name = state['name']
        self._concrete_cache = state['_concrete_cache']
        self._written = None

    def __copy__(self):
        if False:
            while True:
                i = 10
        return ArrayProxy(array=self)

    @property
    def written(self):
        if False:
            for i in range(10):
                print('nop')
        if self._written is None:
            written = set()
            array = self._array
            offset = 0
            while not isinstance(array, ArrayVariable):
                if isinstance(array, ArraySlice):
                    offset += array._slice_offset
                else:
                    written.add(array.index - offset)
                array = array.array
            assert isinstance(array, ArrayVariable)
            self._written = written
        return self._written

    def is_known(self, index):
        if False:
            print('Hello World!')
        if isinstance(index, Constant) and index.value in self._concrete_cache:
            return BoolConstant(value=True)
        is_known_index = BoolConstant(value=False)
        written = self.written
        if isinstance(index, Constant):
            for i in written:
                if isinstance(i, Constant) and index.value == i.value:
                    return BoolConstant(value=True)
                is_known_index = BoolOr(a=is_known_index.cast(index == i), b=is_known_index)
            return is_known_index
        for known_index in written:
            is_known_index = BoolOr(a=is_known_index.cast(index == known_index), b=is_known_index)
        return is_known_index

    def get(self, index, default=None):
        if False:
            i = 10
            return i + 15
        if default is None:
            default = self._default
        index = self.cast_index(index)
        if self.index_max is not None:
            from .visitors import simplify
            index = simplify(BitVecITE(size=self.index_bits, condition=index < 0, true_value=self.index_max + index + 1, false_value=index))
        if isinstance(index, Constant) and index.value in self._concrete_cache:
            return self._concrete_cache[index.value]
        if default is not None:
            default = self.cast_value(default)
            is_known = self.is_known(index)
            if isinstance(is_known, Constant) and is_known.value == False:
                return default
        else:
            return self._array.select(index)
        value = self._array.select(index)
        return BitVecITE(size=self._array.value_bits, condition=is_known, true_value=value, false_value=default)

class ArraySelect(BitVec):
    __xslots__: Tuple[str, ...] = ('_operands',)

    def __init__(self, *, array: 'Array', index: 'BitVec', **kwargs):
        if False:
            while True:
                i = 10
        assert isinstance(array, Array)
        assert index.size == array.index_bits
        self._operands = (array, index)
        kwargs.setdefault('taint', frozenset({y for x in self._operands for y in x.taint}))
        super().__init__(size=array.value_bits, **kwargs)

    @property
    def array(self):
        if False:
            return 10
        return self._operands[0]

    @property
    def index(self):
        if False:
            i = 10
            return i + 15
        return self._operands[1]

    @property
    def operands(self):
        if False:
            return 10
        return self._operands

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'<ArraySelect obj with index={self.index}:\n{self.array}>'

class BitVecSignExtend(BitVecOperation):
    __xslots__: Tuple[str, ...] = ('extend',)

    def __init__(self, *, operand: 'BitVec', size_dest: int, **kwargs):
        if False:
            i = 10
            return i + 15
        assert size_dest >= operand.size
        super().__init__(size=size_dest, operands=(operand,), **kwargs)
        self.extend = size_dest - operand.size

class BitVecZeroExtend(BitVecOperation):
    __xslots__: Tuple[str, ...] = ('extend',)

    def __init__(self, *, size_dest: int, operand: 'BitVec', **kwargs):
        if False:
            return 10
        assert size_dest >= operand.size
        super().__init__(size=size_dest, operands=(operand,), **kwargs)
        self.extend = size_dest - operand.size

class BitVecExtract(BitVecOperation):
    __xslots__: Tuple[str, ...] = ('_begining', '_end')

    def __init__(self, *, operand: 'BitVec', offset: int, size: int, **kwargs):
        if False:
            print('Hello World!')
        assert offset >= 0 and offset + size <= operand.size
        super().__init__(size=size, operands=(operand,), **kwargs)
        self._begining = offset
        self._end = offset + size - 1

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        return self.operands[0]

    @property
    def begining(self):
        if False:
            for i in range(10):
                print('nop')
        return self._begining

    @property
    def end(self):
        if False:
            print('Hello World!')
        return self._end

class BitVecConcat(BitVecOperation):

    def __init__(self, *, size_dest: int, operands: Tuple, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert all((isinstance(x, BitVec) for x in operands))
        assert size_dest == sum((x.size for x in operands))
        super().__init__(size=size_dest, operands=operands, **kwargs)

class BitVecITE(BitVecOperation):

    def __init__(self, *, size: int, condition: Union['Bool', bool], true_value: 'BitVec', false_value: 'BitVec', **kwargs):
        if False:
            return 10
        assert true_value.size == size
        assert false_value.size == size
        super().__init__(size=size, operands=(condition, true_value, false_value), **kwargs)

    @property
    def condition(self):
        if False:
            return 10
        return self.operands[0]

    @property
    def true_value(self):
        if False:
            print('Hello World!')
        return self.operands[1]

    @property
    def false_value(self):
        if False:
            for i in range(10):
                print('nop')
        return self.operands[2]
Constant = (BitVecConstant, BoolConstant)
Variable = (BitVecVariable, BoolVariable, ArrayVariable)
Operation = (BitVecOperation, BoolOperation, ArrayOperation, ArraySelect)