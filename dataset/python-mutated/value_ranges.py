import dataclasses
import itertools
import sympy
from sympy.logic.boolalg import BooleanAtom, Boolean as SympyBoolean
import operator
import math
import logging
import torch
from typing import Union, Dict, Optional, SupportsFloat
from torch._prims_common import dtype_to_type
from .interp import sympy_interp
log = logging.getLogger(__name__)
__all__ = ['ValueRanges', 'ValueRangeAnalysis', 'bound_sympy']

class ValueRangeError(RuntimeError):
    pass

def simple_sympify(e):
    if False:
        while True:
            i = 10
    if isinstance(e, bool):
        return sympy.true if e else sympy.false
    elif isinstance(e, int):
        return sympy.Integer(e)
    elif isinstance(e, float):
        if math.isinf(e):
            return sympy.oo if e > 0 else -sympy.oo
        return sympy.Float(e)
    elif isinstance(e, sympy.Expr):
        assert e.is_constant(), e
        assert e != sympy.nan
        return e
    elif isinstance(e, BooleanAtom):
        return e
    else:
        raise AssertionError(f'not simple sympy type {type(e)}: {e}')

def sympy_generic_le(lower, upper):
    if False:
        i = 10
        return i + 15
    if isinstance(lower, sympy.Expr):
        assert isinstance(upper, sympy.Expr)
        return lower <= upper
    else:
        assert isinstance(lower, SympyBoolean) and isinstance(upper, SympyBoolean)
        return not (lower and (not upper))

@dataclasses.dataclass(frozen=True)
class ValueRanges:
    lower: Union[sympy.Expr, SympyBoolean]
    upper: Union[sympy.Expr, SympyBoolean]
    is_bool: bool

    def __init__(self, lower, upper):
        if False:
            for i in range(10):
                print('nop')
        lower = simple_sympify(lower)
        upper = simple_sympify(upper)
        if not sympy_generic_le(lower, upper):
            raise ValueRangeError(f'Invalid ranges [{lower}:{upper}]')
        object.__setattr__(self, 'lower', lower)
        object.__setattr__(self, 'upper', upper)
        object.__setattr__(self, 'is_bool', isinstance(lower, SympyBoolean))
        assert isinstance(upper, SympyBoolean) == self.is_bool

    def __contains__(self, x):
        if False:
            i = 10
            return i + 15
        x = simple_sympify(x)
        return sympy_generic_le(self.lower, x) and sympy_generic_le(x, self.upper)

    def tighten(self, other) -> 'ValueRanges':
        if False:
            i = 10
            return i + 15
        'Given two ValueRanges, returns their intersection'
        return self & other

    def __and__(self, other) -> 'ValueRanges':
        if False:
            while True:
                i = 10
        if other == ValueRanges.unknown():
            return self
        if self == ValueRanges.unknown():
            return other
        assert self.is_bool == other.is_bool, (self, other)
        if self.is_bool:
            range = ValueRanges(sympy.Or(self.lower, other.lower), sympy.And(self.upper, other.upper))
        else:
            range = ValueRanges(sympy.Max(self.lower, other.lower), sympy.Min(self.upper, other.upper))
        return range

    def __or__(self, other) -> 'ValueRanges':
        if False:
            while True:
                i = 10
        if ValueRanges.unknown() in (self, other):
            return ValueRanges.unknown()
        assert self.is_bool == other.is_bool, (self, other)
        if self.is_bool:
            range = ValueRanges(sympy.And(self.lower, other.lower), sympy.Or(self.upper, other.upper))
        else:
            range = ValueRanges(sympy.Min(self.lower, other.lower), sympy.Max(self.upper, other.upper))
        return range

    def is_singleton(self) -> bool:
        if False:
            print('Hello World!')
        return self.lower == self.upper

    @classmethod
    def unknown(cls):
        if False:
            print('Hello World!')
        return cls(-sympy.oo, sympy.oo)

    @classmethod
    def wrap(cls, arg):
        if False:
            print('Hello World!')
        if isinstance(arg, ValueRanges):
            return arg
        return ValueRanges(arg, arg)

    @classmethod
    def increasing_map(cls, x, fn):
        if False:
            print('Hello World!')
        'Increasing: x <= y => f(x) <= f(y).'
        x = cls.wrap(x)
        return ValueRanges(fn(x.lower), fn(x.upper))

    @classmethod
    def decreasing_map(cls, x, fn):
        if False:
            print('Hello World!')
        'Decreasing: x <= y => f(x) >= f(y).'
        x = cls.wrap(x)
        return ValueRanges(fn(x.upper), fn(x.lower))

    @classmethod
    def monotone_map(cls, x, fn):
        if False:
            print('Hello World!')
        "It's increasing or decreasing."
        x = cls.wrap(x)
        l = fn(x.lower)
        u = fn(x.upper)
        return ValueRanges(min(l, u), max(l, u))

    @classmethod
    def convex_min_zero_map(cls, x, fn):
        if False:
            i = 10
            return i + 15
        'Fn is convex and has a minimum at 0.'
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges(0, max(fn(x.lower), fn(x.upper)))
        else:
            return cls.monotone_map(x, fn)

    @classmethod
    def coordinatewise_increasing_map(cls, x, y, fn):
        if False:
            while True:
                i = 10
        "\n        It's increasing on each coordinate.\n\n        Mathematically:\n        For every 1 <= i <= n and x_i <= y_i we have that\n        f(x1, .., xn) <= f(x1, , yi, ..., xn)\n        "
        (x, y) = (cls.wrap(x), cls.wrap(y))
        return ValueRanges(fn(x.lower, y.lower), fn(x.upper, y.upper))

    @classmethod
    def coordinatewise_monotone_map(cls, x, y, fn):
        if False:
            return 10
        "It's increasing or decreasing on each coordinate."
        (x, y) = (cls.wrap(x), cls.wrap(y))
        products = [fn(a, b) for (a, b) in itertools.product([x.lower, x.upper], [y.lower, y.upper])]
        return ValueRanges(min(products), max(products))

class SymPyValueRangeAnalysis:
    """
    It gives bounds on a SymPy operator given bounds on its arguments
    See the function `bound_sympy` for a function that applies this logic to a full SymPy expression
    """

    @staticmethod
    def constant(value, dtype):
        if False:
            print('Hello World!')
        is_python = isinstance(value, (int, float, bool))
        assert is_python or isinstance(value, (BooleanAtom, sympy.Integer, sympy.Number))
        if isinstance(value, SupportsFloat) and math.isnan(value):
            return ValueRanges.unknown()
        if is_python:
            type_ = dtype_to_type(dtype)
            value = type_(value)
        elif dtype == torch.bool:
            assert isinstance(value, BooleanAtom)
        elif dtype.is_floating_point:
            assert not value.is_finite or value.is_real
        else:
            assert value.is_integer
        return ValueRanges.wrap(value)

    @staticmethod
    def not_(a):
        if False:
            i = 10
            return i + 15
        a = ValueRanges.wrap(a)
        assert a.is_bool
        return ValueRanges.decreasing_map(a, sympy.Not)

    @staticmethod
    def or_(a, b):
        if False:
            i = 10
            return i + 15
        return ValueRanges.coordinatewise_increasing_map(a, b, sympy.Or)

    @staticmethod
    def and_(a, b):
        if False:
            print('Hello World!')
        return ValueRanges.coordinatewise_increasing_map(a, b, sympy.And)

    @staticmethod
    def eq(a, b):
        if False:
            print('Hello World!')
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if a.is_singleton() and b.is_singleton() and (a.lower == b.lower):
            return ValueRanges.wrap(sympy.true)
        elif a.lower > b.upper or b.lower > a.upper:
            return ValueRanges.wrap(sympy.false)
        return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def ne(cls, a, b):
        if False:
            i = 10
            return i + 15
        return cls.not_(cls.eq(a, b))

    @classmethod
    def lt(cls, a, b):
        if False:
            return 10
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        assert a.is_bool == b.is_bool
        if a.is_bool:
            return cls.and_(cls.not_(a), b)
        else:
            if a.upper < b.lower:
                return ValueRanges.wrap(sympy.true)
            elif a.lower >= b.upper:
                return ValueRanges.wrap(sympy.false)
            return ValueRanges(sympy.false, sympy.true)

    @classmethod
    def gt(cls, a, b):
        if False:
            print('Hello World!')
        return cls.lt(b, a)

    @classmethod
    def le(cls, a, b):
        if False:
            return 10
        return cls.not_(cls.gt(a, b))

    @classmethod
    def ge(cls, a, b):
        if False:
            return 10
        return cls.not_(cls.lt(a, b))

    @staticmethod
    def add(a, b):
        if False:
            while True:
                i = 10
        return ValueRanges.coordinatewise_increasing_map(a, b, operator.add)

    @classmethod
    def mul(cls, a, b):
        if False:
            while True:
                i = 10
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        assert a.is_bool == b.is_bool
        if a.is_bool:
            return cls.and_(a, b)

        def safe_mul(a, b):
            if False:
                i = 10
                return i + 15
            if a == 0:
                return a
            elif b == 0:
                return b
            else:
                return a * b
        return ValueRanges.coordinatewise_monotone_map(a, b, safe_mul)

    @classmethod
    def div(cls, a, b):
        if False:
            while True:
                i = 10
        return cls.truediv(a, b)

    @staticmethod
    def truediv(a, b):
        if False:
            i = 10
            return i + 15
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or ((-sympy.oo in a or sympy.oo in a) and (-sympy.oo in b or sympy.oo in b)):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(a, b, operator.truediv)

    @staticmethod
    def floordiv(a, b):
        if False:
            return 10
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if 0 in b or ((-sympy.oo in a or sympy.oo in a) and (-sympy.oo in b or sympy.oo in b)):
            return ValueRanges.unknown()
        else:
            return ValueRanges.coordinatewise_monotone_map(a, b, operator.floordiv)

    @staticmethod
    def mod(x, y):
        if False:
            i = 10
            return i + 15
        x = ValueRanges.wrap(x)
        y = ValueRanges.wrap(y)
        if x.is_singleton() and y.is_singleton() and (y.lower != 0):
            return ValueRanges.wrap(x.lower % y.lower)
        if y.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges(0, y.upper)

    @classmethod
    def modular_indexing(cls, a, b, c):
        if False:
            for i in range(10):
                print('nop')
        return cls.mod(cls.floordiv(a, b), c)

    @classmethod
    def is_non_overlapping_and_dense_indicator(cls, *args):
        if False:
            while True:
                i = 10
        return ValueRanges.unknown()

    @classmethod
    def pow(cls, a, b):
        if False:
            print('Hello World!')

        def is_integer(val):
            if False:
                i = 10
                return i + 15
            return isinstance(val, int) or (hasattr(val, 'is_integer') and val.is_integer)
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)
        if not b.is_singleton():
            return ValueRanges.unknown()
        b = b.lower
        if a.is_singleton():
            a = a.lower
            r = a ** b
            if not r.is_finite:
                return ValueRanges.unknown()
            return ValueRanges.wrap(r)
        if b == 0:
            if not a.lower.is_finite:
                return ValueRanges.unknown()
            type_ = sympy.Float if a.lower.is_real else sympy.Integer
            return ValueRanges.wrap(type_(1))
        if b < 0:
            a = cls.reciprocal(a)
            b = -b
        if a == ValueRanges.unknown():
            return ValueRanges.unknown()
        if not is_integer(b):
            if a.lower >= 0:
                return ValueRanges.increasing_map(a, lambda x: x ** b)
            else:
                return ValueRanges.unknown()
        elif b % 2 == 0:
            return ValueRanges.convex_min_zero_map(a, lambda x: x ** b)
        else:
            return ValueRanges.increasing_map(a, lambda x: x ** b)

    @staticmethod
    def reciprocal(x):
        if False:
            return 10
        " Needed as it's used in pow, but it won't appear on a SymPy expression "
        x = ValueRanges.wrap(x)
        if 0 in x:
            return ValueRanges.unknown()
        else:
            return ValueRanges.decreasing_map(x, lambda y: 1 / y)

    @staticmethod
    def abs(x):
        if False:
            while True:
                i = 10
        return ValueRanges.convex_min_zero_map(x, abs)

    @staticmethod
    def exp(x):
        if False:
            return 10
        return ValueRanges.increasing_map(x, sympy.functions.elementary.exponential.exp)

    @staticmethod
    def log(x):
        if False:
            return 10
        x = ValueRanges.wrap(x)
        if x.lower <= 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, sympy.log)

    @classmethod
    def minimum(cls, a, b):
        if False:
            print('Hello World!')
        return cls.min_or_max(a, b, sympy.Min)

    @classmethod
    def maximum(cls, a, b):
        if False:
            print('Hello World!')
        return cls.min_or_max(a, b, sympy.Max)

    @staticmethod
    def min_or_max(a, b, fn):
        if False:
            return 10
        a = ValueRanges.wrap(a)
        b = ValueRanges.wrap(b)

        def fn_(x, y):
            if False:
                i = 10
                return i + 15
            if x.is_Integer and y.is_Integer:
                result_type = sympy.Integer
            elif x.is_rational and y.is_rational:
                result_type = sympy.Rational
            else:
                assert x.is_real or not x.is_finite or y.is_real or (not y.is_finite)
                result_type = sympy.Float
            return fn(result_type(x), result_type(y))
        return ValueRanges.coordinatewise_increasing_map(a, b, fn_)

    @classmethod
    def floor(cls, x):
        if False:
            print('Hello World!')
        return ValueRanges.increasing_map(x, sympy.functions.elementary.integers.floor)

    @classmethod
    def ceil(cls, x):
        if False:
            for i in range(10):
                print('nop')
        return ValueRanges.increasing_map(x, sympy.functions.elementary.integers.ceiling)

    @staticmethod
    def sqrt(x):
        if False:
            for i in range(10):
                print('nop')
        x = ValueRanges.wrap(x)
        if x.lower < 0:
            return ValueRanges.unknown()
        return ValueRanges.increasing_map(x, sympy.sqrt)

    @staticmethod
    def where(a, b, c):
        if False:
            for i in range(10):
                print('nop')
        b = ValueRanges.wrap(b)
        c = ValueRanges.wrap(c)
        assert a.is_bool
        assert b.is_bool == c.is_bool
        if b.is_bool:
            return ValueRanges(sympy.And(b.lower, c.lower), sympy.Or(b.upper, c.upper))
        else:
            return ValueRanges(sympy.Min(b.lower, c.lower), sympy.Max(b.upper, c.upper))

    @staticmethod
    def expr_cond_pair(a, b):
        if False:
            i = 10
            return i + 15
        assert b.is_bool, f"expect cond_expr's ValueRange to be a boolean range but got {b}"
        return (a, b)

    @staticmethod
    def piecewise(*ranges):
        if False:
            while True:
                i = 10
        init_range = None
        for (expr_range, cond_range) in ranges:
            if sympy.true in cond_range:
                if init_range is None:
                    init_range = expr_range
                else:
                    init_range = init_range | expr_range
        return init_range

class ValueRangeAnalysis(SymPyValueRangeAnalysis):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.name = 'ValueRangeAnalysis'
        boolean_operators = ('xor', 'logical_and', 'logical_or', 'logical_not')
        for op in boolean_operators:
            setattr(self, op, self.bool_handler)

    @staticmethod
    def bool_handler(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return ValueRanges(sympy.false, sympy.true)

    @staticmethod
    def default_handler(*args, **kwargs):
        if False:
            return 10
        return ValueRanges.unknown()

    def load(self, name: str, index: sympy.Expr):
        if False:
            for i in range(10):
                print('nop')
        return ValueRanges.unknown()

    def store(self, name, index, value, mode=None):
        if False:
            i = 10
            return i + 15
        return

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        if False:
            i = 10
            return i + 15
        return ValueRanges.unknown()

    def index_expr(self, index, dtype):
        if False:
            while True:
                i = 10
        assert isinstance(index, ValueRanges)
        return index

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype: Optional[torch.dtype]=None):
        if False:
            for i in range(10):
                print('nop')
        x = ValueRanges.wrap(x)
        if dtype == torch.bool:
            if x.is_singleton():
                return ValueRanges.wrap(x.lower != 0)
            elif 0 not in x:
                return ValueRanges.wrap(sympy.true)
            else:
                return ValueRanges(sympy.false, sympy.true)

        def cast(x, dtype):
            if False:
                for i in range(10):
                    print('nop')
            if dtype.is_floating_point:
                return sympy.Float(x)
            else:
                try:
                    return sympy.Integer(x)
                except TypeError:
                    return x
        if x.is_bool:
            if x.is_singleton():
                val = 1 if x.lower else 0
                return ValueRanges.wrap(cast(val, dtype))
            else:
                return ValueRanges(cast(0, dtype), cast(1, dtype))
        else:
            return ValueRanges(cast(x.lower, dtype), cast(x.upper, dtype))

    @staticmethod
    def square(x):
        if False:
            i = 10
            return i + 15
        return ValueRanges.convex_min_zero_map(x, lambda y: y * y)

    @staticmethod
    def neg(x):
        if False:
            print('Hello World!')
        return ValueRanges.decreasing_map(x, operator.neg)

    @classmethod
    def truncdiv(cls, a, b):
        if False:
            return 10
        x = cls.truediv(a, b)
        if x == ValueRanges.unknown():
            return x

        def trunc(x):
            if False:
                for i in range(10):
                    print('nop')
            return sympy.Integer(x) if x.is_finite else x
        return ValueRanges.increasing_map(x, trunc)

    @classmethod
    def sub(cls, a, b):
        if False:
            i = 10
            return i + 15
        return cls.add(a, cls.neg(b))

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        log.debug('unhandled ValueRange op %s', name)
        return self.default_handler

def bound_sympy(expr: sympy.Expr, ranges: Optional[Dict[sympy.Symbol, ValueRanges]]=None) -> ValueRanges:
    if False:
        i = 10
        return i + 15
    if isinstance(expr, sympy.Number):
        return ValueRanges.wrap(expr)
    ranges = ranges or {}
    context = torch._guards.TracingContext.try_get()
    if context and context.fake_mode.shape_env:
        ranges = {**ranges, **context.fake_mode.shape_env.var_to_range}
    unbounded_vars = expr.free_symbols - ranges.keys()
    if unbounded_vars:
        unbounded_ranges: Dict[sympy.Symbol, ValueRanges] = {}
        for s in unbounded_vars:
            assert s.is_integer
            if s.is_positive:
                lower = 1
            elif s.is_nonnegative:
                lower = 0
            else:
                lower = -math.inf
            unbounded_ranges[s] = ValueRanges(lower, math.inf)
        ranges = {**ranges, **unbounded_ranges}
    return sympy_interp(SymPyValueRangeAnalysis, ranges, expr)