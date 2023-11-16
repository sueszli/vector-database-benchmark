"""
This file does three things:
- Contains the definition of SymNode
- Installs all the magic methods into SymBool, SymFloat, SymFloat at import time
- Does not depend on sympy at import time

As this file is imported from within torch/__init__.py we do not want it to depend on SymPy
to avoid having to load SymPy at import time, as doing so is *very* slow.
"""
import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
import torch
from torch import sym_float, sym_ite, sym_max, sym_min, sym_not, SymBool, SymFloat, SymInt
from torch.fx.experimental._sym_dispatch_mode import handle_sym_dispatch, sym_function_mode
if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
log = logging.getLogger(__name__)
__all__ = ['SymNode', 'method_to_operator', 'magic_methods', 'sym_sqrt']
SymTypes = (SymInt, SymFloat, SymBool)

class SymNode:
    """
    This is a type erased SymInt/SymFloat which we use to do actual operations.
    End users don't touch this.  Magic methods are NOT defined on this object.
    """

    def __init__(self, expr, shape_env, pytype, hint: Optional[Union[int, float]], constant=None, fx_node=None):
        if False:
            return 10
        self._expr = expr
        self.shape_env = shape_env
        self.pytype = pytype
        self._hint = hint
        self.constant: Optional[Union[int, float, bool]] = constant
        self.fx_node = fx_node if self.shape_env._translation_validation_enabled else None

    def with_shape_env(self, shape_env: 'ShapeEnv') -> 'SymNode':
        if False:
            i = 10
            return i + 15
        return SymNode(self._expr, shape_env, self.pytype, self._hint, self.constant, self.fx_node)

    @property
    def expr(self):
        if False:
            while True:
                i = 10
        return self.shape_env.replace(self._expr)

    def _update_hint(self):
        if False:
            return 10
        r = self.shape_env._maybe_evaluate_static(self.expr, compute_hint=True)
        if r is not None:
            self._hint = self.pytype(r)

    @property
    def hint(self):
        if False:
            return 10
        if self._hint is None:
            self._update_hint()
        return self._hint

    def has_hint(self):
        if False:
            while True:
                i = 10
        if self._hint is None:
            self._update_hint()
        return self._hint is not None

    def require_hint(self, fallback=None):
        if False:
            while True:
                i = 10
        if self._hint is None:
            self._update_hint()
        if self._hint is None:
            if fallback is not None:
                return fallback
            return self.shape_env.size_hint(self.expr)
        return self._hint

    def maybe_as_int(self):
        if False:
            while True:
                i = 10
        if self.expr.is_number:
            return int(self.expr)
        else:
            return None

    def is_int(self):
        if False:
            i = 10
            return i + 15
        return self.pytype is int

    def is_float(self):
        if False:
            i = 10
            return i + 15
        return self.pytype is float

    def is_bool(self):
        if False:
            return 10
        return self.pytype is bool

    def wrap_int(self, num):
        if False:
            for i in range(10):
                print('nop')
        assert type(num) is int
        import sympy
        return SymNode(sympy.Integer(num), self.shape_env, int, num, constant=num, fx_node=num)

    def wrap_float(self, num):
        if False:
            for i in range(10):
                print('nop')
        assert type(num) is float
        import sympy
        return SymNode(sympy.Float(num), self.shape_env, float, num, constant=num, fx_node=num)

    def wrap_bool(self, num):
        if False:
            return 10
        assert type(num) is bool
        import sympy
        return SymNode(sympy.true if num else sympy.false, self.shape_env, bool, num, constant=num, fx_node=num)

    def clone(self):
        if False:
            print('Hello World!')
        return self

    def str(self):
        if False:
            i = 10
            return i + 15
        return f'{self.expr}'

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.str()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.str()

    def abs(self) -> 'SymNode':
        if False:
            i = 10
            return i + 15
        return self._abs()

    def add(self, other) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._add(other)

    def sub(self, other) -> 'SymNode':
        if False:
            return 10
        return self._sub(other)

    def mul(self, other) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._mul(other)

    def mod(self, other) -> 'SymNode':
        if False:
            i = 10
            return i + 15
        return self._mod(other)

    def pow(self, other) -> 'SymNode':
        if False:
            i = 10
            return i + 15
        return self._pow(other)

    def and_(self, other) -> 'SymNode':
        if False:
            for i in range(10):
                print('nop')
        return self._and_(other)

    def or_(self, other) -> 'SymNode':
        if False:
            return 10
        return self._or_(other)

    def truediv(self, other) -> 'SymNode':
        if False:
            for i in range(10):
                print('nop')
        return self._truediv(other)

    def floordiv(self, other) -> 'SymNode':
        if False:
            i = 10
            return i + 15
        return self._floordiv(other)

    def lshift(self, other) -> 'SymNode':
        if False:
            for i in range(10):
                print('nop')
        return self._lshift(other)

    def rshift(self, other) -> 'SymNode':
        if False:
            return 10
        return self._rshift(other)

    def sym_not(self) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._sym_not()

    def eq(self, other) -> 'SymNode':
        if False:
            i = 10
            return i + 15
        return self._eq(other)

    def ne(self, other) -> 'SymNode':
        if False:
            i = 10
            return i + 15
        return self._ne(other)

    def gt(self, other) -> 'SymNode':
        if False:
            i = 10
            return i + 15
        return self._gt(other)

    def lt(self, other) -> 'SymNode':
        if False:
            return 10
        return self._lt(other)

    def le(self, other) -> 'SymNode':
        if False:
            for i in range(10):
                print('nop')
        return self._le(other)

    def ge(self, other) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._ge(other)

    def floor(self) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._floor()

    def sym_float(self) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._sym_float()

    def sym_int(self) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._sym_int()

    def ceil(self) -> 'SymNode':
        if False:
            for i in range(10):
                print('nop')
        return self._ceil()

    def neg(self) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._neg()

    def sym_min(self, other) -> 'SymNode':
        if False:
            i = 10
            return i + 15
        return self._sym_min(other)

    def sym_max(self, other) -> 'SymNode':
        if False:
            for i in range(10):
                print('nop')
        return self._sym_max(other)

    def sym_ite(self, then_val, else_val) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._sym_ite(then_val, else_val)

    def sym_sqrt(self) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._sym_sqrt()

    def is_contiguous(self, sizes, strides) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._is_contiguous(sizes, strides)

    def is_channels_last_contiguous_2d(self, sizes, strides) -> 'SymNode':
        if False:
            print('Hello World!')
        return self._is_channels_last_contiguous_2d(sizes, strides)

    def is_channels_last_contiguous_3d(self, sizes, strides) -> 'SymNode':
        if False:
            while True:
                i = 10
        return self._is_channels_last_contiguous_3d(sizes, strides)

    def is_channels_last_strides_2d(self, sizes, strides) -> 'SymNode':
        if False:
            return 10
        return self._is_channels_last_strides_2d(sizes, strides)

    def is_channels_last_strides_3d(self, sizes, strides) -> 'SymNode':
        if False:
            while True:
                i = 10
        return self._is_channels_last_strides_3d(sizes, strides)

    def is_non_overlapping_and_dense_indicator(self, sizes, strides) -> 'SymNode':
        if False:
            return 10
        return self._is_non_overlapping_and_dense_indicator(sizes, strides)

    def sym_or(self, other):
        if False:
            return 10
        return self.or_(other)

    def sym_and(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.and_(other)

    def is_non_overlapping_and_dense(self, sizes, strides):
        if False:
            while True:
                i = 10
        return self.is_non_overlapping_and_dense_indicator(sizes, strides).eq(to_node(self, 1))

    def int_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.guard_int('', 0)

    def guard_int(self, file, line):
        if False:
            return 10
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
        try:
            return int(r)
        except Exception:
            log.warning('Failed to convert to int: %s', r)
            raise

    def guard_float(self, file, line):
        if False:
            print('Hello World!')
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
        try:
            return float(r)
        except Exception:
            log.warning('Failed to convert to float: %s', r)
            raise

    def guard_bool(self, file, line):
        if False:
            while True:
                i = 10
        r = self.shape_env.evaluate_expr(self.expr, self.hint, fx_node=self.fx_node)
        try:
            return bool(r)
        except Exception:
            log.warning('Failed to convert to bool: %s', r)
            raise

    def expect_true(self, file, line):
        if False:
            print('Hello World!')
        if self.has_hint():
            return self.guard_bool(file, line)
        return self.shape_env.defer_runtime_assert(self.expr, f'{file}:{line}', fx_node=self.fx_node)

    def expect_size(self, file, line):
        if False:
            while True:
                i = 10
        from torch.fx.experimental.symbolic_shapes import _advise_is_size
        b = self.ge(self.wrap_int(0))
        r = b.expect_true(file, line)
        if r and (not self.has_hint()):
            _advise_is_size(SymInt(self))
        return r

    def bool_(self):
        if False:
            i = 10
            return i + 15
        return self.guard_bool('', 0)

    def is_symbolic(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def singleton_int(self):
        if False:
            return 10
        return None

    def is_constant(self):
        if False:
            print('Hello World!')
        return False
unary_magic_methods = {'abs', 'sym_float', 'ceil', 'floor', 'neg', 'sym_sqrt', 'sym_not'}
only_bool_magic_methods = {'and', 'or', 'sym_not', 'sym_ite'}
also_bool_magic_methods = {'eq'}
bool_magic_methods = only_bool_magic_methods | also_bool_magic_methods
magic_methods_on_math = {'ceil', 'floor'}
magic_methods_on_submodule = {'sym_float', 'sym_sqrt', 'sym_min', 'sym_max', 'sym_not', 'sym_ite'}
magic_methods_on_operator_with_trailing_underscore = {'and', 'or'}
always_float_magic_methods = {'truediv', 'sym_float', 'sym_sqrt', 'pow'}
always_int_magic_methods = {'ceil', 'floor'}
always_bool_magic_methods = {'eq', 'ne', 'gt', 'lt', 'le', 'ge', 'and', 'or', 'sym_not', 'is_non_overlapping_and_dense'}

def _sympy_truediv(a, b):
    if False:
        for i in range(10):
            print('nop')
    from torch.utils._sympy.functions import TrueDiv
    return TrueDiv(a, b)

def _sympy_floordiv(a, b):
    if False:
        i = 10
        return i + 15
    from torch.utils._sympy.functions import FloorDiv
    return FloorDiv(a, b)

def _sympy_mod(a, b):
    if False:
        return 10
    from torch.utils._sympy.functions import Mod
    return Mod(a, b)

def _sympy_pow(a, b):
    if False:
        print('Hello World!')
    from torch.utils._sympy.functions import Pow
    return Pow(a, b)

def _sympy_and(a, b):
    if False:
        for i in range(10):
            print('nop')
    import sympy
    return sympy.And(a, b)

def _sympy_or(a, b):
    if False:
        i = 10
        return i + 15
    import sympy
    return sympy.Or(a, b)

def _sympy_lshift(a, b):
    if False:
        for i in range(10):
            print('nop')
    from torch.utils._sympy.functions import LShift
    return LShift(a, b)

def _sympy_rshift(a, b):
    if False:
        while True:
            i = 10
    from torch.utils._sympy.functions import RShift
    return RShift(a, b)
reflectable_magic_methods = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b, 'mod': _sympy_mod, 'pow': _sympy_pow, 'and': _sympy_and, 'or': _sympy_or, 'truediv': _sympy_truediv, 'floordiv': _sympy_floordiv, 'lshift': _sympy_lshift, 'rshift': _sympy_rshift}

def _floor_ceil_helper(a, fn):
    if False:
        print('Hello World!')
    import sympy
    if isinstance(a, sympy.Mul):
        aa = a.args
        if len(aa) == 2 and isinstance(aa[0], sympy.Float) and aa[1].is_integer:
            coef = sympy.Integer(aa[0])
            if aa[0] == coef:
                return coef * aa[1]
    if isinstance(a, sympy.Float) and a == sympy.Integer(a) or isinstance(a, sympy.Integer):
        return sympy.Integer(a)
    return fn(a)

def _sympy_floor(a):
    if False:
        i = 10
        return i + 15
    import sympy
    return _floor_ceil_helper(a, sympy.floor)

def _sympy_ceil(a):
    if False:
        print('Hello World!')
    import sympy
    return _floor_ceil_helper(a, sympy.ceiling)

def _sympy_eq(a, b):
    if False:
        while True:
            i = 10
    import sympy
    return sympy.Eq(a, b)

def _sympy_ne(a, b):
    if False:
        print('Hello World!')
    import sympy
    return sympy.Ne(a, b)

def _sympy_gt(a, b):
    if False:
        i = 10
        return i + 15
    import sympy
    return sympy.Gt(a, b)

def _sympy_lt(a, b):
    if False:
        print('Hello World!')
    import sympy
    return sympy.Lt(a, b)

def _sympy_le(a, b):
    if False:
        while True:
            i = 10
    import sympy
    return sympy.Le(a, b)

def _sympy_ge(a, b):
    if False:
        for i in range(10):
            print('nop')
    import sympy
    return sympy.Ge(a, b)

def _sympy_min(a, b):
    if False:
        print('Hello World!')
    import sympy
    return sympy.Min(a, b)

def _sympy_max(a, b):
    if False:
        for i in range(10):
            print('nop')
    import sympy
    return sympy.Max(a, b)

def _sympy_ite(a, t, f):
    if False:
        for i in range(10):
            print('nop')
    import sympy
    return sympy.Piecewise((t, a), (f, True))

def _sympy_sqrt(a):
    if False:
        print('Hello World!')
    import sympy
    return sympy.sqrt(a)

def _sympy_abs(a):
    if False:
        while True:
            i = 10
    import sympy
    return sympy.Abs(a)
magic_methods = {**reflectable_magic_methods, 'sym_not': lambda a: ~a, 'eq': _sympy_eq, 'ne': _sympy_ne, 'gt': _sympy_gt, 'lt': _sympy_lt, 'le': _sympy_le, 'ge': _sympy_ge, 'floor': _sympy_floor, 'sym_float': lambda a: a, 'ceil': _sympy_ceil, 'neg': lambda a: -a, 'sym_min': _sympy_min, 'sym_max': _sympy_max, 'sym_ite': _sympy_ite, 'sym_sqrt': _sympy_sqrt, 'abs': _sympy_abs}

def sym_sqrt(a):
    if False:
        return 10
    if hasattr(a, '__sym_sqrt__'):
        return a.__sym_sqrt__()
    return math.sqrt(a)

def sympy_is_contiguous(sizes, strides):
    if False:
        return 10
    dim = len(sizes)
    return sympy_is_contiguous_generic(sizes, strides, list(range(dim - 1, -1, -1)))

def sympy_is_contiguous_generic(sizes, strides, dim_order):
    if False:
        for i in range(10):
            print('nop')
    import sympy
    dim = len(sizes)
    if len(dim_order) != dim:
        return sympy.false
    is_contiguous = sympy.true
    z = sympy.Integer(1)
    for d in dim_order:
        is_contiguous &= sympy.Eq(sizes[d], sympy.Integer(1)) | sympy.Eq(strides[d], z)
        z *= sizes[d]
    for d in range(dim):
        is_contiguous |= sympy.Eq(sizes[d], sympy.Integer(0))
    return is_contiguous

def sympy_is_channels_last_contiguous_2d(sizes, strides):
    if False:
        i = 10
        return i + 15
    return sympy_is_contiguous_generic(sizes, strides, [1, 3, 2, 0])

def sympy_is_channels_last_contiguous_3d(sizes, strides):
    if False:
        for i in range(10):
            print('nop')
    return sympy_is_contiguous_generic(sizes, strides, [1, 4, 3, 2, 0])

def sympy_is_channels_last_strides_generic(sizes, strides, dim_order):
    if False:
        print('Hello World!')
    import sympy
    dim = len(sizes)
    if dim != len(dim_order):
        return sympy.false
    m = sympy.Integer(0)
    r = sympy.true
    r &= sympy.Ne(strides[1], 0)
    for d in dim_order:
        r &= sympy.Ne(sizes[d], 0) & (strides[d] >= m)
        if d == 0:
            r &= sympy.Ne(m, strides[1])
        m = strides[d] * sympy.Max(sizes[d], 1)
    return r

def sympy_is_channels_last_strides_2d(sizes, strides):
    if False:
        for i in range(10):
            print('nop')
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 3, 2, 0])

def sympy_is_channels_last_strides_3d(sizes, strides):
    if False:
        while True:
            i = 10
    return sympy_is_channels_last_strides_generic(sizes, strides, [1, 4, 3, 2, 0])

def _sympy_is_non_overlapping_and_dense_indicator(sizes, strides):
    if False:
        i = 10
        return i + 15
    from torch.utils._sympy.functions import IsNonOverlappingAndDenseIndicator
    return IsNonOverlappingAndDenseIndicator(*sizes, *strides)
sizes_strides_methods = {'is_contiguous': sympy_is_contiguous, 'is_channels_last_contiguous_2d': sympy_is_channels_last_contiguous_2d, 'is_channels_last_contiguous_3d': sympy_is_channels_last_contiguous_3d, 'is_channels_last_strides_2d': sympy_is_channels_last_strides_2d, 'is_channels_last_strides_3d': sympy_is_channels_last_strides_3d, 'is_non_overlapping_and_dense_indicator': _sympy_is_non_overlapping_and_dense_indicator}
alternate_impl_if_hinted_methods = {'sym_min': builtins.min, 'sym_max': builtins.max}

def to_node(self, num):
    if False:
        return 10
    if isinstance(num, SymTypes):
        return num.node
    elif type(num) is bool:
        return self.wrap_bool(num)
    elif type(num) is int:
        return self.wrap_int(num)
    elif type(num) is float:
        return self.wrap_float(num)
    else:
        return NotImplemented

def wrap_node(x):
    if False:
        print('Hello World!')
    if isinstance(x, SymNode) and x.constant is not None:
        return x.constant
    if x.is_int():
        return SymInt(x)
    elif x.is_float():
        return SymFloat(x)
    elif x.is_bool():
        return SymBool(x)
    else:
        raise AssertionError(f'unrecognized return type {x}')

def method_to_operator(method):
    if False:
        while True:
            i = 10
    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f'{method}_'
    else:
        method_attr = method
    if method in magic_methods_on_submodule:
        op = getattr(torch.fx.experimental.sym_node, method_attr)
    elif method in magic_methods_on_math:
        op = getattr(math, method_attr)
    else:
        op = getattr(operator, method_attr)
    return op

def _make_node_magic(method, func):
    if False:
        for i in range(10):
            print('nop')
    func = lru_cache(256)(func)
    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f'{method}_'
    else:
        method_attr = method

    def binary_magic_impl(self, other):
        if False:
            i = 10
            return i + 15
        from torch.fx.experimental.symbolic_shapes import safe_expand
        op = method_to_operator(method)
        out_hint = None
        if self.hint is not None and other.hint is not None:
            out_hint = op(self.hint, other.hint)
        alternate_impl = alternate_impl_if_hinted_methods.get(method)
        if alternate_impl and out_hint is not None:
            return to_node(self, alternate_impl(wrap_node(self), wrap_node(other)))
        if sym_function_mode():
            return to_node(self, handle_sym_dispatch(op, (wrap_node(self), wrap_node(other)), {}))
        assert isinstance(other, SymNode)
        try:
            out = func(self.expr, other.expr)
        except Exception:
            log.warning('failed to eval %s(%s, %s)', method, self.expr, other.expr)
            raise
        out = safe_expand(out)
        pytype: Type
        if method in always_float_magic_methods:
            pytype = float
        elif method in always_bool_magic_methods:
            pytype = bool
        elif self.pytype is float or other.pytype is float:
            pytype = float
        else:
            pytype = self.pytype
        (fx_node, _) = self.shape_env.create_fx_call_function(op, (self.fx_node, other.fx_node))
        return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)

    def unary_magic_impl(self):
        if False:
            i = 10
            return i + 15
        from torch.fx.experimental.symbolic_shapes import safe_expand
        op = method_to_operator(method)
        if sym_function_mode():
            return to_node(self, handle_sym_dispatch(op, (wrap_node(self),), {}))
        expr = self.expr
        if method == 'floor' or method == 'ceiling':
            expr = self.shape_env._simplify_floor_div(expr)
        try:
            out = func(expr)
        except Exception:
            log.warning('failed to eval %s(%s)', method, expr)
            raise
        out_hint = None
        if self.hint is not None:
            out_hint = op(self.hint)
        out = safe_expand(out)
        pytype: Type
        if method in always_int_magic_methods:
            pytype = int
        elif method in always_float_magic_methods:
            pytype = float
        else:
            pytype = self.pytype
        (fx_node, _) = self.shape_env.create_fx_call_function(op, (self.fx_node,))
        return SymNode(out, self.shape_env, pytype, out_hint, fx_node=fx_node)
    if method in unary_magic_methods:
        setattr(SymNode, f'_{method_attr}', unary_magic_impl)
    elif method == 'sym_ite':

        def sym_ite_impl(pred_node, then_node, else_node):
            if False:
                for i in range(10):
                    print('nop')
            from torch.fx.experimental.symbolic_shapes import safe_expand
            out_hint = then_node.hint if pred_node.hint else else_node.hint
            if sym_function_mode():
                return to_node(pred_node, handle_sym_dispatch(sym_ite, (wrap_node(pred_node), wrap_node(then_node), wrap_node(else_node)), {}))
            try:
                out = func(pred_node.expr, then_node.expr, else_node.expr)
            except Exception:
                log.warning('failed to eval %s(%s, %s, %s)', method, pred_node.expr, then_node.expr, else_node.expr)
                raise
            out = safe_expand(out)
            (fx_node, _) = pred_node.shape_env.create_fx_call_function(sym_ite, (pred_node.fx_node, then_node.fx_node, else_node.fx_node))
            return SymNode(out, pred_node.shape_env, then_node.pytype, out_hint, fx_node=fx_node)
        setattr(SymNode, f'_{method_attr}', sym_ite_impl)
    else:
        setattr(SymNode, f'_{method_attr}', binary_magic_impl)

def _make_node_sizes_strides(method, func):
    if False:
        while True:
            i = 10

    def sizes_strides_impl(self, sizes, strides):
        if False:
            return 10
        op = getattr(sys.modules[__name__], method)
        if sym_function_mode():
            return to_node(self, handle_sym_dispatch(op, ([wrap_node(s) for s in sizes], [wrap_node(s) for s in strides]), {}))
        size_exprs = [s.expr for s in sizes]
        stride_exprs = [s.expr for s in strides]
        try:
            out = func(size_exprs, stride_exprs)
        except Exception:
            log.warning('failed to eval %s(%s, %s)', method, size_exprs, stride_exprs)
            raise
        size_hints = []
        out_hint = None
        for s in sizes:
            if s.hint is None:
                break
            size_hints.append(s.hint)
        else:
            stride_hints = []
            for s in strides:
                if s.hint is None:
                    break
                stride_hints.append(s.hint)
            else:
                out_hint = op(size_hints, stride_hints)
        pytype: Type
        if method.endswith('_indicator'):
            pytype = int
        else:
            pytype = bool
        return SymNode(out, self.shape_env, pytype, out_hint)
    setattr(SymNode, f'_{method}', sizes_strides_impl)

    def sizes_strides_user(sizes, strides):
        if False:
            print('Hello World!')
        import sympy
        from torch.fx.experimental.symbolic_shapes import eval_is_non_overlapping_and_dense
        for a in itertools.chain(sizes, strides):
            if isinstance(a, SymInt):
                return wrap_node(getattr(a.node, method)([to_node(a.node, b) for b in sizes], [to_node(a.node, b) for b in strides]))
        if method == 'is_non_overlapping_and_dense_indicator':
            return eval_is_non_overlapping_and_dense(sizes, strides)
        else:
            return bool(func([sympy.sympify(a) for a in sizes], [sympy.sympify(a) for a in strides]))
    if not hasattr(sys.modules[__name__], method):
        setattr(sys.modules[__name__], method, sizes_strides_user)
for (method, func) in magic_methods.items():
    _make_node_magic(method, func)
for (method, func) in sizes_strides_methods.items():
    _make_node_sizes_strides(method, func)

def _make_user_magic(method, user_type):
    if False:
        for i in range(10):
            print('nop')
    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f'{method}_'
    else:
        method_attr = method

    def get_constant(x: Union[SymInt, int, SymFloat, float, SymBool, bool]):
        if False:
            i = 10
            return i + 15
        if isinstance(x, (int, float, bool)):
            return x
        if isinstance(x, SymBool):
            return x.node.guard_bool('', 0)
        raise AssertionError('expect to be called with constant SymBools')

    def is_constant(x):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(x, (int, float, bool)):
            return True
        if isinstance(x, (SymInt, SymFloat, SymBool)):
            return x.node.is_constant()
        return False

    def unary_magic_impl(self):
        if False:
            i = 10
            return i + 15
        if is_constant(self):
            return method_to_operator(method)(get_constant(self))
        return wrap_node(getattr(self.node, method_attr)())

    def binary_magic_impl(self, other):
        if False:
            for i in range(10):
                print('nop')
        if is_constant(self):
            return method_to_operator(method)(get_constant(self), other)
        if is_constant(other):
            other = get_constant(other)
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        ret = wrap_node(getattr(self.node, method_attr)(other_node))
        return get_constant(ret) if is_constant(ret) else ret

    def rbinary_magic_impl(self, other):
        if False:
            print('Hello World!')
        if is_constant(self):
            return method_to_operator(method)(get_constant(self), other)
        if is_constant(other):
            other = get_constant(other)
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        ret = wrap_node(getattr(other_node, method_attr)(self.node))
        return get_constant(ret) if is_constant(ret) else ret
    if method in unary_magic_methods:
        setattr(user_type, f'__{method}__', unary_magic_impl)
    elif method == 'sym_ite':

        def sym_ite_magic_impl(pred, then_val, else_val):
            if False:
                while True:
                    i = 10
            pred_node = pred.node
            then_node = to_node(pred_node, then_val)
            else_node = to_node(pred_node, else_val)
            if then_node is NotImplemented or else_node is NotImplemented:
                return NotImplemented
            assert isinstance(then_node, SymNode) and isinstance(else_node, SymNode) and (then_node.pytype == else_node.pytype)
            ret = wrap_node(getattr(pred.node, method_attr)(then_node, else_node))
            return get_constant(ret) if ret.node.is_constant() else ret
        setattr(user_type, f'__{method}__', sym_ite_magic_impl)
    else:
        setattr(user_type, f'__{method}__', binary_magic_impl)
        if method in reflectable_magic_methods:
            setattr(user_type, f'__r{method}__', rbinary_magic_impl)
for (method, func) in magic_methods.items():
    if method in only_bool_magic_methods:
        _make_user_magic(method, SymBool)
        continue
    if method in also_bool_magic_methods:
        _make_user_magic(method, SymBool)
    _make_user_magic(method, SymInt)
    _make_user_magic(method, SymFloat)
del method
del func