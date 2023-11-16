import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import FakeTensorMeta, ShapeEnvEvent, record_shapeenv_event, replay_shape_env_events, shape_env_check_state_equal
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
InputList = List
DimList = List
log = logging.getLogger(__name__)

class GuardOnDataDependentSymNode(RuntimeError):
    pass
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence
aten = torch._ops.ops.aten
__all__ = ['has_symbolic_sizes_strides', 'create_contiguous', 'ShapeEnv', 'is_concrete_int', 'guard_int', 'guard_float', 'guard_scalar', 'hint_int', 'SYMPY_INTERP', 'free_symbols', 'is_symbol_binding_fx_node', 'is_concrete_bool', 'SHAPEENV_EVENT_KEY', 'CURRENT_NODE_KEY', 'has_free_symbols', 'sym_eq']
SHAPEENV_EVENT_KEY = 'shapeenv_event'
CURRENT_NODE_KEY = 'current_node'

@lru_cache(None)
def uninteresting_files():
    if False:
        for i in range(10):
            print('nop')
    import torch._inductor.sizevars
    import torch._library.abstract_impl
    mods = [sys.modules[__name__], torch.fx.experimental.recording, torch.fx.experimental.sym_node, torch, torch._inductor.sizevars, torch._library.abstract_impl]
    return {inspect.getfile(m) for m in mods}

class ConstraintViolationError(RuntimeError):
    pass

def has_symbolic_sizes_strides(elem):
    if False:
        for i in range(10):
            print('nop')
    return elem._has_symbolic_sizes_strides

def create_contiguous(shape):
    if False:
        i = 10
        return i + 15
    strides = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])
    return list(reversed(strides))

def hint_int(a, fallback=None):
    if False:
        return 10
    '\n    Retrieve the hint for an int (based on the underlying real values as observed\n    at runtime).  If no hint is available (e.g., because data dependent shapes),\n    if fallback is not None, use that instead (otherwise raise an error).\n    '
    if isinstance(a, torch.SymInt):
        return a.node.require_hint(fallback)
    assert type(a) is int, a
    return a

def has_hint(a):
    if False:
        while True:
            i = 10
    if isinstance(a, SymTypes):
        return a.node.has_hint()
    return True

def is_concrete_int(a: Union[int, SymInt]):
    if False:
        while True:
            i = 10
    ' Utility to check if underlying object\n    in SymInt is concrete value. Also returns\n    true if integer is passed in.\n\n    Args:\n        a (SymInt or int): Object to test if it int\n    '
    assert isinstance(a, (SymInt, int))
    if isinstance(a, int):
        return True
    if isinstance(a.node.expr, sympy.core.numbers.Integer):
        return True
    return False

def is_concrete_bool(a: Union[bool, SymBool]):
    if False:
        for i in range(10):
            print('nop')
    ' Utility to check if underlying object\n    in SymBool is concrete value. Also returns\n    true if integer is passed in.\n    Args:\n        a (SymBool or bool): Object to test if it bool\n    '
    assert isinstance(a, (SymBool, bool))
    if isinstance(a, bool):
        return True
    if isinstance(a.node.expr, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)):
        return True
    return False

def tensor_has_hints(t):
    if False:
        print('Hello World!')
    return all((has_hint(s) for s in t.size()))

def _iterate_exprs(val: Union[SymInt, torch.Tensor]) -> Iterable[sympy.Expr]:
    if False:
        return 10
    if isinstance(val, SymTypes):
        if is_symbolic(val):
            yield val.node.expr
    elif isinstance(val, sympy.Expr):
        yield val
    elif isinstance(val, (int, float, bool)):
        pass
    elif isinstance(val, torch.Tensor):
        yield from _iterate_exprs(val.size())
        yield from _iterate_exprs(val.stride())
        yield from _iterate_exprs(val.storage_offset())
    elif isinstance(val, (tuple, list)):
        for s in val:
            yield from _iterate_exprs(s)
    else:
        raise AssertionError(f'cannot extract sympy expressions from {val} {type(val)}')

def free_symbols(val: Union[SymInt, torch.Tensor]) -> Set[sympy.Symbol]:
    if False:
        i = 10
        return i + 15
    itr = _iterate_exprs(val)
    try:
        first_expr = next(itr)
    except StopIteration:
        return set()
    return first_expr.free_symbols.union(*(e.free_symbols for e in itr))

def has_free_symbols(val: Union[SymInt, torch.Tensor]) -> bool:
    if False:
        print('Hello World!')
    'Faster version of bool(free_symbols(val))'
    return not all((e.is_number for e in _iterate_exprs(val)))

def free_unbacked_symbols(x):
    if False:
        while True:
            i = 10
    return {s for s in free_symbols(x) if s.name.startswith('i')}

def is_symbol_binding_fx_node(node) -> Optional[sympy.Symbol]:
    if False:
        i = 10
        return i + 15
    if node.op == 'placeholder' and 'val' in node.meta and isinstance(node.meta['val'], torch.SymInt) and isinstance(node.meta['val'].node.expr, sympy.Symbol):
        return node.meta['val'].node.expr
    return None

def find_symbol_binding_fx_nodes(graph):
    if False:
        for i in range(10):
            print('nop')
    return {node.meta['val'].node.expr: node for node in graph.nodes if is_symbol_binding_fx_node(node)}

def definitely_true(a):
    if False:
        while True:
            i = 10
    "\n    Returns True only if we can tell that a is True, possibly introducing\n    a guard in the process.  If a depends on some unbacked SymInt, we may\n    return False even though there may exist a possible value of the SymInt\n    that would cause the expression to return True.\n\n    When is it appropriate to use definitely_true?  First, if you can use\n    a higher level combinator like parallel_or/parallel_and, prefer using\n    those instead, they are definitely safe (modulo short-circuiting).\n    Second, it can be used if the program would behave equivalently if\n    definitely_true always returned False (parallel_or/parallel_and are\n    examples of this pattern, modulo short-circuiting).  Finally, it even\n    be OK if the program wouldn't behave equivalently, so long as the\n    change is semantics preserving.  It can be semantics preserving if\n    the program errors in more cases than it did previously (but otherwise\n    behaves identically), or if it changes some quantity in a way that\n    doesn't matter (e.g., strides often fall in this bucket.)\n    "
    if isinstance(a, SymBool):
        if a.node.has_hint():
            return guard_bool(a)
        else:
            return False
    return bool(a)

def definitely_false(a):
    if False:
        return 10
    '\n    Returns True only if we can tell that a is False, possibly introducing\n    a guard in the process.  If a depends on some unbacked SymInt, we may\n    return False even though there may exist a possible value of the SymInt\n    that would cause the expression a to be False.  See definitely_true\n    for more usage guidance.\n    '
    if isinstance(a, SymBool):
        if a.node.has_hint():
            return not guard_bool(a)
        else:
            return False
    return not bool(a)

def parallel_or(*args):
    if False:
        print('Hello World!')
    '\n    Evaluate the logical OR of several arguments, avoiding guarding on\n    unbacked SymInts if another argument is definitely True.\n    '
    if any((definitely_true(a) for a in args)):
        return True
    return any(args)

def parallel_and(*args):
    if False:
        return 10
    '\n    Evaluate the logical FALSE of several arguments, avoiding guarding on\n    unbacked SymInts if another argument is definitely False.\n    '
    if any((definitely_false(a) for a in args)):
        return False
    return all(args)

def sym_eq(x, y):
    if False:
        print('Hello World!')
    '\n    Like ==, but when run on list/tuple, it will recursively test equality\n    and use sym_and to join the results together, without guarding.\n    '
    if isinstance(x, tuple) and isinstance(y, tuple) or (isinstance(x, list) and isinstance(y, list)):
        if len(x) != len(y):
            return False
        return functools.reduce(operator.and_, map(sym_eq, x, y), True)
    elif isinstance(x, (int, torch.SymInt)) and isinstance(y, (int, torch.SymInt)):
        return x == y
    else:
        raise AssertionError(f'unexpected sym_eq between {type(x)} {type(y)}')

def guard_scalar(a):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(a, (SymBool, bool)):
        return guard_bool(a)
    elif isinstance(a, (SymInt, int)):
        return guard_int(a)
    elif isinstance(a, (SymFloat, float)):
        return guard_float(a)
    else:
        raise AssertionError(f'unrecognized scalar {a}')

@record_shapeenv_event()
def _constrain_symbol_range(shape_env, s: sympy.Symbol, compiler_min: int, compiler_max: int, runtime_min: int, runtime_max: int):
    if False:
        print('Hello World!')
    log.debug('_constrain_symbol_range %s [%s, %s] [%s, %s]', s, compiler_min, compiler_max, runtime_min, runtime_max)
    if (r := shape_env.var_to_range.get(s, None)):
        shape_env.var_to_range[s] = ValueRanges(builtins.max(r.lower, compiler_min), builtins.min(r.upper, compiler_max))
    else:
        shape_env.var_to_range[s] = ValueRanges(compiler_min, compiler_max)
    if (r := shape_env.runtime_var_to_range.get(s, None)):
        shape_env.runtime_var_to_range[s] = ValueRanges(builtins.max(r.lower, runtime_min), builtins.min(r.upper, runtime_max))
    else:
        shape_env.runtime_var_to_range[s] = ValueRanges(runtime_min, runtime_max)

def _advise_is_size(a):
    if False:
        return 10
    "\n    Don't use this directly; use torch._check_is_size instead.\n\n    This is a softer version of _constrain_range_for_size (with min=0,\n    max=Inf).  Instead of forcibly constraining a variable (and erroring if we\n    failed to constrain it), it will simply advise us that a size is\n    constrained in some way.  We will always defer a runtime assert for this\n    constraint if we cannot prove it at compile-time, but we we only\n    *sometimes* learn useful extra information at compile-time with this\n    information.  This is in contrast to constrain_range_for_size, where if\n    you don't call that on a fresh unbacked symint, chances are we will choke.\n\n    TODO: Make Dynamo handle this appropriately if this is seen in Dynamo-ed\n    code.  Right now this is only really used in code with AOTAutograd trace\n    through, so it is not a big problem that this isn't supported, but in\n    principle all of this code should be Dynamo'able too.\n\n    TODO: I didn't support min/max because I didn't have a use case where this\n    actually helped.  In principle we can support it, it just makes the\n    implementation below more complicated.\n    "
    assert a >= 0
    if isinstance(a, SymInt) and isinstance(a.node, SymNode) and (not a.node.has_hint()) and isinstance(a.node.expr, sympy.Symbol):
        _constrain_range_for_size(a)

@record_shapeenv_event()
def _constrain_range_for_size(a, min: Optional[int]=None, max: Optional[int]=None):
    if False:
        i = 10
        return i + 15
    '\n    This function is NOT INTENDED to be used by itself.\n    '
    if isinstance(a, (SymFloat, SymBool)):
        raise ValueError('Constraining SymFloat/SymBool is nyi')
    assert isinstance(a, SymInt), 'can only constrain range for SymInt'
    assert isinstance(a.node.expr, sympy.Symbol), 'constraining non-Symbols NYI'
    if min is None:
        min = 0
    if max is None:
        max = sympy.oo
    if max <= 2:
        raise ValueError(f'Maximum value to constrain_as_size must be greater than 2, but was {max}')
    if max < min:
        raise ValueError("Maximum value to constrain_as_size can't be less than the specified min value, received min={min} and max={max}")
    compiler_min = 2 if min < 2 else min
    _constrain_symbol_range(a.node.shape_env, a.node.expr, compiler_min=compiler_min, compiler_max=max, runtime_min=min, runtime_max=max)

@record_shapeenv_event()
def constrain_range(a, *, min: Optional[int], max: Optional[int]=None):
    if False:
        print('Hello World!')
    "\n    Applies a constraint that the passed in SymInt must lie between min-max\n    inclusive-inclusive, WITHOUT introducing a guard on the SymInt (meaning\n    that it can be used on unbacked SymInts).  If min/max are None, we assume\n    that the dimension is unbounded in that direction.  Repeated application\n    of constrain_range intersects the ranges.  This is a fairly low level API\n    that doesn't have a lot of safety guarantees (TODO: provide higher level\n    APIs).\n\n    Currently, we use this API in the following circumstance: when we allocate\n    an unbacked SymInt, denoting an integer quantity which is data dependent,\n    we ordinarily do not know anything about what values it may take.  This\n    means that any sort of guard on it will immediately fail.  However, in\n    many cases, we know something about the unbacked SymInt: for example, we\n    know that nonzero(x).size(0) must be >= 0.  We use constrain_range to\n    narrow the possible range, declaring that negative symbols are impossible.\n    This permits to definitely answer True to queries like 'nnz >= 0', even if\n    we don't know what the actual (hinted) value of 'nnz' is.  In fact, we\n    actually use constrain_range to unsoundly discharge common guards: for an\n    unbacked SymInt produced by nonzero, we will also assume that it is not\n    equal to 0/1 (even though these are perfectly possible values at runtime),\n    because we generally expect graphs that are valid for N=2 to also be valid\n    for N=1.\n\n    .. warning::\n        If you use constrain_range in the context of tracing, we do NOT check\n        that the constraint was actually valid at runtime!  In fact, we\n        cannot (easily) do so, as we currently unsoundly assume that unbacked\n        SymInt can never be zero/one, even if it may actually take on these\n        values at runtime (we assume that a graph that is valid for N=2 will\n        also be valid for N=1).\n    "
    if min is None:
        min = -sympy.oo
    if max is None:
        max = sympy.oo
    if max < min:
        raise ValueError("Maximum value to constrain_as_size can't be less than the specified min value, received min={min} and max={max}")
    if isinstance(a, int):
        if not min <= a <= max:
            raise ValueError(f'Invalid value {a} for range [{min}:{max}]')
        return
    if isinstance(a.node.expr, sympy.Integer):
        if not min <= int(a.node.expr) <= max:
            raise ValueRangeError(f'Invalid value {int(a.node.expr)} for range [{min}:{max}]')
        return
    assert isinstance(a.node.expr, sympy.Symbol), 'constraining non-Symbols NYI'
    _constrain_symbol_range(a.node.shape_env, a.node.expr, compiler_min=min, compiler_max=max, runtime_min=min, runtime_max=max)

@record_shapeenv_event()
def constrain_unify(a, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given two SymInts, constrain them so that they must be equal.  NB:\n    this will not work with SymInts that represent nontrivial expressions\n    (yet!)\n    '
    if not isinstance(a, SymInt):
        if not isinstance(b, SymInt):
            assert a == b
        else:
            assert isinstance(b.node.expr, sympy.Symbol), 'constraining non-Symbols NYI'
            shape_env = b.node.shape_env
            shape_env.replacements[b.node.expr] = sympy.Integer(a)
    else:
        assert isinstance(a.node.expr, sympy.Symbol), 'constraining non-Symbols NYI'
        shape_env = a.node.shape_env
        if not isinstance(b, SymInt):
            shape_env.replacements[a.node.expr] = sympy.Integer(b)
        else:
            assert a.node.shape_env is b.node.shape_env
            assert isinstance(b.node.expr, sympy.Symbol), 'constraining non-Symbols NYI'
            new_var = shape_env._find(a.node.expr)
            shape_env.replacements[b.node.expr] = new_var

def expect_true(a, skip: int=0):
    if False:
        while True:
            i = 10
    if isinstance(a, SymBool):
        frame = inspect.currentframe()
        for _ in range(skip + 1):
            frame = frame.f_back
        return a.node.expect_true(frame.f_code.co_filename, frame.f_lineno)
    assert type(a) is bool, a
    return a

def guard_bool(a):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(a, SymBool):
        return a.node.guard_bool('', 0)
    assert type(a) is bool, a
    return a

def guard_int(a):
    if False:
        while True:
            i = 10
    if isinstance(a, SymInt):
        return a.node.guard_int('', 0)
    assert type(a) is int, a
    return a

def guard_float(a):
    if False:
        print('Hello World!')
    if isinstance(a, SymFloat):
        return a.node.guard_float('', 0)
    assert isinstance(a, float), a
    return a

def fx_placeholder_vals(gm):
    if False:
        return 10
    return [n.meta['val'] for n in gm.graph.nodes if n.op == 'placeholder']

def fx_placeholder_targets(gm):
    if False:
        return 10
    return [n.target for n in gm.graph.nodes if n.op == 'placeholder']

def eval_guards(gm, *args, ignore_static=True):
    if False:
        i = 10
        return i + 15
    return gm.shape_env.evaluate_guards_for_args(fx_placeholder_vals(gm), args, ignore_static=ignore_static)

def bind_symbols(gm, *args):
    if False:
        print('Hello World!')
    return gm.shape_env.bind_symbols(fx_placeholder_vals(gm), args)

def _assert_bound_is_rational(expr: sympy.Expr, bound: ValueRanges):
    if False:
        return 10
    '\n    We assert that the bounds are either Boolean, or not finite, or can be computed\n    in exact prevision via rational arithmetic.\n    The only exception to this is the rare case when the user calls `sqrt(s0)`\n    sqrt is turned into sympy.Pow so we just match for that (it matches more things, but still)\n    '
    assert bound.lower.is_rational or bound.lower.is_Boolean or (not bound.lower.is_finite) or expr.has(sympy.Pow), (bound, expr)
    assert bound.upper.is_rational or bound.upper.is_Boolean or (not bound.upper.is_finite) or expr.has(sympy.Pow), (bound, expr)

class DimDynamic(Enum):
    """
    Controls how to perform symbol allocation for a dimension.  It is always
    sound to default this to DYNAMIC, but the policies DUCK and STATIC can
    result in better trace-time and compile-time performance, as they reduce
    the number of allocated symbols and generally make your graph more static.

    NB: If we notice you've applied a constraint to the dimension, we will
    force it to DYNAMIC for simplicity.

    DimDynamic is controlled by a variety of higher level UX features.
    Currently:

    - In eager mode, the default policy is DUCK.
        - The default is changed to STATIC with assume_static_by_default.
        - An individual dim is marked DYNAMIC if you mark_dynamic_dim.
    - In export mode, the default policy is STATIC.
        - An individual dim is marked DYNAMIC if you mention it as dynamic_dim
          in the constraints kwarg.
    """
    DYNAMIC = 0
    DUCK = 1
    STATIC = 2

@dataclass(frozen=True)
class Constraint:
    warn_only: bool

@dataclass(frozen=True)
class StrictMinMaxConstraint(Constraint):
    """
    For clients: the size at this dimension must be within 'vr' (which
    specifies a lower and upper bound, inclusive-inclusive) AND it
    must be non-negative and should not be 0 or 1 (but see NB below).

    For backends: there must not be any guards on this dimension which
    are not implied by the given lower and upper bound.  Regardless of
    the lower bound, the backend can assume the size is non-negative
    and that it is not 0 or 1.

    An unbounded StrictMinMaxConstraint can be thought of as a strict version
    of "RelaxedUnspecConstraint".

    NB: Export will often unsoundly assume that a graph works for 0/1, even
    though at trace time we assumed size is not 0 or 1.  The idea is that
    if we produce a graph that works for a range of values, it will be OK
    for N=0/1 too.
    """
    vr: ValueRanges

    def render(self, source: Source):
        if False:
            while True:
                i = 10
        return f'{self.vr.lower} <= {source.name()} <= {self.vr.upper}'

@dataclass(frozen=True)
class RelaxedUnspecConstraint(Constraint):
    """
    For clients: no explicit constraint; constraint is whatever is implicitly
    inferred by guards from tracing.

    For backends: there must exist at least TWO possible values for the
    size at this dimension which satisfy the guards for this dimension.

    In other words, this constraint helps us distinguish between "we don't
    care if this dimension specializes or not" versus "this dimension must be
    unspecialized."  However, this constraint doesn't say very much about what
    specialization is permitted; for example, if we guard on a size being
    even, this would still be acceptable under an unspec constraint.  This
    makes RelaxedUnspecConstraint useful for eager mode, where your backend compiler
    may add constraints to otherwise dynamic dimensions; we can't assert that
    there are NO guards as this is brittle because compilers should be able to
    add extra constraints.  If you want to assert that there are no guards,
    use StrictMinMaxConstraint with an unbounded ValueRanges.
    """

    def render(self, source: Source):
        if False:
            while True:
                i = 10
        return f'RelaxedUnspecConstraint({source.name()})'
DimConstraint = Union[StrictMinMaxConstraint, RelaxedUnspecConstraint, None]

@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    """
    Given pairs of sources corresponding to pairs of dynamic dimensions that
    are specified equal, represent them in a union-find data structure so that
    we can efficiently check whether two such sources are transitively equal.
    """
    source_pairs: List[Tuple[Source, Source]]

    def __post_init__(self):
        if False:
            i = 10
            return i + 15
        object.__setattr__(self, '_parents', {})
        for (source1, source2) in self.source_pairs:
            self._union(self._find(source1), self._find(source2))

    def _find(self, source):
        if False:
            for i in range(10):
                print('nop')
        if source in self._parents:
            return self._find(self._parents[source])
        else:
            return source

    def _union(self, root1, root2):
        if False:
            print('Hello World!')
        if root1 != root2:
            self._parents[root1] = root2

    def render(self):
        if False:
            while True:
                i = 10
        buf = ', '.join((f'{source1.name()} == {source2.name()}' for (source1, source2) in self.source_pairs))
        return '{' + buf + '}'

    def is_equal(self, source1, source2):
        if False:
            print('Hello World!')
        return self._find(source1) == self._find(source2)

def is_symbolic(val: Union[int, SymInt, float, SymFloat, bool, SymBool]) -> bool:
    if False:
        return 10
    if isinstance(val, (int, float, bool)):
        return False
    return val.node.is_symbolic()
IndicatorTypes = (IsNonOverlappingAndDenseIndicator,)

@lru_cache(256)
def safe_expand(r):
    if False:
        print('Hello World!')
    if hasattr(r, 'expand'):
        try:
            return sympy.expand(r)
        except RecursionError:
            log.warning('RecursionError in sympy.expand(%s)', r)
            return r
    else:
        return r

def error():
    if False:
        while True:
            i = 10
    raise AssertionError("shouldn't be hit")

def eval_is_non_overlapping_and_dense(sizes, strides):
    if False:
        for i in range(10):
            print('nop')
    return int(guard_bool(_eval_is_non_overlapping_and_dense(sizes, strides)))

def _eval_is_non_overlapping_and_dense(sizes, strides):
    if False:
        print('Hello World!')
    dim = len(sizes)
    if dim == 1:
        return strides[0] == 1 or sizes[0] < 2
    lengths_and_strides = sorted(zip(sizes, strides), key=operator.itemgetter(1))
    expected_stride = 1
    for (length, stride) in lengths_and_strides:
        if length == 1:
            continue
        if stride != expected_stride:
            return False
        expected_stride *= length
    return True

def cast_symbool_to_symint_guardless(symbool: torch.SymBool) -> torch.SymInt:
    if False:
        for i in range(10):
            print('nop')
    int_sym = sympy.Piecewise((1, symbool.node.expr), (0, True))
    return symbool.node.shape_env.create_symintnode(int_sym, hint=int(symbool.node.require_hint()))
SYMPY_INTERP = {'Abs': operator.abs, 'Eq': operator.eq, 'Ne': operator.ne, 'Gt': operator.gt, 'Lt': operator.lt, 'Le': operator.le, 'Ge': operator.ge, 'Min': min, 'Max': max, 'Mod': operator.mod, 'FloorDiv': operator.floordiv, 'TrueDiv': operator.truediv, 'IsNonOverlappingAndDenseIndicator': eval_is_non_overlapping_and_dense, 'floor': math.floor, 'ceiling': math.ceil, 'cast_symbool_to_symint_guardless': cast_symbool_to_symint_guardless}

def _lru_cache(fn, maxsize=None):
    if False:
        print('Hello World!')
    '\n    Wrapper around lru_cache that clears when new info about shapes has been\n    updated.\n\n    Use lru_cache if the output is always the same, regardless of the\n    constraints we know now (i.e. evaluate_expr)\n\n    Use _lru_cache otherwise.\n\n    Also note that this depends on _update_version_counter being called on the\n    shape environment whenever the constraints are updated, otherwise the cache\n    will not be cleared.\n    '
    fn_cache = lru_cache(maxsize)(fn)
    prior_version = 0
    if config.validate_shape_env_verison_key:
        prior_key = None

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal prior_version, prior_key
            if prior_key is None:
                prior_key = self._get_key()
            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter
                prior_key = self._get_key()
            else:
                assert prior_key == self._get_key(), 'ShapeEnv cache key changed without version being updated!'
            return fn_cache(self, *args, **kwargs)
    else:

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            nonlocal prior_version
            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter
            return fn_cache(self, *args, **kwargs)
    wrapper.cache_clear = fn_cache.cache_clear
    wrapper.cache_info = fn_cache.cache_info
    return wrapper

@dataclass(frozen=True)
class RuntimeAssert:
    expr: sympy.Expr
    msg: str = field(repr=False)
    stack: str = field(repr=False)

class ShapeGuardPrinter(StrPrinter):

    def __init__(self, symbol_to_source, source_ref, var_to_sources):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.symbol_to_source = symbol_to_source
        self.source_ref = source_ref
        self.var_to_sources = var_to_sources

    def _print_Symbol(self, expr) -> str:
        if False:
            while True:
                i = 10
        assert isinstance(expr, sympy.Symbol), str(type(expr))

        def repr_symbol_to_source():
            if False:
                print('Hello World!')
            return repr({symbol: [s.name() for s in sources] for (symbol, sources) in self.symbol_to_source.items()})
        assert self.symbol_to_source.get(expr), f'{expr} (could be from {[s.name() for s in self.var_to_sources[expr]]}) not in {repr_symbol_to_source()}.  If this assert is failing, it could be due to the issue described in https://github.com/pytorch/pytorch/pull/90665'
        return self.source_ref(self.symbol_to_source[expr][0])

class LoggingShapeGuardPrinter(ShapeGuardPrinter):

    def __init__(self, var_to_sources):
        if False:
            i = 10
            return i + 15
        super().__init__(var_to_sources, lambda n: n.name(), var_to_sources)

class DynamicDimConstraintPrinter(StrPrinter):
    """
    Printer for dynamic dim constraints.
    - Instead of t.size()[d] it prints dynamic_dim(t, d)
    - Instead of Eq(_, _), Mod(_, _), etc. it prints _ == _, _ % _, etc.

    We use this to suggest code for specifying dynamic dim constraints.
    """

    def __init__(self, symbol_to_source, source_name_to_debug_name):
        if False:
            while True:
                i = 10
        super().__init__()
        self.symbol_to_source = symbol_to_source
        self.source_name_to_debug_name = source_name_to_debug_name

    def print_source(self, source) -> str:
        if False:
            return 10
        if self.source_name_to_debug_name:
            return source.name()
        return f'dynamic_dim({source.base.name()}, {source.idx})'

    def _print_Symbol(self, expr) -> str:
        if False:
            return 10
        assert isinstance(expr, sympy.Symbol), str(type(expr))
        return self.print_source(self.symbol_to_source[expr][0])

    def _print_Relational(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return '{} {} {}'.format(self.parenthesize(expr.lhs, precedence(expr)), expr.rel_op, self.parenthesize(expr.rhs, precedence(expr)))

class DimConstraints:
    """
    Custom solver for a system of constraints on symbolic dimensions.
    Solutions are "static" values or simplified "dynamic" constraints.
    """

    def __init__(self, symbol_to_source, var_to_val, marked_dynamic, source_name_to_debug_name):
        if False:
            i = 10
            return i + 15
        self._univariate_inequalities: Dict[sympy.Symbol, Set[sympy.Expr]] = defaultdict(set)
        self._symbols_with_equalities: Set[sympy.Symbol] = set()
        self._substitutions: Dict[sympy.Symbol, sympy.Integer] = {}
        self._var_to_val: Dict[sympy.Symbol, sympy.Integer] = var_to_val
        self._congruences: Set[sympy.Expr] = defaultdict(set)
        self._multivariate_inequalities: Set[sympy.Expr] = set()
        self._symbolic_equivalences: List[Tuple[Source, sympy.Expr]] = []
        self._static_results: Set[str] = set()
        self._dynamic_results: Set[str] = set()
        self._dcp = DynamicDimConstraintPrinter(symbol_to_source, source_name_to_debug_name)
        self._inconsistencies: List[str] = []
        self._marked_dynamic = marked_dynamic

    def rewrite_with_congruences(self, s, expr):
        if False:
            for i in range(10):
                print('nop')
        '\n        Eliminate expressions of the form b // d and b % d while adding congruences of the form b % d == k.\n        This leaves rational operators (in particular of the form b / d) that our inequality solver can handle.\n        We solve the added congruences separately (using our congruence solver, see below).\n        '

        def mod_handler(*args):
            if False:
                return 10
            (base, divisor) = args
            (base, divisor) = (self.rewrite_with_congruences(s, base), self.rewrite_with_congruences(s, divisor))
            mod_reduced = base.subs(self._var_to_val) % divisor.subs(self._var_to_val)
            congruence = (base - mod_reduced) % divisor
            if congruence != 0:
                self._congruences[s].add(congruence)
            return mod_reduced

        def floor_div_handler(*args):
            if False:
                print('Hello World!')
            (base, divisor) = args
            (base, divisor) = (self.rewrite_with_congruences(s, base), self.rewrite_with_congruences(s, divisor))
            mod_reduced = base.subs(self._var_to_val) % divisor.subs(self._var_to_val)
            congruence = (base - mod_reduced) % divisor
            if congruence != 0:
                self._congruences[s].add(congruence)
            return (base - mod_reduced) / divisor
        if expr.has(Mod):
            expr = expr.replace(Mod, mod_handler)
        if expr.has(FloorDiv):
            expr = expr.replace(FloorDiv, floor_div_handler)
        return expr

    def add(self, expr) -> bool:
        if False:
            print('Hello World!')
        if expr == sympy.true:
            return True
        orig_expr = expr
        orig_reduced = orig_expr.subs(self._var_to_val)
        if orig_reduced == sympy.false:
            self._inconsistencies.append(f'{orig_expr} is inconsistent!')
        free_symbols = expr.free_symbols
        assert free_symbols, f'Did not expect constraint with no free variables: {expr}'
        if len(free_symbols) > 1:
            self._multivariate_inequalities.add(expr)
        else:
            s = next(iter(free_symbols))
            expr = self.rewrite_with_congruences(s, expr)
            if expr == sympy.true:
                return True
            reduced = expr.subs(self._var_to_val)
            if reduced == sympy.false:
                self._inconsistencies.append(f'{expr}, obtained by rewriting {orig_expr} with congruences, is inconsistent!')
            if isinstance(expr, sympy.Eq):
                self._symbols_with_equalities.add(s)
            self._univariate_inequalities[s].add(expr)
        return False

    def add_equality(self, source, expr):
        if False:
            i = 10
            return i + 15
        if expr.is_number:
            self._static_results.add(f'{source.name()} == {expr}')
        else:
            self._symbolic_equivalences.append((source, expr))

    def reduce_congruences(self):
        if False:
            return 10
        reduced_congruences = {}
        for (s, congruences) in self._congruences.items():
            remainder_modulus_pairs = []
            congruences_to_check = set()
            for congruence in congruences:
                (base, divisor) = congruence.args
                tmp = sympy.Symbol('tmp', integer=True)
                (symbol, solution) = sympy.solve_linear(base - divisor * tmp, symbols=[s])
                if s == symbol:
                    (modulus, remainder) = sympy.polys.polytools.div(solution, tmp)
                    if isinstance(modulus, sympy.Integer) and isinstance(remainder, sympy.Integer):
                        remainder = remainder % modulus
                        remainder_modulus_pairs.append((remainder, modulus))
                        continue
                congruences_to_check.add(congruence)
            if remainder_modulus_pairs:
                (remainder, modulus) = sympy.ntheory.modular.solve_congruence(*remainder_modulus_pairs)
                reduced_congruences[s] = {(s - remainder) % modulus}
                substitution = {s: modulus * sympy.Symbol('tmp', integer=True) + remainder}
                reduced_congruences[s].update((congruence for congruence in congruences_to_check if not sympy.checksol(congruence, substitution)))
            else:
                reduced_congruences[s] = congruences_to_check
        return reduced_congruences

    def raise_inconsistencies(self):
        if False:
            i = 10
            return i + 15
        if self._inconsistencies:
            msg = '\n'.join(self._inconsistencies)
            self._inconsistencies.clear()
            raise ValueError(f'The following inconsistencies were found:\n{msg}')

    def _force_specialization(self, s):
        if False:
            while True:
                i = 10
        val = self._var_to_val[s]
        self._static_results.add(f'{self._dcp.symbol_to_source[s][0].name()} == {val}')
        self._substitutions[s] = val

    def specialize_divisor_symbols(self):
        if False:
            print('Hello World!')
        for expr in self._multivariate_inequalities:
            for atom in expr.atoms(FloorDiv, Mod):
                (_, divisor) = atom.args
                for s in divisor.free_symbols:
                    self._force_specialization(s)
        multivariate_inequalities = self._multivariate_inequalities
        self._multivariate_inequalities = set()
        for expr in multivariate_inequalities:
            self.add(expr.subs(self._substitutions))
        self.raise_inconsistencies()
        self._univariate_inequalities = {s: exprs for (s, exprs) in self._univariate_inequalities.items() if s not in self._substitutions}
        self._congruences = {s: congruences for (s, congruences) in self._congruences.items() if s not in self._substitutions}

    def solve(self, disable_congruences=True, disable_equivalences=True):
        if False:
            print('Hello World!')
        self.raise_inconsistencies()
        while self._symbols_with_equalities:
            s = self._symbols_with_equalities.pop()
            exprs = self._univariate_inequalities.pop(s)
            solution = sympy.solvers.inequalities.reduce_inequalities(exprs, s)
            if isinstance(solution, sympy.And):
                solution = next((arg for arg in solution.args if isinstance(arg, sympy.Eq)), solution)
            assert isinstance(solution, sympy.Eq), f'Expected an equality constraint for {s}, got {solution}'
            (symbol, val) = solution.args
            assert symbol == s, f'Expected a constraint on {s} instead of on {symbol}'
            self._static_results.add(f'{self._dcp.symbol_to_source[s][0].name()} == {val}')
            self._substitutions[s] = val
            multivariate_inequalities = self._multivariate_inequalities
            self._multivariate_inequalities = set()
            for expr in multivariate_inequalities:
                self.add(expr.subs(s, self._substitutions[s]))
            self.raise_inconsistencies()
        self.specialize_divisor_symbols()
        reduced_congruences = self.reduce_congruences()
        for (s, congruences) in reduced_congruences.items():
            for congruence in congruences:
                if s not in self._substitutions or not sympy.checksol(congruence, {s: self._substitutions[s]}):
                    if disable_congruences:
                        self._force_specialization(s)
                        self._univariate_inequalities.pop(s, None)
                    else:
                        self._dynamic_results.add(self._dcp.doprint(sympy.Eq(congruence, 0)))
        for (s, exprs) in self._univariate_inequalities.items():
            try:
                solution = sympy.solvers.inequalities.reduce_inequalities(exprs, s)
                if isinstance(solution, sympy.And):
                    for arg in solution.args:
                        self._dynamic_results.add(self._dcp.doprint(arg))
                else:
                    self._dynamic_results.add(self._dcp.doprint(solution))
            except NotImplementedError as e:
                log.warning('Failed to reduce inequalities: %s', e)
                for expr in exprs:
                    self._dynamic_results.add(self._dcp.doprint(expr))
        symbolic_equivalences = self._symbolic_equivalences
        self._symbolic_equivalences = []
        for (source, expr) in symbolic_equivalences:
            if disable_equivalences and (not isinstance(expr, sympy.Symbol)):
                for s in expr.free_symbols:
                    self._force_specialization(s)
                    sexpr = self._dcp._print_Symbol(s)
                    self._dynamic_results = {r for r in self._dynamic_results if sexpr not in r}
            self.add_equality(source, expr.subs(self._substitutions))
        for (source, expr) in self._symbolic_equivalences:
            self._dynamic_results.add(f'{self._dcp.print_source(source)} == {self._dcp.doprint(expr)}')

    def forced_specializations(self):
        if False:
            while True:
                i = 10

        def debug_name(src):
            if False:
                return 10
            name = src.name()
            if self._dcp.source_name_to_debug_name:
                return f'{self._dcp.source_name_to_debug_name[name]} = {name}'
            else:
                return name
        return {debug_name(self._dcp.symbol_to_source[s][0]): val for (s, val) in self._substitutions.items() if s in self._marked_dynamic}

    def remove_redundant_dynamic_results(self):
        if False:
            i = 10
            return i + 15
        candidates_for_removal = []
        dynamic_results = set()
        for dc in self._dynamic_results:
            dc_ = re.sub('2 <= dynamic_dim(.+)', 'dynamic_dim\\1', dc)
            if dc != dc_:
                candidates_for_removal.append(dc_)
            else:
                dynamic_results.add(dc_)
        for dc in candidates_for_removal:
            found = False
            for other_dc in dynamic_results:
                if dc in other_dc:
                    found = True
            if not found:
                dynamic_results.add(dc)
        self._dynamic_results = dynamic_results

    def prettify_results(self, original_signature: inspect.Signature, constraint_violation_error=None, forced_specializations=None):
        if False:
            i = 10
            return i + 15
        if self._dcp.source_name_to_debug_name:

            def transform(s):
                if False:
                    for i in range(10):
                        print('nop')
                for (k, v) in self._dcp.source_name_to_debug_name.items():
                    s = s.replace(k, v)
                return s
            results = defaultdict(dict)

            def flip(op):
                if False:
                    i = 10
                    return i + 15
                if op == '<=':
                    return '>='
                if op == '>=':
                    return '<='
                if op == '<':
                    return '>'
                if op == '>':
                    return '<'
                assert op == '=='
                return op

            def relation_with_digit(expr, op, digit):
                if False:
                    for i in range(10):
                        print('nop')
                if op == '<=':
                    results[expr]['max'] = digit
                elif op == '<':
                    results[expr]['max'] = digit - 1
                elif op == '>=':
                    results[expr]['min'] = digit
                elif op == '>':
                    results[expr]['min'] = digit + 1
                else:
                    assert op == '=='
                    results[expr]['eq'] = digit
            for s in self._static_results.union(self._dynamic_results):
                t = transform(s)
                if t == s:
                    continue
                (left, op, right) = t.split(' ')
                if op == '==' and left == right:
                    continue
                if right.isdigit():
                    relation_with_digit(left, op, int(right))
                elif left.isdigit():
                    relation_with_digit(right, flip(op), int(left))
                else:
                    assert op == '=='
                    results[left]['eq'] = right
            buf = ''
            debug_names = set()
            if forced_specializations:
                debug_names.update((k.split(' = ')[0] for k in forced_specializations.keys()))
                buf += f"Specializations unexpectedly required ({', '.join(debug_names)})! For more information, run with TORCH_LOGS=dynamic.\n"
                for (s, val) in forced_specializations.items():
                    buf += f'  - {s} must be specialized to {val} because the guards generated for it are too complex.\n'
            dims = []
            others = []
            match = None
            if constraint_violation_error:
                match = re.search('Constraints violated \\((.*)\\)', constraint_violation_error.args[0])
            if match is not None:
                debug_names.update(match.expand('\\1').split(', '))
            for (k, c) in results.items():
                if k not in debug_names:
                    continue
                if 'eq' in c:
                    other = c['eq']
                    if isinstance(other, int):
                        others.append(f'{k} = None  # {other}')
                    else:
                        others.append(f'{k} = {other}')
                else:
                    min_ = c.get('min', None)
                    if min_ == 2:
                        min_ = None
                    max_ = c.get('max', None)
                    if min_ is not None and max_ is not None:
                        dims.append(f"{k} = Dim('{k}', min={min_}, max={max_})")
                    elif min_ is not None:
                        dims.append(f"{k} = Dim('{k}', min={min_})")
                    elif max_ is not None:
                        dims.append(f"{k} = Dim('{k}', max={max_})")
                    else:
                        dims.append(f"{k} = Dim('{k}')")
            buf += '\nSuggested fixes:\n  '
            buf += '\n  '.join(dims + others)
            return buf

        def extract_and_rewrite_local(dc):
            if False:
                while True:
                    i = 10
            match = re.search("L\\['(.+?)'\\]", dc)
            if match is None:
                return
            arg = match.expand('\\1')
            dc = re.sub("L\\['(.+?)'\\]", '\\1', dc)
            return (arg, dc)

        def group(results, args_index):
            if False:
                i = 10
                return i + 15
            groups = defaultdict(list)
            for dc in results:
                local = extract_and_rewrite_local(dc)
                if local is None:
                    continue
                (arg, dc) = local
                if arg in args_index:
                    groups[args_index[arg]].append(dc)
                else:
                    continue
            sorted_groups = []
            for (idx, dcs) in sorted(groups.items()):
                (_, arg) = idx
                sorted_groups.append((arg, sorted(dcs)))
            return sorted_groups
        signature = original_signature.replace(return_annotation=inspect.Signature.empty)
        args_index = {}
        for (i, arg) in enumerate(signature.parameters.keys()):
            args_index[arg] = (i, arg)

        def print_results(grouped, indent, result_fn):
            if False:
                i = 10
                return i + 15
            nonlocal buf
            space = False
            for (arg, results) in grouped:
                if space:
                    buf += '\n'
                else:
                    space = True
                buf += f'\n{indent}# {arg}:'
                for result in results:
                    buf += f'\n{indent}{result_fn(result)}'
        buf = ''
        if forced_specializations:
            buf += 'Some dynamic dimensions need to be specialized because the constraints inferred for them are too complex to specify.\n'
            for (s, val) in forced_specializations.items():
                buf += f'  - {s}, which was marked dynamic, must be specialized to {val}.\n'
        indent = 4 * ' '
        if self._static_results:
            grouped_static_results = group(self._static_results, args_index)
            buf += '\nThe following dimensions have been specialized and CANNOT be dynamic.'
            buf += f'\n```\ndef specializations{str(signature)}:'
            print_results(grouped_static_results, indent, lambda result: f'assert {result}')
            buf += '\n```\n'
        if self._dynamic_results:
            grouped_dynamic_results = group(self._dynamic_results, args_index)
            buf += '\nThe following dimensions CAN be dynamic.'
            buf += '\nPlease use the following code to specify the constraints they must satisfy:'
            buf += f'\n```\ndef specify_constraints{str(signature)}:'
            buf += f'\n{indent}return ['
            print_results(grouped_dynamic_results, indent * 2, lambda result: f'{result},')
            buf += f'\n{indent}]\n```\n'
        return buf
TLS = threading.local()

class ShapeEnv:

    def __init__(self, *, should_record_events: Optional[bool]=None, tracked_fakes: Optional[List[Any]]=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        self._init(**kwargs)
        kwargs['should_record_events'] = False
        from torch.fx.experimental.validator import translation_validation_enabled
        self._translation_validation_enabled = translation_validation_enabled()
        self.should_record_events = should_record_events if should_record_events is not None else self._translation_validation_enabled and (not config.translation_validation_no_bisect)
        self.check_recorded_events = self.should_record_events and config.check_shape_env_recorded_events
        self.is_recording = not self.should_record_events
        self.tracked_fakes = tracked_fakes
        self.events: List[ShapeEnvEvent] = [ShapeEnvEvent(ShapeEnv, kwargs=kwargs)] if self.should_record_events else []

    def _init(self, *, allow_scalar_outputs=True, allow_dynamic_output_shape_ops=True, assume_static_by_default=False, specialize_zero_one=True, duck_shape=True, co_fields=None):
        if False:
            i = 10
            return i + 15
        self.allow_scalar_outputs = allow_scalar_outputs
        self.allow_dynamic_output_shape_ops = allow_dynamic_output_shape_ops
        self.guards: List[ShapeGuard] = []
        self.var_to_val: Dict[sympy.Symbol, sympy.Integer] = {}
        self.var_to_range: Dict[sympy.Symbol, ValueRanges] = {}
        self.source_name_to_debug_name: Dict[str, str] = {}
        self.runtime_var_to_range: Dict[sympy.Symbol, ValueRanges] = {}
        self.var_to_sources: Dict[sympy.Symbol, List[Source]] = {}
        self.var_to_stack: Dict[sympy.Symbol, CapturedTraceback] = {}
        self.var_to_guards: Dict[sympy.Symbol, Tuple[Optional[ShapeGuard], Optional[ShapeGuard]]] = {}
        self.replacements: Dict[sympy.Symbol, sympy.Expr] = {}
        self.divisible: Set[sympy.Expr] = set()
        self.val_to_var: Dict[int, sympy.Expr] = {}
        if specialize_zero_one:
            self.val_to_var = {0: sympy.Integer(0), 1: sympy.Integer(1)}
        self.unbacked_symfloat_counter = itertools.count()
        self.unbacked_symint_counter = itertools.count()
        self.deferred_runtime_asserts: Dict[sympy.Symbol, List[RuntimeAssert]] = {}
        self.num_deferred_runtime_asserts = 0
        self.assume_static_by_default = assume_static_by_default
        self.specialize_zero_one = specialize_zero_one
        self.duck_shape = duck_shape
        self.log = log
        self.log.info('create_env')
        self.frozen = False
        self.dim_constraints: Optional[DimConstraints] = None
        self.counter = collections.Counter()
        self.co_fields = co_fields if co_fields else {}
        self._prev_cache_key = self._get_key()
        self._version_counter = 0
        self.fx_node_cache: Dict[Tuple[Callable, Tuple[Any, ...]], torch.fx.Node] = {}
        self.source_to_symbol: Dict[str, sympy.Symbol] = {}
        from torch.fx.experimental.validator import translation_validation_enabled
        self._translation_validation_enabled = translation_validation_enabled()
        if self._translation_validation_enabled:
            from torch.fx.experimental.validator import TranslationValidator
            self.validator = TranslationValidator()
            self.graph = torch.fx.Graph()
            self.graph.inserting_before(self.graph.output(None))
            self.name_to_node: Dict[str, torch.fx.Node] = {}

    def check_equal(self, other: 'ShapeEnv') -> None:
        if False:
            print('Hello World!')
        non_state_variable_names = ('counter', 'log', 'var_to_stack', 'fx_node_cache', 'graph', 'validator', 'check_recorded_events', 'should_record_events', 'is_recording', 'tracked_fakes', 'events', 'source_name_to_debug_name', '_prev_cache_key', '_version_counter')

        def map_value(key: str, value: Any) -> Any:
            if False:
                print('Hello World!')
            if key in ('unbacked_symfloat_counter', 'unbacked_symint_counter'):
                from copy import copy
                return next(copy(value))
            elif key == 'guards':
                return [g.expr for g in value]
            elif key == 'var_to_guards':
                return {s: (lb.expr if lb is not None else None, ub.expr if ub is not None else None) for (s, (lb, ub)) in value.items()}
            elif key == 'deferred_runtime_asserts':
                return {s: [ra.expr for ra in ras] for (s, ras) in value.items()}
            elif key == 'name_to_node':
                return set(value.keys())
            return value
        shape_env_check_state_equal(self, other, non_state_variable_names, map_value)

    def snapshot_tracked_fakes(self) -> Optional[List[Any]]:
        if False:
            i = 10
            return i + 15
        if self.tracked_fakes is None:
            return None
        from torch._dynamo.variables.builder import TrackedFake

        def maybe_transform_fake(fake: TrackedFake):
            if False:
                while True:
                    i = 10
            inner_fake = fake.fake if isinstance(fake.fake, torch.SymInt) else FakeTensorMeta.from_fake(fake.fake)
            return TrackedFake(inner_fake, fake.source, fake.constraint_dims)
        return [maybe_transform_fake(fake) for fake in self.tracked_fakes]

    def inc_tracked_fakes_length(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.tracked_fakes_length += 1

    def set_tracked_fakes_length(self, i: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.tracked_fakes_length = i

    def last_event_index(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self.events) - 1

    @contextmanager
    def recording(self):
        if False:
            while True:
                i = 10
        self.is_recording = True
        try:
            yield
        finally:
            self.is_recording = False

    @record_shapeenv_event()
    def freeze(self):
        if False:
            print('Hello World!')
        self.frozen = True

    def _create_symbol_for_source(self, source: Source) -> Optional[sympy.Symbol]:
        if False:
            print('Hello World!')
        if not self._translation_validation_enabled:
            return None
        srcname = source.name()
        if source not in self.source_to_symbol:
            self.source_to_symbol[srcname] = sympy.Symbol(srcname, integer=True)
        return self.source_to_symbol[srcname]

    def _add_z3var(self, symbol: sympy.Symbol, type: Type) -> None:
        if False:
            i = 10
            return i + 15
        if self._translation_validation_enabled:
            self.validator.add_var(symbol, type)

    def _add_target_expr(self, expr) -> None:
        if False:
            print('Hello World!')
        if self._translation_validation_enabled:
            self.validator.add_target_expr(expr)

    def _add_assertion(self, expr) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._translation_validation_enabled:
            self.validator.add_assertion(expr)

    def _check_translation_validate(self) -> None:
        if False:
            return 10
        if self._translation_validation_enabled:
            self.validator.validate()

    @record_shapeenv_event()
    def create_fx_call_function(self, op: Callable, args: Tuple) -> Tuple[Optional[torch.fx.Node], bool]:
        if False:
            while True:
                i = 10
        node_key = (op, args)
        fresh = False
        if self._translation_validation_enabled and node_key not in self.fx_node_cache:
            from torch.fx.experimental.validator import z3op
            if any((a is None for a in args)):
                assert all((not isinstance(a, torch.fx.Node) for a in args))
                return (None, fresh)
            fresh = True
            lifted_op = z3op(op, self.validator)
            assert all((a is not None for a in args)), f'missing arg in FX graph ({op.__name__}): {args}'
            node = self.fx_node_cache[node_key] = self.graph.call_function(lifted_op, args)
            self.name_to_node[node.name] = node
        return (self.fx_node_cache.get(node_key, None), fresh)

    def create_fx_placeholder_and_z3var(self, symbol: sympy.Symbol, type: Type) -> Optional[torch.fx.Node]:
        if False:
            print('Hello World!')
        if not self._translation_validation_enabled:
            return None
        node_key = (self.graph.placeholder, (symbol,))
        if node_key not in self.fx_node_cache:
            self._add_z3var(symbol, type)
            mangled_name = re.sub('[^a-zA-Z0-9]', '_', re.sub('[()]', '', symbol.name))
            node = self.fx_node_cache[node_key] = self.graph.placeholder(mangled_name)
            self.name_to_node[node.name] = node
            node.meta['symbol'] = symbol
        return self.fx_node_cache[node_key]

    def remove_fx_node(self, node: Optional[torch.fx.Node]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._translation_validation_enabled and node is not None:
            self.name_to_node.pop(node.name)
            self.graph.erase_node(node)

    def add_fx_node_metadata(self, node: torch.fx.Node) -> None:
        if False:
            while True:
                i = 10
        from torch._dynamo.utils import get_current_node
        if self.should_record_events:
            node.meta[SHAPEENV_EVENT_KEY] = self.last_event_index()
            node.meta[CURRENT_NODE_KEY] = get_current_node()

    def _suppress_guards_tls(self):
        if False:
            while True:
                i = 10
        return getattr(TLS, 'suppress_guards', False)

    @record_shapeenv_event()
    def suppress_guards_enter(self):
        if False:
            i = 10
            return i + 15
        TLS.suppress_guards = True

    @record_shapeenv_event()
    def suppress_guards_exit(self):
        if False:
            print('Hello World!')
        TLS.suppress_guards = False

    @contextmanager
    def suppress_guards(self):
        if False:
            while True:
                i = 10
        self.suppress_guards_enter()
        try:
            yield
        finally:
            self.suppress_guards_exit()

    def _get_key(self):
        if False:
            while True:
                i = 10
        '\n        Defines the current "state" of the guards we\'ve accumulated in this ShapeEnv.\n        Determines when we need to invalidate our cache\n        '
        return (len(self.replacements), len(self.divisible), self.num_deferred_runtime_asserts)

    def _update_version_counter(self):
        if False:
            while True:
                i = 10
        cur_key = self._get_key()
        if self._prev_cache_key != cur_key:
            self._prev_cache_key = cur_key
            self._version_counter += 1

    def _produce_dyn_sizes(self, ex_size: Sequence[int], source: Source, dynamic_dims: DimList[DimDynamic], constraint_dims: DimList[DimConstraint]) -> List[sympy.Expr]:
        if False:
            return 10
        return self._produce_dyn_sizes_from_int_tuple(tuple(ex.size()), source, dynamic_dims, constraint_dims)

    def _produce_dyn_sizes_from_int_tuple(self, tensor_size: Tuple[int], source: Source, dynamic_dims: DimList[DimDynamic], constraint_dims: List[DimConstraint]) -> List[sympy.Expr]:
        if False:
            return 10
        assert all((not is_symbolic(val) for val in tensor_size)), f'Expect size to be a plain tuple of ints but got {tensor_size}'
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        size = []
        for (i, val) in enumerate(tensor_size):
            size.append(self.create_symbol(val, TensorPropertySource(source, TensorProperty.SIZE, i), dynamic_dims[i], constraint_dims[i]))
        return size

    def create_symbolic_sizes_strides_storage_offset(self, ex: torch.Tensor, source: Source, *, dynamic_dims: Optional[DimList[DimDynamic]]=None, constraint_dims: Optional[DimList[DimConstraint]]=None):
        if False:
            return 10
        '\n        Returns a list of symbolic sizes and strides for the given tensor.\n        We try our best to express stride in terms of the sizes, so as to not\n        introduce new symbolic variables.\n        '
        assert not ex.is_nested

        def maybe_specialize_sym_int_with_hint(maybe_sym) -> int:
            if False:
                return 10
            assert isinstance(maybe_sym, (int, torch.SymInt))
            if is_symbolic(maybe_sym):
                assert maybe_sym.node.shape_env is not self, 'expect the symbol is created from an shape env other than current one.'
                return maybe_sym.node.require_hint()
            return maybe_sym
        ex_size = tuple((maybe_specialize_sym_int_with_hint(sz) for sz in ex.size()))
        ex_stride = tuple((maybe_specialize_sym_int_with_hint(sd) for sd in ex.stride()))
        ex_storage_offset = maybe_specialize_sym_int_with_hint(ex.storage_offset())
        return self._create_symbolic_sizes_strides_storage_offset(ex_size, ex_stride, ex_storage_offset, [_is_dim_dynamic(ex, i) for i in range(ex.dim())], source, dynamic_dims=dynamic_dims, constraint_dims=constraint_dims)

    @record_shapeenv_event()
    def _create_symbolic_sizes_strides_storage_offset(self, ex_size: Sequence[int], ex_stride: Sequence[int], ex_storage_offset: int, is_dim_dynamic: Sequence[bool], source: Source, *, dynamic_dims: Optional[DimList[DimDynamic]]=None, constraint_dims: Optional[DimList[DimConstraint]]=None):
        if False:
            for i in range(10):
                print('nop')
        dim = len(ex_size)
        if constraint_dims is None:
            constraint_dims = [None] * dim
        if dynamic_dims is None:
            dynamic_dims = []
            for i in range(dim):
                if is_dim_dynamic[i]:
                    r = DimDynamic.DYNAMIC
                elif self.assume_static_by_default:
                    r = DimDynamic.STATIC
                else:
                    r = DimDynamic.DUCK
                dynamic_dims.append(r)
            dynamic_dims = [DimDynamic.DUCK] * dim
        dynamic_strides_offset = DimDynamic.STATIC if all((r == DimDynamic.STATIC for r in dynamic_dims)) else DimDynamic.DUCK
        assert len(dynamic_dims) == dim, f'{len(dynamic_dims)} != {dim}'
        assert len(constraint_dims) == dim
        from torch._dynamo.source import TensorPropertySource, TensorProperty
        size: List[sympy.Expr] = self._produce_dyn_sizes_from_int_tuple(ex_size, source, dynamic_dims, constraint_dims)
        stride: List[Optional[sympy.Expr]] = [None] * len(size)
        for (i, val) in enumerate(ex_stride):
            if val in (0, 1):
                stride[i] = sympy.Integer(val)
        while any((x is None for x in stride)):
            candidates = {ex_size[i] * ex_stride[i]: size[i] * stride[i] for i in range(len(size)) if stride[i] is not None and ex_stride[i] >= 0}
            val_list = sorted([(ex_stride[i], i) for i in range(len(stride)) if stride[i] is None])
            for (_, i) in val_list:
                if stride[i] is None and ex_stride[i] in candidates:
                    stride[i] = candidates[ex_stride[i]]
                    candidates[ex_size[i] * ex_stride[i]] = size[i] * stride[i]
            if any((x is None for x in stride)):
                (val, i) = min([(ex_stride[i], i) for i in range(len(stride)) if stride[i] is None])
                stride[i] = self.create_symbol(val, TensorPropertySource(source, TensorProperty.STRIDE, i), dynamic_dim=dynamic_strides_offset, constraint_dim=None)
        assert all((x is not None for x in stride))
        sym_sizes = [self.create_symintnode(sym, hint=hint, source=TensorPropertySource(source, TensorProperty.SIZE, i)) for (i, (sym, hint)) in enumerate(zip(size, ex_size))]
        sym_stride = []
        for (i, stride_expr) in enumerate(stride):
            assert stride_expr is not None
            sym_stride.append(self.create_symintnode(stride_expr, hint=ex_stride[i], source=TensorPropertySource(source, TensorProperty.STRIDE, i)))
        sym_storage_offset = self.create_symintnode(self.create_symbol(ex_storage_offset, TensorPropertySource(source, TensorProperty.STORAGE_OFFSET), dynamic_dim=dynamic_strides_offset, constraint_dim=None), hint=ex_storage_offset, source=TensorPropertySource(source, TensorProperty.STORAGE_OFFSET))
        return (tuple(sym_sizes), tuple(sym_stride), sym_storage_offset)

    @record_shapeenv_event()
    def create_symintnode(self, sym: 'sympy.Expr', *, hint: Optional[int], source: Optional[Source]=None):
        if False:
            return 10
        if self._translation_validation_enabled and source is not None:
            symbol = self._create_symbol_for_source(source)
            assert symbol is not None
            fx_node = self.create_fx_placeholder_and_z3var(symbol, int)
            self._add_assertion(sympy.Eq(symbol, sym))
        else:
            fx_node = None
        if isinstance(sym, sympy.Integer):
            if hint is not None:
                assert int(sym) == hint
            return int(sym)
        return SymInt(SymNode(sym, self, int, hint, fx_node=fx_node))

    @record_shapeenv_event()
    def create_unspecified_symint_and_symbol(self, value, source, dynamic_dim):
        if False:
            i = 10
            return i + 15
        return self.create_symintnode(self.create_unspecified_symbol(value, source=source, dynamic_dim=dynamic_dim), hint=value, source=source)

    def create_symboolnode(self, sym: 'sympy.Expr'):
        if False:
            return 10
        return SymBool(SymNode(sym, self, bool, None))

    @record_shapeenv_event()
    def create_unbacked_symfloat(self):
        if False:
            i = 10
            return i + 15
        symbol: sympy.Symbol = sympy.Symbol(f'f{next(self.unbacked_symfloat_counter)}')
        self.counter['create_unbacked_symbol'] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        self.var_to_range[symbol] = ValueRanges.unknown()
        fx_node = self.create_fx_placeholder_and_z3var(symbol, float)
        return SymFloat(SymNode(symbol, self, float, None, fx_node=fx_node))

    @record_shapeenv_event()
    def create_unbacked_symint(self):
        if False:
            while True:
                i = 10
        symbol: sympy.Symbol = sympy.Symbol(f'i{next(self.unbacked_symint_counter)}', integer=True)
        self.counter['create_unbacked_symbol'] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        vr = self.var_to_range[symbol] = self._default_unspecified_value_range()
        fx_node = self.create_fx_placeholder_and_z3var(symbol, int)
        (fsummary, user_tb, maybe_user_loc) = self._get_stack_summary()
        log.info('create_unbacked_symbol %s [%s, %s]%s (%s)', symbol, vr.lower, vr.upper, maybe_user_loc, format_frame(fsummary))
        return SymInt(SymNode(symbol, self, int, None, fx_node=fx_node))

    def is_unbacked_symint(self, symbol: sympy.Symbol) -> bool:
        if False:
            while True:
                i = 10
        return str(symbol).startswith('i')

    @record_shapeenv_event()
    def create_unbacked_symbool(self):
        if False:
            while True:
                i = 10
        symbol: sympy.Symbol = sympy.Symbol(f'i{next(self.unbacked_symint_counter)}', integer=True)
        self.counter['create_unbacked_symbol'] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        self.var_to_range[symbol] = ValueRanges(0, 1)
        fx_node = self.create_fx_placeholder_and_z3var(symbol, bool)
        return SymBool(SymNode(sympy.Eq(symbol, 1), self, bool, None, fx_node=fx_node))

    @record_shapeenv_event()
    def create_unspecified_symbol(self, val: Union[int, SymInt], source: Source, dynamic_dim: DimDynamic=DimDynamic.DUCK, constraint_dim: DimConstraint=None) -> 'sympy.Expr':
        if False:
            i = 10
            return i + 15
        return self.create_symbol(val, source, dynamic_dim, constraint_dim, positive=None, do_not_specialize_zero_one=True)

    @record_shapeenv_event()
    def create_symbol(self, val: int, source: Source, dynamic_dim: DimDynamic=DimDynamic.DUCK, constraint_dim: DimConstraint=None, positive: Optional[bool]=True, do_not_specialize_zero_one: bool=False) -> 'sympy.Expr':
        if False:
            i = 10
            return i + 15
        if do_not_specialize_zero_one:
            specialize_zero_one = False
        else:
            specialize_zero_one = self.specialize_zero_one
        assert isinstance(source, Source), f'{type(source)} {source}'
        assert not (positive and val < 0), f'positive set for negative value: {val}'
        if constraint_dim is not None:
            dynamic_dim = DimDynamic.DYNAMIC
        if dynamic_dim is DimDynamic.STATIC:
            return sympy.Integer(val)
        elif dynamic_dim is DimDynamic.DUCK:
            duck = self.duck_shape
        elif dynamic_dim is DimDynamic.DYNAMIC:
            duck = False
        else:
            raise AssertionError(f'unhandled dynamic_dim {dynamic_dim}')
        if val in (0, 1) and specialize_zero_one:
            r = self.val_to_var[val]
        elif not duck or val not in self.val_to_var:
            sympy_expr = sympy.Symbol(f's{len(self.var_to_val)}', positive=positive, integer=True)
            if isinstance(val, int):
                self.var_to_val[sympy_expr] = sympy.Integer(val)
            else:
                self.var_to_val[sympy_expr] = SingletonInt(val.node.singleton_int(), coeff=val.node.singleton_coeff())
            self.var_to_sources[sympy_expr] = []
            self._add_z3var(sympy_expr, int)
            if duck:
                self.val_to_var[val] = sympy_expr
            if isinstance(val, int):
                if positive:
                    self._add_assertion(sympy_expr > 1)
                    self.var_to_range[sympy_expr] = self._default_value_range()
                else:
                    self.var_to_range[sympy_expr] = self._default_unspecified_value_range()
                if isinstance(constraint_dim, StrictMinMaxConstraint):
                    assert not duck
                    self.var_to_range[sympy_expr] &= constraint_dim.vr
                vr = self.var_to_range[sympy_expr]
                if val not in vr:
                    raise ConstraintViolationError(f'{val} not in range [{vr.lower}, {vr.upper}]')
                self.runtime_var_to_range[sympy_expr] = vr
                range_str = f'[{vr.lower}, {vr.upper}]'
            else:
                range_str = ''
            r = sympy_expr
            self.log.info('create_symbol %s = %s for %s %s', sympy_expr, val, source.name(), range_str)
            self.counter['create_symbol'] += 1
        else:
            r = self.val_to_var[val]
            self.log.debug('create_symbol %s duck sized %s', r, source.name())
        if isinstance(r, sympy.Symbol):
            self.var_to_sources[r].append(source)
        return r

    def debug_name(self, source):
        if False:
            while True:
                i = 10
        src_name = source.name()
        return self.source_name_to_debug_name.get(src_name, src_name)

    def render_range_for_constraint_violation(self, source, c):
        if False:
            while True:
                i = 10
        if isinstance(c, StrictMinMaxConstraint):
            (lower, upper) = (c.vr.lower, c.vr.upper)
            default = self._default_value_range()
            if lower <= default.lower:
                lower = None
            if upper >= default.upper:
                upper = None
            c_render = f'{self.debug_name(source)} = {source.name()} in the specified range'
            if lower is not None and upper is not None:
                c_render += f' {lower} <= {self.debug_name(source)} <= {upper}'
            elif lower is None and upper is not None:
                c_render += f' {self.debug_name(source)} <= {upper}'
            elif lower is not None and upper is None:
                c_render += f' {lower} <= {self.debug_name(source)}'
            return c_render
        return c.render(source)

    def produce_guards(self, placeholders, sources, source_ref=lambda n: n.name(), *, constraint_inputs: Optional[InputList[Union[DimConstraint, Optional[DimList[DimConstraint]]]]]=None, equalities_inputs: Optional[Set[Tuple[Source, Source]]]=None, _simplified=False, ignore_static=True) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        self.log.info('produce_guards')
        if self.check_recorded_events:
            shape_env = replay_shape_env_events(self.events)
            self.check_equal(shape_env)
        assert len(placeholders) == len(sources)
        Tensorlike = (torch.Tensor, FakeTensorMeta)
        if constraint_inputs is None:
            constraint_inputs = [[None] * t.dim() if isinstance(t, Tensorlike) else None for t in placeholders]
        else:
            assert len(constraint_inputs) == len(placeholders)
            for (i, (t, constraint)) in enumerate(zip(placeholders, constraint_inputs)):
                if isinstance(t, Tensorlike):
                    if constraint is None:
                        constraint_inputs[i] = [None] * t.dim()
                    else:
                        assert len(constraint) == t.dim()
                else:
                    assert isinstance(t, (SymInt, int))
                    assert not isinstance(constraint, list)
        from torch._dynamo.source import TensorPropertySource, TensorProperty, NegateSource
        input_guards = []
        symbol_to_source = collections.defaultdict(list)
        symbol_to_constraints = collections.defaultdict(set)
        constraint_violations: List[Tuple[bool, Callable[[], str]]] = []

        def record_constraint_violation(warn_only, debug_name, msg, hint=None):
            if False:
                for i in range(10):
                    print('nop')
            constraint_violations.append((warn_only, debug_name, lambda : f'{msg}{hint()}' if hint else msg))

        def is_dim(src):
            if False:
                i = 10
                return i + 15
            return isinstance(src, TensorPropertySource) and src.prop is TensorProperty.SIZE
        if equalities_inputs:
            source_index = {}
            for (i, src) in enumerate(sources):
                source_index[src.name()] = i

            def get_symbol(tensor_dim_src):
                if False:
                    for i in range(10):
                        print('nop')
                fake = placeholders[source_index[tensor_dim_src.base.name()]]
                symint = fake.shape[tensor_dim_src.idx]
                assert isinstance(symint, torch.SymInt)
                return symint.node.expr
            for (src1, src2) in equalities_inputs.source_pairs:
                (s1, s2) = (get_symbol(src1), get_symbol(src2))
                concrete_val = self.evaluate_expr(sympy.Eq(s1, s2))
                if not concrete_val:
                    raise ConstraintViolationError(f'{src1.name()} = {self.var_to_val[s1]} is not equal to {src2.name()} = {self.var_to_val[s2]}')

        def track_symint(source, val, constraint=None):
            if False:
                print('Hello World!')
            log.debug('track_symint %s %s %s', LazyString(source.name), val, constraint)
            assert not isinstance(val, SymInt) or is_symbolic(val)
            if isinstance(val, SymInt) and val.node.maybe_as_int() is not None:
                val = val.node.maybe_as_int()
            if isinstance(val, SymInt):
                s = val.node.expr
                if isinstance(s, sympy.Symbol):
                    symbol_to_source[s].append(source)
                    if constraint is not None:
                        symbol_to_constraints[s].add(constraint)
                elif isinstance(-s, sympy.Symbol):
                    symbol_to_source[-s].append(NegateSource(source))
                else:
                    constraint_violated = False
                    if isinstance(constraint, StrictMinMaxConstraint):
                        constraint_violated = True
                    elif isinstance(constraint, RelaxedUnspecConstraint):
                        if s.is_number:
                            i = int(s)
                            if i not in (0, 1):
                                constraint_violated = True
                        else:
                            constraint_violated = True
                    if constraint_violated:

                        def hint(s):
                            if False:
                                print('Hello World!')
                            sexpr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(s)
                            return f'{sexpr}.'
                        var_with_range = self.render_range_for_constraint_violation(source, constraint)
                        msg = f'Not all values of {var_with_range} are valid because {self.debug_name(source)} was inferred to be equal to '
                        record_constraint_violation(constraint.warn_only, self.debug_name(source), msg, hint=functools.partial(hint, s))
                input_guards.append((source, s))
            else:
                s = sympy.Integer(val)
                input_guards.append((source, s))
                constraint_violated = False
                if isinstance(constraint, StrictMinMaxConstraint):
                    constraint_violated = True
                elif isinstance(constraint, RelaxedUnspecConstraint):
                    if val not in (0, 1):
                        constraint_violated = True
                if constraint_violated:
                    var_with_range = self.render_range_for_constraint_violation(source, constraint)
                    msg = f'Not all values of {var_with_range} are valid because {self.debug_name(source)} was inferred to be a constant ({val}).'
                    record_constraint_violation(constraint.warn_only, self.debug_name(source), msg)
        for (t, source, constraint) in zip(placeholders, sources, constraint_inputs):
            if isinstance(source, str):
                from torch._dynamo.source import LocalSource
                source = LocalSource(source)
            assert isinstance(source, Source)
            if t is None:
                continue
            if isinstance(t, (SymInt, int)):
                track_symint(source, t)
                continue
            assert isinstance(t, Tensorlike)
            sources_and_tensors = [(source, t)]
            if is_traceable_wrapper_subclass(t):
                (attrs, _) = t.__tensor_flatten__()
                from torch._dynamo.source import AttrSource
                inner_sources_and_tensors = [(AttrSource(source, attr), getattr(t, attr)) for attr in attrs]
                if t.is_nested:
                    sources_and_tensors.extend(inner_sources_and_tensors)
                else:
                    sources_and_tensors = inner_sources_and_tensors
            for (src, curr_t) in sources_and_tensors:
                for (i, ss) in enumerate(curr_t.size()):
                    property_source = TensorPropertySource(src, TensorProperty.SIZE, i)
                    track_symint(property_source, ss, constraint[i])
                if not t.is_nested:
                    for (i, ss) in enumerate(curr_t.stride()):
                        track_symint(TensorPropertySource(src, TensorProperty.STRIDE, i), ss)
                    track_symint(TensorPropertySource(src, TensorProperty.STORAGE_OFFSET), curr_t.storage_offset())
        exprs = []
        self.dim_constraints = DimConstraints(symbol_to_source, self.var_to_val, set(symbol_to_constraints.keys()), self.source_name_to_debug_name)
        if not _simplified:
            for (source, expr) in input_guards:
                if self._translation_validation_enabled:
                    srcname = source.name()
                    if srcname in self.source_to_symbol:
                        self._add_target_expr(sympy.Eq(self.source_to_symbol[srcname], expr))
                if isinstance(expr, sympy.Symbol) and symbol_to_source.get(expr) and (source == symbol_to_source[expr][0]):
                    continue
                if ignore_static and isinstance(source, TensorPropertySource):
                    if expr.is_number:
                        self.log.debug('Skipping guard %s', f'{source_ref(source)} == {expr}')
                        continue
                if is_dim(source):
                    self.dim_constraints.add_equality(source, expr)
                sexpr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(expr)
                exprs.append(f'{source_ref(source)} == {sexpr}')
                if isinstance(expr, sympy.Symbol) and expr in symbol_to_constraints and isinstance(source, TensorPropertySource) and (source.prop is TensorProperty.SIZE) and equalities_inputs and (not equalities_inputs.is_equal(source, symbol_to_source[expr][0])):
                    msg = f'The values of {self.debug_name(source)} = {source.name()} and {self.debug_name(symbol_to_source[expr][0])} = {symbol_to_source[expr][0].name()} must always be equal.'
                    record_constraint_violation(equalities_inputs.warn_only, self.debug_name(source), msg)
        issued = set()

        def issue_guard(guard: ShapeGuard) -> None:
            if False:
                return 10
            expr = self.simplify(guard.expr)
            if expr in issued:
                return
            issued.add(expr)
            try:
                is_trivial = False
                if any((is_dim(source) for s in expr.free_symbols for source in symbol_to_source[s])):
                    is_trivial = self.dim_constraints.add(expr)
                guard_expr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(expr)
                exprs.append(guard_expr)
                self._add_target_expr(expr)
                if not is_trivial and len(expr.free_symbols) == 1:
                    symbol = next(iter(expr.free_symbols))
                    source = symbol_to_source[symbol][0]
                    constraints = symbol_to_constraints[symbol]
                    for c in constraints:
                        if isinstance(c, StrictMinMaxConstraint):
                            var_with_range = self.render_range_for_constraint_violation(source, c)
                            msg = f'Not all values of {var_with_range} satisfy the generated guard {guard_expr}.'
                            record_constraint_violation(c.warn_only, self.debug_name(source), msg)
                        elif isinstance(c, RelaxedUnspecConstraint):
                            pass
                        else:
                            raise AssertionError(f'unrecognized constraint {c}')
            except Exception:
                self.log.warning('Failing guard allocated at: \n%s', ''.join(guard.stack.format()))
                raise
        for guard in self.guards:
            if self._maybe_evaluate_static(guard.expr) is not None:
                continue
            issue_guard(guard)
        for (symbol, guards) in self.var_to_guards.items():
            if symbol not in symbol_to_source:
                continue
            for guard in guards:
                if guard is not None:
                    issue_guard(guard)
        if not _simplified:
            for (symbol, sources) in symbol_to_source.items():
                r = self.runtime_var_to_range.get(symbol)
                if r is None:
                    if symbol not in self.var_to_range:
                        continue
                    r = self.var_to_range[symbol]
                assert sources
                assert symbol.is_integer
                (g_lower, g_upper) = self.var_to_guards.get(symbol, (None, None))
                bounds = []
                if r.lower != -sympy.oo and g_lower is None:
                    if any((is_dim(source) for source in sources)):
                        self.dim_constraints.add(sympy.Ge(symbol, r.lower))
                    bounds.append(str(r.lower))
                bounds.append(source_ref(sources[0]))
                if r.upper != sympy.oo and r.upper < sys.maxsize - 1 and (g_upper is None):
                    if any((is_dim(source) for source in sources)):
                        self.dim_constraints.add(sympy.Le(symbol, r.upper))
                    bounds.append(str(r.upper))
                if len(bounds) > 1:
                    exprs.append(' <= '.join(bounds))
        if constraint_violations:
            warn_msgs = []
            error_msgs = []
            debug_names = set()
            for (warn_only, debug_name, msg) in constraint_violations:
                if warn_only:
                    msg = f'  {len(warn_msgs) + 1}. {msg()}'
                    warn_msgs.append(msg)
                else:
                    msg = f'  - {msg()}'
                    error_msgs.append(msg)
                    debug_names.add(debug_name)
            if len(error_msgs) > 0:
                debug_names = ', '.join(debug_names)
                err = '\n'.join(error_msgs)
                raise ConstraintViolationError(f'Constraints violated ({debug_names})! For more information, run with TORCH_LOGS=dynamic.\n{err}')
            elif len(warn_msgs) > 0:
                log.debug('%s Warning only constraints violated', len(warn_msgs))
        signpost_event('dynamic', 'produce_guards', {**self.co_fields, **self.counter, 'num_guards': len(exprs), 'free_symbols': sum((1 for v in symbol_to_source.values() if v))})
        if self._translation_validation_enabled:
            from torch.fx.experimental.validator import PopulateValidator
            for ras in self.deferred_runtime_asserts.values():
                for ra in ras:
                    self._add_target_expr(ra.expr)
            for (sym, vr) in self.var_to_range.items():
                if vr.lower != -sympy.oo:
                    self._add_target_expr(sympy.Le(vr.lower, sym))
                if vr.upper != sympy.oo:
                    self._add_target_expr(sympy.Le(sym, vr.upper))
            with fx_traceback.preserve_node_meta():
                PopulateValidator(self.graph, self.validator).run()
        self._check_translation_validate()
        return exprs

    def produce_guards_expression(self, placeholders, ignore_static=True):
        if False:
            print('Hello World!')
        '\n        Expected to be used with evaluate_guards_expression(). Produces the guards\n        for the given placeholders and returns a string expression to be evaluated\n        by evaluate_guards_expression given concrete values for the placeholders.\n        '
        from torch._dynamo.source import LocalSource
        arg_names = [f't{i}' for i in range(len(placeholders))]
        guards = self.produce_guards(placeholders, [LocalSource(a) for a in arg_names], ignore_static=ignore_static)
        if guards:
            return ' and '.join(guards)
        return None

    def evaluate_guards_expression(self, code, args):
        if False:
            while True:
                i = 10
        '\n        Expected to be used with produce_guards_expression(). Evaluates an expression\n        generated by produce_guards_expression for the given concrete args.\n        '
        arg_names = [f't{i}' for i in range(len(args))]
        return eval(code, SYMPY_INTERP, {'L': dict(zip(arg_names, args))})

    def evaluate_guards_for_args(self, placeholders, args, *, ignore_static=True):
        if False:
            i = 10
            return i + 15
        code = self.produce_guards_expression(placeholders, ignore_static=ignore_static)
        if code:
            return self.evaluate_guards_expression(code, args)
        return True

    def bind_symbols(self, placeholders, args):
        if False:
            i = 10
            return i + 15
        bindings: Dict[sympy.Symbol, int] = {}

        def bind_symint(arg, val):
            if False:
                i = 10
                return i + 15
            if isinstance(val, SymInt):
                s = val.node.expr
                if isinstance(s, sympy.Symbol):
                    if s in bindings:
                        assert bindings[s] == arg, f'{bindings[s]} != {arg}'
                    else:
                        bindings[s] = arg
                elif isinstance(-s, sympy.Symbol):
                    if -s in bindings:
                        assert bindings[-s] == -arg, f'{bindings[-s]} != {-arg}'
                    else:
                        bindings[-s] = -arg
        for (t, arg) in zip(placeholders, args):
            if t is None:
                continue
            if isinstance(t, SymInt):
                bind_symint(arg, t)
                continue
            assert isinstance(t, torch.Tensor)
            for (i, s) in enumerate(t.size()):
                bind_symint(arg.size(i), s)
            for (i, s) in enumerate(t.stride()):
                bind_symint(arg.stride(i), s)
            bind_symint(arg.storage_offset(), t.storage_offset())
        return bindings

    def get_nontrivial_guards(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.simplify(guard.expr) for guard in self.guards if self._maybe_evaluate_static(guard.expr) is None]

    def format_guards(self, verbose=False):
        if False:
            while True:
                i = 10

        def format_tb(tb):
            if False:
                return 10
            if not verbose:
                return ''
            return f"\n   Guarded at:\n{''.join(('   ' + l for l in tb.format()))}"
        return '\n'.join((f' - {guard.expr}{format_tb(guard.stack)}' for guard in self.guards))

    def get_shape_groups(self):
        if False:
            for i in range(10):
                print('nop')
        shape_groups = collections.defaultdict(list)
        for (k, v) in self.replacements.items():
            shape_groups[v].append(k)
        return shape_groups

    @_lru_cache
    def _maybe_evaluate_static(self, expr: 'sympy.Expr', *, unbacked_only: bool=False, compute_hint: bool=False) -> 'Optional[sympy.Expr]':
        if False:
            while True:
                i = 10
        '\n        Tries to evaluate expr without introducing guards\n\n        If unbacked_only == True, then we only do substitutions on\n        unbacked SymInts (leaving regular hinted integers alone).  This could\n        result in an expression that still contains backed SymInts, which you\n        could then potentially guard on.\n\n        Use compute_hint == True if you are trying to compute a non-binding\n        hint for the particular hint values of backed SymInts, e.g., if\n        s0 happens to be 3 this run, compute_hint will subsitute s0 with 3.\n        '
        expr = self.simplify(expr)
        if compute_hint:
            expr = expr.xreplace(self.var_to_val)
        symbols = list(expr.free_symbols)
        for s in symbols:
            if s in self.var_to_val:
                continue
            subst = {}
            if s in self.deferred_runtime_asserts:
                for ra in self.deferred_runtime_asserts[s]:
                    if compute_hint:
                        e = ra.expr.xreplace(self.var_to_val)
                    else:
                        e = ra.expr
                    subst[e] = sympy.true
                    subst[sympy.Not(e)] = sympy.false
            expr = expr.subs(subst)
        new_shape_env = {}
        new_range_env = {}
        for (idx, k) in enumerate(symbols):
            if isinstance(self.var_to_val.get(k, None), SingletonInt):
                continue
            vr = self.var_to_range[k]
            if vr.lower < (-sys.maxsize - 1) // 2 or (unbacked_only and k in self.var_to_val):
                new_range_env[k] = vr
                continue
            s = sympy.Symbol(f'shape_{idx}', positive=True, integer=True)
            offset = vr.lower - 1
            new_shape_env[k] = s + offset
            new_range_env[s] = SymPyValueRangeAnalysis.add(vr, -offset)

        def replace(expr, repl):
            if False:
                return 10
            return expr.xreplace(repl)
        try:
            new_expr = replace(expr, new_shape_env)
        except RecursionError:
            log.warning('RecursionError in sympy.xreplace(%s, %s)', expr, new_shape_env)
            self.counter['sympy_recursion_error'] += 1
            return None
        floor_div_replace = {}
        for atom in new_expr.atoms(FloorDiv):
            floor_div_replace[atom] = sympy.floor(atom.args[0] / atom.args[1])
        new_expr = safe_expand(new_expr.xreplace(floor_div_replace))
        if new_expr.is_number:
            return new_expr
        out = bound_sympy(new_expr, new_range_env)
        _assert_bound_is_rational(new_expr, out)
        if out.is_singleton():
            return out.lower
        return new_expr if unbacked_only else None

    @_lru_cache
    def replace(self, expr: 'sympy.Expr') -> 'sympy.Expr':
        if False:
            while True:
                i = 10
        replacements = {s: self._find(cast(sympy.Symbol, s)) for s in expr.free_symbols}
        return safe_expand(expr.xreplace(replacements))

    @_lru_cache
    def _update_divisible(self):
        if False:
            while True:
                i = 10
        new_divisible = set()
        for k in self.divisible:
            res = self.replace(k)
            if not res.is_number:
                new_divisible.add(k)
        self.divisible = new_divisible
        self._update_version_counter()

    @_lru_cache
    def simplify(self, expr: 'sympy.Expr') -> 'sympy.Expr':
        if False:
            print('Hello World!')
        expr = self.replace(expr)
        if expr.has(FloorDiv):
            self._update_divisible()
            div_replacements = {}
            for atom in expr.atoms(FloorDiv):
                (base, divisor) = atom.args
                if isinstance(divisor, FloorDiv):
                    (base1, divisor1) = divisor.args
                    if self.replace(Mod(base, divisor)) in self.divisible and base == base1 and (self.replace(Mod(base1, divisor1)) in self.divisible):
                        div_replacements[atom] = divisor1
            expr = expr.xreplace(div_replacements)
            expr = safe_expand(expr)
        if expr.has(FloorDiv):
            div_replacements = {}
            pows = expr.atoms(sympy.Pow)
            rationals = expr.atoms(sympy.Rational).difference(expr.atoms(sympy.Integer))
            for fd in expr.atoms(FloorDiv):
                (base, divisor) = fd.args
                if self.replace(Mod(base, divisor)) in self.divisible:
                    div_replacements[fd] = base / divisor
            new_expr = expr.xreplace(div_replacements)
            new_expr = safe_expand(new_expr)
            new_pows = new_expr.atoms(sympy.Pow)
            new_rationals = new_expr.atoms(sympy.Rational).difference(new_expr.atoms(sympy.Integer))
            if new_pows.issubset(pows) and new_rationals.issubset(rationals):
                expr = new_expr
        return expr

    @lru_cache(256)
    def size_hint(self, expr: 'sympy.Expr', *, allow_none=False):
        if False:
            return 10
        '\n        Gets a size hint for a given expression from the underlying shapes we had.\n        Does not introduce a guard, so only use this when you can guarantee that\n        your code is still valid for arbitrary shapes (such as optimization decisions)\n        '
        result_expr = safe_expand(expr).xreplace(self.var_to_val)
        if not result_expr.is_number:
            r = self._maybe_evaluate_static(result_expr, compute_hint=True)
            if r is not None:
                return r
            if allow_none:
                return None
            raise self._make_data_dependent_error(result_expr, expr)
        return result_expr

    @lru_cache(256)
    def has_hint(self, expr: 'sympy.Expr'):
        if False:
            print('Hello World!')
        result_expr = safe_expand(expr).xreplace(self.var_to_val)
        return result_expr.is_number or self._maybe_evaluate_static(result_expr) is not None

    def _make_data_dependent_error(self, expr, unhinted_expr):
        if False:
            return 10
        for s in expr.free_symbols:
            stacktrace = ''.join(self.var_to_stack[s].format())
            self.log.debug("Data dependent variable '%s' allocated at:\n%s", s, stacktrace)
        return GuardOnDataDependentSymNode(f"It appears that you're trying to get a value out of symbolic int/float whose value is data-dependent (and thus we do not know the true value.)  The expression we were trying to evaluate is {expr} (unhinted: {unhinted_expr}).  Scroll up to see where each of these data-dependent accesses originally occurred.")

    def _set_replacement(self, a: 'sympy.Symbol', expr: 'sympy.Expr') -> None:
        if False:
            print('Hello World!')
        '\n        Adds or updates a replacement for a symbol.\n        Use this instead of `self.replacements[a] = expr`.\n        '
        if config.print_specializations and isinstance(expr, (sympy.Integer, sympy.Float)):
            if a not in self.replacements or expr != self.replacements[a]:
                self.log.warning('Specializing %s to %s', self.var_to_sources[a][0].name(), expr)
                self.log.debug('SPECIALIZATION', stack_info=True)
        log.info('set_replacement %s = %s', a, expr)
        self.replacements[a] = expr
        self._update_version_counter()
        self._add_target_expr(sympy.Eq(a, expr))

    def _add_divisible(self, expr: 'sympy.Expr'):
        if False:
            for i in range(10):
                print('nop')
        self.divisible.add(expr)
        self._update_version_counter()

    @_lru_cache
    @record_shapeenv_event()
    def _find(self, a: 'sympy.Symbol') -> 'sympy.Expr':
        if False:
            for i in range(10):
                print('nop')
        '\n        Implements a DSU-like algorithm to find the variable that represents a\n        Also handles transitive non-identity replacements.\n\n        a: b + c\n        c: d\n        '
        if a not in self.replacements:
            return a
        res = self.replacements[a]
        cur_replace = {s: self._find(s) for s in res.free_symbols}
        self._set_replacement(a, self.replacements[a].xreplace(cur_replace))
        return self.replacements[a]

    @lru_cache(256)
    def _maybe_guard_eq(self, expr: Union['sympy.Eq', 'sympy.Ne'], concrete_bool: bool) -> None:
        if False:
            while True:
                i = 10
        '\n        Evaluates the result of an eq call. If true, uses information to\n        simplify shapes (i.e. a == b or a % 5 == 0)\n        '
        assert type(concrete_bool) is bool
        if isinstance(expr, sympy.Eq):
            if not concrete_bool:
                return
        elif isinstance(expr, sympy.Ne):
            if concrete_bool:
                return
        free = list(expr.free_symbols)
        assert len(free) > 0, f'The expression should not be static by this point: {expr}'
        if len(free) > 5:
            return
        free = sorted(free, key=lambda x: (self.size_hint(x, allow_none=True) or sys.maxsize, x.name), reverse=True)
        lhs = expr.lhs
        rhs = expr.rhs
        if not expr.has(Mod):
            try:
                floor_div_atoms = lhs.atoms(FloorDiv).union(rhs.atoms(FloorDiv))
                if len(floor_div_atoms) > 0 and any((a.divisor != 1 for a in floor_div_atoms)):
                    raise NotImplementedError
                if isinstance(lhs, sympy.Symbol) and free_unbacked_symbols(lhs):
                    self._set_replacement(lhs, self._find(rhs))
                elif isinstance(rhs, sympy.Symbol) and free_unbacked_symbols(rhs):
                    self._set_replacement(rhs, self._find(lhs))
                else:
                    r = try_solve(expr, free[0], floordiv_inequality=False)
                    if r is not None and all((t.is_integer for t in sympy.preorder_traversal(r[1]))):
                        new_var = self._find(r[1])
                        ok = False
                        if self.is_unbacked_symint(free[0]):
                            ok = len(free_unbacked_symbols(new_var)) <= 1
                        else:
                            ok = len(free_unbacked_symbols(new_var)) == 0
                        if ok:
                            self._set_replacement(cast(sympy.Symbol, free[0]), new_var)
            except NotImplementedError:
                pass
        if expr.has(Mod):
            mod_expr = next(iter(expr.atoms(Mod)))
            try:
                r = try_solve(expr, mod_expr, floordiv_inequality=False)
                if r is not None and r[1] == 0:
                    self._add_divisible(mod_expr)
                    (p, q) = mod_expr.args
                    if isinstance(q, sympy.Number) and isinstance(p, sympy.Mul) and (len(p.args) == 2):
                        (c, i0) = p.args
                        if isinstance(c, sympy.Number) and isinstance(i0, sympy.Symbol) and self.is_unbacked_symint(i0):
                            d = q / sympy.gcd(q, c)
                            i1 = self.create_unbacked_symint().node.expr
                            self.var_to_range[i1] = SymPyValueRangeAnalysis.truediv(self.var_to_range[i0], ValueRanges.wrap(d))
                            self.runtime_var_to_range[i1] = SymPyValueRangeAnalysis.truediv(self.runtime_var_to_range[i0], ValueRanges.wrap(d))
                            self._set_replacement(i0, d * i1)
            except NotImplementedError:
                pass
        return

    def _default_value_range(self) -> ValueRanges:
        if False:
            print('Hello World!')
        lower = 2 if self.specialize_zero_one else 0
        return ValueRanges(lower, sys.maxsize - 1)

    def _default_unspecified_value_range(self) -> ValueRanges:
        if False:
            print('Hello World!')
        return ValueRanges(-sys.maxsize - 1, sys.maxsize)

    @_lru_cache
    def _simplify_floor_div(self, expr):
        if False:
            while True:
                i = 10
        floor_divs = tuple(expr.atoms(FloorDiv))
        for fd in reversed(floor_divs):
            (base, divisor) = fd.args
            mod_expr = Mod(base, divisor)
            eq_expr = sympy.Eq(mod_expr, 0)
            self.evaluate_expr(eq_expr)
        return self.simplify(expr)

    def _check_frozen(self, expr, concrete_val):
        if False:
            for i in range(10):
                print('nop')
        if self.frozen:
            self.counter['ignored_backward_guard'] += 1
            signpost_event('dynamic', 'evaluate_expr_frozen', {**self.co_fields, 'ignored_guard': f'{expr} == {concrete_val}', 'version': 2})
            log.warning('Ignored guard %s == %s, this could result in accuracy problems', expr, concrete_val)

    def _get_stack_summary(self):
        if False:
            print('Hello World!')
        fsummary = None
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_code.co_filename not in uninteresting_files():
                    fsummary = traceback.FrameSummary(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
                    break
                frame = frame.f_back
        finally:
            del frame
        maybe_user_loc = ''
        user_tb = TracingContext.extract_stack()
        if user_tb:
            maybe_user_loc = ' at ' + format_frame(user_tb[-1])
        return (fsummary, user_tb, maybe_user_loc)

    def _log_guard(self, prefix: str, g):
        if False:
            print('Hello World!')
        if self.log.isEnabledFor(logging.INFO):
            (fsummary, user_tb, maybe_user_loc) = self._get_stack_summary()
            is_debug = self.log.isEnabledFor(logging.DEBUG)
            maybe_extra_debug = ''
            if is_debug and user_tb:
                maybe_extra_debug = '\nUser Stack (most recent call last):\n' + '  (snipped, see stack below for prefix)\n' + ''.join(traceback.format_list(user_tb))
            self.log.info('%s %s [guard added]%s (%s)%s', prefix, g, maybe_user_loc, format_frame(fsummary), maybe_extra_debug, stack_info=is_debug)

    @lru_cache(256)
    @record_shapeenv_event(save_tracked_fakes=True)
    def evaluate_expr(self, orig_expr: 'sympy.Expr', hint=None, fx_node=None):
        if False:
            i = 10
            return i + 15
        '\n        Given an expression, evaluates it, adding guards if necessary\n        '
        if hint is None:
            concrete_val = self.size_hint(orig_expr)
        else:
            concrete_val = sympy.sympify(hint)
        node = None
        fresh = False
        if self._translation_validation_enabled and fx_node is not None and (not self._suppress_guards_tls()):
            if concrete_val is sympy.true:
                (node, fresh) = self.create_fx_call_function(torch._assert, (fx_node,))
            elif concrete_val is sympy.false:
                (neg, _) = self.create_fx_call_function(operator.not_, (fx_node,))
                (node, fresh) = self.create_fx_call_function(torch._assert, (neg,))
            else:
                (eql, _) = self.create_fx_call_function(operator.eq, (fx_node, concrete_val))
                (node, fresh) = self.create_fx_call_function(torch._assert, (eql,))
            assert node is not None
            if fresh:
                self.add_fx_node_metadata(node)
        guard = None
        tb = None
        try:
            if orig_expr.is_number:
                self.log.debug('eval %s [trivial]', orig_expr)
                if isinstance(hint, (int, bool)):
                    assert orig_expr == hint, f'{orig_expr} != {hint}'
                return orig_expr
            expr = orig_expr
            static_expr = self._maybe_evaluate_static(expr)
            if static_expr is not None:
                self.log.debug('eval %s == %s [statically known]', orig_expr, static_expr)
                if isinstance(hint, (int, bool)):
                    assert static_expr == hint, f'{static_expr} != {hint}'
                return static_expr
            if not expr.free_symbols <= self.var_to_val.keys():
                new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
                if not new_expr.free_symbols <= self.var_to_val.keys():
                    raise self._make_data_dependent_error(expr.xreplace(self.var_to_val), expr)
                expr = new_expr
            self._check_frozen(expr, concrete_val)
            if config.inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY and isinstance(hint, bool) and isinstance(expr, (sympy.Eq, sympy.Ne)):
                expr = sympy.Not(expr)
            if isinstance(expr, (sympy.Eq, sympy.Ne)):
                self._maybe_guard_eq(expr, bool(concrete_val))
            elif isinstance(concrete_val, sympy.Integer):
                self._maybe_guard_eq(sympy.Eq(expr, concrete_val), True)
            if concrete_val is sympy.true:
                g = expr
            elif concrete_val is sympy.false:
                g = sympy.Not(expr)
            else:
                g = sympy.Eq(expr, concrete_val)
            if not self._suppress_guards_tls():
                stack = CapturedTraceback.extract(skip=1)
                guard = ShapeGuard(g, stack)
                self.guards.append(guard)
        except Exception:
            if fresh:
                self.remove_fx_node(node)
            raise
        else:
            if not self._suppress_guards_tls():
                assert guard is not None
                self.refine_ranges(guard)
                self._log_guard('eval', g)
            else:
                self.log.debug('eval %s [guard suppressed]', g)
        return concrete_val

    def cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        for g in self.guards:
            g.stack.cleanup()
        for s in self.var_to_stack.values():
            s.cleanup()
        for ras in self.deferred_runtime_asserts.values():
            for ra in ras:
                ra.stack.cleanup()

    @record_shapeenv_event(save_tracked_fakes=True)
    def defer_runtime_assert(self, orig_expr: 'sympy.Expr', msg, fx_node=None):
        if False:
            i = 10
            return i + 15
        expr = orig_expr
        static_expr = self._maybe_evaluate_static(expr)
        if static_expr is not None:
            self.log.debug('runtime_assert %s == %s [statically known]', orig_expr, static_expr)
            return static_expr
        new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
        if new_expr.free_symbols <= self.var_to_val.keys():
            return self.evaluate_expr(new_expr, fx_node=fx_node)
        if self._translation_validation_enabled and fx_node is not None and (not self._suppress_guards_tls()):
            (node, fresh) = self.create_fx_call_function(torch._assert, (fx_node,))
            assert node is not None
            if fresh:
                self.add_fx_node_metadata(node)
        self._check_frozen(expr, sympy.true)
        if isinstance(expr, sympy.Eq):
            self._maybe_guard_eq(expr, True)
        if not self._suppress_guards_tls():
            stack = CapturedTraceback.extract(skip=1)
            ra = RuntimeAssert(expr, msg, stack)
            cands = sorted([s for s in expr.free_symbols if s.name.startswith('i')], key=lambda s: int(s.name[1:]))
            self.deferred_runtime_asserts.setdefault(cands[-1], []).append(ra)
            self.num_deferred_runtime_asserts += 1
            self._update_version_counter()
            self._log_guard('runtime_assert', expr)
        else:
            self.log.debug('runtime_assert %s [guard suppressed]', expr)
        return True

    def refine_ranges(self, guard: ShapeGuard) -> None:
        if False:
            while True:
                i = 10
        expr = self.simplify(guard.expr)
        for symbol in expr.free_symbols:
            assert isinstance(symbol, sympy.Symbol)
            if isinstance(self.var_to_val.get(symbol, None), SingletonInt):
                continue
            r = try_solve(expr, symbol)
            if r is None or not (symbol.is_integer and r[1].is_integer):
                continue
            (r_expr, rhs) = r
            vr = self.var_to_range[symbol]
            (lower, upper) = (vr.lower, vr.upper)
            rhs_vr = bound_sympy(rhs, self.var_to_range)
            _assert_bound_is_rational(rhs, rhs_vr)
            (lower_guard, upper_guard) = self.var_to_guards.get(symbol, (None, None))
            if lower < rhs_vr.lower and isinstance(r_expr, (sympy.Eq, sympy.Ge, sympy.Gt)):
                lower = rhs_vr.lower + int(isinstance(r_expr, sympy.Gt))
                lower_guard = guard
            if upper > rhs_vr.upper and isinstance(r_expr, (sympy.Eq, sympy.Le, sympy.Lt)):
                upper = rhs_vr.upper - int(isinstance(r_expr, sympy.Lt))
                upper_guard = guard
            if vr == ValueRanges(lower, upper):
                continue
            self.var_to_range[symbol] = ValueRanges(lower, upper)
            self.var_to_guards[symbol] = (lower_guard, upper_guard)
            self._maybe_evaluate_static.cache_clear()

def _is_int(expr):
    if False:
        for i in range(10):
            print('nop')
    return isinstance(expr, SymInt) and expr.node.expr.is_number

def _is_dim_dynamic(t, d):
    if False:
        return 10
    return hasattr(t, '_dynamo_dynamic_indices') and d in t._dynamo_dynamic_indices