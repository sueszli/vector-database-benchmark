import functools
import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import bound_sympy
from .utils import sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
log = logging.getLogger(__name__)

class SizeVarAllocator:

    def __init__(self, shape_env=None):
        if False:
            return 10
        super().__init__()
        if shape_env is None:
            shape_env = ShapeEnv()
        self.shape_env = shape_env
        self.var_to_val = self.shape_env.var_to_val
        self.replacements: Dict[sympy.Symbol, Expr] = self.shape_env.replacements
        self.precomputed_replacements: Dict[Expr, sympy.Symbol] = dict()
        self.inv_precomputed_replacements: Dict[sympy.Symbol, Expr] = dict()
        self.stride_vars = self.make_stride_vars_cache()
        self.simplify_with_ranges = self.make_simplify_with_ranges_cache()
        self._simplify_loops = self.make_simplify_loops_cache()

    def simplify(self, expr: Expr):
        if False:
            i = 10
            return i + 15
        return sympy.expand(expr).xreplace(self.replacements)

    def make_simplify_with_ranges_cache(self) -> Callable[[Expr, VarRanges], Expr]:
        if False:
            while True:
                i = 10
        '\n        self._simplify_with_ranges() can be expensive, cache its results\n        '
        cache: Dict[Tuple[Any, ...], Expr] = dict()
        replacement_count = len(self.replacements)

        def simplify_with_ranges(expr: Expr, var_ranges: VarRanges) -> Expr:
            if False:
                i = 10
                return i + 15
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                cache.clear()
                replacement_count = len(self.replacements)
            key = (expr, *var_ranges.items())
            result = cache.get(key, None)
            if result is None:
                result = self._simplify_with_ranges(expr, var_ranges)
                cache[key] = result
            return result
        return simplify_with_ranges

    def make_simplify_loops_cache(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        self._simplify_with_ranges() can be expensive, cache its results\n        '
        cache: Dict[Tuple[Any, ...], Any] = dict()
        replacement_count = len(self.replacements)

        def simplify_loops(index_vars, sizes, index_formulas):
            if False:
                i = 10
                return i + 15
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                cache.clear()
                replacement_count = len(self.replacements)
            key = (*index_vars, *sizes, *index_formulas)
            result = cache.get(key, None)
            if result is None:
                result = self._simplify_loops_impl(index_vars, sizes, index_formulas)
                cache[key] = result
            return result
        return simplify_loops

    def _simplify_with_ranges(self, expr: Expr, var_ranges: VarRanges) -> Expr:
        if False:
            while True:
                i = 10
        '\n        Simplify indexing expression with knowledge of the ranges of\n        iteration variables.\n        '
        expr = join_dimensions(self.simplify(expr))
        original_expr = expr

        def remove_zero_terms(base, divisor):
            if False:
                while True:
                    i = 10
            'Symbols smaller than the divisor are zero'
            for v in base.free_symbols:
                if v in var_ranges:
                    rest = sympy.Wild('_rest', exclude=[v])
                    m = base.match(v + rest)
                    if m and v not in m[rest].free_symbols:
                        gcd = sympy.gcd(m[rest], divisor)
                        if gcd == divisor:
                            if self.statically_known_leq(var_ranges[v], divisor):
                                base = m[rest]
            return base

        def visit_indexing_div(base, divisor):
            if False:
                i = 10
                return i + 15
            return FloorDiv(remove_zero_terms(base, divisor), divisor)

        def visit_modular_indexing(base, divisor, modulus):
            if False:
                print('Hello World!')
            base = remove_zero_terms(base, divisor)
            base_pos = True
            if isinstance(base, ModularIndexing):
                base_s = base.args[2] - 1
            elif not base.has(ModularIndexing):
                iter_ranges_zero = {k: 0 for (k, v) in var_ranges.items()}
                base_lowest = sympy_subs(base, iter_ranges_zero)
                if self.statically_known_leq(0, base_lowest):
                    base_pos = True
                else:
                    base_pos = False
                iter_ranges = {k: v - 1 for (k, v) in var_ranges.items()}
                base_s = sympy_subs(base, iter_ranges)
            else:
                base_s = base
            if self.statically_known_lt(base_s, modulus * divisor) and base_pos:
                return FloorDiv(base, divisor)
            return ModularIndexing(base, divisor, modulus)
        if expr.has(ModularIndexing):
            expr = expr.replace(ModularIndexing(sympy.Wild('base'), sympy.Wild('divisor'), sympy.Wild('modulus')), visit_modular_indexing)
        if expr.has(FloorDiv):
            expr = expr.replace(FloorDiv(sympy.Wild('base'), sympy.Wild('divisor')), visit_indexing_div)
        if expr != original_expr:
            return self._simplify_with_ranges(expr, var_ranges)
        return expr

    def _simplify_loops_impl(self, index_vars: List[sympy.Symbol], sizes, index_formulas):
        if False:
            return 10
        '\n        Try to remove as many axis from loop iterations as possible, by:\n            1) removing size==1 dimensions\n            2) fuse contiguous dimensions into a single loop\n            If channel_last = True, we will prevent the last dim fused with other dims\n        '
        sizes = list(map(self.simplify, sizes))
        strides = [self.stride_vars(x, index_vars) for x in index_formulas]
        assert len(sizes) == len(strides[0]), (len(sizes), len(strides[0]))
        for i in range(len(sizes)):
            if sizes[i] == 1:
                sizes[i] = None

        def can_merge_dims(a, b):
            if False:
                print('Hello World!')
            for k in range(len(strides)):
                if self.simplify(strides[k][a] * sizes[a]) == self.simplify(strides[k][b]):
                    va = index_vars[a]
                    vb = index_vars[b]
                    v = sympy_symbol('_merge_tester')
                    expr1 = sympy_subs(index_formulas[k], {va: v * sizes[a], vb: 0})
                    expr2 = sympy_subs(index_formulas[k], {va: 0, vb: v})
                    if self.simplify(expr1) == self.simplify(expr2):
                        continue
                return False
            return True
        changed = True
        while changed:
            changed = False
            for (i, j) in itertools.product(reversed(range(len(sizes))), reversed(range(len(sizes)))):
                if i == j or sizes[i] is None or sizes[j] is None:
                    continue
                if can_merge_dims(i, j):
                    changed = True
                    sizes[i] = sizes[i] * sizes[j]
                    sizes[j] = None

        def reindex(index):
            if False:
                for i in range(10):
                    print('nop')
            it = list(reversed(index))
            new_index = []
            for size in sizes:
                if size is None:
                    new_index.append(sympy.Integer(0))
                else:
                    new_index.append(it.pop())
            assert not it
            return new_index

        def prune(index):
            if False:
                for i in range(10):
                    print('nop')
            assert len(index) == len(sizes)
            return [i for (i, s) in zip(index, sizes) if s is not None]
        return ([x for x in sizes if x is not None], reindex, prune)

    def is_expr_static_and_true(self, expr: Union[Expr, int]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if expr in (True, False):
            return bool(expr)
        try:
            simplified = self.shape_env._maybe_evaluate_static(expr)
            if simplified is not None:
                return bool(simplified)
        except Exception:
            log.debug('Could not simplify %s', expr)
        return False

    def statically_known_equals(self, left: Expr, right: Expr) -> bool:
        if False:
            while True:
                i = 10
        '\n        Returns a bool indicating if it is sound to optimize as if left and right are equal.\n        '
        return self.is_expr_static_and_true(sympy.Eq(left, right))

    def statically_known_list_equals(self, left: List[Expr], right: List[Expr]) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns a bool indicating if it is sound to optimize as if left and right lists are equal.\n        '
        if len(left) != len(right):
            return False
        if all((self.statically_known_equals(l, r) for (l, r) in zip(left, right))):
            return True
        return False

    def statically_known_leq(self, left: Expr, right: Expr) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns a bool indicating if it is sound to optimize as if left is less than or equal to right.\n        '
        expr = left <= right
        return self.is_expr_static_and_true(expr)

    def statically_known_lt(self, left: Expr, right: Expr) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns a bool indicating if it is sound to optimize as if left is less than right.\n        '
        expr = left < right
        return self.is_expr_static_and_true(expr)

    def statically_known_multiple_of(self, numerator: Expr, denominator: Expr) -> bool:
        if False:
            return 10
        '\n        Return a bool indicating if it is sound to optimize for the numerator being a multiple of the denominator.\n        '
        expr = sympy.Eq(numerator % denominator, 0)
        return self.is_expr_static_and_true(expr)

    def guard_equals(self, left: Expr, right: Expr) -> Expr:
        if False:
            i = 10
            return i + 15
        if isinstance(left, Expr):
            left = sympy_subs(left, self.inv_precomputed_replacements)
        if isinstance(right, Expr):
            right = sympy_subs(right, self.inv_precomputed_replacements)
        assert self.shape_env.evaluate_expr(sympy.Eq(left, right))
        return left

    def guard_leq(self, left: Expr, right: Expr) -> None:
        if False:
            print('Hello World!')
        return self.guard_lt(left, right + 1)

    def guard_lt(self, left: Expr, right: Expr) -> None:
        if False:
            return 10
        assert self.shape_env.evaluate_expr(sympy.Lt(left, right))

    def evaluate_expr(self, left: Union[Expr, sympy.logic.boolalg.Boolean]) -> bool:
        if False:
            print('Hello World!')
        assert isinstance(left, (Expr, sympy.logic.boolalg.Boolean)), type(left)
        return self.shape_env.evaluate_expr(sympy.sympify(left))

    def evaluate_min(self, left: Expr, right: Expr) -> Expr:
        if False:
            print('Hello World!')
        'return the smaller of left and right, and guard on that choice'
        lv = self.size_hint(left)
        rv = self.size_hint(right)
        if lv == rv:
            return self.guard_equals(left, right)
        elif lv < rv:
            self.guard_lt(left, right)
            return left
        else:
            self.guard_lt(right, left)
            return right

    def evaluate_static_shape(self, left: Expr) -> int:
        if False:
            print('Hello World!')
        right = self.size_hint(left)
        self.guard_equals(left, sympy.Integer(right))
        return int(right)

    def evaluate_static_shapes(self, left: List[Expr]) -> List[int]:
        if False:
            while True:
                i = 10
        return [self.evaluate_static_shape(x) for x in left]

    def symbolic_hint(self, expr: Expr) -> Expr:
        if False:
            i = 10
            return i + 15
        if not isinstance(expr, Expr):
            assert isinstance(expr, int)
            return expr
        free_symbols = expr.free_symbols
        if not free_symbols:
            return int(expr)
        while any((s.name.startswith('ps') for s in free_symbols)):
            expr = sympy_subs(expr, self.inv_precomputed_replacements)
            free_symbols = expr.free_symbols
        return sympy_subs(expr, self.var_to_val)

    def size_hint(self, expr: Expr, *, fallback: Optional[int]=None) -> int:
        if False:
            print('Hello World!')
        out = self.symbolic_hint(expr)
        if not isinstance(out, (int, sympy.Integer)) and fallback is not None:
            sym_vrs = {s: self.shape_env.var_to_range.get(s, None) for s in expr.free_symbols}
            if all((vr is not None for vr in sym_vrs.values())):
                expr_vr = bound_sympy(expr, sym_vrs)
                lower = self.size_hint(expr_vr.lower)
                upper = self.size_hint(expr_vr.upper)
                fallback = min(max(fallback, lower), upper)
            return fallback
        try:
            return int(out)
        except Exception:
            log.debug('failed on: %s', out)
            raise

    def size_hints(self, exprs: List[Expr], *, fallback: Optional[int]=None) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        return tuple((self.size_hint(x, fallback=fallback) for x in exprs))

    def _lru_cache(self, fn, maxsize=None):
        if False:
            return 10
        '\n        Wrapper around functools.lru_cache that clears when replacements\n        has been invalidated.\n        '
        fn_cache = functools.lru_cache(maxsize)(fn)
        prior_len = len(self.replacements)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            nonlocal prior_len
            if prior_len != len(self.replacements):
                prior_len = len(self.replacements)
                fn_cache.cache_clear()
            return fn_cache(*args, **kwargs)
        return wrapper

    def make_stride_vars_cache(self):
        if False:
            return 10
        cache = self._lru_cache(self._stride_vars)

        def stride_vars(index: Expr, vars: List[sympy.Symbol], support_vars: Optional[List[sympy.Symbol]]=None) -> List[Expr]:
            if False:
                while True:
                    i = 10
            if not support_vars:
                support_vars = vars
            return cache(index, tuple(vars), tuple(support_vars))
        return stride_vars

    def _stride_vars(self, index: Expr, vars: List[sympy.Symbol], support_vars: List[sympy.Symbol]) -> List[Expr]:
        if False:
            return 10
        'Convert an indexing expression back into strides\n\n        NOTE: This is only valid if the index is a standard strided offset\n        calculation. e.g. 10 * ModularIndexing(i0 + 1, 1, 2) would give a\n        stride of -10 because the index wraps around after the first element\n\n        '
        strides = []
        index = self.simplify(index)
        index = index - sympy_subs(index, {v: sympy.Integer(0) for v in support_vars if v != 0})
        for i in range(len(vars)):
            index_dim = sympy_subs(index, {support_vars[j]: sympy.Integer(0) for j in range(len(support_vars)) if vars[i] != support_vars[j] and support_vars[j] != 0})
            v = vars[i]
            if v == 0:
                strides.append(sympy.Integer(0))
            else:
                strides.append(sympy_subs(index_dim, {v: sympy.Integer(1)}) - sympy_subs(index_dim, {v: sympy.Integer(0)}))
        return strides

    def offset_var(self, index: Expr, vars: List[sympy.Symbol]) -> Expr:
        if False:
            while True:
                i = 10
        'Extract offset part of an indexing expression'
        index = self.simplify(index)
        return sympy_subs(index, {v: sympy.Integer(0) for v in vars if v != 0})

    def stride_hints(self, index: Expr, vars: List[sympy.Symbol], support_vars: Optional[List[sympy.Symbol]]=None) -> List[int]:
        if False:
            print('Hello World!')
        for v in index.free_symbols:
            if v.name.startswith('indirect'):
                index = sympy_subs(index, {v: 0})
        result = []
        for s in self.stride_vars(index, vars, support_vars):
            try:
                result.append(self.size_hint(s))
            except TypeError:
                result.append(0)
        return result

    def stride_order(self, index: Expr, vars: List[sympy.Symbol]) -> List[int]:
        if False:
            print('Hello World!')
        strides = tuple(map(abs, self.stride_hints(index, vars)))
        order = list(range(len(strides)))
        order.sort(key=lambda x: (strides[x] == 0, strides[x]))
        return order

    def lookup_precomputed_size(self, expr: Expr) -> sympy.Symbol:
        if False:
            i = 10
            return i + 15
        if expr not in self.precomputed_replacements:
            sym = sympy_symbol(f'ps{len(self.precomputed_replacements)}')
            self.precomputed_replacements[expr] = sym
            self.inv_precomputed_replacements[sym] = expr
        return self.precomputed_replacements[expr]

    def free_symbols(self) -> Set[sympy.Symbol]:
        if False:
            return 10
        return set(self.var_to_val.keys()) - set(self.replacements.keys())

def join_dimensions(expr: Expr) -> Expr:
    if False:
        print('Hello World!')
    if not isinstance(expr, sympy.Add) or not expr.has(ModularIndexing):
        return expr
    return _join_dimensions_cached(expr)

@functools.lru_cache(256)
def _join_dimensions_cached(expr: Expr) -> Expr:
    if False:
        return 10
    '\n    ModularIndexing(i0, 1, 32) + 32 * ModularIndexing(i0, 32, 4)\n    becomes\n    ModularIndexing(i0, 1, 128)\n    ModularIndexing(i0, 1, 32) + 32 * FloorDiv(i0, 32)\n    becomes i0\n\n\n    This type of pattern can come from view operations\n    '
    assert isinstance(expr, sympy.Add)
    scale = sympy.Wild('scale', exclude=[0])
    base = sympy.Wild('base')
    divisor = sympy.Wild('divisor')
    mod1 = sympy.Wild('modulus')
    mod2 = sympy.Wild('modulus2')
    for term1 in expr.args:
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            for term2 in expr.args:
                m2 = term2.match(m1[scale] * m1[mod1] * ModularIndexing(m1[base], m1[divisor] * m1[mod1], mod2))
                if m2 and term1 != term2:
                    expr = join_dimensions(expr - term1 - term2 + m1[scale] * ModularIndexing(m1[base], m1[divisor], m1[mod1] * m2[mod2]))
                    return expr
    for term1 in expr.args:
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            for term2 in expr.args:
                m2 = term2.match(m1[scale] * m1[mod1] * FloorDiv(m1[base], m1[divisor] * m1[mod1]))
                if m2 is not None:
                    expr = join_dimensions(expr - term1 - term2 + m1[scale] * FloorDiv(m1[base], m1[divisor]))
                    return expr
    return expr

class SimplifyIndexing(V.WrapperHandler):
    """
    A wrapper around .virtualize.ops that uses var range information to
    simplify ModularIndexing/FloorDiv.
    """

    def __init__(self, inner, var_ranges: VarRanges):
        if False:
            while True:
                i = 10
        super().__init__(inner)
        self.name = 'SimplifyIndexing'
        self._simplify: Callable[[Expr], Expr] = lambda index: V.graph.sizevars.simplify_with_ranges(index, var_ranges)

    def load(self, name: str, index: sympy.Expr):
        if False:
            while True:
                i = 10
        return self._inner.load(name, self._simplify(index))

    def store(self, name, index, value, mode=None):
        if False:
            for i in range(10):
                print('nop')
        return self._inner.store(name, self._simplify(index), value, mode=mode)

    def store_reduction(self, name, index, value):
        if False:
            print('Hello World!')
        return self._inner.store_reduction(name, self._simplify(index), value)

    def index_expr(self, index, dtype):
        if False:
            i = 10
            return i + 15
        return self._inner.index_expr(self._simplify(index), dtype)