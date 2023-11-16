from __future__ import annotations
import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
from typing import Any, Counter, Dict, Iterable, List, Optional, Set, Tuple
import sympy
import torch
import torch._logging
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import MemoryDep, StarDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseScheduling
from ..triton_heuristics import AutotuneHint
from ..utils import do_bench, get_fused_kernel_name, get_kernel_metadata, green_text, is_welford_reduction, next_power_of_2, Placeholder, sympy_product, sympy_subs, sympy_symbol, unique, yellow_text
from ..virtualized import ops, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import CSEVariable, DeferredLine, free_symbol_startswith, IndentedBuffer, index_prevent_reordering, Kernel, OpOverrides, PythonPrinter, SizeArg, TensorArg
from .triton_utils import config_of, signature_of, signature_to_meta
log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, 'perf_hints')
schedule_log = torch._logging.getArtifactLogger(__name__, 'schedule')
fusion_log = torch._logging.getArtifactLogger(__name__, 'fusion')

class TritonPrinter(PythonPrinter):

    def _print_floor(self, expr):
        if False:
            for i in range(10):
                print('nop')
        assert len(expr.args) == 1
        return f'tl.math.floor({self.paren(self._print(expr.args[0]))})'

    def _helper_sqrt(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return f'tl.math.sqrt({self.paren(self._print(expr))}.to(tl.float32))'

    def _print_Where(self, expr):
        if False:
            i = 10
            return i + 15
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f'tl.where({c}, {p}, {q})'

    def _print_Min(self, expr):
        if False:
            i = 10
            return i + 15
        nargs = len(expr.args)
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        mid = len(expr.args) // 2
        a = self._print(sympy.Min(*expr.args[:mid]))
        b = self._print(sympy.Min(*expr.args[mid:]))
        return f'tl.math.min({a}, {b})'

    def _print_Max(self, expr):
        if False:
            for i in range(10):
                print('nop')
        nargs = len(expr.args)
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        mid = len(expr.args) // 2
        a = self._print(sympy.Max(*expr.args[:mid]))
        b = self._print(sympy.Max(*expr.args[mid:]))
        return f'tl.math.max({a}, {b})'

    def _print_Abs(self, expr):
        if False:
            return 10
        assert len(expr.args) == 1
        return f'tl.abs({self._print(expr.args[0])})'
texpr = TritonPrinter().doprint
pexpr = PythonPrinter().doprint

def triton_compute_type(dtype):
    if False:
        for i in range(10):
            print('nop')
    triton_type_name = str(dtype).split('.')[-1]
    if triton_type_name == 'bool':
        triton_type_name = 'int1'
    elif triton_type_name in ('float16', 'bfloat16'):
        triton_type_name = 'float32'
    elif triton_type_name == 'float8_e4m3fn':
        triton_type_name = 'float8e4nv'
    elif triton_type_name == 'float8_e5m2':
        triton_type_name = 'float8e5'
    return f'tl.{triton_type_name}'

def triton_acc_type(dtype):
    if False:
        print('Hello World!')
    if is_integer_dtype(dtype) and dtype.is_signed:
        nbits = 64 if dtype == torch.int64 else 32
        return f'tl.int{nbits}'
    return triton_compute_type(dtype)

def triton_constant(value):
    if False:
        print('Hello World!')
    if value == float('inf'):
        return 'float("inf")'
    elif value == float('-inf'):
        return 'float("-inf")'
    elif math.isnan(value):
        return 'float("nan")'
    return repr(value)

class TritonCSEVariable(CSEVariable):

    def __init__(self, name, bounds: ValueRanges):
        if False:
            print('Hello World!')
        super().__init__(name, bounds)
        self.mask_vars: Set[str] = set()

    def update_on_args(self, name, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        if name == 'where':
            return
        for arg in args:
            if isinstance(arg, TritonCSEVariable):
                self.mask_vars.update(arg.mask_vars)
            elif isinstance(arg, sympy.Symbol) and arg.name[0] in 'xyr':
                self.mask_vars.update({f'{arg.name[0]}mask'})

class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype: Optional[torch.dtype]=None):
        if False:
            return 10

        def _get_min_elements_per_thread(src_dtype: torch.dtype, dst_dtype: torch.dtype) -> int:
            if False:
                i = 10
                return i + 15
            if src_dtype == dst_dtype:
                return 0
            fp8_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2}
            assert not (src_dtype in fp8_dtypes and dst_dtype in fp8_dtypes and (src_dtype != dst_dtype)), 'Conversions between float8_e5m2 and float8_e4m3fn is not supported!'
            if src_dtype == torch.float8_e5m2 or dst_dtype == torch.float8_e5m2:
                return 4
            if src_dtype == torch.float8_e4m3fn or dst_dtype == torch.float8_e4m3fn:
                return 2
            return 0
        if src_dtype is not None:
            V.kernel.min_elem_per_thread = max(_get_min_elements_per_thread(src_dtype, dtype), V.kernel.min_elem_per_thread)
        if dtype == torch.bool:
            return f'({x} != 0)'
        elif dtype == torch.uint8:
            return f'{x}.to(tl.int8).to(tl.uint8)'
        return f'{x}.to({triton_compute_type(dtype)})'

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype):
        if False:
            print('Hello World!')
        return f'{x}.to({triton_compute_type(dtype)}, bitcast=True)'

    @classmethod
    def constant(cls, value, dtype):
        if False:
            print('Hello World!')
        if dtype == torch.uint8:
            tmp = cls.constant(value, torch.int16)
            return cls.to_dtype(tmp, dtype)
        type_ = torch._prims_common.dtype_to_type(dtype)
        triton_val = triton_constant(type_(value))
        triton_type = triton_compute_type(dtype)
        if triton_type == 'tl.float32':
            return triton_val
        ndim = V.kernel.triton_tensor_ndim()
        shape = [1] * ndim
        return f'tl.full({shape}, {triton_val}, {triton_type})'

    @staticmethod
    def abs(x):
        if False:
            return 10
        return f'tl.abs({x})'

    @staticmethod
    def libdevice_abs(x):
        if False:
            while True:
                i = 10
        return f'tl.math.abs({x})'

    @staticmethod
    def exp(x):
        if False:
            print('Hello World!')
        return f'tl.exp({x})'

    @staticmethod
    def libdevice_exp(x):
        if False:
            i = 10
            return i + 15
        return f'tl.math.exp({x})'

    @staticmethod
    def exp2(x):
        if False:
            i = 10
            return i + 15
        return f'tl.math.exp2({x})'

    @staticmethod
    def expm1(x):
        if False:
            return 10
        return f'tl.math.expm1({x})'

    @staticmethod
    def sqrt(x):
        if False:
            i = 10
            return i + 15
        return f'tl.sqrt({x})'

    @staticmethod
    def libdevice_sqrt(x):
        if False:
            return 10
        return f'tl.math.sqrt({x})'

    @staticmethod
    def relu(x):
        if False:
            return 10
        bug = config.triton.inject_relu_bug_TESTING_ONLY
        if bug == 'compile_error':
            return 'compile error!'
        elif bug == 'runtime_error':
            return f'triton_helpers.device_assert_then({x} == 0, "injected assert fail", {x})'
        elif bug == 'accuracy':
            return f'{x} + 1'
        elif bug is None:
            return ops.maximum('0', x)
        else:
            raise AssertionError(f'unrecognized config triton.inject_relu_bug_TESTING_ONLY = {bug!r}')

    @staticmethod
    def minimum(a, b):
        if False:
            i = 10
            return i + 15
        return f'triton_helpers.minimum({a}, {b})'

    @staticmethod
    def maximum(a, b):
        if False:
            i = 10
            return i + 15
        return f'triton_helpers.maximum({a}, {b})'

    @staticmethod
    def where(a, b, c):
        if False:
            return 10
        return f'tl.where({a}, {b}, {c})'

    @staticmethod
    def cos(x):
        if False:
            return 10
        return f'tl.cos({x})'

    @staticmethod
    def libdevice_cos(x):
        if False:
            for i in range(10):
                print('nop')
        return f'tl.math.cos({x})'

    @staticmethod
    def sin(x):
        if False:
            print('Hello World!')
        return f'tl.sin({x})'

    @staticmethod
    def libdevice_sin(x):
        if False:
            while True:
                i = 10
        return f'tl.math.sin({x})'

    @classmethod
    def index_expr(cls, expr, dtype):
        if False:
            i = 10
            return i + 15
        (index_str, mask_vars, mask, expand_str) = V.kernel.indexing(expr)
        var = V.kernel.cse.generate(V.kernel.compute, index_str)
        if dtype not in {torch.int32, torch.int64}:
            var = V.kernel.cse.generate(V.kernel.compute, cls.to_dtype(var, dtype))
        var.mask_vars = mask_vars
        return var

    @staticmethod
    def masked(mask, body, other):
        if False:
            return 10
        with V.kernel.mask_loads(mask) as new_mask:
            result = body()
        other = V.kernel.cse.generate(V.kernel.compute, f'tl.full({result}.shape, {triton_constant(other)}, {result}.dtype)')
        return ops.where(new_mask, result, other)

    @staticmethod
    def lgamma(x):
        if False:
            for i in range(10):
                print('nop')
        return f'tl.math.lgamma({x})'

    @staticmethod
    def erf(x):
        if False:
            for i in range(10):
                print('nop')
        return f'tl.math.erf({x})'

    @staticmethod
    def cosh(x):
        if False:
            print('Hello World!')
        return f'tl.math.cosh({x})'

    @staticmethod
    def sinh(x):
        if False:
            return 10
        return f'tl.math.sinh({x})'

    @staticmethod
    def acos(x):
        if False:
            return 10
        return f'tl.math.acos({x})'

    @staticmethod
    def acosh(x):
        if False:
            i = 10
            return i + 15
        return f'tl.math.acosh({x})'

    @staticmethod
    def asin(x):
        if False:
            i = 10
            return i + 15
        return f'tl.math.asin({x})'

    @staticmethod
    def asinh(x):
        if False:
            print('Hello World!')
        return f'tl.math.asinh({x})'

    @staticmethod
    def atan2(x, y):
        if False:
            while True:
                i = 10
        return f'tl.math.atan2({x}, {y})'

    @staticmethod
    def atan(x):
        if False:
            while True:
                i = 10
        return f'tl.math.atan({x})'

    @staticmethod
    def atanh(x):
        if False:
            print('Hello World!')
        return f'tl.math.atanh({x})'

    @staticmethod
    def copysign(x, y):
        if False:
            print('Hello World!')
        return f'tl.math.copysign({x}, {y})'

    @staticmethod
    def erfc(x):
        if False:
            print('Hello World!')
        return f'tl.math.erfc({x})'

    @staticmethod
    def erfinv(x):
        if False:
            print('Hello World!')
        return f'tl.math.erfinv({x})'

    @staticmethod
    def hypot(x, y):
        if False:
            return 10
        return f'tl.math.hypot({x}, {y})'

    @staticmethod
    def log10(x):
        if False:
            while True:
                i = 10
        return f'tl.math.log10({x})'

    @staticmethod
    def nextafter(x, y):
        if False:
            i = 10
            return i + 15
        return f'tl.math.nextafter({x}, {y})'

    @staticmethod
    def logical_and(a, b):
        if False:
            while True:
                i = 10
        return f'{a} & {b}'

    @staticmethod
    def logical_not(a):
        if False:
            return 10
        return f'{a} == 0'

    @staticmethod
    def logical_or(a, b):
        if False:
            while True:
                i = 10
        return f'{a} | {b}'

    @staticmethod
    def logical_xor(a, b):
        if False:
            return 10
        return f'({a} ^ {b})'

    @staticmethod
    def bitwise_and(a, b):
        if False:
            for i in range(10):
                print('nop')
        return f'{a} & {b}'

    @staticmethod
    def bitwise_not(a):
        if False:
            print('Hello World!')
        return f'~{a}'

    @staticmethod
    def bitwise_or(a, b):
        if False:
            while True:
                i = 10
        return f'{a} | {b}'

    @staticmethod
    def bitwise_xor(a, b):
        if False:
            return 10
        return f'{a} ^ {b}'

    @staticmethod
    def bitwise_left_shift(a, b):
        if False:
            print('Hello World!')
        return f'{a} << {b}'

    @staticmethod
    def bitwise_right_shift(a, b):
        if False:
            while True:
                i = 10
        return f'{a} >> {b}'

    @staticmethod
    def rand(seed, offset):
        if False:
            i = 10
            return i + 15
        offset = f'({offset}).to(tl.uint32)'
        return f'tl.rand({seed}, {offset})'

    @staticmethod
    def randn(seed, offset):
        if False:
            print('Hello World!')
        offset = f'({offset}).to(tl.uint32)'
        return f'tl.randn({seed}, {offset})'

    @staticmethod
    def randint64(seed, offset, low, high):
        if False:
            print('Hello World!')
        offset = f'({offset}).to(tl.uint32)'
        return f'triton_helpers.randint64({seed}, {offset}, {low}, {high})'

    @staticmethod
    def load_seed(name, offset):
        if False:
            i = 10
            return i + 15
        var = V.kernel.args.input(name)
        return f"tl.load({var} + {V.kernel.args.seed_offset('load_seed_offset', offset)})"

    @staticmethod
    def rsqrt(x):
        if False:
            return 10
        return f'tl.math.rsqrt({x})'

    @staticmethod
    def log1p(x):
        if False:
            while True:
                i = 10
        return f'tl.math.log1p({x})'

    @staticmethod
    def tan(x):
        if False:
            return 10
        return f'tl.math.tan({x})'

    @staticmethod
    def tanh(x):
        if False:
            while True:
                i = 10
        return f'tl.math.tanh({x})'

    @staticmethod
    def sigmoid(x):
        if False:
            for i in range(10):
                print('nop')
        return f'tl.sigmoid({x})'

    @staticmethod
    def libdevice_sigmoid(x):
        if False:
            print('Hello World!')
        return f'1/(1 + tl.math.exp(-({x})))'

    @staticmethod
    def signbit(x):
        if False:
            i = 10
            return i + 15
        return f'tl.math.signbit({x}) if ({x}).dtype is tl.float32 else {x} < 0'

    @staticmethod
    def fmod(a, b):
        if False:
            i = 10
            return i + 15
        return f'tl.math.fmod({a}, {b})'

    @staticmethod
    def pow(a, b):
        if False:
            for i in range(10):
                print('nop')
        return f'tl.math.pow({a}, {b})'

    @staticmethod
    def log(x):
        if False:
            i = 10
            return i + 15
        return f'tl.log({x})'

    @staticmethod
    def libdevice_log(x):
        if False:
            i = 10
            return i + 15
        return f'tl.math.log({x})'

    @staticmethod
    def isinf(x):
        if False:
            return 10
        return f'tl.math.isinf({x}).to(tl.int1)'

    @staticmethod
    def isnan(x):
        if False:
            print('Hello World!')
        return f'tl.math.isnan({x}).to(tl.int1)'

    @staticmethod
    def round(x):
        if False:
            i = 10
            return i + 15
        return f'tl.math.nearbyint({x})'

    @staticmethod
    def floor(x):
        if False:
            print('Hello World!')
        return f'tl.math.floor({x})'

    @staticmethod
    def floordiv(a, b):
        if False:
            for i in range(10):
                print('nop')
        quot = f'{a} // {b}'
        rem = f'{a} % {b}'
        return f'tl.where(({a} < 0) != ({b} < 0), tl.where({rem} != 0, {quot} - 1, {quot}), {quot})'

    @staticmethod
    def sign(x):
        if False:
            return 10

        def to_int(s):
            if False:
                return 10
            return f'{s}.to(tl.int8)'
        left = to_int(ops.lt('0', x))
        right = to_int(ops.lt(x, '0'))
        sub = ops.sub(left, right)
        return f'{sub}.to({x}.dtype)'

    @staticmethod
    def trunc(x):
        if False:
            i = 10
            return i + 15
        return f'tl.math.trunc({x})'

    @staticmethod
    def truncdiv(a, b):
        if False:
            i = 10
            return i + 15
        return f'{a} // {b}'

    @staticmethod
    def ceil(x):
        if False:
            return 10
        return f'tl.math.ceil({x})'

@dataclasses.dataclass
class IterationRanges:
    """
    Each range tree represents multiple sets of iteration indexing
    in a single tiled dimension in the output kernel.

    If you have two loops ranges one (4, 3, 2) and another (4, 6),
    then the range tree will be:
            4 (i0)
        3 (i1)  6 (i3)
        2 (i2)
    Where i0 is shared between both loops, but then the split into
    different indexing vars.  All loop ranges must iterate over
    the same number of elements.
    """

    def __init__(self, name: str, var_list: List[sympy.Symbol], var_ranges: Dict[sympy.Symbol, sympy.Expr], numel: sympy.Expr, prefix: str, *, kernel: TritonKernel, divisor=sympy.Integer(1), length=sympy.Integer(1)):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.name = name
        self.var_list = var_list
        self.var_ranges = var_ranges
        self.numel = numel
        self.prefix = prefix
        self.divisor = divisor
        self.length = length
        self.kernel = kernel

    def is_loop(self):
        if False:
            while True:
                i = 10
        return self.prefix == 'r' and (not self.kernel.persistent_reduction)

class IterationRangesRoot(IterationRanges):

    def __init__(self, name: str, numel: sympy.Expr, prefix: str, index: int, kernel: TritonKernel, pid_cache=None):
        if False:
            for i in range(10):
                print('nop')
        if pid_cache is None:
            pid_cache = {}
        super().__init__(name=name, var_list=[], var_ranges={}, numel=numel, prefix=prefix, kernel=kernel)
        self.index = index
        self.nodes: Dict[sympy.Expr, IterationRangesEntry] = {}
        self.pid_cache: Dict[str, str] = pid_cache

    def cache_clear(self):
        if False:
            return 10
        for node in self.nodes.values():
            node.cache_clear()

    def lookup(self, divisor, length):
        if False:
            return 10
        '\n        Lookup a given RangeTreeEntry, creating it if needed\n        '
        if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
            expr = FloorDiv(sympy_symbol(f'{self.prefix}index'), divisor)
        else:
            expr = ModularIndexing(sympy_symbol(f'{self.prefix}index'), divisor, length)
        if expr not in self.nodes:
            node = IterationRangesEntry(f'{self.prefix}{next(V.kernel.iter_vars_count)}', divisor, length, expr, self)
            V.kernel.range_tree_nodes[node.symbol()] = node
            self.var_list.append(node.symbol())
            self.var_ranges[node.symbol()] = length
            self.nodes[expr] = node
        return self.nodes[expr]

    def construct_entries(self, lengths: List[sympy.Expr]):
        if False:
            print('Hello World!')
        divisor = sympy.Integer(1)
        itervars = []
        for length in reversed(lengths):
            itervars.append(self.lookup(divisor, length))
            divisor = divisor * length
        return list(reversed(itervars))

    def construct(self, lengths: List[sympy.Expr]):
        if False:
            i = 10
            return i + 15
        return [e.symbol() for e in self.construct_entries(lengths)]

    def vars_and_sizes(self, index: sympy.Expr):
        if False:
            while True:
                i = 10
        'Figure out vars from this tree used in index'
        nodes = [V.kernel.range_tree_nodes.get(s) for s in index.free_symbols]
        nodes = [n for n in nodes if n and n.prefix == self.prefix]
        nodes.sort(key=lambda x: V.graph.sizevars.size_hint(x.divisor))
        divisor = sympy.Integer(1)
        index_vars = []
        sizes = []

        def add(node):
            if False:
                i = 10
                return i + 15
            nonlocal divisor
            index_vars.append(node.symbol())
            sizes.append(node.length)
            divisor = divisor * node.length
        for node in nodes:
            if not V.graph.sizevars.statically_known_equals(node.divisor, divisor):
                add(self.lookup(divisor, FloorDiv(node.divisor, divisor)))
                divisor = node.divisor
            add(node)
        if not V.graph.sizevars.statically_known_equals(self.numel, divisor):
            add(self.lookup(divisor, FloorDiv(self.numel, divisor)))
        return (list(reversed(index_vars)), list(reversed(sizes)))

    def ranges_code(self):
        if False:
            print('Hello World!')
        size = self.kernel.indexing_size_str(self.index, self.prefix)
        index_dtype = self.kernel.index_dtype
        convert = f'.to({index_dtype})' if index_dtype != 'tl.int32' else ''
        return f'tl.arange(0, {self.prefix.upper()}BLOCK){size}{convert}'

    def scalar_code(self, value):
        if False:
            while True:
                i = 10
        index_dtype = self.kernel.index_dtype
        ndim = self.kernel.triton_tensor_ndim()
        size = [1] * ndim
        return f'tl.full({size}, {value}, {index_dtype})'

    def get_pid(self):
        if False:
            while True:
                i = 10
        key = f'tl.program_id({self.index})'
        pid = self.pid_cache.get(key, key)
        if self.kernel.index_dtype != 'tl.int32':
            return f'{pid}.to({self.kernel.index_dtype})'
        return pid

    def codegen_header(self, code, no_x_dim=False):
        if False:
            i = 10
            return i + 15
        x = self.prefix
        if self.is_loop():
            code.writeline(f'{self.name} = {x}offset + {x}base')
        elif x == 'r' and self.kernel.persistent_reduction:
            code.writeline(f'{self.name} = {self.ranges_code()}')
        else:
            if not no_x_dim:
                line = f'{x}offset + {self.ranges_code()}'
            else:
                line = self.scalar_code(f'{x}offset')
            code.writelines([f'{x}offset = {self.get_pid()} * {x.upper()}BLOCK', f'{self.name} = {line}'])
        code.writeline(f'{x}mask = {self.name} < {x}numel')

class IterationRangesEntry(IterationRanges):

    def __init__(self, name: str, divisor: sympy.Expr, length: sympy.Expr, expr: sympy.Expr, parent: IterationRanges):
        if False:
            print('Hello World!')
        super().__init__(name=name, numel=parent.numel / length, var_list=parent.var_list, var_ranges=parent.var_ranges, prefix=parent.prefix, divisor=divisor, length=length, kernel=parent.kernel)
        self.parent = parent
        self.codegen = functools.lru_cache(None)(self._codegen)
        self.expr = expr

    def set_name(self, name):
        if False:
            while True:
                i = 10
        self.codegen = lambda : name
        self.codegen.cache_clear = lambda : None
        self.name = name

    def cache_clear(self):
        if False:
            while True:
                i = 10
        self.codegen.cache_clear()

    def writeline(self, line):
        if False:
            return 10
        if self.is_loop():
            V.kernel.indexing_code.writeline(line)
        else:
            V.kernel.body.writeline(line)

    def _codegen(self):
        if False:
            print('Hello World!')
        self.writeline(f'{self.name} = ' + texpr(V.kernel.rename_indexing(self.expr)))
        return self.name

    def precomputed_args(self):
        if False:
            i = 10
            return i + 15
        precomputed_args: List[sympy.Expr] = []
        if isinstance(self.expr, sympy.Symbol):
            return precomputed_args
        assert isinstance(self.expr, (FloorDiv, ModularIndexing)), type(self.expr)
        for arg in self.expr.args[1:]:
            if not isinstance(arg, (sympy.Integer, sympy.Symbol)):
                symbols = arg.free_symbols
                if len(symbols) > 0 and all((s.name.startswith('s') for s in symbols)):
                    precomputed_args.append(arg)
        return precomputed_args

    def symbol(self):
        if False:
            while True:
                i = 10
        return sympy_symbol(self.name)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.name)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return self.name == other.name

class TritonKernel(Kernel):
    overrides = TritonOverrides
    sexpr = pexpr

    def __init__(self, *groups, index_dtype, mutations=None, pid_cache=None, reduction_hint=ReductionHint.DEFAULT, min_elem_per_thread=0):
        if False:
            for i in range(10):
                print('nop')
        if pid_cache is None:
            pid_cache = {}
        super().__init__()
        self.numels = [V.graph.sizevars.simplify(s) for s in groups]
        self.mutations = mutations
        self.range_trees: List[IterationRangesRoot] = []
        self.range_tree_nodes = {}
        self.iter_vars_count = itertools.count()
        self.inside_reduction = self.numels[-1] != 1
        self.body = IndentedBuffer()
        self.indexing_code = IndentedBuffer()
        self.suffix: IndentedBuffer = IndentedBuffer()
        self.outside_loop_vars = set()
        self.reduction_hint = reduction_hint
        self.index_dtype = index_dtype
        self.min_elem_per_thread = min_elem_per_thread
        self.last_usage = set()
        self.persistent_reduction = self.should_use_persistent_reduction()
        self.no_x_dim = self.reduction_hint == ReductionHint.INNER and self.persistent_reduction and (len(self.numels) == 2) and (self.numels[-1] >= 256)
        self.initialize_range_tree(pid_cache)
        self.autotune_hints: Set[AutotuneHint] = set()

        @functools.lru_cache(None)
        def simplify_indexing(index: sympy.Expr):
            if False:
                print('Hello World!')
            index = V.graph.sizevars.simplify_with_ranges(index, self.var_ranges())
            for tree in self.range_trees:
                index = self.combine_contiguous_dims(index, tree)
            return index
        self.simplify_indexing = simplify_indexing

    def should_use_persistent_reduction(self):
        if False:
            print('Hello World!')
        '\n        Heuristic to set self.persistent_reduction and add guards\n        if needed.\n        '
        if not (self.inside_reduction and config.triton.persistent_reductions):
            return False
        threshold = {ReductionHint.INNER: 1024}.get(self.reduction_hint, 64)
        last_numel = self.numels[-1]
        if not isinstance(last_numel, (int, sympy.Integer)):
            return False
        hint = V.graph.sizevars.size_hint(last_numel)
        if hint > threshold:
            return False
        V.graph.sizevars.guard_leq(self.numels[-1], next_power_of_2(hint))
        return True

    def set_last_usage(self, nodes):
        if False:
            while True:
                i = 10
        if not self.inside_reduction or self.persistent_reduction:
            return
        self.last_usage = set(itertools.chain.from_iterable((n.last_usage for n in nodes if n is not EnableReduction)))

    def initialize_range_tree(self, pid_cache):
        if False:
            return 10
        names = list(reversed(['xindex', 'yindex', 'zindex'][:len(self.numels) - 1])) + ['rindex']
        for i in range(len(self.numels)):
            pid_idx = i if names[i][0] == 'r' else 'xyz'.find(names[i][0])
            self.range_trees.append(IterationRangesRoot(names[i], self.numels[i], names[i][0], pid_idx, self, pid_cache))
        for tree in self.range_trees:
            if not tree.is_loop():
                tree.codegen_header(self.body, self.no_x_dim)
        if self.inside_reduction and self.range_trees[-1].is_loop():
            self.body.writeline(f'rbase = {self.range_trees[-1].ranges_code()}')

    def disable_reduction(self):
        if False:
            for i in range(10):
                print('nop')

        @contextlib.contextmanager
        def ctx():
            if False:
                return 10
            if self.numels[-1] == 1:
                assert not self.inside_reduction
                yield
                return
            if not self.persistent_reduction:
                self.codegen_body()
            self.inside_reduction = False
            try:
                yield
                if not self.persistent_reduction:
                    self.codegen_body()
            finally:
                self.inside_reduction = True
        return ctx()

    def set_ranges(self, *lengths):
        if False:
            print('Hello World!')
        assert len(lengths) == len(self.range_trees)
        return [ranges.construct(length) for (length, ranges) in zip(lengths, self.range_trees)]

    @staticmethod
    def _split_iteration_ranges(groups: Iterable[sympy.Expr], lengths: List[List[sympy.Expr]]):
        if False:
            i = 10
            return i + 15
        sv = V.graph.sizevars
        new_ranges: List[List[sympy.Expr]] = [[] for _ in groups]
        remaining = [sv.simplify(g) for g in groups]
        var_count = itertools.count()

        def add_range(i, expr):
            if False:
                for i in range(10):
                    print('nop')
            expr = sv.simplify(expr)
            if not sv.statically_known_multiple_of(remaining[i], expr):
                raise CantSplit()
            remaining[i] = FloorDiv(remaining[i], expr)
            new_ranges[i].append(expr)
            return next(var_count)

        def make_combined(size, idx1, idx2):
            if False:
                i = 10
                return i + 15

            def getter(flat_vars):
                if False:
                    for i in range(10):
                        print('nop')
                return size * flat_vars[idx1] + flat_vars[idx2]
            return getter
        return_getters_groups = []
        current_group = 0
        for length_group in lengths:
            return_getters = []
            for size in length_group:
                if sv.statically_known_equals(size, 1):
                    return_getters.append(lambda _: sympy.Integer(0))
                    continue
                while current_group < len(remaining) and sv.size_hint(remaining[current_group]) == 1:
                    current_group += 1
                if sv.size_hint(size) > sv.size_hint(remaining[current_group]):
                    if not sv.statically_known_multiple_of(size, remaining[current_group]):
                        raise CantSplit()
                    size1 = remaining[current_group]
                    size2 = FloorDiv(size, remaining[current_group])
                    return_getters.append(make_combined(size2, add_range(current_group, size1), add_range(current_group + 1, size2)))
                else:
                    return_getters.append(operator.itemgetter(add_range(current_group, size)))
            return_getters_groups.append(return_getters)
        assert all((V.graph.sizevars.size_hint(s) == 1 for s in remaining)), f'failed to set ranges {remaining} {lengths}'
        return (new_ranges, return_getters_groups)

    @classmethod
    def is_compatible(cls, groups: Iterable[sympy.Expr], lengths: List[List[sympy.Expr]]):
        if False:
            for i in range(10):
                print('nop')
        try:
            cls._split_iteration_ranges(groups, lengths)
            return True
        except CantSplit:
            return False

    def split_and_set_ranges(self, lengths: List[List[sympy.Expr]]):
        if False:
            for i in range(10):
                print('nop')
        '\n        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).\n\n        To do this we need to split up the iteration space of i0 into something like:\n            for i1 in s0:\n              for i2 in s1:\n                i0 = i1*s1 + i2\n                ....\n\n        This function matches and resplits lengths to the groups of\n        this kernel to enable tiled + non-tiled fusions.\n        '
        groups = [rt.numel for rt in self.range_trees]
        if not self.inside_reduction:
            groups[-1] = sympy.Integer(1)
        if len(lengths) == len(self.range_trees) and all((V.graph.sizevars.simplify(sympy_product(x) - g) == 0 for (x, g) in zip(lengths, groups))):
            return self.set_ranges(*lengths)
        (new_ranges, return_getters_groups) = self._split_iteration_ranges(groups, lengths)
        itervars = list(itertools.chain(*self.set_ranges(*new_ranges)))
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    def is_indirect_indexing(self, index: sympy.Expr):
        if False:
            return 10
        return free_symbol_startswith(index, 'tmp')

    def is_broadcasted(self, index: sympy.Expr):
        if False:
            while True:
                i = 10
        if self.is_indirect_indexing(index):
            return False
        index_numels = [1] * len(self.numels)
        for symbol in index.free_symbols:
            if symbol not in self.range_tree_nodes:
                continue
            entry = self.range_tree_nodes[symbol]
            index_numels[entry.parent.index] *= entry.length
        simplify = V.graph.sizevars.simplify
        return any((simplify(idx_range) != simplify(iter_range) for (idx_range, iter_range) in zip(index_numels, self.numels)))

    def combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
        if False:
            print('Hello World!')
        '\n        More aggressive simplification to merge contiguous dims\n        '
        if isinstance(index, (sympy.Integer, sympy.Symbol)):
            return index
        (index_vars, sizes) = tree.vars_and_sizes(index)
        if len(sizes) <= 1:
            return index
        (new_sizes, reindex, prune) = V.graph.sizevars._simplify_loops(index_vars, sizes, index_prevent_reordering([index], index_vars, sizes))
        if new_sizes == sizes:
            return index
        new_index_vars = tree.construct(new_sizes)
        new_index = sympy_subs(index, dict(zip(index_vars, reindex(new_index_vars))))
        return new_index

    def index_to_str(self, index: sympy.Expr) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert an index expr to a string that can be used in triton code.\n        e.g. a sympy expression "s2" may actually appear as "ks1" in the triton kernel.\n\n        Index expressions often need to be passed in as arguments to the triton kernel.\n        Rename_indexing and codegen_indexing keep track of the needed indices and add\n        new parameters to the function signature.\n        '
        return texpr(self.rename_indexing(self.codegen_indexing(index)))

    def indexing(self, index: sympy.Expr, *, copy_shape=None, dense_indexing=False, override_mask=None):
        if False:
            return 10
        '\n        Compute the index and mask to pass to tl.load() or tl.store()\n        '
        index = self.simplify_indexing(index)
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
            index = index.subs(V.graph.sizevars.precomputed_replacements)
        if len(index.atoms(sympy.ceiling)):
            for a in index.atoms(sympy.ceiling):
                symbols = a.free_symbols
                if len(symbols) > 0 and all((s.name.startswith('s') or s.name.startswith('ps') for s in symbols)):
                    replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                    index = sympy_subs(index, replacements)
        index_vars = index.free_symbols
        index = self.simplify_indexing(index)
        index_str = self.index_to_str(index)
        mask_vars: Set[str] = set()
        for var in index_vars:
            assert isinstance(var, sympy.Symbol)
            if override_mask:
                pass
            elif var.name.startswith('tmp'):
                cse_var = self.cse.varname_map[var.name]
                mask_vars.update(cse_var.mask_vars)
            elif var.name.startswith(('s', 'ps')):
                pass
            else:
                assert var.name[0] in 'xyr', var.name
                mask_vars.add(f'{var.name[0]}mask')
        need_dense = (config.triton.dense_indexing or dense_indexing or self._load_mask is not None) and index != 0
        have_dense = True
        have_loop_vars = False
        dense_mask_vars = set()
        for tree in self.range_trees:
            if tree.prefix == 'r' and (not self.inside_reduction):
                continue
            if index_vars.intersection(tree.var_list):
                have_loop_vars = True
            else:
                have_dense = False
            dense_mask_vars.add(f'{tree.prefix}mask')
        expand_str = None
        if isinstance(index, sympy.Integer):
            expand_str = f'{copy_shape}.shape' if copy_shape else self.dense_size_str()
            index_str = f'tl.full({expand_str}, {index_str}, tl.int32)'
            return (index_str, set(), 'None', expand_str)
        if need_dense and (not have_dense):
            expand_str = f'{copy_shape}.shape' if copy_shape else self.dense_size_str()
            index_str = f'tl.broadcast_to({index_str}, {expand_str})'
            mask_vars = dense_mask_vars
        elif not have_loop_vars and copy_shape:
            index_str = f'tl.broadcast_to({index_str}, {copy_shape}.shape)'
            mask_vars = dense_mask_vars
        if override_mask:
            mask_vars = {override_mask}
        if self._load_mask:
            mask_vars.add(self._load_mask)
        self.filter_masks(mask_vars)
        mask_str = ' & '.join(sorted(map(str, mask_vars))) if mask_vars else 'None'
        return (index_str, mask_vars, mask_str, expand_str)

    def filter_masks(self, mask_vars):
        if False:
            while True:
                i = 10
        for tree in self.range_trees:
            if V.graph.sizevars.statically_known_equals(tree.numel, 1):
                mask_vars.discard(f'{tree.prefix}mask')
                continue
            if tree.prefix.upper() not in config.triton.max_block:
                continue
            max_block = config.triton.max_block[tree.prefix.upper()]
            if V.graph.sizevars.statically_known_multiple_of(tree.numel, max_block):
                mask_vars.discard(f'{tree.prefix}mask')

    def var_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        return dict(itertools.chain.from_iterable((tree.var_ranges.items() for tree in self.range_trees)))

    def codegen_indexing(self, expr: sympy.Expr):
        if False:
            for i in range(10):
                print('nop')
        expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
        for sym in sorted(expr.free_symbols, key=str):
            if sym in self.range_tree_nodes:
                replacements = {}
                for ps in self.range_tree_nodes[sym].precomputed_args():
                    replacements[ps] = V.graph.sizevars.lookup_precomputed_size(ps)
                if len(replacements) > 0:
                    self.range_tree_nodes[sym].expr = sympy_subs(self.range_tree_nodes[sym].expr, replacements)
                self.range_tree_nodes[sym].codegen()
        return expr

    @contextlib.contextmanager
    def mask_loads(self, mask):
        if False:
            while True:
                i = 10
        'Context manager to add an additional mask to tl.load/store'
        prior = self._load_mask
        if prior:
            mask = self.cse.generate(self.compute, f'{mask} & {prior}')
        self._load_mask = mask
        try:
            yield mask
        finally:
            self._load_mask = prior

    def generate_assert(self, check):
        if False:
            i = 10
            return i + 15
        return torch.version.hip is None and super().generate_assert(check)

    def load_mask(self, var):
        if False:
            i = 10
            return i + 15
        mask = ''
        mask_vars = set(var.mask_vars)
        if self._load_mask:
            mask_vars.add(self._load_mask)
        if mask_vars:
            mask = f'{next(iter(mask_vars))}' if len(mask_vars) == 1 else f"({' & '.join((str(v) for v in mask_vars))})"
        return mask

    @property
    def assert_function(self):
        if False:
            for i in range(10):
                print('nop')
        return 'tl.device_assert'

    def get_strides_of_load(self, index: sympy.Expr):
        if False:
            for i in range(10):
                print('nop')
        '\n        This gets the stride of the index for each of the tiling variables\n        (technically, it does it at index 0)\n\n        For example, if\n        xindex = x0 + 512*x1 + 1024*r0\n        x0 = (xindex//512)\n        x1 = (xindex % 512)\n        r0 = rindex // 1024\n\n        this function would return\n        {xindex: 512, rindex: 1024}\n        '
        index_to_tile_indexes = {k: v.expr for (k, v) in self.range_tree_nodes.items()}
        index_in_tile_vars = sympy_subs(index, index_to_tile_indexes)
        strides = {}
        for range_tree in self.range_trees:
            s = sympy_symbol(range_tree.name)
            strides[s] = sympy_subs(index_in_tile_vars, {s: 1}) - sympy_subs(index_in_tile_vars, {s: 0})
        return strides

    def load(self, name: str, index: sympy.Expr):
        if False:
            i = 10
            return i + 15
        var = self.args.input(name)
        indirect_indexing = self.is_indirect_indexing(index)
        original_index = index
        (index, mask_vars, mask, expand_str) = self.indexing(index)
        is_coalesced = any((i == 1 for i in self.get_strides_of_load(original_index).values()))
        if self.is_broadcasted(original_index):
            ep = ", eviction_policy='evict_last'"
        elif not is_coalesced:
            ep = ", eviction_policy='evict_last'"
        elif self.inside_reduction and (not self.persistent_reduction):
            if name in self.args.inplace_buffers:
                names = set(self.args.inplace_buffers[name].other_names)
            else:
                names = {name}
            last_use = len(names & self.last_usage) > 0
            evict_last = not last_use and ('rmask' in mask or indirect_indexing)
            if evict_last:
                ep = ", eviction_policy='evict_last'"
            else:
                ep = ", eviction_policy='evict_first'"
        else:
            ep = ''
        if ('tmp' in mask or 'rmask' in mask) and V.graph.get_dtype(name) != torch.bool:
            other = ', other=0.0'
        else:
            other = ''
        append_broadcast = None
        if V.graph.is_unspec_arg(name):
            line = var
        else:
            if isinstance(original_index, sympy.Integer):
                line = f'tl.load({var} + ({original_index}))'
                append_broadcast = expand_str
            else:
                line = f'tl.load({var} + ({index}), {mask}{ep}{other})'
            dtype = V.graph.get_dtype(name)
            if dtype in (torch.float16, torch.bfloat16):
                line += '.to(tl.float32)'
            if dtype == torch.bool and torch.version.hip is None:
                line += '.to(tl.int1)'
        if 'tmp' in mask:
            load_buffer = self.compute
        elif self.inside_reduction and (not self.persistent_reduction) and ('rmask' not in mask) and (not indirect_indexing):
            load_buffer = self.body
        else:
            load_buffer = self.loads
        result_var = self.cse.generate(load_buffer, line)
        assert isinstance(result_var, TritonCSEVariable)
        result_var.mask_vars = mask_vars
        if append_broadcast:
            line = f'tl.broadcast_to({result_var}, {append_broadcast})'
            result_var = self.cse.generate(load_buffer, line)
        if not self.inside_reduction or 'rmask' not in mask:
            self.outside_loop_vars.add(result_var)
        return result_var

    def store(self, name, index, value, mode=None):
        if False:
            return 10
        var = self.args.output(name)
        indirect_indexing = self.is_indirect_indexing(index)
        original_index = index
        (index, mask_vars, mask, expand_str) = self.indexing(index, dense_indexing=True)
        is_inplace = name in self.args.inplace_buffers
        is_broadcasted = self.is_broadcasted(original_index)
        if is_inplace and is_broadcasted:
            self.stores.writeline(DeferredLine(name, 'tl.debug_barrier()'))
        if mode is None:
            line = f'tl.store({var} + ({index}), {value}, {mask})'
        elif mode == 'atomic_add':
            line = f'tl.atomic_add({var} + ({index}), {value}, {mask})'
        else:
            raise NotImplementedError(f'store mode={mode}')
        self.stores.writeline(DeferredLine(name, line))
        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

    def bucketize(self, values: CSEVariable, offsets_name: str, offsets_size: sympy.Expr, indexing_dtype: torch.dtype, right: bool):
        if False:
            while True:
                i = 10
        '\n        See [Note: Inductor bucketize op]\n        '
        self.autotune_hints.add(AutotuneHint.ELEMENTS_PER_WARP_32)
        offsets_ptr = self.args.input(offsets_name)
        block_size = self.dense_size_str()
        offsets_size_str = self.index_to_str(offsets_size)
        if indexing_dtype == torch.int32:
            triton_dtype = 'tl.int32'
        elif indexing_dtype == torch.int64:
            triton_dtype = 'tl.int64'
        else:
            raise NotImplementedError('Bucketize only supports indexing with int32 and int64')
        result = self.cse.generate(self.compute, f'triton_helpers.bucketize_binary_search({values}, {offsets_ptr}, {triton_dtype}, {right}, {offsets_size_str}, {block_size})')
        return result

    def reduction_resize(self, value):
        if False:
            while True:
                i = 10
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f'triton_helpers.promote_to_tensor({value})'
        sizes = [':'] * ndims
        sizes[-1] = 'None'
        return f"{value}[{', '.join(sizes)}]"

    @staticmethod
    def _map_tuple_or_scalar(fn, value):
        if False:
            print('Hello World!')
        if isinstance(value, tuple):
            return tuple(map(fn, value))
        return fn(value)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if False:
            while True:
                i = 10
        assert self.inside_reduction
        masks = {f'{tree.prefix}mask' for tree in self.range_trees}
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
        reduction_range_prefix = self.range_trees[-1].prefix
        reduction_sizes = ['None' for _ in self.range_trees]
        reduction_sizes[-1] = ':'
        dense_size_str = self.dense_size_str()
        value = self._map_tuple_or_scalar(lambda v: self.cse.generate(self.compute, f'tl.broadcast_to({v}, {dense_size_str})'), value)

        def final_reduction(value):
            if False:
                print('Hello World!')
            use_helper = reduction_type in {'any', 'max', 'min', 'prod'}
            module = 'triton_helpers' if use_helper else 'tl'
            if reduction_type in {'max', 'min'}:
                return self.reduction_resize(f'{module}.{reduction_type}2({value}, {dim})')
            return self.reduction_resize(f'{module}.{reduction_type}({value}, {dim})')

        def final_argreduce(buffer, result_var, value, index):
            if False:
                i = 10
                return i + 15
            buffer.splice(f"                _, {result_var}_tmp = triton_helpers.{root_op}_with_index({value}, {index}, {dim})\n                {result_var} = {self.reduction_resize(f'{result_var}_tmp')}\n                ")
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]
        dim = len(self.range_trees) - 1 - int(bool(self.no_x_dim))
        acc_type = triton_acc_type(src_dtype)
        result_var: Any = self.cse.newvar()
        result_var.mask_vars = {var for var in masks if var[0] != 'r'}
        cond = ' & '.join(masks)
        if self.persistent_reduction:
            default = ir.Reduction.default_value(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(triton_constant, default)

            def _mask_value(value, default):
                if False:
                    return 10
                return self.cse.generate(self.compute, f'tl.where({cond}, {value}, {default})')
            if isinstance(value, tuple):
                masked_value = [_mask_value(v, d) for (v, d) in zip(value, default)]
            else:
                masked_value = _mask_value(value, default)
            if reduction_type in {'argmax', 'argmin'}:
                accumulator_index = self.cse.generate(self.compute, f'tl.broadcast_to({reduction_range_prefix}index, {masked_value}.shape)')
                root_op = {'argmax': 'max', 'argmin': 'min'}[reduction_type]
                final_argreduce(self.compute, result_var, masked_value, accumulator_index)
            elif reduction_type == 'welford_reduce':
                sum_ = ops.reduction(dtype, dtype, 'sum', value)
                self.inside_reduction = False
                rnumel = ops.index_expr(self.numels[-1], dtype)
                mean = ops.truediv(sum_, rnumel)
                self.inside_reduction = True
                dx = ops.sub(value, mean)
                dx2 = ops.mul(dx, dx)
                m2 = ops.reduction(dtype, dtype, 'sum', dx2)
                result_var = (mean, m2, rnumel)
            elif reduction_type == 'welford_combine':
                (mean, m2, weight) = masked_value
                welford = f'triton_helpers.welford({mean}, {m2}, {weight}, {dim})'
                (mean, m2, weight) = (self.cse.newvar() for _ in range(3))
                self.compute.writeline(f'{mean}, {m2}, {weight} = {welford}')
                result_var = tuple((self.cse.generate(self.compute, self.reduction_resize(var_name)) for var_name in (mean, m2, weight)))
            else:
                result_var = self.cse.generate(self.compute, final_reduction(masked_value))
        else:
            accumulator = f'_{result_var}'
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(triton_constant, default)
            if not isinstance(default, tuple):
                self.body.writeline(f'{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})')
            if reduction_type in {'argmax', 'argmin'}:
                accumulator_index = f'_{result_var}_index'
                long_max = torch.iinfo(torch.int64).max
                self.body.writeline(f'{accumulator_index} = tl.full({self.dense_size_str()}, {long_max}, tl.int64)')
                root_op = {'argmax': 'max', 'argmin': 'min'}[reduction_type]
                self.compute.splice(f'                {accumulator}_next, {accumulator_index}_next = triton_helpers.{root_op}imum_with_index(\n                    {accumulator}, {accumulator_index}, {value}, {reduction_range_prefix}index\n                )\n                {accumulator} = tl.where({cond}, {accumulator}_next, {accumulator})\n                {accumulator_index} = tl.where({cond}, {accumulator_index}_next, {accumulator_index})\n                ')
                final_argreduce(self.suffix, result_var, accumulator, accumulator_index)
            elif is_welford_reduction(reduction_type):
                accumulator = f'{result_var}_mean'
                accumulator_m2 = f'{result_var}_m2'
                accumulator_weight = f'{result_var}_weight'
                self.body.writeline(f'{accumulator} = tl.zeros({self.dense_size_str()}, {acc_type})')
                self.body.writeline(f'{accumulator_m2} = tl.zeros({self.dense_size_str()}, {acc_type})')
                self.body.writeline(f'{accumulator_weight} = tl.zeros({self.dense_size_str()}, {acc_type})')
                if reduction_type == 'welford_combine':
                    (mean, m2, weight) = value
                    self.compute.splice(f'                    {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_combine(\n                        {accumulator}, {accumulator_m2}, {accumulator_weight},\n                        {mean}, {m2}, {weight}\n                    )\n                    ')
                else:
                    assert reduction_type == 'welford_reduce'
                    self.compute.splice(f'                    {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_reduce(\n                        {value}, {accumulator}, {accumulator_m2}, {accumulator_weight},\n                    )\n                    ')
                self.compute.splice(f'                {accumulator} = tl.where({cond}, {accumulator}_next, {accumulator})\n                {accumulator_m2} = tl.where({cond}, {accumulator_m2}_next, {accumulator_m2})\n                {accumulator_weight} = tl.where({cond}, {accumulator_weight}_next, {accumulator_weight})\n                ')
                result_mean = result_var
                result_m2 = self.cse.newvar()
                result_weight = self.cse.newvar()
                self.suffix.splice(f"                {result_mean}_tmp, {result_m2}_tmp, {result_weight}_tmp = triton_helpers.welford(\n                    {accumulator}, {accumulator_m2}, {accumulator_weight}, {dim}\n                )\n                {result_mean} = {self.reduction_resize(f'{result_mean}_tmp')}\n                {result_m2} = {self.reduction_resize(f'{result_m2}_tmp')}\n                {result_weight} = {self.reduction_resize(f'{result_weight}_tmp')}\n                ")
                result_var = (result_mean, result_m2, result_weight)
            else:
                combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
                updated = combine_fn(accumulator, value)
                self.compute.writeline(f'{accumulator} = tl.where({cond}, {updated}, {accumulator})')
                if src_dtype == torch.bool:
                    accumulator = f'{accumulator}.to(tl.int8)'
                    result_type = triton_compute_type(dtype)
                    self.suffix.writeline(f'{result_var} = {final_reduction(accumulator)}.to({result_type})')
                else:
                    self.suffix.writeline(f'{result_var} = {final_reduction(accumulator)}')
        self.cse.reduction_cache[cache_key] = result_var
        if isinstance(result_var, tuple):
            self.outside_loop_vars |= set(result_var)
        else:
            self.outside_loop_vars.add(result_var)
        return result_var

    def store_reduction(self, name, index, value):
        if False:
            for i in range(10):
                print('nop')
        assert self.inside_reduction
        self.inside_reduction = False
        (index, mask_vars, mask, _) = self.indexing(index)
        assert 'rmask' not in index
        self.inside_reduction = True
        var = self.args.output(name)
        self.suffix.writeline(DeferredLine(name, f'tl.store({var} + ({index}), {value}, {mask})'))

    def codegen_body(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Concat output code from index_code, loads, compute, stores,\n        suffix into self.body.\n\n        For pointwise kernels, this is called just once at the end.\n\n        For reduction kernels, this generates a loop over the reduction\n        axis.\n        '
        if not (self.indexing_code or self.loads or self.stores or self.compute or self.suffix):
            return
        if self.inside_reduction and (not self.persistent_reduction):
            self.body.writeline('for roffset in range(0, rnumel, RBLOCK):')
            with self.body.indent():
                self.range_trees[-1].codegen_header(self.body)
                self.body.splice(self.indexing_code)
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)
            self.cse.invalidate(self.outside_loop_vars)
            self.range_trees[-1].cache_clear()
        else:
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.body.splice(self.suffix)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.suffix.clear()

    def codegen_kernel_benchmark(self):
        if False:
            i = 10
            return i + 15
        result = IndentedBuffer()
        (argdefs, call_args, signature) = self.args.python_argdefs()
        result.writelines(['', '', 'def get_args():'])
        with result.indent():
            name_cnt = itertools.count()
            var_names = []
            for (arg_name, arg_sig) in zip(call_args, signature):
                var_name = f'arg_{next(name_cnt)}'
                buf = V.graph.get_buffer(arg_name)
                if buf:
                    result.writeline(f"{var_name} = rand_strided({V.graph.sizevars.size_hints(buf.get_size())}, {V.graph.sizevars.size_hints(buf.get_stride())}, device='{buf.get_device()}', dtype={buf.get_dtype()})")
                elif arg_name in V.graph.constants:
                    const_tensor = V.graph.constants[arg_name]
                    result.writeline(f"{var_name} = rand_strided({V.graph.sizevars.size_hints(const_tensor.size())}, {V.graph.sizevars.size_hints(const_tensor.stride())}, device='{const_tensor.device}', dtype={const_tensor.dtype})")
                elif isinstance(arg_sig, SizeArg):
                    symval_hint = V.graph.sizevars.size_hint(arg_sig.expr)
                    if 'seed_offset' in arg_sig.name:
                        symval_hint = 0
                    result.writeline(f'{var_name} = {symval_hint}')
                else:
                    raise KeyError(f"Don't find the buffer or const tensor for {arg_name}")
                var_names.append(var_name)
            result.writeline(f"return {', '.join(var_names)},")
        result.writelines(['\n', '\n', 'def call(args):'])
        grid = []
        extra_args = []
        extra_args_str = None
        index = V.graph.scheduler.current_device.index
        with result.indent():
            result.writeline(f'with torch.cuda._DeviceGuard({index}):')
            with result.indent():
                result.writeline(f'torch.cuda.set_device({index})')
                for tree in self.range_trees:
                    expr = pexpr(V.graph.sizevars.size_hint(tree.numel))
                    if tree.prefix != 'r' or self.inside_reduction:
                        extra_args.append(expr)
                    if tree.prefix != 'r':
                        grid.append(expr)
                stream_name = f'stream{index}'
                result.writeline(f'{stream_name} = get_cuda_stream({index})')
                extra_args_str = ', '.join(map(str, extra_args)) + ', '
                result.writeline(f"{str(Placeholder.KERNEL_NAME)}.run(*args, {extra_args_str}grid=grid({', '.join(grid)}), stream={stream_name})")
        result.writelines(['\n', '\n', 'def benchmark_all_configs(args):'])
        with result.indent():
            result.writeline(f'with torch.cuda._DeviceGuard({index}):')
            with result.indent():
                result.writeline(f'torch.cuda.set_device({index})')
                result.writeline(f"return {str(Placeholder.KERNEL_NAME)}.benchmark_all_configs(*args, {extra_args_str}grid=grid({', '.join(grid)}))")
        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        result.writelines(['\n', '\n', "if __name__ == '__main__':"])
        with result.indent():
            result.writeline('from torch._inductor.utils import get_num_bytes')
            result.writeline('from triton.testing import do_bench')
            result.writeline('')
            result.writeline('args = get_args()')
            result.writeline('ms = do_bench(lambda: call(args), rep=40, fast_flush=True)')
            result.writeline(f'num_gb = get_num_bytes(*args, num_in_out_args={ninplace_args}) / 1e9')
            result.writeline('gb_per_s = num_gb / (ms / 1e3)')
            result.writeline('print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")')
        return result

    def codegen_kernel(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        from triton import next_power_of_2
        code = IndentedBuffer()
        size_hints = []
        for numel in self.numels:
            numel_hint = V.graph.sizevars.symbolic_hint(numel)
            if not isinstance(numel_hint, (int, sympy.Integer)):
                size_hint = 8192
            else:
                size_hint = next_power_of_2(int(numel_hint))
            size_hints.append(size_hint)
        if self.persistent_reduction:
            assert self.inside_reduction
            heuristics = 'persistent_reduction'
        elif self.inside_reduction:
            heuristics = 'reduction'
        else:
            size_hints.pop()
            heuristics = 'pointwise'
        if name is None:
            code.splice(f'\n                    import triton\n                    import triton.language as tl\n                    from torch._inductor.ir import ReductionHint\n                    from torch._inductor.ir import TileHint\n                    from torch._inductor.triton_heuristics import AutotuneHint, {heuristics}\n                    from torch._inductor.utils import instance_descriptor\n                    from torch._inductor import triton_helpers\n                ')
            if config.benchmark_kernel:
                code.splice('\n                        from torch._dynamo.testing import rand_strided\n                        from torch._C import _cuda_getCurrentRawStream as get_cuda_stream\n                        import torch\n                        from torch._inductor.triton_heuristics import grid\n                    ')
        (argdefs, _, signature) = self.args.python_argdefs()
        for (i, arg) in enumerate(signature):
            if isinstance(arg, SizeArg) and arg.expr in V.graph.sizevars.inv_precomputed_replacements:
                signature[i] = SizeArg(arg.name, V.graph.sizevars.inv_precomputed_replacements[arg.expr])
        mutated_args = set()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if mutation in self.args.inplace_buffers and mutation not in V.graph.removed_buffers and (mutation not in self.removed_buffers):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)
        triton_meta = {'signature': signature_to_meta(signature, size_dtype=self.index_dtype), 'device': V.graph.scheduler.current_device.index, 'device_type': V.graph.scheduler.current_device.type, 'constants': {}}
        inductor_meta = {'autotune_hints': set(self.autotune_hints), 'kernel_name': str(Placeholder.DESCRIPTIVE_NAME), 'mutated_arg_names': mutated_args}
        for tree in self.range_trees:
            if tree.prefix != 'r' or self.inside_reduction:
                sizearg = SizeArg(f'{tree.prefix}numel', tree.numel)
                signature.append(sizearg)
                triton_meta['signature'][len(argdefs)] = signature_of(sizearg, size_dtype=self.index_dtype)
                argdefs.append(f'{tree.prefix}numel')
        triton_meta['configs'] = [config_of(signature)]
        for tree in self.range_trees:
            if tree.prefix == 'r' and (not self.inside_reduction or self.persistent_reduction):
                continue
            if tree.prefix == 'x' and self.no_x_dim:
                continue
            argdefs.append(f'{tree.prefix.upper()}BLOCK : tl.constexpr')
        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f'\n                @{heuristics}(\n                    size_hints={size_hints!r},\n                    reduction_hint={reduction_hint},\n                    filename=__file__,\n                    triton_meta={triton_meta!r},\n                    inductor_meta={inductor_meta!r}\n                )\n                @triton.jit\n            '
        else:
            tile_hint = ''
            if len(size_hints) == 2:
                if len(signature) == 4:
                    tile_hint = 'tile_hint=TileHint.SQUARE,'
                else:
                    tile_hint = 'tile_hint=TileHint.DEFAULT,'
            heuristics_line = f'\n                @{heuristics}(\n                    size_hints={size_hints!r}, {tile_hint}\n                    filename=__file__,\n                    triton_meta={triton_meta!r},\n                    inductor_meta={inductor_meta!r},\n                    min_elem_per_thread={self.min_elem_per_thread}\n                )\n                @triton.jit\n            '
        code.splice(heuristics_line)
        code.writeline(f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):")
        self.codegen_body()
        with code.indent():
            self.codegen_static_numels(code)
            for (old, new) in self.args.aliases():
                code.writeline(f'{old} = {new}')
            code.splice(self.body)
        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark())
        return code.getvalue()

    def codegen_static_numels(self, code):
        if False:
            print('Hello World!')
        "\n        We get a small speedup from hard coding numels if they are static.\n\n        This code stomps on the passed-in values by writing an constant to the top of the kernel.\n\n        In a kernel like:\n        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):\n\n        We would add\n        xnumel = 4096\n        rnumel = 768\n\n        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes\n        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream\n        knows that its a static numel, as that you just plop a constant into the kernel.\n        "
        for tree in self.range_trees:
            if tree.prefix != 'r' or self.inside_reduction:
                simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    code.writeline(f'{tree.prefix}numel = {int(simplified_tree_numel)}')
            if tree.prefix == 'r' and self.persistent_reduction:
                simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    val = int(simplified_tree_numel)
                else:
                    continue
                val = next_power_of_2(val)
                code.writeline(f'RBLOCK: tl.constexpr = {val}')
            if tree.prefix == 'x' and self.no_x_dim:
                code.writeline('XBLOCK: tl.constexpr = 1')

    def triton_tensor_ndim(self):
        if False:
            return 10
        no_x_dim = int(bool(self.no_x_dim))
        no_r_dim = self.numels[-1] == 1
        return len(self.range_trees) - no_x_dim - no_r_dim

    def indexing_size_str(self, i=None, x=None):
        if False:
            while True:
                i = 10
        no_x_dim = int(bool(self.no_x_dim))
        sizes = ['None'] * self.triton_tensor_ndim()
        if i is not None:
            idx = i - no_x_dim
            sizes[idx] = ':'
        return f"[{', '.join(sizes)}]"

    def dense_size_str(self):
        if False:
            while True:
                i = 10
        sizes = []
        for tree in self.range_trees:
            if self.no_x_dim and tree.prefix == 'x':
                continue
            if tree.prefix != 'r' or self.inside_reduction:
                sizes.append(f'{tree.prefix.upper()}BLOCK')
            elif tree.prefix == 'r' and tree.numel != 1:
                sizes.append('1')
        if sizes[0:3] == ['ZBLOCK', 'YBLOCK', 'XBLOCK']:
            sizes[0:3] = reversed(sizes[0:3])
        if sizes[0:2] == ['YBLOCK', 'XBLOCK']:
            sizes[0:2] = reversed(sizes[0:2])
        return f"[{', '.join(sizes)}]"

    def call_kernel(self, name: str, node: Optional[IRNode]=None):
        if False:
            while True:
                i = 10
        wrapper = V.graph.wrapper_code
        (_, call_args, _) = self.args.python_argdefs()
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + '.item()'
        grid = []
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = tree.numel
            else:
                expr = wrapper.generate_numel_expr(name, tree)
            if tree.prefix != 'r' or self.inside_reduction:
                call_args.append(expr)
            if tree.prefix != 'r':
                grid.append(expr)
        grid = wrapper.generate_default_grid(name, grid)
        wrapper.generate_kernel_call(name, call_args, grid, V.graph.scheduler.current_device.index, cuda=True, triton=True)

    def codegen_nan_check(self):
        if False:
            while True:
                i = 10
        if not config.nan_asserts:
            return
        wrapper = V.graph.wrapper_code
        (_, call_args, arg_types) = self.args.python_argdefs()
        for (arg, arg_type) in zip(call_args, arg_types):
            if isinstance(arg_type, TensorArg):
                line = f'assert not {arg}.isnan().any().item()'
                wrapper.writeline(line)
                line = f'assert not {arg}.isinf().any().item()'
                wrapper.writeline(line)

    def warn_mix_layout(self, kernel_name):
        if False:
            print('Hello World!')
        '\n        Print message if the kernel have mixed layout inputs.\n        Only care about 4D tensor for now.\n        '
        if len(self.args.input_buffers) == 1 and len(self.args.output_buffers) == 1 and (len(self.args.inplace_buffers) == 0):
            return
        (argdefs, call_args, signature) = self.args.python_argdefs()
        uniform_stride_order = None
        for arg_name in call_args:
            buf = V.graph.get_buffer(arg_name)
            if buf and len(buf.layout.size) == 4:
                if len([x for x in buf.layout.size if x == 1]) == 3:
                    continue
                stride_order = ir.get_stride_order(buf.layout.stride)
                if uniform_stride_order is None:
                    uniform_stride_order = stride_order
                elif uniform_stride_order != stride_order:
                    msg = yellow_text(f'Expected stride order {uniform_stride_order}, but found stride order' + f' {stride_order} for kernel {kernel_name}')
                    log.warning(msg)
                    stride_order_list = [ir.get_stride_order(V.graph.get_buffer(name).layout.stride) if V.graph.get_buffer(name) else None for name in call_args]
                    size_list = [V.graph.get_buffer(name).layout.size if V.graph.get_buffer(name) else None for name in call_args]
                    source_list = ['GraphInput' if name in V.graph.graph_inputs else 'IntermediateBuffer' if name in V.graph.name_to_buffer else None for name in call_args]
                    msg = yellow_text(f'  param names {argdefs}\n  buf names {call_args}\n  strides {stride_order_list}' + f'\n  sizes {size_list}\n  sources {source_list}\n')
                    log.warning(msg)
                    return
        msg = green_text(f'All the inputs for the triton kernel {kernel_name} have uniform layout')
        log.warning(msg)

    def create_cse_var(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return TritonCSEVariable(*args, **kwargs)

class TritonScheduling(BaseScheduling):

    def __init__(self, scheduler):
        if False:
            return 10
        self.scheduler = scheduler

    def group_fn(self, sizes):
        if False:
            print('Hello World!')
        return tuple((V.graph.sizevars.simplify(sympy_product(s)) for s in sizes))

    def can_fuse(self, node1, node2):
        if False:
            for i in range(10):
                print('nop')
        '\n        Hook called by Scheduler to determine if the Triton backend\n        can fuse node1 and node2.  These nodes might already be\n        FusedSchedulerNodes.\n        '
        if isinstance(node1, scheduler.ForeachKernelSchedulerNode) or isinstance(node2, scheduler.ForeachKernelSchedulerNode):
            return scheduler.ForeachKernelSchedulerNode.can_fuse(node1, node2)
        (_, (numel1, rnumel1)) = node1.group
        (_, (numel2, rnumel2)) = node2.group
        if node1.is_reduction() and node2.is_reduction():
            reduction_can_fuse = numel1 == numel2 and rnumel1 == rnumel2
            if not reduction_can_fuse:
                fusion_log.debug('cannot fuse (triton:1): numel/rnumel mismatch (reduce) (%s, %s), (%s, %s)', numel1, numel2, rnumel1, rnumel2)
            return reduction_can_fuse
        if not node1.is_reduction() and (not node2.is_reduction()):
            if not (numel1 == numel2 and rnumel1 == rnumel2):
                fusion_log.debug('cannot fuse (triton:2): numel/rnumel mismatch (non-reduce) (%s, %s), (%s, %s)', numel1, numel2, rnumel1, rnumel2)
                return False
            if node1.is_template():
                is_triton_template = isinstance(node1.node, TritonTemplateBuffer)
                if not is_triton_template:
                    fusion_log.debug('cannot fuse (triton:3): is not TritonTemplateBuffer %s', node1)
                return is_triton_template
            tiling1 = self.select_tiling(node1.get_nodes(), numel1, rnumel1)
            tiling2 = self.select_tiling(node2.get_nodes(), numel1, rnumel1)
            tiling3 = self.select_tiling(node1.get_nodes() + node2.get_nodes(), numel1, rnumel1)
            if config.triton.tiling_prevents_pointwise_fusion:
                cond = True
                if len(tiling1) > 2:
                    if len(tiling2) > 2:
                        cond = tiling1 == tiling2 == tiling3
                    else:
                        cond = tiling1 == tiling3
                elif len(tiling2) > 2:
                    cond = tiling2 == tiling3
                if not cond:
                    fusion_log.debug('cannot fuse (triton:4): tiling mismatch (%s, %s, %s)', tiling1, tiling2, tiling3)
                    return cond
            return True
        if not node1.is_reduction() and node2.is_reduction():
            assert rnumel1 == 1 and rnumel2 != 1
            if numel1 == numel2 * rnumel2:
                if not all((TritonKernel.is_compatible((numel2, rnumel2), n.get_ranges()) for n in node1.get_nodes())):
                    fusion_log.debug('cannot fuse (triton:5): nodes numel/rnumel incompatibility')
                    return False
                if config.triton.tiling_prevents_reduction_fusion and (not node1.is_template()):
                    is_reduction_tiling_valid = self.select_tiling(node1.get_nodes(), numel1) in ((numel1, 1), (numel2, rnumel2, 1))
                    if not is_reduction_tiling_valid:
                        fusion_log.debug('cannot fuse (triton:6): invalid tiling for reduction')
                    return is_reduction_tiling_valid
                return True
            return numel1 == numel2
        assert node1.is_reduction() and (not node2.is_reduction())
        return self.can_fuse_horizontal(node2, node1)
    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def generate_node_schedule(self, nodes, numel, rnumel):
        if False:
            while True:
                i = 10
        node_schedule: List[Any] = []
        current_loop_writes: Set[str] = set()
        is_current_reductions = set()
        done = set()

        def fits_in_main_body(n):
            if False:
                for i in range(10):
                    print('nop')
            (_, (node_numel, node_rnumel)) = n.group
            return node_numel == numel and node_rnumel == rnumel or (node_numel == numel * rnumel and node_rnumel == 1)

        def fits_outside_reduction(n):
            if False:
                return 10
            (_, (node_numel, node_rnumel)) = n.group
            return node_numel == numel and node_rnumel == 1 and (rnumel != 1)

        @contextlib.contextmanager
        def end_current_reduction_loop():
            if False:
                return 10
            if current_loop_writes:
                for other_node in nodes[index + 1:]:
                    if node not in done and fits_in_main_body(other_node) and (not current_loop_writes & other_node.ancestors):
                        done.add(node)
                        current_loop_writes.add(node.get_name())
                        is_current_reductions.add(node.is_reduction())
                        node_schedule.append(node)
            if node_schedule and node_schedule[-1] is EnableReduction:
                node_schedule.pop()
            else:
                node_schedule.append(DisableReduction)
            yield
            node_schedule.append(EnableReduction)
            current_loop_writes.clear()
            is_current_reductions.clear()
        for (index, node) in enumerate(nodes):
            if node in done:
                continue
            done.add(node)

            def requires_closing_previous_reduction(node, node_schedule):
                if False:
                    print('Hello World!')
                if rnumel == 1:
                    return False
                if not current_loop_writes & node.ancestors:
                    return False
                assert node_schedule and (not isinstance(node_schedule[-1], (EnableReduction, DisableReduction)))
                return True in is_current_reductions
            if fits_in_main_body(node):
                if requires_closing_previous_reduction(node, node_schedule):
                    with end_current_reduction_loop():
                        pass
                current_loop_writes.add(node.get_name())
                is_current_reductions.add(node.is_reduction())
                node_schedule.append(node)
            elif fits_outside_reduction(node):
                with end_current_reduction_loop():
                    node_schedule.append(node)
            else:
                raise NotImplementedError(f'unexpected group: ({numel}, {rnumel}) != {node.group[1]}')
        return node_schedule

    def codegen_nodes(self, nodes):
        if False:
            while True:
                i = 10
        '\n        Given a set of pre-fused nodes, generate a Triton kernel.\n        '
        (_, (numel, rnumel)) = max(nodes, key=lambda x: int(x.is_reduction())).group
        node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
        schedule_log.debug('Schedule:\n %s', node_schedule)
        return self.codegen_node_schedule(node_schedule, numel, rnumel)

    @staticmethod
    def reduction_hint(node):
        if False:
            while True:
                i = 10
        assert node.is_reduction()
        if all((dep.is_contiguous() for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes))):
            return ReductionHint.INNER
        else:
            return node.node.data.reduction_hint

    @staticmethod
    def can_use_32bit_indexing(numel: sympy.Expr, buffers: Iterable[ir.Buffer]) -> bool:
        if False:
            i = 10
            return i + 15
        int_max = torch.iinfo(torch.int32).max
        size_hint = V.graph.sizevars.size_hint
        has_hint = V.graph.sizevars.shape_env.has_hint

        def within_32bit(e):
            if False:
                print('Hello World!')
            if V.graph.sizevars.is_expr_static_and_true(e <= int_max):
                return True
            return has_hint(e) and size_hint(e) <= int_max
        if not within_32bit(numel):
            return False
        buf_sizes = [buf.get_layout().storage_size() for buf in buffers if not isinstance(buf.get_layout(), ir.MultiOutputLayout)]
        if not all((within_32bit(size) for size in buf_sizes)):
            return False
        V.graph.sizevars.guard_leq(numel, int_max)
        for size in buf_sizes:
            V.graph.sizevars.guard_leq(size, int_max)
        return True

    @staticmethod
    def select_index_dtype(node_schedule, numel, reduction_numel):
        if False:
            for i in range(10):
                print('nop')
        buffer_names = set()
        for node in node_schedule:
            if not isinstance(node, scheduler.BaseSchedulerNode):
                continue
            buffer_names.update(node.get_names())
            buffer_names.update(node.used_buffer_names())

        def _get_buffer(name: str) -> ir.Buffer:
            if False:
                for i in range(10):
                    print('nop')
            if name in V.graph.name_to_buffer:
                return V.graph.name_to_buffer[name]
            elif name in V.graph.graph_inputs:
                return V.graph.graph_inputs[name]
            elif name in V.graph.constants:
                data = V.graph.constants[name]
                return ir.ConstantBuffer(name, ir.FixedLayout(data.device, data.dtype, *V.graph.static_sizes_strides(data)))
            raise RuntimeError(f'Failed to find buffer matching name {name}')
        buffers = [_get_buffer(name) for name in buffer_names]
        total_numel = numel * reduction_numel
        if TritonScheduling.can_use_32bit_indexing(total_numel, buffers):
            return 'tl.int32'
        return 'tl.int64'

    def get_kernel_args(self, node_schedule, numel, reduction_numel):
        if False:
            i = 10
            return i + 15
        reductions = list(filter(lambda n: n not in (EnableReduction, DisableReduction) and n.is_reduction(), node_schedule))
        if len(reductions) > 0:
            hints = [self.reduction_hint(n) for n in reductions]
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT
        else:
            reduction_hint_val = ReductionHint.DEFAULT
        mutations = set()
        for node in node_schedule:
            if hasattr(node, 'get_mutations'):
                mutations.update(node.get_mutations())
        index_dtype = self.select_index_dtype(node_schedule, numel, reduction_numel)
        return (reduction_hint_val, mutations, index_dtype)

    def codegen_comment(self, node_schedule):
        if False:
            print('Hello World!')
        wrapper = V.graph.wrapper_code
        (origins, detailed_origins) = get_kernel_metadata(node_schedule, wrapper)
        if origins:
            wrapper.writeline(origins)
        if config.debug_fusion:
            from torch._inductor.scheduler import BaseSchedulerNode, ForeachKernelSchedulerNode
            if not any((isinstance(n, ForeachKernelSchedulerNode) for n in node_schedule)):
                node_names = [n.get_name() for n in node_schedule if isinstance(n, BaseSchedulerNode)]
                wrapper.writeline(f"{wrapper.comment} Fused node name list: {', '.join(node_names)}")

    def codegen_node_schedule(self, node_schedule, numel, reduction_numel):
        if False:
            i = 10
            return i + 15
        tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
        (reduction_hint_val, mutations, index_dtype) = self.get_kernel_args(node_schedule, numel, reduction_numel)
        kernel = TritonKernel(*tiled_groups, reduction_hint=reduction_hint_val, mutations=mutations, index_dtype=index_dtype)
        self.codegen_node_schedule_with_kernel(node_schedule, kernel)
        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()
            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.mark_run()
        kernel_name = self.define_kernel(src_code, node_schedule)
        log.debug('Generating kernel code with kernel_name: %s', kernel_name)
        self.codegen_comment(node_schedule)
        kernel.call_kernel(kernel_name)
        kernel.codegen_nan_check()
        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        if config.warn_mix_layout:
            kernel.warn_mix_layout(kernel_name)
        if V.graph.wrapper_code.supports_intermediate_hooks and config.generate_intermediate_hooks:
            live_outs = kernel.args.live_output_buffers()
            for node in node_schedule:
                if not isinstance(node, scheduler.BaseSchedulerNode):
                    continue
                name = node.get_name()
                if name not in live_outs:
                    continue
                origin_node = node.node.get_origin_node()
                if origin_node is not None:
                    counters['inductor']['intermediate_hooks'] += 1
                    V.graph.wrapper_code.writeline(f'run_intermediate_hooks({origin_node.name!r}, {name})')
        self.scheduler.free_buffers()

    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
        if False:
            i = 10
            return i + 15

        def current_reduction_nodes(nodes):
            if False:
                return 10
            return itertools.takewhile(lambda n: n is not DisableReduction, nodes)
        with kernel:
            stack = contextlib.ExitStack()
            kernel.set_last_usage(current_reduction_nodes(node_schedule))
            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.decide_inplace_update()
            for (i, node) in enumerate(node_schedule):
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                    kernel.set_last_usage(current_reduction_nodes(node_schedule[i:]))
                else:
                    indexing_dtype_strength_reduction(node._body)
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    node.codegen(index_vars)

    def define_kernel(self, src_code, node_schedule):
        if False:
            i = 10
            return i + 15
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = get_fused_kernel_name(node_schedule, config.triton.descriptive_names) if config.triton.descriptive_names else ''
            kernel_category = get_kernel_category_by_source_code(src_code)[:3]
            kernel_name = '_'.join(['triton', kernel_category, fused_name, wrapper.next_kernel_suffix()])
            wrapper.src_to_kernel[src_code] = kernel_name
            subs_name = kernel_name if config.triton.unique_kernel_names else 'triton_'
            src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
            src_code = src_code.replace(str(Placeholder.KERNEL_NAME), subs_name)
            src_code = src_code.replace('#pragma CMT', '#')
            (basename, _, kernel_path) = get_path(code_hash(src_code), 'py')
            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.triton({subs_name!r}, '''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline("''')")
            metadata_comment = f'# kernel path: {kernel_path}'
            (origins, detailed_origins) = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += '\n' + origins + '\n' + detailed_origins
            wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)
        return kernel_name

    def codegen_template(self, template_node, epilogue_nodes):
        if False:
            for i in range(10):
                print('nop')
        '\n        Codegen a triton template\n        '
        (_, (numel, rnumel)) = template_node.group
        assert rnumel == 1
        (kernel, render) = template_node.node.make_kernel_render(template_node.node)
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()
            partial_code = render()
            for node in epilogue_nodes:
                node.codegen(kernel.split_and_set_ranges(node.get_ranges()))
        with V.set_kernel_handler(kernel):
            src_code = partial_code if isinstance(partial_code, str) else partial_code.finalize()
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code, node_schedule)
        self.codegen_comment(node_schedule)
        kernel.call_kernel(kernel_name, template_node.node)
        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        self.scheduler.free_buffers()

    def codegen_sync(self):
        if False:
            print('Hello World!')
        V.graph.wrapper_code.writeline('torch.cuda.synchronize()')

    def codegen_foreach(self, foreach_node):
        if False:
            for i in range(10):
                print('nop')
        from .triton_foreach import ForeachKernel
        for partitions_with_metadata in ForeachKernel.horizontal_partition(foreach_node.get_subkernel_nodes(), self):
            kernel = ForeachKernel()
            for (nodes, tiled_groups, numel, rnumel) in partitions_with_metadata:
                node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
                (reduction_hint_val, mutations, index_dtype) = self.get_kernel_args(node_schedule, numel, rnumel)
                subkernel = kernel.create_sub_kernel(*tiled_groups, reduction_hint=reduction_hint_val, mutations=mutations, index_dtype=index_dtype)
                self.codegen_node_schedule_with_kernel(node_schedule, subkernel)
                with V.set_kernel_handler(subkernel):
                    for node in node_schedule:
                        if node not in (EnableReduction, DisableReduction):
                            node.mark_run()
                V.graph.removed_buffers |= subkernel.removed_buffers
                V.graph.inplaced_to_remove |= subkernel.inplaced_to_remove
            src_code = kernel.codegen_kernel()
            kernel_name = self.define_kernel(src_code, [foreach_node])
            self.codegen_comment([foreach_node])
            kernel.call_kernel(V.graph.wrapper_code, kernel_name)
        self.scheduler.free_buffers()

    @staticmethod
    @functools.lru_cache(32)
    def candidate_tilings(node):
        if False:
            for i in range(10):
                print('nop')
        (ranges, reduction_ranges) = node.get_ranges()
        if len(ranges) <= 1:
            return ()
        rw = node.pointwise_read_writes()
        assert len(rw.range_vars) == len(ranges)
        dep_sources = [rw.reads, rw.writes]
        assert all((isinstance(dep, (MemoryDep, StarDep)) for dep in itertools.chain(*dep_sources)))
        deps = [dep for dep in itertools.chain(*dep_sources) if dep.name not in V.graph.removed_buffers and isinstance(dep, MemoryDep)]
        write_names = {dep.name for dep in rw.writes}
        tilings: List[CandidateTiling] = []
        for dep in deps:
            strides = V.graph.sizevars.stride_hints(dep.index, rw.range_vars)
            assert len(strides) == len(ranges)
            try:
                split = strides.index(1) + 1
                if split == len(ranges):
                    continue
                if all((s == 0 for s in strides[split:])):
                    continue
            except ValueError:
                continue
            tiled_groups = (V.graph.sizevars.simplify(sympy_product(ranges[:split])), V.graph.sizevars.simplify(sympy_product(ranges[split:])))
            score = V.graph.sizevars.size_hint(sympy_product((size for (size, stride) in zip(ranges, strides) if stride != 0)))
            if dep.name in write_names:
                score *= 2
            if CandidateTiling.is_good_size(tiled_groups[0]):
                score *= 2
            if CandidateTiling.is_good_size(tiled_groups[1]):
                score *= 2
            if V.graph.sizevars.size_hint(score - sympy_product(itertools.chain(ranges, reduction_ranges))) >= 0:
                tilings.append(CandidateTiling(tiled_groups, score, dep.name))
        return tilings

    @classmethod
    def select_tiling(cls, node_schedule, numel, reduction_numel=sympy.Integer(1)):
        if False:
            i = 10
            return i + 15
        '\n        Heuristics to decide how to tile kernels.\n        Currently, we tile based on stride-1 dimensions.\n\n        Returns:\n            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`\n\n        '
        if reduction_numel != 1 or config.triton.max_tiles <= 1:
            if perf_hint_log.level <= logging.WARNING:
                for node in EnableReduction.filter(node_schedule):
                    if len(cls.candidate_tilings(node)) > 0:
                        perf_hint_log.info('reduction over non-contiguous dims')
                        break
            return (numel, reduction_numel)
        seen_names = set()
        candidate_tiles: Counter[Any] = collections.Counter()
        for node in EnableReduction.filter(node_schedule):
            for tiling in cls.candidate_tilings(node):
                if tiling.name in seen_names:
                    continue
                seen_names.add(tiling.name)
                candidate_tiles[tiling.tiling] += tiling.score
        ranked_tilings = [tiling for (tiling, score) in candidate_tiles.most_common()]
        if config.triton.max_tiles >= 3:
            for i in range(1, len(ranked_tilings)):
                (a0, a1) = ranked_tilings[0]
                (b0, b1) = ranked_tilings[i]
                if V.graph.sizevars.size_hint(a1 - b1) == 0:
                    continue
                if V.graph.sizevars.size_hint(a1 - b1) < 0:
                    (a0, a1) = ranked_tilings[i]
                    (b0, b1) = ranked_tilings[0]
                assert V.graph.sizevars.size_hint(a1 - b1) > 0
                if V.graph.sizevars.statically_known_multiple_of(a1, b1):
                    tiling = (a0, FloorDiv(a1, b1), b1)
                    ranked_tilings = [tiling] + ranked_tilings
                    break
        if len(ranked_tilings) > 1:
            perf_hint_log.info('possibly bad tiling: %s', ranked_tilings)
        for tiled_groups in ranked_tilings:
            new_groups = (*tiled_groups, reduction_numel)
            if all((TritonKernel.is_compatible(new_groups, node.get_ranges()) for node in node_schedule if isinstance(node, scheduler.SchedulerNode))):
                return new_groups
        return (numel, reduction_numel)

    def flush(self):
        if False:
            print('Hello World!')
        pass

    def benchmark_fused_nodes(self, nodes):
        if False:
            for i in range(10):
                print('nop')
        (_, (numel, rnumel)) = max(nodes, key=lambda x: int(x.is_reduction())).group
        node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
        tiled_groups = self.select_tiling(node_schedule, numel, rnumel)
        (reduction_hint_val, mutations, index_dtype) = self.get_kernel_args(node_schedule, numel, rnumel)
        kernel = TritonKernel(*tiled_groups, reduction_hint=reduction_hint_val, mutations=mutations, index_dtype=index_dtype)
        for n in nodes:
            n.last_usage = set()
        self.codegen_node_schedule_with_kernel(node_schedule, kernel)
        with config.patch('benchmark_kernel', True), V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), 'triton_')
        mod = PyCodeCache.load(src_code)

        def cache_file_path():
            if False:
                i = 10
                return i + 15
            return os.path.splitext(mod.__file__)[0] + '.kernel_perf'

        def load_cache():
            if False:
                print('Hello World!')
            path = cache_file_path()
            if os.path.exists(path):
                with open(path) as fd:
                    return float(fd.read())
            return None

        def store_cache():
            if False:
                return 10
            path = cache_file_path()
            with open(path, 'w') as fd:
                fd.write(str(ms))
        log.debug('kernel src code for %s written to: %s', {n.get_name() for n in nodes}, mod.__file__)
        ms = load_cache()
        if ms is not None:
            return (ms, mod.__file__)
        args = mod.get_args()
        call = mod.call
        wrapped_jit_function = mod.triton_
        call(wrapped_jit_function.clone_args(*args)[0])
        launchers = wrapped_jit_function.launchers
        assert len(launchers) == 1
        if launchers[0].n_spills > 0:
            ms = float('inf')
        else:
            ms = do_bench(lambda : call(wrapped_jit_function.clone_args(*args)[0]))
        log.debug('The fused kernel for %s took %.3f ms to run', {n.get_name() for n in nodes}, ms)
        store_cache()
        return (ms, mod.__file__)

@dataclasses.dataclass
class CandidateTiling:
    tiling: Tuple[sympy.Expr, sympy.Expr]
    score: int
    name: Optional[str] = None

    @staticmethod
    def is_good_size(s):
        if False:
            i = 10
            return i + 15
        'Somewhat arbitrary heuristic used to boost scores for some sizes'
        s = V.graph.sizevars.size_hint(s)
        return s >= 32 and s % 32 == 0

class DisableReduction:
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """

class EnableReduction:
    """
    Marker to end a DisableReduction block.
    """

    @staticmethod
    def filter(node_schedule):
        if False:
            return 10
        '\n        Get the nodes from node_schedule skipping those in a\n        DisableReduction block.\n        '
        disabled = False
        for node in node_schedule:
            if node in (EnableReduction, DisableReduction):
                disabled = node is DisableReduction
            elif disabled:
                pass
            else:
                yield node

class CantSplit(Exception):
    pass