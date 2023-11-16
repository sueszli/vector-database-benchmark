import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import cache_on_self, get_fused_kernel_name, is_welford_reduction, sympy_product, sympy_subs, sympy_symbol
from ..virtualized import ops, V
from .common import BracesBuffer, CppWrapperKernelArgs, CSE, CSEVariable, DataTypePropagation, DeferredLine, DTYPE_TO_COMPUTATION_DTYPE, ExprPrinter, IndentedBuffer, Kernel, KernelArgs, OpOverrides, OptimizationContext
schedule_log = torch._logging.getArtifactLogger(__name__, 'schedule')
DTYPE_TO_CPP = {torch.float32: 'float', torch.float64: 'double', torch.float16: 'half', torch.int64: 'long', torch.int32: 'int', torch.int16: 'short', torch.int8: 'signed char', torch.uint8: 'unsigned char', torch.bool: 'bool', torch.bfloat16: 'bfloat16', torch.complex64: 'complex64'}
DTYPE_TO_ATEN = {torch.float32: 'at::kFloat', torch.float64: 'at::kDouble', torch.float16: 'at::kHalf', torch.int64: 'at::kLong', torch.int32: 'at::kInt', torch.int16: 'at::kShort', torch.int8: 'at::kChar', torch.uint8: 'at::kByte', torch.bool: 'at::kBool', torch.bfloat16: 'at::kBFloat16', torch.complex64: 'at::kComplexFloat'}
DEVICE_TO_ATEN = {'cpu': 'at::kCPU', 'cuda': 'at::kCUDA'}
INDEX_TYPE = 'long'
NATIVE_OMP_RTYPES = {'+', '*', '^', '||', 'min', 'max'}
RTYPE_TO_CPP = {'sum': '+', 'prod': '*', 'xor_sum': '^', 'min': 'min', 'max': 'max', 'argmin': 'argmin', 'argmax': 'argmax', 'any': '||', 'welford_reduce': 'welford', 'welford_combine': 'welford'}
VECTORIZABLE_RTYPES = {'max', 'min', 'sum', 'prod', 'xor_sum', 'welford_reduce', 'welford_combine'}
PYTHON_TO_CPP = {'Tensor': 'at::Tensor', 'int': 'long', 'float': 'double', 'bool': 'bool', 'str': 'std::string', 'ScalarType': 'c10::ScalarType', 'MemoryFormat': 'at::MemoryFormat', 'Layout': 'at::Layout', 'Device': 'at::Device', 'number': 'at::Scalar'}
CONTAINER_PYTHON_TO_CPP = {'List': 'std::vector', 'Optional': 'c10::optional'}
DTYPE_LOWP_FP = [torch.bfloat16, torch.float16]

def reduction_init(reduction_type, dtype):
    if False:
        i = 10
        return i + 15
    if dtype in DTYPE_LOWP_FP:
        dtype = torch.float32
    if reduction_type in ('xor_sum', 'sum', 'any'):
        return 0
    if reduction_type == 'prod':
        return 1
    if reduction_type in {'max', 'argmax'}:
        return f'-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()' if is_float_dtype(dtype) else f'std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()'
    if reduction_type in {'min', 'argmin'}:
        return f'std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()' if is_float_dtype(dtype) else f'std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()'
    if is_welford_reduction(reduction_type):
        return f'Welford<{DTYPE_TO_CPP[dtype]}>()'
    raise AssertionError(reduction_type)

def reduction_init_vec(reduction_type, dtype):
    if False:
        return 10
    scalar_type = DTYPE_TO_CPP[DTYPE_TO_COMPUTATION_DTYPE[dtype]]
    vec_type = f'at::vec::Vectorized<{scalar_type}>'
    if is_welford_reduction(reduction_type):
        return f'Welford<{vec_type}>()'
    scalar_init = reduction_init(reduction_type, dtype)
    return f'{vec_type}({scalar_init})'

def reduction_acc_type(reduction_type, dtype):
    if False:
        while True:
            i = 10
    assert reduction_type not in {'argmin', 'argmax'}
    scalar_type = DTYPE_TO_CPP[DTYPE_TO_COMPUTATION_DTYPE[dtype]]
    if is_welford_reduction(reduction_type):
        return f'Welford<{scalar_type}>'
    return scalar_type

def reduction_acc_type_vec(reduction_type, dtype):
    if False:
        print('Hello World!')
    assert reduction_type not in {'argmin', 'argmax'}
    scalar_type = DTYPE_TO_CPP[DTYPE_TO_COMPUTATION_DTYPE[dtype]]
    vec_type = f'at::vec::Vectorized<{scalar_type}>'
    if is_welford_reduction(reduction_type):
        return f'Welford<{vec_type}>'
    return vec_type

def reduction_combine(reduction_type, var, next_value):
    if False:
        while True:
            i = 10
    if reduction_type == 'sum':
        return f'{var} + {next_value}'
    if reduction_type == 'prod':
        return f'{var} * {next_value}'
    if reduction_type == 'xor_sum':
        return f'{var} ^ {next_value}'
    if reduction_type == 'any':
        return f'{var} || {next_value}'
    if reduction_type in ('min', 'max'):
        return f'{reduction_type}_propagate_nan({var}, {next_value})'
    if reduction_type == 'welford_reduce':
        return f'welford_combine({var}, {next_value})'
    if reduction_type == 'welford_combine':
        if isinstance(next_value, tuple):
            (mean, m2, weight) = next_value
        else:
            (mean, m2, weight) = reduction_project(reduction_type, next_value)
        return f'welford_combine({var}, {{{mean}, {m2}, {weight}}})'
    raise AssertionError(reduction_type)

def reduction_combine_vec(reduction_type, var, next_value):
    if False:
        while True:
            i = 10
    if reduction_type == 'max':
        return f'at::vec::maximum({var}, {next_value})'
    elif reduction_type == 'min':
        return f'at::vec::minimum({var}, {next_value})'
    elif reduction_type == 'sum':
        return f'{var} + {next_value}'
    elif reduction_type == 'prod':
        return f'{var} * {next_value}'
    elif reduction_type == 'xor_sum':
        return f'{var} ^ {next_value}'
    elif reduction_type == 'welford_reduce':
        return f'welford_combine({var}, {next_value})'
    elif reduction_type == 'welford_combine':
        if isinstance(next_value, tuple):
            (mean, m2, weight) = next_value
        else:
            (mean, m2, weight) = reduction_project(reduction_type, next_value)
        return f'welford_combine({var}, {{{mean}, {m2}, {weight}}})'
    else:
        raise NotImplementedError()

def reduction_project(reduction_type, acc):
    if False:
        for i in range(10):
            print('nop')
    if is_welford_reduction(reduction_type):
        return (f'{acc}.mean', f'{acc}.m2', f'{acc}.weight')
    elif reduction_type in {'argmin', 'argmax'}:
        return f'{acc}.index'
    return acc
index_value_name_counter = 1

def argmax_argmin_prefix(reduction_type, src_dtype, tmpvar):
    if False:
        i = 10
        return i + 15
    global index_value_name_counter
    struct_name = f'IndexValue_{index_value_name_counter}'
    index_value_name_counter += 1
    prefix = [f'struct {struct_name} {{size_t index; {DTYPE_TO_CPP[src_dtype]} value;}};', f'{struct_name} {tmpvar}{{0, {reduction_init(reduction_type, src_dtype)}}};']
    if reduction_type == 'argmax':
        prefix.extend(['#if !defined(__clang_major__) || __clang_major__ > 9', f'#pragma omp declare reduction(argmax : {struct_name} :\\', '    omp_out.value = omp_in.value < omp_out.value ? omp_out.value : omp_in.value,\\', '    omp_out.index = omp_in.value < omp_out.value ? omp_out.index : omp_in.index)\\', f'\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})', '#endif'])
    elif reduction_type == 'argmin':
        prefix.extend(['#if !defined(__clang_major__) || __clang_major__ > 9', f'#pragma omp declare reduction(argmin : {struct_name} :\\', '    omp_out.value = omp_in.value > omp_out.value ? omp_out.value : omp_in.value,\\', '    omp_out.index = omp_in.value > omp_out.value ? omp_out.index : omp_in.index)\\', f'\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})', '#endif'])
    return prefix

def parallel_num_threads():
    if False:
        return 10
    threads = config.cpp.threads
    if threads < 1:
        threads = torch.get_num_threads()
    return threads

@functools.lru_cache
def stride_at(var: sympy.Symbol, index: sympy.Expr):
    if False:
        while True:
            i = 10
    replacement = {var: var + 1}
    new_index = sympy_subs(index, replacement)
    return sympy.simplify(new_index - index)

class CppPrinter(ExprPrinter):

    def _print_Integer(self, expr):
        if False:
            i = 10
            return i + 15
        return f'{int(expr)}L'

    def _print_Where(self, expr):
        if False:
            i = 10
            return i + 15
        c = self.paren(self.doprint(expr.args[0]))
        p = self.paren(self.doprint(expr.args[1]))
        q = self.paren(self.doprint(expr.args[2]))
        return f'{c} ? {p} : {q}'

    def _print_ModularIndexing(self, expr):
        if False:
            while True:
                i = 10
        (x, div, mod) = expr.args
        x = self.paren(self.doprint(x))
        if div != 1:
            div = self.paren(self.doprint(div))
            if expr.is_integer:
                x = f'c10::div_floor_integer({x}, {div})'
            else:
                x = f'c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))'
        mod = self.paren(self.doprint(mod))
        return f'static_cast<{INDEX_TYPE}>({x}) % static_cast<{INDEX_TYPE}>({mod})'

    def _print_FloorDiv(self, expr):
        if False:
            i = 10
            return i + 15
        (x, div) = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        if expr.is_integer:
            return f'c10::div_floor_integer({x}, {div})'
        return f'c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))'

    def _print_floor(self, expr):
        if False:
            for i in range(10):
                print('nop')
        assert len(expr.args) == 1
        r = f'std::floor({self._print(expr.args[0])})'
        return f'static_cast<{INDEX_TYPE}>({r})' if expr.is_integer else r

    def _print_Pow(self, expr):
        if False:
            for i in range(10):
                print('nop')
        (base, exp) = expr.args
        base = self._print(base)
        if exp == 0.5 or exp == -0.5:
            r = f'std::sqrt({base})' if exp == 0.5 else f'1.0/std::sqrt({base})'
            return f'static_cast<INDEX_TYPE>({r})' if expr.is_integer else r
        assert exp.is_integer
        exp = int(exp)
        if exp > 0:
            r = '*'.join([self.paren(base)] * exp)
        elif exp < 0:
            r = '1.0/' + self.paren('*'.join([self.paren(base)] * abs(exp)))
        else:
            r = '1.0'
        return f'static_cast<{INDEX_TYPE}>({r})' if expr.is_integer else r

    def _print_Rational(self, expr):
        if False:
            print('Hello World!')
        if expr.q == 1:
            r = f'{expr.p}'
        else:
            r = f'{expr.p}.0/{expr.q}.0'
        return f'static_cast<{INDEX_TYPE}>({r})' if expr.is_integer else r

    def _print_ceiling(self, expr):
        if False:
            return 10
        assert len(expr.args) == 1
        r = f'std::ceil({self._print(expr.args[0])})'
        return f'static_cast<{INDEX_TYPE}>({r})' if expr.is_integer else r

    def _print_Min(self, expr):
        if False:
            return 10
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f'std::min({args[0]}, {args[1]})'
        else:
            il = '{' + ', '.join(args) + '}'
            return f'std::min({il})'

    def _print_Max(self, expr):
        if False:
            return 10
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f'std::max({args[0]}, {args[1]})'
        else:
            il = '{' + ', '.join(args) + '}'
            return f'std::max({il})'

    def _print_Abs(self, expr):
        if False:
            for i in range(10):
                print('nop')
        assert len(expr.args) == 1
        return f'std::abs({self._print(expr.args[0])})'
cexpr = CppPrinter().doprint

def cexpr_index(index):
    if False:
        while True:
            i = 10
    return f'static_cast<{INDEX_TYPE}>({cexpr(index)})'

class RecordOptimizationContext:

    def __init__(self, func_name: str=''):
        if False:
            return 10
        self.func_name = func_name
        self.current_node: Optional[torch.fx.Node] = None
        self.opt_ctx: Optional[OptimizationContext] = None

    def __enter__(self):
        if False:
            return 10
        assert V.interpreter
        assert V.interpreter.current_node
        self.current_node = V.interpreter.current_node
        assert self.current_node is not None
        if OptimizationContext.key in self.current_node.meta:
            self.opt_ctx = self.current_node.meta[OptimizationContext.key]
        else:
            self.opt_ctx = OptimizationContext()
        assert self.opt_ctx is not None
        self.opt_ctx.ops_name = self.func_name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        assert self.current_node
        assert self.opt_ctx
        self.current_node.meta[OptimizationContext.key] = self.opt_ctx

    def get_opt_ctx(self):
        if False:
            i = 10
            return i + 15
        return self.opt_ctx

    def get_fx_node(self):
        if False:
            while True:
                i = 10
        assert self.current_node
        return self.current_node

def get_opt_ctx(node: torch.fx.Node) -> OptimizationContext:
    if False:
        print('Hello World!')
    return node.meta.get(OptimizationContext.key, None)

def get_current_node_opt_ctx() -> OptimizationContext:
    if False:
        while True:
            i = 10
    assert V.interpreter.current_node
    return get_opt_ctx(V.interpreter.current_node)

class CppVecOverrides(OpOverrides):
    """Map element-wise ops to aten vectorization C++"""

    @staticmethod
    def add(a, b):
        if False:
            i = 10
            return i + 15
        return f'{a} + {b}'

    @staticmethod
    def sub(a, b):
        if False:
            print('Hello World!')
        return f'{a} - {b}'

    @staticmethod
    def mul(a, b):
        if False:
            return 10
        return f'{a} * {b}'

    @staticmethod
    def truediv(a, b):
        if False:
            i = 10
            return i + 15
        return f'{a} / {b}'

    @staticmethod
    def abs(x):
        if False:
            return 10
        return f'{x}.abs()'

    @staticmethod
    def sin(x):
        if False:
            for i in range(10):
                print('nop')
        return f'{x}.sin()'

    @staticmethod
    def cos(x):
        if False:
            while True:
                i = 10
        return f'{x}.cos()'

    @staticmethod
    def exp(x):
        if False:
            i = 10
            return i + 15
        return f'{x}.exp()'

    @staticmethod
    def exp2(x):
        if False:
            while True:
                i = 10
        return f'{x}.exp2()'

    @staticmethod
    def expm1(x):
        if False:
            return 10
        vec_one = f'decltype({x})(1)'
        return f'{x}.exp() - {vec_one}'

    @staticmethod
    def erf(x):
        if False:
            return 10
        return f'{x}.erf()'

    @staticmethod
    def erfc(x):
        if False:
            print('Hello World!')
        return f'{x}.erfc()'

    @staticmethod
    def erfinv(x):
        if False:
            i = 10
            return i + 15
        return f'{x}.erfinv()'

    @staticmethod
    def sqrt(x):
        if False:
            i = 10
            return i + 15
        return f'{x}.sqrt()'

    @staticmethod
    def eq(x, y):
        if False:
            while True:
                i = 10
        return f'to_float_mask({x} == {y})'

    @staticmethod
    def ne(x, y):
        if False:
            print('Hello World!')
        return f'to_float_mask({x} != {y})'

    @staticmethod
    def lt(x, y):
        if False:
            return 10
        return f'to_float_mask({x} < {y})'

    @staticmethod
    def gt(x, y):
        if False:
            print('Hello World!')
        return f'to_float_mask({x} > {y})'

    @staticmethod
    def le(x, y):
        if False:
            for i in range(10):
                print('nop')
        return f'to_float_mask({x} <= {y})'

    @staticmethod
    def ge(x, y):
        if False:
            print('Hello World!')
        return f'to_float_mask({x} >= {y})'

    @staticmethod
    def and_(x, y):
        if False:
            for i in range(10):
                print('nop')
        return f'{x} & {y}'

    @staticmethod
    def rsqrt(x):
        if False:
            return 10
        return f'{x}.rsqrt()'

    @staticmethod
    def pow(a, b):
        if False:
            return 10
        return f'{a}.pow({b})'

    @staticmethod
    def log(x):
        if False:
            for i in range(10):
                print('nop')
        return f'{x}.log()'

    @staticmethod
    def round(x):
        if False:
            for i in range(10):
                print('nop')
        return f'{x}.round()'

    @staticmethod
    def floor(x):
        if False:
            for i in range(10):
                print('nop')
        return f'{x}.floor()'

    @staticmethod
    def ceil(x):
        if False:
            while True:
                i = 10
        return f'{x}.ceil()'

    @staticmethod
    def trunc(x):
        if False:
            for i in range(10):
                print('nop')
        return f'{x}.trunc()'

    @staticmethod
    def fmod(a, b):
        if False:
            print('Hello World!')
        return f'{a}.fmod({b})'

    @staticmethod
    def lgamma(x):
        if False:
            while True:
                i = 10
        return f'{x}.lgamma()'

    @staticmethod
    def logical_and(a, b):
        if False:
            for i in range(10):
                print('nop')
        return f'({a} != 0) & ({b} != 0)'

    @staticmethod
    def logical_not(a):
        if False:
            return 10
        return f'{a} == 0'

    @staticmethod
    def logical_or(a, b):
        if False:
            for i in range(10):
                print('nop')
        return f'({a} != 0) | ({b} != 0)'

    @staticmethod
    def logical_xor(a, b):
        if False:
            while True:
                i = 10
        return f'({a} != 0) ^ ({b} != 0)'

    @staticmethod
    def tan(a):
        if False:
            i = 10
            return i + 15
        return f'{a}.tan()'

    @staticmethod
    def tanh(a):
        if False:
            while True:
                i = 10
        vec_one = f'decltype({a})(1)'
        vec_two = f'decltype({a})(2)'
        vec_minus_two = f'decltype({a})(-2)'
        return f'{vec_two} / ({vec_one} + ({vec_minus_two} * {a}).exp()) - {vec_one}'

    @staticmethod
    def reciprocal(a):
        if False:
            i = 10
            return i + 15
        return f'{a}.reciprocal()'

    @staticmethod
    def atan(x):
        if False:
            return 10
        return f'{x}.atan()'

    @staticmethod
    def acos(x):
        if False:
            return 10
        return f'{x}.acos()'

    @staticmethod
    def asin(x):
        if False:
            print('Hello World!')
        return f'{x}.asin()'

    @staticmethod
    def cosh(x):
        if False:
            for i in range(10):
                print('nop')
        return f'{x}.cosh()'

    @staticmethod
    def sinh(x):
        if False:
            while True:
                i = 10
        return f'{x}.sinh()'

    @staticmethod
    def log10(x):
        if False:
            return 10
        return f'{x}.log10()'

    @staticmethod
    def nextafter(x):
        if False:
            while True:
                i = 10
        return f'{x}.nextafter()'

    @staticmethod
    def copysign(a, b):
        if False:
            for i in range(10):
                print('nop')
        return f'{a}.copysign({b})'

    @staticmethod
    def atan2(a, b):
        if False:
            print('Hello World!')
        return f'{a}.atan2({b})'

    @staticmethod
    def hypot(a, b):
        if False:
            for i in range(10):
                print('nop')
        return f'{a}.hypot({b})'

    @staticmethod
    def atanh(x):
        if False:
            for i in range(10):
                print('nop')
        vec_one = f'decltype({x})(1)'
        vec_one_half = f'decltype({x})(0.5)'
        return f'{vec_one_half} * (({vec_one} + {x})/({vec_one} - {x})).log()'

    @staticmethod
    def asinh(x):
        if False:
            print('Hello World!')
        vec_one = f'decltype({x})(1)'
        return f'({x} + ({vec_one} + {x}*{x}).sqrt()).log()'

    @staticmethod
    def acosh(x):
        if False:
            for i in range(10):
                print('nop')
        vec_one = f'decltype({x})(1)'
        return f'({x} + ({x}*{x} - {vec_one}).sqrt()).log()'

    @staticmethod
    def constant(val, dtype):
        if False:
            print('Hello World!')
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx
        proposed_dtype = opt_ctx.dtype
        assert proposed_dtype in [torch.float, torch.int32]
        if val == float('inf'):
            quote = f'std::numeric_limits<{DTYPE_TO_CPP[proposed_dtype]}>::infinity()'
        elif val == float('-inf'):
            quote = f'-std::numeric_limits<{DTYPE_TO_CPP[proposed_dtype]}>::infinity()'
        elif math.isnan(val):
            quote = f'std::numeric_limits<{DTYPE_TO_CPP[proposed_dtype]}>::quiet_NaN()'
        elif val is True or val is False:
            quote = f'static_cast<{DTYPE_TO_CPP[proposed_dtype]}>({str(val).lower()})'
        else:
            quote = f'static_cast<{DTYPE_TO_CPP[proposed_dtype]}>({repr(val)})'
        return f'at::vec::Vectorized<{DTYPE_TO_CPP[proposed_dtype]}>({quote})'

    @staticmethod
    def relu(x):
        if False:
            for i in range(10):
                print('nop')
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        if bug == 'compile_error':
            return 'compile error!'
        elif bug == 'runtime_error':
            return f'{x}; throw 1'
        elif bug == 'accuracy':
            return f'{x} + decltype({x})(1)'
        elif bug is None:
            return f'at::vec::clamp_min({x}, decltype({x})(0))'
        else:
            raise AssertionError(f'unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}')

    @staticmethod
    def sigmoid(x):
        if False:
            return 10
        return f'decltype({x})(1)/(decltype({x})(1) + {x}.neg().exp())'

    @staticmethod
    def neg(x):
        if False:
            while True:
                i = 10
        return f'{x}.neg()'

    @staticmethod
    def floordiv(a, b):
        if False:
            i = 10
            return i + 15
        _t = f'decltype({a})'
        quot = f'{a} / {b}'
        rem = f'{a} % {b}'
        return f'(({a} < {_t}(0)) != ({b} < {_t}(0)) ? ({rem} != {_t}(0) ? {quot} - {_t}(1) : {quot}) : {quot})'

    @staticmethod
    def truncdiv(a, b):
        if False:
            i = 10
            return i + 15
        return f'{a} / {b}'

    @staticmethod
    def minimum(a, b):
        if False:
            while True:
                i = 10
        return f'at::vec::minimum({a}, {b})'

    @staticmethod
    def maximum(a, b):
        if False:
            while True:
                i = 10
        return f'at::vec::maximum({a}, {b})'

    @staticmethod
    def square(a):
        if False:
            i = 10
            return i + 15
        return f'{a} * {a}'

    @staticmethod
    def where(a, b, c):
        if False:
            for i in range(10):
                print('nop')
        return f'decltype({b})::blendv({c}, {b}, {a})'

    @staticmethod
    def sign(x):
        if False:
            for i in range(10):
                print('nop')
        code = BracesBuffer()
        vec_zero = f'decltype({x})(0)'
        vec_one = f'decltype({x})(1)'
        blendv = f'decltype({x})::blendv({vec_zero}, {vec_one}, {vec_zero} < {x})'
        left = V.kernel.cse.newvar()
        code.writeline(f'auto {left} = {blendv};')
        blendv = f'decltype({x})::blendv({vec_zero}, {vec_one}, {x} < {vec_zero})'
        right = V.kernel.cse.newvar()
        code.writeline(f'auto {right} = {blendv};')
        result = V.kernel.cse.newvar()
        code.writeline(f'auto {result} = {left} - {right};')
        V.kernel.compute.splice(code)
        return result

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None):
        if False:
            for i in range(10):
                print('nop')
        assert dtype in [torch.bool, torch.float, torch.bfloat16, torch.float16, torch.uint8], f'{__name__} does not support {dtype}'
        node: torch.fx.Node = V.interpreter.current_node
        assert node and isinstance(node, torch.fx.Node)
        opt_ctx_x = get_opt_ctx(node.args[1])
        assert opt_ctx_x
        if opt_ctx_x.dtype in (torch.float, torch.float32) and dtype == torch.bool:
            return f'vec_convert_to_mask({x})'
        if opt_ctx_x.dtype == torch.bool and dtype in (torch.float, torch.float32):
            return f'mask_convert_to_float({x})'
        if opt_ctx_x.dtype in (torch.float, torch.float32) and dtype in DTYPE_LOWP_FP:
            return f'cvt_fp32_to_lowp_fp<{DTYPE_TO_CPP[dtype]}>({x})'
        if opt_ctx_x.dtype in DTYPE_LOWP_FP and dtype in (torch.float, torch.float32):
            return f'cvt_lowp_fp_to_fp32<{DTYPE_TO_CPP[opt_ctx_x.dtype]}>({x})'
        if opt_ctx_x.dtype == torch.uint8 and dtype in (torch.float, torch.float32):
            return f'at::vec::convert_uint8_to_float({x})'
        if opt_ctx_x.dtype in (torch.float, torch.float32) and dtype == torch.uint8:
            return f'at::vec::convert_float_to_uint8({x})'
        return f'({x})'

    @staticmethod
    def log1p(x):
        if False:
            while True:
                i = 10
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == 'accuracy':
            return f'{x} + decltype({x})(1)'
        elif bug is None:
            return f'{x}.log1p()'
        else:
            raise AssertionError(f'unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}')

    @staticmethod
    def masked(mask, body, other):
        if False:
            print('Hello World!')
        code = BracesBuffer()
        var = V.kernel.cse.newvar()
        with V.kernel.masked(mask) as new_mask:
            code.writeline(f'auto {var} = [&]')
            with V.kernel.swap_buffers(code), code.indent():
                result = body()
                code.writeline(f'return {result};')
        code.writeline(';')
        V.kernel.compute.splice(code)
        if other == float('-inf'):
            other_code = 'at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())'
        elif other == float('inf'):
            other_code = 'at::vec::Vectorized<float>(std::numeric_limits<float>::infinity())'
        elif math.isnan(other):
            other_code = 'at::vec::Vectorized<float>(std::numeric_limits<float>::quiet_NaN())'
        else:
            other_code = f'at::vec::Vectorized<float>({other!r})'
        type = f'decltype({var}())'
        float_mask = f'to_float_mask({new_mask})'
        return f'{type}::blendv({other_code}, {var}(), {float_mask})'

    @staticmethod
    def index_expr(expr, dtype):
        if False:
            i = 10
            return i + 15
        assert dtype == torch.int64
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx
        assert opt_ctx.dtype == torch.int32
        assert opt_ctx.is_most_inner_loop_irrevelant
        return f'at::vec::Vectorized<int>(static_cast<int>({cexpr(V.kernel.rename_indexing(expr))}))'

class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def add(a, b):
        if False:
            print('Hello World!')
        return f'decltype({a})({a} + {b})'

    @staticmethod
    def sub(a, b):
        if False:
            while True:
                i = 10
        return f'decltype({a})({a} - {b})'

    @staticmethod
    def mul(a, b):
        if False:
            print('Hello World!')
        return f'decltype({a})({a} * {b})'

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None):
        if False:
            while True:
                i = 10
        assert dtype in DTYPE_TO_CPP, f'{dtype} missing from {__name__}.DTYPE_TO_CPP'
        return f'c10::convert<{DTYPE_TO_CPP[dtype]}>({x})'

    @staticmethod
    def to_dtype_bitcast(x, dtype):
        if False:
            for i in range(10):
                print('nop')
        assert dtype in DTYPE_TO_CPP, f'{dtype} missing from {__name__}.DTYPE_TO_CPP'
        return f'c10::bit_cast<{DTYPE_TO_CPP[dtype]}>({x})'

    @staticmethod
    def abs(x):
        if False:
            while True:
                i = 10
        return f'std::abs({x})'

    @staticmethod
    def sin(x):
        if False:
            return 10
        return f'std::sin({x})'

    @staticmethod
    def cos(x):
        if False:
            while True:
                i = 10
        return f'std::cos({x})'

    @staticmethod
    def neg(x):
        if False:
            for i in range(10):
                print('nop')
        return f'decltype({x})(-{x})'

    @staticmethod
    def exp(x):
        if False:
            i = 10
            return i + 15
        return f'std::exp({x})'

    @staticmethod
    def exp2(x):
        if False:
            for i in range(10):
                print('nop')
        return f'std::exp2({x})'

    @staticmethod
    def expm1(x):
        if False:
            print('Hello World!')
        return f'std::expm1({x})'

    @staticmethod
    def erf(x):
        if False:
            while True:
                i = 10
        return f'std::erf({x})'

    @staticmethod
    def erfc(x):
        if False:
            print('Hello World!')
        return f'std::erfc({x})'

    @staticmethod
    def erfinv(x):
        if False:
            while True:
                i = 10
        return f'calc_erfinv({x})'

    @staticmethod
    def sqrt(x):
        if False:
            while True:
                i = 10
        return f'std::sqrt({x})'

    @staticmethod
    def rsqrt(x):
        if False:
            print('Hello World!')
        return f'1 / std::sqrt({x})'

    @staticmethod
    def log1p(x):
        if False:
            while True:
                i = 10
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == 'accuracy':
            return f'{x} + decltype({x})(1)'
        elif bug is None:
            return f'std::log1p({x})'
        else:
            raise AssertionError(f'unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}')

    @staticmethod
    def tan(x):
        if False:
            return 10
        return f'std::tan({x})'

    @staticmethod
    def tanh(x):
        if False:
            print('Hello World!')
        return f'std::tanh({x})'

    @staticmethod
    def signbit(x):
        if False:
            return 10
        return f'std::signbit({x})'

    @staticmethod
    def pow(a, b):
        if False:
            i = 10
            return i + 15
        return f'std::pow({a}, {b})'

    @staticmethod
    def log(x):
        if False:
            print('Hello World!')
        return f'std::log({x})'

    @staticmethod
    def round(x):
        if False:
            i = 10
            return i + 15
        return f'std::nearbyint({x})'

    @staticmethod
    def floor(x):
        if False:
            while True:
                i = 10
        return f'std::floor({x})'

    @staticmethod
    def floordiv(a, b):
        if False:
            while True:
                i = 10
        quot = f'{a} / {b}'
        rem = f'{a} % {b}'
        return f'(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})'

    @staticmethod
    def ceil(x):
        if False:
            for i in range(10):
                print('nop')
        return f'std::ceil({x})'

    @staticmethod
    def trunc(x):
        if False:
            while True:
                i = 10
        return f'std::trunc({x})'

    @staticmethod
    def truncdiv(a, b):
        if False:
            i = 10
            return i + 15
        return f'{a} / {b}'

    @staticmethod
    def fmod(a, b):
        if False:
            print('Hello World!')
        return f'std::fmod({a}, {b})'

    @staticmethod
    def isinf(x):
        if False:
            for i in range(10):
                print('nop')
        return f'std::isinf({x})'

    @staticmethod
    def isnan(x):
        if False:
            for i in range(10):
                print('nop')
        return f'std::isnan({x})'

    @staticmethod
    def lgamma(x):
        if False:
            print('Hello World!')
        return f'std::lgamma({x})'

    @staticmethod
    def acos(x):
        if False:
            print('Hello World!')
        return f'std::acos({x})'

    @staticmethod
    def acosh(x):
        if False:
            while True:
                i = 10
        return f'std::acosh({x})'

    @staticmethod
    def cosh(x):
        if False:
            while True:
                i = 10
        return f'std::cosh({x})'

    @staticmethod
    def sinh(x):
        if False:
            while True:
                i = 10
        return f'std::sinh({x})'

    @staticmethod
    def asin(x):
        if False:
            for i in range(10):
                print('nop')
        return f'std::asin({x})'

    @staticmethod
    def asinh(x):
        if False:
            while True:
                i = 10
        return f'std::asinh({x})'

    @staticmethod
    def atan2(x, y):
        if False:
            while True:
                i = 10
        return f'std::atan2({x}, {y})'

    @staticmethod
    def atan(x):
        if False:
            while True:
                i = 10
        return f'std::atan({x})'

    @staticmethod
    def atanh(x):
        if False:
            for i in range(10):
                print('nop')
        return f'std::atanh({x})'

    @staticmethod
    def copysign(x, y):
        if False:
            while True:
                i = 10
        return f'std::copysign({x}, {y})'

    @staticmethod
    def hypot(x, y):
        if False:
            print('Hello World!')
        return f'std::hypot({x}, {y})'

    @staticmethod
    def log10(x):
        if False:
            return 10
        return f'std::log10({x})'

    @staticmethod
    def nextafter(x, y):
        if False:
            return 10
        return f'std::nextafter({x}, {y})'

    @staticmethod
    def relu(x):
        if False:
            i = 10
            return i + 15
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        if bug == 'compile_error':
            return 'compile error!'
        elif bug == 'runtime_error':
            return f'{x}; throw 1'
        elif bug == 'accuracy':
            return f'{x} + decltype({x})(1)'
        elif bug is None:
            return f'{x} * ({x}>0)'
        else:
            raise AssertionError(f'unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}')

    @staticmethod
    def minimum(a, b):
        if False:
            return 10
        return f'min_propagate_nan({a}, {b})'

    @staticmethod
    def maximum(a, b):
        if False:
            i = 10
            return i + 15
        return f'max_propagate_nan({a}, {b})'

    @staticmethod
    def where(a, b, c):
        if False:
            print('Hello World!')
        return f'{a} ? {b} : {c}'

    @staticmethod
    def mod(a, b):
        if False:
            return 10
        return f'mod({a}, {b})'

    @staticmethod
    def constant(val, dtype):
        if False:
            return 10
        if dtype in DTYPE_LOWP_FP:
            dtype = torch.float32
        if val == float('inf'):
            return f'std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()'
        elif val == float('-inf'):
            return f'-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()'
        elif math.isnan(val):
            return f'std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::quiet_NaN()'
        elif val is True or val is False:
            return ops.to_dtype(str(val).lower(), dtype)
        return ops.to_dtype(repr(val), dtype)

    @staticmethod
    def index_expr(expr, dtype):
        if False:
            for i in range(10):
                print('nop')
        return ops.to_dtype(cexpr(V.kernel.rename_indexing(expr)), dtype)

    @staticmethod
    def masked(mask, body, other):
        if False:
            return 10
        code = BracesBuffer()
        body_var = V.kernel.cse.newvar()
        code.writeline(f'auto {body_var} = [&]')
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f'return {result};')
        code.writeline(';')
        V.kernel.compute.splice(code)
        type = f'decltype({body_var}())'
        if other == float('-inf'):
            other_code = f'-std::numeric_limits<{type}>::infinity()'
        elif other == float('inf'):
            other_code = f'std::numeric_limits<{type}>::infinity()'
        elif isinstance(other, bool):
            other_code = f'static_cast<{type}>({str(other).lower()})'
        elif math.isnan(other):
            other_code = f'std::numeric_limits<{type}>::quiet_NaN()'
        else:
            other_code = f'static_cast<{type}>({repr(other)})'
        return f'{mask} ? {body_var}() : {other_code}'

    @staticmethod
    def logical_and(a, b):
        if False:
            i = 10
            return i + 15
        return f'{a} && {b}'

    @staticmethod
    def logical_not(a):
        if False:
            for i in range(10):
                print('nop')
        return f'!{a}'

    @staticmethod
    def logical_or(a, b):
        if False:
            i = 10
            return i + 15
        return f'{a} || {b}'

    @staticmethod
    def logical_xor(a, b):
        if False:
            for i in range(10):
                print('nop')
        return f'{a} != {b}'

    @staticmethod
    def bitwise_and(a, b):
        if False:
            return 10
        return f'decltype({a})({a} & {b})'

    @staticmethod
    def bitwise_not(a):
        if False:
            while True:
                i = 10
        return f'decltype({a})(~{a})'

    @staticmethod
    def bitwise_or(a, b):
        if False:
            print('Hello World!')
        return f'decltype({a})({a} | {b})'

    @staticmethod
    def bitwise_xor(a, b):
        if False:
            return 10
        return f'decltype({a})({a} ^ {b})'

    @staticmethod
    def bitwise_left_shift(a, b):
        if False:
            while True:
                i = 10
        return f'decltype({a})({a} << {b})'

    @staticmethod
    def bitwise_right_shift(a, b):
        if False:
            for i in range(10):
                print('nop')
        return f'decltype({a})({a} >> {b})'

    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr):
        if False:
            return 10
        return f'normalized_rand_cpu({seed}, {offset})'

    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr):
        if False:
            print('Hello World!')
        return f'randn_cpu({seed}, {offset})'

    @staticmethod
    def randint64(seed: sympy.Expr, offset: sympy.Expr, low, high):
        if False:
            print('Hello World!')
        return f'randint64_cpu({seed}, {offset}, {low}, {high})'

    @staticmethod
    def sigmoid(x):
        if False:
            print('Hello World!')
        return f'decltype({x})(1) / (decltype({x})(1) + std::exp(-{x}))'

    @staticmethod
    def sign(x):
        if False:
            while True:
                i = 10
        code = BracesBuffer()
        left = V.kernel.cse.newvar()
        right = V.kernel.cse.newvar()
        result = V.kernel.cse.newvar()
        scalar_zero = f'decltype({x})(0)'
        scalar_one = f'decltype({x})(1)'
        code.writeline(f'auto {left} = {x} > 0 ? {scalar_one} : {scalar_zero};')
        code.writeline(f'auto {right} = {x} < 0 ? {scalar_one} : {scalar_zero};')
        code.writeline(f'auto {result} = {left} - {right};')
        V.kernel.compute.splice(code)
        return result

class CppKernel(Kernel):
    overrides = CppOverrides
    sexpr = cexpr
    newvar_prefix = 'auto '
    suffix = ';'

    def __init__(self, args, num_threads):
        if False:
            print('Hello World!')
        super().__init__(args)
        self.call_ranges: Optional[Tuple[sympy.Expr, ...]] = None
        self.ranges: List[sympy.Expr] = []
        self.itervars: List[sympy.Symbol] = []
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.reduction_var_map = {}
        self.reduction_cse = CSE(self.newvar_prefix, self.suffix, name_prefix='tmp_acc')
        self.preloads = IndentedBuffer()
        self.poststores = IndentedBuffer()
        self.num_threads = num_threads
        self.reduction_omp_dec: Dict[Tuple[str, str], str] = {}

    @contextlib.contextmanager
    def masked(self, mask):
        if False:
            i = 10
            return i + 15
        'Context manager to add an additional mask to loads and stores.'
        prior = self._load_mask
        if prior:
            mask = self.cse.generate(self.compute, f'{mask} & {prior}')
        self._load_mask = mask
        try:
            yield mask
        finally:
            self._load_mask = prior

    def scale_index_with_offset(self, index: sympy.Expr, scale=1, itervar_idx=-1, offset=0):
        if False:
            while True:
                i = 10
        var = self.itervars[itervar_idx]
        replacement = {var: var * scale + offset}
        new_index = sympy_subs(index, replacement)
        return new_index

    def index_to_str(self, index: sympy.Expr) -> str:
        if False:
            while True:
                i = 10
        '\n        Convert an index expr to a string that can be used in cpp code.\n        e.g. a sympy expression "s2" may actually appear as "ks1" in the cpp kernel.\n        '
        return cexpr(self.rename_indexing(index))

    def load(self, name: str, index: sympy.Expr):
        if False:
            print('Hello World!')
        var = self.args.input(name)
        index = self.rename_indexing(index)
        line = f'{var}[{cexpr_index(index)}]'
        if V.graph.get_dtype(name) in [torch.float16]:
            line = f'static_cast<float>({line})'
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        if False:
            for i in range(10):
                print('nop')
        assert 'buf' in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        if mode is None:
            line = f'{var}[{cexpr_index(index)}] = {value};'
        elif mode == 'atomic_add':
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                line = f'{var}[{cexpr_index(index)}] += {value};'
            else:
                line = f'atomic_add(&{var}[{cexpr_index(index)}], {value});'
        else:
            raise NotImplementedError(f'store mode={mode}')
        self.stores.writeline(DeferredLine(name, line))

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if False:
            i = 10
            return i + 15
        argmax_or_argmin = reduction_type in {'argmax', 'argmin'}
        reduction_key = (src_dtype, reduction_type, value)
        if reduction_key in self.reduction_cse.reduction_cache:
            return self.reduction_cse.reduction_cache[reduction_key]
        acc = self.reduction_cse.generate(self.loads, f'reduction {reduction_key}', write=False)
        self.reduction_var_map[acc] = reduction_type
        if argmax_or_argmin:
            self.reduction_prefix.writelines(argmax_argmin_prefix(reduction_type, src_dtype, acc))
            compare_op = '<' if reduction_type == 'argmax' else '>'
            assert self.reduction_depth is not None
            index = self.itervars[self.reduction_depth]
            for i in range(self.reduction_depth + 1, len(self.itervars)):
                index = index * self.ranges[i] + self.itervars[i]
            self.stores.writelines([f'if ({acc}.value {compare_op} {value}) {{', f'    {acc}.index = {cexpr_index(index)}; {acc}.value = {value};', '}'])
        else:
            acc_type = reduction_acc_type(reduction_type, dtype)
            if (reduction_type, acc_type) not in self.reduction_omp_dec:
                if RTYPE_TO_CPP[reduction_type] not in NATIVE_OMP_RTYPES:
                    self.reduction_prefix.splice(f"    #pragma omp declare reduction(    {RTYPE_TO_CPP[reduction_type]}:{acc_type}:    omp_out = {reduction_combine(reduction_type, 'omp_out', 'omp_in')})     initializer(omp_priv={{{reduction_init(reduction_type, dtype)}}})\n                ")
                self.reduction_omp_dec[reduction_type, acc_type] = RTYPE_TO_CPP[reduction_type]
            self.reduction_prefix.writeline(f'{acc_type} {acc} = {reduction_init(reduction_type, dtype)};')
            self.stores.writeline(f'{acc} = {reduction_combine(reduction_type, acc, value)};')
        result = reduction_project(reduction_type, acc)
        self.reduction_cse.reduction_cache[reduction_key] = result
        return result

    def store_reduction(self, name, index, value):
        if False:
            for i in range(10):
                print('nop')
        index = self.rename_indexing(index)
        var = self.args.output(name)
        self.reduction_suffix.writeline(DeferredLine(name, f'{var}[{cexpr_index(index)}] = {value};'))

    def set_ranges(self, lengths, reduction_lengths):
        if False:
            for i in range(10):
                print('nop')
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(reduction_lengths), f'{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}'
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy_symbol(f'x{n}') for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (self.itervars[:self.reduction_depth], self.itervars[self.reduction_depth:])

    def size_hint(self):
        if False:
            for i in range(10):
                print('nop')
        return V.graph.sizevars.size_hint(sympy_product(self.call_ranges), fallback=8192)

    def codegen_loops_impl(self, loop_nest, code, worksharing):
        if False:
            for i in range(10):
                print('nop')
        threads = parallel_num_threads()
        assert self.call_ranges is not None
        par_depth = self.decide_parallel_depth(self.call_ranges[:loop_nest.max_parallel_depth()], threads)
        with contextlib.ExitStack() as stack:
            if par_depth:
                if loop_nest.is_reduction_only():
                    worksharing.close()
                else:
                    worksharing.parallel(threads)
                loop_nest.mark_parallel(par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            def gen_kernel(kernel):
                if False:
                    print('Hello World!')
                with contextlib.ExitStack() as stack:
                    assert kernel
                    if hasattr(kernel, 'codegen_inner_loops'):
                        code.splice(kernel.preloads)
                        kernel.codegen_inner_loops(code)
                        stack.enter_context(code.indent())
                    code.splice(kernel.loads)
                    code.splice(kernel.compute)
                    code.splice(kernel.stores)
                if hasattr(kernel, 'codegen_inner_loops'):
                    code.splice(kernel.poststores)

            def get_reduction_code_buffer(loops, is_suffix=True):
                if False:
                    print('Hello World!')
                for loop in loops:
                    for kernel in loop.get_kernels():
                        if is_suffix:
                            return kernel.reduction_suffix
                        else:
                            return kernel.reduction_prefix
                return None

            def gen_loops(loops: List[LoopLevel], in_reduction=False):
                if False:
                    while True:
                        i = 10
                with contextlib.ExitStack() as stack_outer:
                    if loops:
                        loop = loops[0]
                        if loop.is_reduction() and (not in_reduction):
                            reduction_prefix = get_reduction_code_buffer(loops, is_suffix=False)
                            if reduction_prefix:
                                stack_outer.enter_context(code.indent())
                            code.splice(reduction_prefix)
                        if loop_nest.is_reduction_only() and loop.parallel:
                            worksharing.parallel(threads)
                    for loop in loops:
                        gen_loop(loop, in_reduction)
                    if loops:
                        loop = loops[0]
                        if loop_nest.is_reduction_only() and loop.parallel:
                            worksharing.close()
                        if loop.is_reduction() and (not in_reduction):
                            code.splice(get_reduction_code_buffer(loops, is_suffix=True))

            def gen_loop(loop: LoopLevel, in_reduction=False):
                if False:
                    return 10
                with contextlib.ExitStack() as stack:
                    loop_lines = loop.lines()
                    if loop_lines is None:
                        return
                    code.writelines(loop_lines)
                    stack.enter_context(code.indent())
                    if loop.inner:
                        gen_loops(loop.inner, loop.is_reduction())
                    else:
                        kernels = loop.get_kernels()
                        assert len(kernels) == 1
                        gen_kernel(kernels[0])
            stack.enter_context(code.indent())
            if loop_nest.root:
                gen_loops(loop_nest.root)
            else:
                gen_kernel(loop_nest.kernel)

    def codegen_loops(self, code, worksharing):
        if False:
            return 10
        loop_nest = LoopNestWithSplit.build(self)
        self.codegen_loops_impl(loop_nest, code, worksharing)

    @property
    def assert_function(self):
        if False:
            return 10
        return 'TORCH_CHECK'

    def decide_parallel_depth(self, ranges, threads):
        if False:
            for i in range(10):
                print('nop')
        seq = self.size_hint()
        par = 1
        depth = 0
        for expr in ranges:
            hint = V.graph.sizevars.size_hint(expr, fallback=8192)
            if par >= 2 * threads or par == threads:
                break
            if seq // threads < config.cpp.min_chunk_size:
                break
            depth += 1
            par *= hint
            seq /= hint
        if config.cpp.dynamic_threads and depth == 0 and (len(ranges) > 0):
            depth = 1
        return depth

    @contextlib.contextmanager
    def write_to_suffix(self):
        if False:
            i = 10
            return i + 15
        prior = (self.loads, self.compute, self.stores, self.cse)
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.cse = self.cse.clone()
        yield
        self.reduction_suffix.splice(self.loads)
        self.reduction_suffix.splice(self.compute)
        self.reduction_suffix.splice(self.stores)
        (self.loads, self.compute, self.stores, self.cse) = prior

class CppVecKernel(CppKernel):
    overrides = CppVecOverrides

    def __init__(self, args, num_threads, tiling_factor=0, tiling_idx=-1, tiling_dtype=torch.float):
        if False:
            i = 10
            return i + 15
        super().__init__(args, num_threads)
        assert codecache.pick_vec_isa()
        if tiling_factor == 0:
            tiling_factor = codecache.pick_vec_isa().nelements(dtype=tiling_dtype)
        self.tiling_factor = tiling_factor
        self.tiling_idx = tiling_idx
        metrics.generated_cpp_vec_kernel_count += 1

    def load(self, name: str, index: sympy.Expr):
        if False:
            while True:
                i = 10
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.input(name)
        index = self.rename_indexing(index)
        dtype = V.graph.get_dtype(name)
        tiling_var = self.itervars[self.tiling_idx]
        is_broadcast = not index.has(tiling_var)
        is_mask = dtype in [torch.bool, torch.uint8] and (not opt_ctx.is_load_uint8_as_float)
        load_mask = f'to_float_mask({self._load_mask})' if self._load_mask else None
        non_contiguous = not is_broadcast and stride_at(tiling_var, index) != 1 or 'tmp' in f'{index}'
        var_expr = f'{var}[{cexpr_index(index)}]' if is_broadcast else f'{var} + {cexpr_index(index)}'
        loadbuf = 'tmpbuf' if non_contiguous else var_expr
        if is_broadcast:
            if is_mask:
                loadbuf = f'flag_to_float_scalar({loadbuf})'
            if dtype in DTYPE_LOWP_FP:
                line = f'at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>({loadbuf})'
            else:
                line = f'at::vec::Vectorized<float>(static_cast<float>({loadbuf}))'
        elif dtype in [torch.uint8] and opt_ctx.is_load_uint8_as_float:
            line = f'masked_load({loadbuf}, {load_mask})' if load_mask else f'at::vec::Vectorized<uint8_t>::loadu_one_fourth({loadbuf})'
        elif is_mask:
            line = f'flag_to_float_vec({loadbuf})'
        elif dtype in DTYPE_LOWP_FP:
            line = f'masked_load({loadbuf}, {load_mask})' if load_mask else f'at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>::loadu({loadbuf}, {self.tiling_factor})'
        else:
            line = f'masked_load({loadbuf}, {load_mask})' if load_mask else f'at::vec::Vectorized<float>::loadu({loadbuf})'
        if non_contiguous:
            tmpbuftype = 'float' if is_mask else f'{DTYPE_TO_CPP[dtype]}'
            tmpbufsize = f'{self.tiling_factor}'
            if dtype in DTYPE_LOWP_FP:
                tmpbufsize += ' * 2'
            tmpbufdeclare = f'__at_align__ {tmpbuftype} tmpbuf[{tmpbufsize}];'
            inner = sympy_symbol(f'{tiling_var}_inner')
            new_index = self.scale_index_with_offset(index, itervar_idx=self.tiling_idx, offset=inner)
            tmpbufdefine = f'for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++) '
            rhs = f'{var}[{cexpr_index(new_index)}]'
            if is_mask:
                rhs = f'flag_to_float_scalar({rhs})'
            tmpbufdefine += f'tmpbuf[{inner}] = {rhs};'
            line = f'([&]() {{ {tmpbufdeclare} {tmpbufdefine} return {line}; }})()'
        return self.cse.generate(self.loads, line)

    def get_vec_store_line(self, value, var, index, dtype):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a store line str that stores `value` into `var` at `index` of `dtype`.\n        :param value: Vectorized type templaterized on `dtype`.\n        :param var: buffer to store into.\n        :index: index into the `var`.\n        '
        tiling_var = self.itervars[self.tiling_idx]
        assert index.has(tiling_var)
        var_expr = f'{var} + {cexpr_index(index)}'
        non_contiguous = stride_at(tiling_var, index) != 1 or 'tmp' in f'{index}'
        if non_contiguous:
            var_expr = 'tmpbuf'
        if dtype == torch.float:
            line = f'{value}.store({var_expr});'
        else:
            line = f'{value}.store({var_expr}, {self.tiling_factor});'
        if non_contiguous:
            inner = sympy_symbol(f'{tiling_var}_inner')
            new_index = self.scale_index_with_offset(index, itervar_idx=self.tiling_idx, offset=inner)
            tmp_bufsize = f'{self.tiling_factor}*sizeof(float)/sizeof({DTYPE_TO_CPP[dtype]})'
            line = f'{{ __at_align__ {DTYPE_TO_CPP[dtype]} tmpbuf[{tmp_bufsize}]; {line} for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++) {var}[{cexpr_index(new_index)}] = tmpbuf[{inner}]; }}'
        return line

    def store(self, name, index, value, mode=None):
        if False:
            while True:
                i = 10
        assert 'buf' in name
        assert mode is None
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.output(name)
        index = self.rename_indexing(index)
        self.stores.writeline(DeferredLine(name, self.get_vec_store_line(value, var, index, V.graph.get_dtype(name))))

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if False:
            return 10
        assert reduction_type in {'max', 'min', 'sum', 'prod', 'xor_sum', 'welford_reduce', 'welford_combine'}
        assert dtype == torch.float
        assert src_dtype == torch.float
        vec_ns = 'at::vec'
        vec = f'{vec_ns}::Vectorized<{DTYPE_TO_CPP[dtype]}>'
        acc_type = reduction_acc_type(reduction_type, dtype)
        acc_type_vec = reduction_acc_type_vec(reduction_type, dtype)
        if (reduction_type, acc_type) not in self.reduction_omp_dec:
            if RTYPE_TO_CPP[reduction_type] not in NATIVE_OMP_RTYPES:
                self.reduction_prefix.splice(f"#pragma omp declare reduction({RTYPE_TO_CPP[reduction_type]}:{acc_type}:omp_out = {reduction_combine(reduction_type, 'omp_out', 'omp_in')}) initializer(omp_priv={{{reduction_init(reduction_type, dtype)}}})\n            ")
            self.reduction_omp_dec[reduction_type, acc_type] = RTYPE_TO_CPP[reduction_type]
        if (reduction_type, acc_type_vec) not in self.reduction_omp_dec:
            self.reduction_prefix.splice(f"#pragma omp declare reduction({RTYPE_TO_CPP[reduction_type]}:{acc_type_vec}:omp_out = {reduction_combine_vec(reduction_type, 'omp_out', 'omp_in')}) initializer(omp_priv={{{reduction_init_vec(reduction_type, dtype)}}})\n            ")
            self.reduction_omp_dec[reduction_type, acc_type_vec] = RTYPE_TO_CPP[reduction_type]
        reduction_key = (src_dtype, reduction_type, value)
        if reduction_key in self.reduction_cse.reduction_cache:
            return self.reduction_cse.reduction_cache[reduction_key]
        acc = self.reduction_cse.generate(self.loads, f'reduction {reduction_key}', write=False)
        acc_vec = f'{acc}_vec'
        self.reduction_var_map[acc_vec] = reduction_type
        self.reduction_prefix.writeline(f'{acc_type} {acc} = {reduction_init(reduction_type, dtype)};')
        self.reduction_prefix.writeline(f'{acc_type_vec} {acc_vec} = {reduction_init_vec(reduction_type, dtype)};')
        self.stores.writeline(f'{acc_vec} = {reduction_combine_vec(reduction_type, acc_vec, value)};')
        tmpvar: Union[str, CSEVariable]
        if self.tiling_idx >= self.reduction_depth:
            if is_welford_reduction(reduction_type):
                next_value = f'welford_vec_reduce_all({acc_vec})'
            else:
                reduce_all_body = '{ return ' + reduction_combine_vec(reduction_type, 'x', 'y') + '; }'
                vec_reduce_all_func = f'{vec_ns}::vec_reduce_all<{DTYPE_TO_CPP[dtype]}>'
                next_value = f'{vec_reduce_all_func}([]({vec}& x, {vec}& y) {reduce_all_body}, {acc_vec})'
            self.reduction_suffix.writeline(f'{acc} = {reduction_combine(reduction_type, acc, next_value)};')
            tmpvar = acc
        else:
            tmpvar = acc_vec
        result = reduction_project(reduction_type, tmpvar)
        self.reduction_cse.reduction_cache[reduction_key] = result
        return result

    def store_reduction(self, name, index, value):
        if False:
            while True:
                i = 10
        index = self.rename_indexing(index)
        var = self.args.output(name)
        out_dtype = V.graph.get_dtype(name)
        dtype = torch.float
        if self.tiling_idx >= self.reduction_depth:
            self.reduction_suffix.writeline(DeferredLine(name, f'{var}[{cexpr_index(index)}] = static_cast<{DTYPE_TO_CPP[out_dtype]}>({value});'))
        else:
            store_lines = []
            if out_dtype != dtype:
                if out_dtype in DTYPE_LOWP_FP and dtype == torch.float:
                    _lowp_fp_tmpvar_vec = f'{DTYPE_TO_CPP[out_dtype]}_{value}'
                    store_lines = [DeferredLine(name, f'auto {_lowp_fp_tmpvar_vec} = cvt_fp32_to_lowp_fp<{DTYPE_TO_CPP[out_dtype]}>({value});')]
                    value = _lowp_fp_tmpvar_vec
                else:
                    raise AssertionError(f'Unsupported reduction type from {dtype} to {out_dtype}')
            store_lines += [DeferredLine(name, self.get_vec_store_line(value, var, index, out_dtype))]
            self.reduction_suffix.writelines(store_lines)

class CppTile2DKernel(CppVecKernel):
    """
    A vector kernel that handles the 2d tiles with the tile size defined in `tiling_factor` on
    the inner-most loop level and one of the outer loop level (`outer_tiling_idx`). When the data
    tile is accessed in a contiguous way from the outer loop axis, a transposition is applied on the
    tile to make the access contiguous from the inner-most loop axis. Then, the same vectorization
    logic from its parent `CppVecKernel` is leveraged for load/store/compute. The transposed tile load
    and store are generated into kernel.preloads and kernel.poststores buffers.

    The loop structure looks like below:
    for ...
      for i_outer ...
        for ...
          for inner_most ...
            // generated by CppTile2DKernel
            float tmp0[16*16]; at::vec::transpose_mxn<...>(tmp0, in_ptr0 + ..., ...); // into kernel.preloads
            float tmp1[16*16]; // into kernel.preloads
            for i_inner ... { // the kernel inner loop
              vectorized loads/compute/stores (e.g., load tmp0, store tmp1) // into kernel.loads/compute/stores
            }
            at::vec::transpose_mxn(out_ptr0 + ..., tmp1, ...) // into kernel.poststores
          for inner_most ... (tail)
            // generated by CppVecKernel
            ...
      for i_outer ... (tail)
        for ...
          for ...
            // generated by CppKernel
            ...
    """

    def __init__(self, args, num_threads, tiling_factor, tiling_indices, tiling_dtype):
        if False:
            i = 10
            return i + 15
        super().__init__(args, num_threads, tiling_factor, tiling_indices[1], tiling_dtype)
        self.tiling_indices = tiling_indices

    def inner_itervar(self):
        if False:
            print('Hello World!')
        return sympy_symbol(f'{self.itervars[self.outer_idx]}_inner')

    def need_vec_transpose(self, index):
        if False:
            for i in range(10):
                print('nop')
        return stride_at(self.itervars[self.outer_idx], index) == 1 and index.has(self.itervars[self.tiling_idx]) and (not stride_at(self.itervars[self.tiling_idx], index).has(self.itervars[self.tiling_idx])) and (not stride_at(self.itervars[self.tiling_idx], index).has(self.itervars[self.outer_idx]))

    def gen_transposed_tile_load_store(self, name, var, index, is_store):
        if False:
            while True:
                i = 10
        dtype = V.graph.get_dtype(name)
        factor = self.tiling_factor
        src = f'{var} + {cexpr_index(index)}'
        dst = '__place_holder__'
        ld_src = f'{cexpr_index(stride_at(self.itervars[self.tiling_idx], index))}'
        ld_dst = f'{factor}'
        if is_store:
            (src, dst) = (dst, src)
            (ld_src, ld_dst) = (ld_dst, ld_src)
        need_define = True
        load_or_store = f'at::vec::transpose_mxn<{DTYPE_TO_CPP[dtype]},{factor},{factor}>({src}, {ld_src}, {dst}, {ld_dst});'
        if is_store:
            tile_var = self.cse.newvar()
        elif load_or_store not in self.cse.cache:
            tile_var = self.cse.generate(self.preloads, load_or_store, write=False)
        else:
            need_define = False
            tile_var = self.cse.cache[load_or_store]
        if need_define:
            define_line = f'{DTYPE_TO_CPP[dtype]} {tile_var}[{factor}*{factor}] __attribute__ ((aligned ({factor})));'
            self.preloads.writeline(define_line)
        load_or_store = load_or_store.replace('__place_holder__', str(tile_var))
        if is_store:
            self.poststores.writeline(DeferredLine(name, load_or_store))
        else:
            self.preloads.writeline(load_or_store)
        return tile_var

    def load(self, name: str, index: sympy.Expr):
        if False:
            i = 10
            return i + 15
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.input(name)
        index = self.rename_indexing(index)
        inner = self.inner_itervar()
        if self.need_vec_transpose(index):
            tile_var = self.gen_transposed_tile_load_store(name, var, index, is_store=False)
            loadbuf = f'{tile_var} + {cexpr_index(inner * self.tiling_factor)}'
            dtype = V.graph.get_dtype(name)
            if dtype in DTYPE_LOWP_FP:
                line = f'at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>::loadu({loadbuf}, {self.tiling_factor})'
            elif V.graph.get_dtype(name) in [torch.uint8] and opt_ctx.is_load_uint8_as_float:
                line = f'at::vec::Vectorized<uint8_t>::loadu_one_fourth({loadbuf})'
            else:
                line = f'at::vec::Vectorized<float>::loadu({loadbuf})'
            return self.cse.generate(self.loads, line)
        else:
            new_index = self.scale_index_with_offset(index, itervar_idx=self.outer_idx, offset=inner)
            return super().load(name, new_index)

    def store(self, name, index, value, mode=None):
        if False:
            return 10
        assert 'buf' in name
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.output(name)
        inner = self.inner_itervar()
        index = self.rename_indexing(index)
        assert mode is None
        if self.need_vec_transpose(index):
            tile_var = self.gen_transposed_tile_load_store(name, var, index, is_store=True)
            storebuf = f'{tile_var} + {cexpr_index(inner * self.tiling_factor)}'
            if V.graph.get_dtype(name) in DTYPE_LOWP_FP:
                line = f'{value}.store({storebuf}, {self.tiling_factor});'
            elif V.graph.get_dtype(name) in [torch.uint8]:
                line = f'{value}.store({storebuf}, {self.tiling_factor});'
            else:
                line = f'{value}.store({storebuf});'
            self.stores.writeline(DeferredLine(name, line))
        else:
            new_index = self.scale_index_with_offset(index, itervar_idx=self.outer_idx, offset=inner)
            super().store(name, new_index, value, mode)

    def codegen_inner_loops(self, code):
        if False:
            while True:
                i = 10
        inner = self.inner_itervar()
        code.writeline(f'for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++)')

    def set_ranges(self, group, reduction_group):
        if False:
            while True:
                i = 10
        vars = super().set_ranges(group, reduction_group)
        (self.outer_idx, self.tiling_idx) = self.tiling_indices if self.tiling_indices[1] < self.reduction_depth else reversed(self.tiling_indices)
        return vars

class CppVecKernelChecker(CppVecKernel):

    def __init__(self, args, num_threads, tiling_factor, tiling_idx=-1):
        if False:
            i = 10
            return i + 15
        super().__init__(args, num_threads, tiling_factor, tiling_idx)
        metrics.generated_kernel_count -= 1
        metrics.generated_cpp_vec_kernel_count -= 1
        self._orig_wrapper_code = None
        self.simd_vec = True
        self.fast_vec_list = []
        for (k, v) in CppVecOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                self.fast_vec_list.append(k)
        self.exit_stack = contextlib.ExitStack()
        self.load_supported_dtypes: List[torch.dtype] = [torch.float, torch.bfloat16, torch.float16, torch.bool, torch.uint8]
        self.store_supported_dtypes: List[torch.dtype] = [torch.float, torch.bfloat16, torch.float16, torch.uint8]
        self.store_dtypes: List[torch.dtype] = []
        self.vec_dtype: torch.dtype = torch.float32

    def disable_vec(self, msg=None):
        if False:
            while True:
                i = 10
        if schedule_log.isEnabledFor(logging.DEBUG):
            schedule_log.debug('Disabled vectorization: %s', msg)
        self.simd_vec = False

    def could_vec(self, name: str, index: sympy.Expr):
        if False:
            while True:
                i = 10
        assert self.itervars is not None
        return len(self.itervars) > 0

    def is_mask(self, name: str, users: Dict[torch.fx.Node, None]):
        if False:
            return 10
        load_type = V.graph.get_dtype(name)
        if load_type == torch.bool:
            return all((user.target in ('where', 'masked') for user in users.keys()))
        elif load_type == torch.uint8:
            '\n            If the load value is torch.uint8, then we only support the loaded\n            value is as the mask.\n            '
            if not all((user.target == 'to_dtype' and user.args[-1] == torch.bool for user in users.keys())):
                return False
            for to_dtype_node in users.keys():
                assert to_dtype_node.target == 'to_dtype'
                if not all((user.target in ('where', 'masked') for user in to_dtype_node.users.keys())):
                    return False
            return True
        else:
            return False

    def is_load_uint8_as_float(self, name: str, users: Dict[torch.fx.Node, None]):
        if False:
            print('Hello World!')
        '\n        Check:\n        1. load_type is torch.uint8\n        2. has 1 user node of target to_dtype\n        3. dtype of to_dtype is torch.float\n        '
        load_type = V.graph.get_dtype(name)
        if load_type is not torch.uint8:
            return False
        if len(users) == 1:
            user = next(iter(users))
            if user.target == 'to_dtype' and user.args[-1] == torch.float:
                return True
            return False
        return False

    def can_store_fp32_as_uint8(self, store_var: str, value_node: torch.fx.Node):
        if False:
            return 10
        '\n        Check:\n        1. store_type is torch.uint8\n        2. value_node is of target to_dtype\n        3. dtype of to_dtype node is torch.uint8\n        '
        store_type = V.graph.get_dtype(store_var)
        if store_type not in [torch.uint8]:
            return False
        if value_node.target == 'to_dtype' and value_node.args[-1] == torch.uint8:
            return True
        return False

    def is_load_integer_scalar_tensor(self, name: str, index: sympy.Expr):
        if False:
            return 10
        load_dtype = V.graph.get_dtype(name)
        buffer = V.graph.get_buffer(name)
        return load_dtype in [torch.int32, torch.int64] and isinstance(buffer, TensorBox) and isinstance(buffer.data, StorageBox) and (len(buffer.data.layout.size) == 0) and (index == 0)

    def load(self, name: str, index: sympy.Expr):
        if False:
            return 10
        with RecordOptimizationContext(__name__) as node_ctx:
            load_dtype = V.graph.get_dtype(name)
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            opt_ctx.dtype = load_dtype
            opt_ctx.is_load_as_mask = self.is_mask(name, node_ctx.get_fx_node().users)
            opt_ctx.is_load_uint8_as_float = self.is_load_uint8_as_float(name, node_ctx.get_fx_node().users)
            var = self.cse.newvar()
            if load_dtype in [torch.bool, torch.uint8] and (not (opt_ctx.is_load_as_mask or opt_ctx.is_load_uint8_as_float)):
                if not opt_ctx.is_load_as_mask:
                    self.disable_vec(f'{load_dtype} not loaded as mask')
                elif not opt_ctx.is_load_uint8_as_float:
                    self.disable_vec(f'{load_dtype} not loaded as float')
                return var
            if load_dtype not in self.load_supported_dtypes and (not self.is_load_integer_scalar_tensor(name, index)):
                self.disable_vec(f'{load_dtype} not supported by load')
                return var
            index = self.rename_indexing(index)
            if self.simd_vec and (not self.could_vec(name, index)):
                self.disable_vec(f'not a loop: {index}')
            return var

    def store(self, name, index, value, mode=None):
        if False:
            print('Hello World!')
        with RecordOptimizationContext(__name__) as node_ctx:
            store_dtype = V.graph.get_dtype(name)
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            opt_ctx.dtype = store_dtype
            store_dtype = torch.float if store_dtype == torch.float32 else store_dtype
            self.store_dtypes.append(store_dtype)
            if store_dtype not in self.store_supported_dtypes:
                self.disable_vec(f'{store_dtype} not supported by store')
                return self.simd_vec
            if store_dtype in [torch.uint8]:
                value_node = node_ctx.get_fx_node().all_input_nodes[-1]
                if not self.can_store_fp32_as_uint8(name, value_node):
                    self.disable_vec('not support store float32 as uint8')
                    return self.simd_vec
            assert 'buf' in name
            index = self.rename_indexing(index)
            if mode:
                self.disable_vec(f'store mode: {mode}')
                return self.simd_vec
            if index.is_number:
                self.disable_vec(f'constant store index: {index}')
            if self.simd_vec and (not self.could_vec(name, index)):
                self.disable_vec(f'not a loop: {index}')
            return self.simd_vec

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if False:
            return 10
        if dtype == torch.float and src_dtype == torch.float and (reduction_type in VECTORIZABLE_RTYPES):
            pass
        else:
            self.disable_vec(f'reduction: dtype {dtype}, src_dtype {src_dtype}, reduction_type {reduction_type}')
        if is_welford_reduction(reduction_type):
            return tuple([self.simd_vec] * 3)
        return self.simd_vec

    def store_reduction(self, name, index, value):
        if False:
            print('Hello World!')
        return self.simd_vec

    def is_supported_cmp(self, node: torch.fx.Node):
        if False:
            while True:
                i = 10

        def get_node_dtype(node):
            if False:
                i = 10
                return i + 15
            if type(node) == torch.fx.Node:
                opt_ctx: OptimizationContext = get_current_node_opt_ctx()
                return opt_ctx.dtype if opt_ctx else None
            else:
                return None

        def get_cmp_dtypes(node: torch.fx.Node):
            if False:
                for i in range(10):
                    print('nop')
            return (get_node_dtype(node.args[-2]), get_node_dtype(node.args[-1]))
        assert len(node.args) >= 2
        if type(node.args[-1]) in [int, float]:
            return True
        if type(node.args[-2]) in [int, float]:
            return False
        (left_dtype, right_dtype) = get_cmp_dtypes(node)
        if left_dtype is None or right_dtype is None:
            return True
        else:
            return left_dtype == right_dtype

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        assert self._orig_wrapper_code is not None
        V.graph.wrapper_code = self._orig_wrapper_code
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self._orig_wrapper_code = V.graph.wrapper_code
        V.graph.wrapper_code = WrapperCodeGen()

        class VecCheckerProxy:
            bin_cmp_ops = ['eq', 'ne', 'le', 'ge', 'lt', 'gt']

            @staticmethod
            def _bin_cmp_op(x, y):
                if False:
                    return 10
                current_node: torch.fx.Node = V.interpreter.current_node
                if not self.is_supported_cmp(current_node):
                    self.disable_vec(f'binary comparison op: {current_node}')
                return self.simd_vec

            @staticmethod
            def __getattr__(name):
                if False:
                    print('Hello World!')

                def inner(*args, **kwargs):
                    if False:
                        return 10
                    if name in VecCheckerProxy.bin_cmp_ops:
                        return VecCheckerProxy._bin_cmp_op(args, kwargs)
                    if name not in self.fast_vec_list:
                        self.disable_vec(f'op: {name}')
                    return self.simd_vec
                return inner

            @staticmethod
            def load(name: str, index: sympy.Expr):
                if False:
                    while True:
                        i = 10
                return self.load(name, index)

            @staticmethod
            def store(name, index, value, mode=None):
                if False:
                    return 10
                return self.store(name, index, value, mode=mode)

            @staticmethod
            def reduction(dtype, src_dtype, reduction_type, value):
                if False:
                    i = 10
                    return i + 15
                return self.reduction(dtype, src_dtype, reduction_type, value)

            @staticmethod
            def store_reduction(name, index, value):
                if False:
                    return 10
                return self.store_reduction(name, index, value)

            @staticmethod
            def constant(val, dtype):
                if False:
                    while True:
                        i = 10
                with RecordOptimizationContext(__name__) as node_ctx:
                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    i32_iinfo = torch.iinfo(torch.int32)
                    if dtype == torch.int64 and val <= i32_iinfo.max and (val >= i32_iinfo.min):
                        opt_ctx.dtype = torch.int32
                    f32_iinfo = torch.finfo(torch.float32)
                    if dtype == torch.double:
                        if val <= f32_iinfo.max and val >= f32_iinfo.min or val == torch.inf or val == -torch.inf:
                            opt_ctx.dtype = torch.float32
                    supported_dtypes = [torch.float32, torch.int32, torch.bfloat16, torch.float16]
                    if opt_ctx.dtype not in supported_dtypes or (opt_ctx.dtype == torch.int32 and (not all((user.target in VecCheckerProxy.bin_cmp_ops for user in node_ctx.current_node.users)))):
                        self.disable_vec(f'constant dtype: {opt_ctx.dtype}')
                    return val

            @staticmethod
            def index_expr(expr, dtype):
                if False:
                    print('Hello World!')
                assert len(self.ranges) == len(self.itervars)
                if not len(self.ranges) or not all((not isinstance(range, sympy.Expr) or sympy.simplify(range).is_number for range in self.ranges)):
                    self.disable_vec(f'index_expr: {expr}, dtype {dtype}')
                    return self.cse.newvar()

                def can_use_int32():
                    if False:
                        print('Hello World!')
                    free_symbols = list(expr.free_symbols)
                    sizes = {k: v for (k, v) in zip(self.itervars, self.ranges) if k in free_symbols}
                    if any((v == 0 for v in sizes.values())):
                        return True
                    vars_ranges = {k: ValueRanges(0, v - 1) for (k, v) in sizes.items()}
                    if not vars_ranges or len(vars_ranges) != len(free_symbols):
                        i32_iinfo = torch.iinfo(torch.int32)
                        return expr.is_number and expr <= i32_iinfo.max and (expr >= i32_iinfo.min)
                    expr_ranges = bound_sympy(expr, vars_ranges)
                    if math.isinf(expr_ranges.lower) or math.isinf(expr_ranges.upper):
                        return False
                    return range_expressable_in_32_bits(ValueRanges(int(expr_ranges.lower), int(expr_ranges.upper) + 1))
                with RecordOptimizationContext(__name__) as node_ctx:
                    assert len(self.ranges) == len(self.itervars)
                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    if dtype == torch.int64 and can_use_int32() and all((user.target in VecCheckerProxy.bin_cmp_ops for user in node_ctx.current_node.users)):
                        opt_ctx.dtype = torch.int32
                    else:
                        opt_ctx.dtype = dtype
                        self.disable_vec(f'index_expr: {expr}, dtype {dtype}')
                    tiling_var = self.itervars[self.tiling_idx]
                    tiling_var_irrelevant = not expr.has(tiling_var)
                    if not tiling_var_irrelevant:
                        self.disable_vec(f'index_expr (tiling var relevant): {expr}, dtype {dtype}')
                    opt_ctx.is_most_inner_loop_irrevelant = tiling_var_irrelevant
                    tmp_var = self.cse.newvar()
                    return tmp_var

            @staticmethod
            def indirect_indexing(index_var, size, check=True):
                if False:
                    print('Hello World!')
                return sympy_symbol(str(index_var))

            @staticmethod
            def masked(mask, body, other):
                if False:
                    for i in range(10):
                        print('nop')
                body()
                return self.cse.newvar()

            @staticmethod
            def to_dtype(x, dtype, src_dtype=None):
                if False:
                    i = 10
                    return i + 15
                with RecordOptimizationContext(__name__) as node_ctx:
                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    opt_ctx.dtype = dtype
                    cur_node = node_ctx.get_fx_node()
                    input_value: torch.fx.Node = cur_node.all_input_nodes[1]
                    if dtype == torch.float:
                        if input_value.target in ['load']:
                            dtype = V.graph.get_dtype(input_value.args[1]) if input_value.target == 'load' else input_value.args[-1]
                            if dtype in [torch.float16, torch.bfloat16, torch.float, torch.uint8]:
                                pass
                            elif dtype in [torch.int32, torch.int64] and input_value.target == 'load':
                                buffer = V.graph.get_buffer(input_value.args[1])
                                if not (isinstance(buffer, TensorBox) and isinstance(buffer.data, StorageBox) and (len(buffer.data.layout.size) == 0)):
                                    self.disable_vec(f'to_dtype: dtype {dtype}')
                            else:
                                self.disable_vec(f'to_dtype: dtype {dtype}')
                    elif dtype in DTYPE_LOWP_FP:
                        if not all((usr.target == 'store' for usr in cur_node.users)):
                            self.disable_vec('to_dtype: bfloat16/float16 expecting users are all stores')
                            return x
                        store_names = [usr.args[1] for usr in cur_node.users]
                        if not all((V.graph.get_dtype(name) in [dtype] for name in store_names)):
                            self.disable_vec('to_dtype: expecting all stores into bfloat16 or float16')
                            return x
                    elif dtype == torch.bool:
                        pass
                    elif dtype == torch.uint8:
                        is_to_uint8_and_store = all((usr.target in ['store'] for usr in cur_node.users))
                        is_to_uint8_and_to_float = all((usr.target in ['to_dtype'] and usr.args[2] == torch.float32 for usr in cur_node.users))
                        if not (is_to_uint8_and_store or is_to_uint8_and_to_float):
                            self.disable_vec(f'to_dtype: dtype {dtype}')
                    else:
                        self.disable_vec(f'to_dtype: dtype {dtype}')
                    return x
        self.exit_stack.enter_context(V.set_ops_handler(VecCheckerProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

class CppKernelProxy(CppKernel):

    def __init__(self, kernel_group):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.kernel_group = kernel_group
        self.loop_nest = None
        self.call_ranges = None
        self.picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()

    def data_type_propagation(self, nodes):
        if False:
            i = 10
            return i + 15
        for _node in nodes:
            assert isinstance(_node, SchedulerNode)
            DataTypePropagation.propagate_scheduler_node(_node)

    def is_lowp_fp_scheduler(self, scheduler_node: SchedulerNode):
        if False:
            i = 10
            return i + 15
        if not isinstance(scheduler_node._body, ir.LoopBody):
            return True
        _lowp_fp_type: Optional[torch.dtype] = None
        DataTypePropagation.propagate_scheduler_node(scheduler_node)
        sub_blocks = [scheduler_node._body.root_block] + list(scheduler_node._body.subblocks.values())
        for sub_block in sub_blocks:
            for _node in sub_block.graph.nodes:
                if _node.op == 'placeholder' or _node.target in ('get_index', 'index_expr'):
                    continue
                if _node.target not in ['load', 'store', 'abs', 'neg', 'output']:
                    return False
                if hasattr(_node, 'meta') and _node.meta:
                    assert OptimizationContext.key in _node.meta
                    opt_ctx: OptimizationContext = _node.meta[OptimizationContext.key]
                    if not opt_ctx.dtype or opt_ctx.dtype not in DTYPE_LOWP_FP:
                        return False
                    if _lowp_fp_type:
                        assert _lowp_fp_type == opt_ctx.dtype, 'scheduler node do not support bf16/fp16 mix'
                    else:
                        _lowp_fp_type = opt_ctx.dtype
                else:
                    return False
        scheduler_node._lowp_fp_type = _lowp_fp_type
        return True

    def legalize_lowp_fp_dtype(self, nodes):
        if False:
            while True:
                i = 10

        def add_to_dtype(sub_graph: torch.fx.Graph):
            if False:
                i = 10
                return i + 15

            def is_lowp_fp_load(node: torch.fx.Node):
                if False:
                    for i in range(10):
                        print('nop')
                if node.target not in ['load']:
                    return False
                assert len(node.args) == 3
                load_dtype = V.graph.get_dtype(node.args[1])
                return load_dtype in DTYPE_LOWP_FP

            def is_lowp_fp_store(node: torch.fx.Node):
                if False:
                    for i in range(10):
                        print('nop')
                if node.target != 'store':
                    return False
                (_, store_var, _, _, _) = node.args
                store_dtype = V.graph.get_dtype(store_var)
                return store_dtype in DTYPE_LOWP_FP
            sub_graph_nodes = list(sub_graph.nodes)
            to_lowp_fp_legalized_nodes = []
            for _node in sub_graph_nodes:
                if is_lowp_fp_load(_node):
                    ops = _node.args[0]
                    with sub_graph.inserting_after(_node):
                        to_type_node = sub_graph.call_method('to_dtype', args=(ops, _node, torch.float))
                        to_type_node_args = to_type_node.args
                        _node.replace_all_uses_with(to_type_node)
                        to_type_node.args = to_type_node_args
                        metrics.cpp_to_dtype_count += 1
                elif is_lowp_fp_store(_node):
                    (ops, name, _, value_var, _) = _node.args
                    dtype = V.graph.get_dtype(name)
                    with sub_graph.inserting_before(_node):
                        to_type_node = sub_graph.call_method('to_dtype', args=(ops, value_var, dtype))
                        _node.replace_input_with(value_var, to_type_node)
                        metrics.cpp_to_dtype_count += 1
                elif _node.target == 'reduction':
                    (ops, dtype, src_dtype, reduction_type, value) = _node.args
                    if src_dtype in DTYPE_LOWP_FP:
                        assert dtype in [torch.float, torch.bfloat16, torch.float16, torch.int64]
                        _node.args = (ops, torch.float if dtype in DTYPE_LOWP_FP else dtype, torch.float, reduction_type, value)
                elif _node.target == 'to_dtype' and _node.args[-1] in DTYPE_LOWP_FP:
                    (ops, x, _) = _node.args
                    to_lowp_fp_legalized_nodes.append(_node)
                    _node.args = (ops, x, torch.float)
                else:
                    pass

            def eliminate_to_dtype(sub_graph: torch.fx.Graph):
                if False:
                    for i in range(10):
                        print('nop')

                def _eliminate_duplicate_to_node(sub_graph: torch.fx.Graph):
                    if False:
                        i = 10
                        return i + 15

                    def _used_by_to(to_node: torch.fx.Node):
                        if False:
                            return 10
                        return all((usr.target == 'to_dtype' for usr in to_node.users))
                    all_to_nodes = [node for node in sub_graph.nodes if node.target == 'to_dtype']
                    all_to_nodes_and_users = [{node: node.users} for node in all_to_nodes if _used_by_to(node)]
                    for node_users in all_to_nodes_and_users:
                        for (node, users) in node_users.items():
                            if node in sub_graph.nodes and (all((usr.args[-1] == node.args[-1] for usr in users)) or (node in to_lowp_fp_legalized_nodes and all((usr.args[-1] in DTYPE_LOWP_FP for usr in users)))):
                                val_node = node.all_input_nodes[-1]
                                node.replace_all_uses_with(val_node)
                                sub_graph.erase_node(node)
                    if sub_graph.owning_module is None:
                        sub_graph.lint()
                _eliminate_duplicate_to_node(sub_graph)
            eliminate_to_dtype(sub_graph)

        def _legalize_lowp_fp(loop_body: ir.LoopBody):
            if False:
                while True:
                    i = 10
            sub_blocks = [loop_body.root_block] + list(loop_body.subblocks.values())
            for sub_block in sub_blocks:
                add_to_dtype(sub_block.graph)
        if all((isinstance(_node, SchedulerNode) and self.is_lowp_fp_scheduler(_node) for _node in nodes)):
            for _node in nodes:
                sub_blocks = [_node._body.root_block] + list(_node._body.subblocks.values())
                for sub_block in sub_blocks:
                    for fx_node in sub_block.graph.nodes:
                        if fx_node.target in ['load', 'store']:
                            assert fx_node.meta
                            assert OptimizationContext.key in fx_node.meta
                            opt_ctx: OptimizationContext = fx_node.meta[OptimizationContext.key]
                            assert opt_ctx.dtype in DTYPE_LOWP_FP
            return
        for _node in nodes:
            assert isinstance(_node, SchedulerNode)
            assert isinstance(_node._body, ir.LoopBody)
            node: SchedulerNode = _node

            def is_memory_copy_scheduler_node(node: SchedulerNode):
                if False:
                    print('Hello World!')
                op_counts = node.read_writes.op_counts
                return len(op_counts) == 2 and 'load' in op_counts and ('store' in op_counts)
            should_legalize = not is_memory_copy_scheduler_node(node)
            if should_legalize:
                body: ir.LoopBody = node._body
                _legalize_lowp_fp(body)

    def codegen_nodes(self, nodes):
        if False:
            while True:
                i = 10
        self.legalize_lowp_fp_dtype(nodes)
        self.data_type_propagation(nodes)
        assert len(nodes) >= 1
        first_node = nodes[0]
        vec_dtype = first_node._lowp_fp_type if all((hasattr(_node, '_lowp_fp_type') and _node._lowp_fp_type == first_node._lowp_fp_type for _node in nodes)) else torch.float
        kernel_group = self.kernel_group
        (_, (group, reduction_group)) = max(nodes, key=lambda x: int(x.is_reduction())).group
        self.set_ranges(group, reduction_group)

        def codegen_kernel(cls, *args):
            if False:
                i = 10
                return i + 15
            with kernel_group.new_kernel(cls, *args) as kernel:
                run(kernel)
                metrics.generated_kernel_count -= 1
                return kernel

        def run(kernel):
            if False:
                print('Hello World!')
            (vars, reduction_vars) = kernel.set_ranges(group, reduction_group)
            in_suffix = False
            for node in nodes:
                if node.group[1] in [(group, reduction_group), (group + reduction_group, ())]:
                    assert not in_suffix
                    node.run(vars, reduction_vars)
                else:
                    in_suffix = True
                    assert node.group[1] == (group, ()), f'unexpected group: {node.group[1]} != {group}, {reduction_group}'
                    with kernel.write_to_suffix():
                        node.run(vars, ())
        scalar_kernel = codegen_kernel(CppKernel)
        V.graph.removed_buffers |= scalar_kernel.removed_buffers
        V.graph.inplaced_to_remove |= scalar_kernel.inplaced_to_remove
        self.loop_nest = LoopNestWithSplit.build(scalar_kernel)
        if not self.picked_vec_isa:
            return

        def select_tiling_indices():
            if False:
                for i in range(10):
                    print('nop')
            all_index = []
            for node in nodes:
                rw = dependencies.extract_read_writes(node._body, *node._sizes)
                all_index += [dep.index for dep in itertools.chain(rw.reads, rw.writes)]
            contig_vars = set()
            contig_vars_list = []
            non_contig_stride_const = set()
            non_contig_stride_other = set()
            for index in all_index:
                for var in index.free_symbols:
                    if not re.search('^d\\d+$', var.name):
                        continue
                    stride = stride_at(var, index)
                    if stride == 1:
                        contig_vars.add(int(var.name[1:]))
                        contig_vars_list.append(int(var.name[1:]))
                    elif all((s.name.startswith('s') for s in stride.free_symbols)):
                        non_contig_stride_const.add(int(var.name[1:]))
                    else:
                        non_contig_stride_other.add(int(var.name[1:]))
            contig_only = contig_vars - non_contig_stride_const - non_contig_stride_other
            if len(contig_vars) == 0:
                return [len(self.itervars) - 1]
            if contig_only:
                return sorted(contig_only)[-1:]
            contig_and_const_stride = (contig_vars & non_contig_stride_const) - non_contig_stride_other
            contig_vars_sorted = sorted(contig_vars)
            if len(contig_vars_sorted) == 2 and contig_vars_sorted[-1] in contig_and_const_stride and (contig_vars_sorted[-1] == len(self.itervars) - 1):
                return contig_vars_sorted
            return sorted(contig_vars_sorted, key=contig_vars_list.count)[-1:]

        def select_tiling(dtype: torch.dtype=torch.float):
            if False:
                for i in range(10):
                    print('nop')
            tiling_factor = self.picked_vec_isa.nelements(dtype=dtype)
            tiling_indices = select_tiling_indices()
            if tiling_indices:
                could_vec = True
                for tiling_indice in tiling_indices:
                    with CppVecKernelChecker(deepcopy(self.kernel_group.args), parallel_num_threads(), tiling_factor, tiling_indice) as vec_checker:
                        run(vec_checker)
                        could_vec = could_vec and vec_checker.simd_vec
                        if not could_vec:
                            break
                if could_vec:
                    if len(tiling_indices) == 1:
                        return ([tiling_factor], tiling_indices)
                    if len(tiling_indices) == 2:
                        return ([tiling_factor, tiling_factor], tiling_indices)
            return ([], [])
        with torch._inductor.config.patch(inplace_buffers=False):
            (tiling_factors, tiling_indices) = select_tiling(vec_dtype)
            assert len(tiling_factors) == len(tiling_indices)
            if len(tiling_indices) == 1:
                (main_loop, tail_loop) = self.loop_nest.split_with_tiling(tiling_indices[0], factor=tiling_factors[0])
                main_loop.set_kernel(codegen_kernel(CppVecKernel, tiling_factors[0], tiling_indices[0], vec_dtype))
                tail_loop.set_kernel(scalar_kernel)
                main_loop.simd_vec = True
                tail_loop.simd_omp = True
                tail_loop.simd_nelements = tiling_factors[0] // 2
            elif len(tiling_indices) == 2:
                assert tiling_indices[1] == len(self.itervars) - 1 and tiling_factors[0] == tiling_factors[1]
                (outer_main_loop, outer_tail_loop) = self.loop_nest.split_with_tiling(tiling_indices[0], factor=tiling_factors[0])
                outer_tail_loop.set_kernel(scalar_kernel)
                (inner_main_loop, inner_tail_loop) = outer_main_loop.split_with_tiling(tiling_indices[1] - tiling_indices[0], factor=tiling_factors[0])
                inner_main_loop.set_kernel(codegen_kernel(CppTile2DKernel, tiling_factors[0], tiling_indices, vec_dtype))
                inner_tail_loop.set_kernel(codegen_kernel(CppVecKernel, tiling_factors[0], tiling_indices[0], vec_dtype))

    def codegen_loops(self, code, worksharing):
        if False:
            print('Hello World!')
        self.codegen_loops_impl(self.loop_nest, code, worksharing)

class CppScheduling(BaseScheduling):

    def __init__(self, scheduler):
        if False:
            for i in range(10):
                print('nop')
        self.scheduler = scheduler
        self.get_kernel_group()

    def group_fn(self, sizes):
        if False:
            i = 10
            return i + 15
        return tuple((tuple(map(V.graph.sizevars.simplify, s)) for s in sizes))

    def get_kernel_group(self):
        if False:
            for i in range(10):
                print('nop')
        from .wrapper import CppWrapperCodeGen
        self.kernel_group: Union[CppWrapperKernelGroup, KernelGroup]
        if isinstance(V.graph.wrapper_code, CppWrapperCodeGen):
            self.kernel_group = CppWrapperKernelGroup()
        else:
            self.kernel_group = KernelGroup()

    def _can_fuse_horizontal_impl(self, node1, node2):
        if False:
            i = 10
            return i + 15
        (_, (vars1, reduce1)) = node1.group
        (_, (vars2, reduce2)) = node2.group
        if vars1 == vars2 and reduce1 == reduce2:
            return True
        if reduce1 == () and vars1 == vars2 + reduce2:
            return True
        return False

    def can_fuse_horizontal(self, node1, node2):
        if False:
            return 10
        if len(node1.get_nodes()) + len(node2.get_nodes()) > config.cpp.max_horizontal_fusion_size:
            return False
        return self._can_fuse_horizontal_impl(node1, node2)

    def can_fuse_vertical(self, node1, node2):
        if False:
            i = 10
            return i + 15
        return self._can_fuse_horizontal_impl(node1, node2) and (not node1.is_reduction())

    def codegen_nodes(self, nodes):
        if False:
            i = 10
            return i + 15
        '\n        Turn an set of pre-fused nodes into a C++ kernel.\n        '
        kernel_group = self.kernel_group
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        cpp_kernel_proxy.codegen_nodes(nodes)
        kernel_group.finalize_kernel(cpp_kernel_proxy, nodes)

    def codegen_sync(self):
        if False:
            while True:
                i = 10
        pass

    def flush(self):
        if False:
            i = 10
            return i + 15
        self.kernel_group.codegen_define_and_call(V.graph.wrapper_code)
        self.get_kernel_group()

class KernelGroup:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.args = KernelArgs()
        self.loops_code = BracesBuffer()
        self.ws = WorkSharing(self.loops_code)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(self.ws)
        self.scheduled_nodes = []

    def new_kernel(self, cls, *args):
        if False:
            print('Hello World!')
        return cls(self.args, parallel_num_threads(), *args)

    def finalize_kernel(self, new_kernel, nodes):
        if False:
            for i in range(10):
                print('nop')
        self.scheduled_nodes += nodes
        code = self.loops_code
        ws = self.ws
        new_kernel.codegen_loops(code, ws)

    def codegen_define_and_call(self, wrapper):
        if False:
            for i in range(10):
                print('nop')
        self.stack.close()
        if not self.scheduled_nodes:
            return
        fused_name = get_fused_kernel_name(self.scheduled_nodes, config.cpp.descriptive_names) if config.cpp.descriptive_names else ''
        kernel_name = '_'.join(['cpp', fused_name, wrapper.next_kernel_suffix()])
        (arg_defs, call_args, arg_types) = self.args.cpp_argdefs()
        arg_defs = ',\n'.ljust(25).join(arg_defs)
        arg_types = ','.join(arg_types)
        code = BracesBuffer()
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform == 'linux'
        if enable_kernel_profile:
            code.writelines(['#include <ATen/record_function.h>'])
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else 'kernel'
        code.writeline(codecache.cpp_prefix())
        code.writeline(f'extern "C" void {kernel_decl_name}({arg_defs})')
        with code.indent():
            if enable_kernel_profile:
                graph_id = V.graph.graph_id
                prefix = 'graph_' + str(graph_id) + '_' if graph_id is not None else ''
                code.writelines([f'RECORD_FUNCTION("{prefix + kernel_name}", c10::ArrayRef<c10::IValue>({{}}));'])
            for (old, new) in self.args.aliases():
                code.writeline(f'auto {old} = {new};')
            code.splice(self.loops_code)
        codecache_def = IndentedBuffer()
        if not V.graph.cpp_wrapper:
            codecache_def.writeline("async_compile.cpp('''")
        codecache_def.splice(code)
        if not V.graph.cpp_wrapper:
            codecache_def.writeline("''')")
        codecache_str = codecache_def.getvalue()
        codecache_str = codecache_str.replace('#pragma CMT', '//')
        wrapper.define_kernel(kernel_name, codecache_str, cuda=False)
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)

class CppWrapperKernelGroup(KernelGroup):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.args = CppWrapperKernelArgs()

class WorkSharing:

    def __init__(self, code):
        if False:
            print('Hello World!')
        self.code = code
        self.in_parallel = False
        self.num_threads = None
        self.stack = contextlib.ExitStack()

    def parallel(self, threads):
        if False:
            for i in range(10):
                print('nop')
        if self.in_parallel and threads != self.num_threads:
            self.close()
        if not self.in_parallel:
            self.num_threads = threads
            self.in_parallel = True
            if config.cpp.dynamic_threads:
                self.code.writeline('#pragma omp parallel')
            else:
                self.code.writeline(f'#pragma omp parallel num_threads({threads})')
            self.stack.enter_context(self.code.indent())

    def single(self):
        if False:
            return 10
        if self.in_parallel:
            self.code.writeline('#pragma omp single')
        return self.in_parallel

    def close(self):
        if False:
            while True:
                i = 10
        self.stack.close()
        self.in_parallel = False

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        self.stack.__exit__(exc_type, exc_val, exc_tb)

@dataclasses.dataclass
class LoopLevel:
    var: Optional[sympy.Expr] = None
    size: Optional[sympy.Expr] = None
    offset: sympy.Expr = sympy.Integer(0)
    steps: sympy.Expr = sympy.Integer(1)
    parallel: int = 0
    simd_omp: bool = False
    simd_vec: bool = False
    collapsed: bool = False
    reduction_var_map: Optional[Dict[str, str]] = None
    parent: Optional['LoopLevel'] = None
    inner: List['LoopLevel'] = dataclasses.field(default_factory=list)
    kernel: Optional[CppKernel] = None

    def __post_init__(self):
        if False:
            return 10
        picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()
        self.simd_nelements: int = picked_vec_isa.nelements() if picked_vec_isa else 0

    def get_kernels(self) -> List[CppKernel]:
        if False:
            return 10
        'Get all kernel objects under this loop level'
        if self.kernel:
            return [self.kernel]
        kernels = []
        for loop in self.inner:
            kernels += loop.get_kernels()
        return kernels

    def set_kernel(self, kernel: CppKernel):
        if False:
            i = 10
            return i + 15
        '\n        Set the kernel under this loop level. No split is allowed under\n        this loop level.\n        '
        if not self.inner:
            self.kernel = kernel
            loop: Optional[LoopLevel] = self
            assert loop is not None
            if loop.is_reduction():
                loop.reduction_var_map = kernel.reduction_var_map.copy()
                loop = loop.parent
                while loop is not None and loop.is_reduction():
                    assert loop.reduction_var_map is not None
                    loop.reduction_var_map.update(kernel.reduction_var_map)
                    loop = loop.parent
            return
        assert len(self.inner) == 1
        self.inner[0].set_kernel(kernel)

    def get_loops_at(self, depth) -> List['LoopLevel']:
        if False:
            i = 10
            return i + 15
        if depth == 0:
            return [self]
        else:
            loops = []
            for loop in self.inner:
                loops += loop.get_loops_at(depth - 1)
            return loops

    def is_reduction(self):
        if False:
            i = 10
            return i + 15
        return bool(self.reduction_var_map)

    def split_with_tiling(self, depth, factor):
        if False:
            i = 10
            return i + 15

        def clone_inner():
            if False:
                print('Hello World!')
            inner = []
            if self.inner:
                for loop in self.inner:
                    inner.append(loop.clone())
            return inner

        def do_split_with_tiling():
            if False:
                return 10
            sympy_factor = sympy.Integer(factor)
            offset = FloorDiv(self.size, sympy_factor) * sympy_factor
            main_loop = LoopLevel(self.var, offset)
            main_loop.steps = sympy_factor
            main_loop.parallel = self.parallel
            main_loop.collapsed = False
            main_loop.reduction_var_map = self.reduction_var_map
            main_loop.inner = clone_inner()
            if main_loop.inner:
                for loop in main_loop.inner:
                    loop.parent = main_loop
            tail_loop = LoopLevel(self.var, self.size)
            tail_loop.offset = offset
            tail_loop.parallel = self.parallel
            tail_loop.collapsed = False
            tail_loop.reduction_var_map = self.reduction_var_map
            tail_loop.inner = clone_inner()
            if tail_loop.inner:
                for loop in tail_loop.inner:
                    loop.parent = tail_loop
            return (main_loop, tail_loop)
        if depth == 0:
            (main_loop, tail_loop) = do_split_with_tiling()
            parent = self.parent
            if parent:
                parent.inner = [main_loop, tail_loop]
                main_loop.parent = parent
                tail_loop.parent = parent
            return (main_loop, tail_loop)
        else:
            assert len(self.inner) == 1
            return self.inner[0].split_with_tiling(depth - 1, factor)

    def clone(self):
        if False:
            for i in range(10):
                print('nop')
        loop = copy(self)
        loop.inner = []
        if self.inner:
            for inner_loop in self.inner:
                inner_loop_clone = inner_loop.clone()
                inner_loop_clone.parent = loop
                loop.inner.append(inner_loop_clone)
        loop.kernel = deepcopy(self.kernel)
        return loop

    def lines(self):
        if False:
            while True:
                i = 10
        offset_expr = cexpr_index(self.offset)
        size_expr = cexpr_index(self.size)
        if config.cpp.no_redundant_loops and offset_expr == size_expr:
            return None
        if self.reduction_var_map:
            reduction = ' ' + ' '.join((f'reduction({RTYPE_TO_CPP[rtype]}:{var})' for (var, rtype) in self.reduction_var_map.items()))
        else:
            reduction = ''
        simd = f'simd simdlen({self.simd_nelements}) ' if self.simd_omp and self.simd_nelements > 1 else ''
        if self.parallel:
            line1 = f'#pragma omp for{reduction} '
            if self.parallel > 1:
                line1 += f' collapse({self.parallel})'
            if self.simd_omp:
                line1 = line1.replace(' for ', f' for {simd}')
        elif self.simd_vec:
            line1 = ''
        elif self.simd_omp:
            line1 = f'#pragma omp {simd}{reduction}'
        elif not self.reduction_var_map and codecache.is_gcc():
            line1 = '#pragma GCC ivdep'
        else:
            line1 = ''
        offset_str = f'{INDEX_TYPE} {self.var}={offset_expr}'
        size_str = f'{self.var}<{size_expr}'
        steps_str = f'{self.var}+={cexpr_index(self.steps)}'
        line2 = f'for({offset_str}; {size_str}; {steps_str})'
        if self.collapsed or not line1:
            return [line2]
        return [line1, line2]

@dataclasses.dataclass
class LoopNestWithSplit:
    """
    A loop-nest like structure but with some loop level split along
    the loop range into the main tiling loop and the tail. It is built
    with the `build` method as a loop nest and then split with
    `split_with_tiling` at some depth.

    A typical case is for vectorization where we typically split at the inner-most
    loop level. A more complicated case is 2D tiling where we split at
    both inner-most and outer levels.
    """
    root: Optional[List[LoopLevel]] = None
    kernel: Optional[CppKernel] = None

    @staticmethod
    def build(kernel: CppKernel):
        if False:
            i = 10
            return i + 15
        'Build a LoopNest with the given `kernel` as the leaf'
        itervars = kernel.itervars
        ranges = kernel.ranges
        reduction_depth = kernel.reduction_depth
        assert reduction_depth is not None
        root: List[LoopLevel] = []
        levels: List[LoopLevel] = root
        loop: Optional[LoopLevel] = None
        for (loop_idx, (var, size)) in enumerate(zip(itervars, ranges)):
            loop = LoopLevel(var, size, parent=loop)
            if loop_idx >= reduction_depth:
                loop.reduction_var_map = kernel.reduction_var_map.copy()
            levels.append(loop)
            levels = loop.inner
        loop_nest = LoopNestWithSplit(root)
        if loop:
            loop.kernel = kernel
        else:
            loop_nest.kernel = kernel
        return loop_nest

    def __bool__(self):
        if False:
            return 10
        return bool(self.root)

    def get_loops_at(self, depth) -> List[LoopLevel]:
        if False:
            for i in range(10):
                print('nop')
        'Get all the loop levels at the given `depth` (most outer loop has depth 0)'
        loops: List[LoopLevel] = []
        assert self.root is not None
        for loop in self.root:
            loops += loop.get_loops_at(depth)
        return loops

    @cache_on_self
    def max_parallel_depth(self):
        if False:
            print('Hello World!')
        '\n        Maximal allowed depth for parallelism:\n        1) Levels without splitting and\n        2) All reduction or non-reduction levels\n        When the loop is split at the top level, the max depth is 1.\n        '
        max_depth = 0
        assert self.root is not None
        loops = self.root
        if len(loops) > 1:
            return 1
        is_reduction = loops[0].is_reduction() if loops else False
        while len(loops) == 1 and loops[0].is_reduction() == is_reduction:
            max_depth += 1
            loops = loops[0].inner
        return max_depth

    def is_reduction_only(self):
        if False:
            while True:
                i = 10
        '\n        Whether all the loops are for reduction. Reduction loops\n        are always the inner most ones.\n        '
        return self.root is not None and len(self.root) > 0 and self.root[0].is_reduction()

    def mark_parallel(self, par_depth):
        if False:
            for i in range(10):
                print('nop')
        assert par_depth <= self.max_parallel_depth(), 'Parallel depth cannot exceed the maximal allowed parallel depth'
        assert self.root is not None
        loops = self.root
        for loop in loops:
            loop.parallel = par_depth
        for i in range(1, par_depth):
            loops = loops[0].inner
            loops[0].collapsed = True

    def split_with_tiling(self, depth, factor):
        if False:
            while True:
                i = 10
        '\n        Split the loop into main and tail loops at given `depth` so that the range\n        of the main loop has range `floor_div(range, factor) * factor` and\n        the tail loop handles the remainder. The main loop is tiled\n        according to the `factor`.\n        '
        loops = self.get_loops_at(depth)
        assert len(loops) == 1
        split_loops = loops[0].split_with_tiling(0, factor)
        if depth == 0:
            self.root = split_loops
        return split_loops