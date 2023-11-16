import math
import re
import dis
import numbers
import os
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import njit, prange, parallel_chunksize, get_parallel_chunksize, set_parallel_chunksize, set_num_threads, get_num_threads, typeof
from numba.core import types, typing, errors, ir, rewrites, typed_passes, inline_closurecall, config, compiler, cpu
from numba.extending import overload_method, register_model, typeof_impl, unbox, NativeValue, models
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import find_callname, guard, build_definitions, get_definition, is_getitem, is_setitem, index_var_of_get_setitem
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.bytecode import ByteCodeIter
from numba.core.compiler import compile_isolated, Flags, CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import TestCase, captured_stdout, MemoryLeakMixin, override_env_config, linux_only, tag, skip_parfors_unsupported, _32bit, needs_blas, needs_lapack, disabled_test, skip_unless_scipy, needs_subprocess
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest

@skip_parfors_unsupported
class TestParforsRunner(TestCase):
    _numba_parallel_test_ = False
    _TIMEOUT = 1800 if platform.machine() != 'aarch64' else 3600
    'This is the test runner for all the parfors tests, it runs them in\n    subprocesses as described above. The convention for the test method naming\n    is: `test_<TestClass>` where <TestClass> is the name of the test class in\n    this module.\n    '

    def runner(self):
        if False:
            i = 10
            return i + 15
        themod = self.__module__
        test_clazz_name = self.id().split('.')[-1].split('_')[-1]
        self.subprocess_test_runner(test_module=themod, test_class=test_clazz_name, timeout=self._TIMEOUT)

    def test_TestParforBasic(self):
        if False:
            return 10
        self.runner()

    def test_TestParforNumericalMisc(self):
        if False:
            print('Hello World!')
        self.runner()

    def test_TestParforNumPy(self):
        if False:
            return 10
        self.runner()

    def test_TestParfors(self):
        if False:
            i = 10
            return i + 15
        self.runner()

    def test_TestParforsBitMask(self):
        if False:
            i = 10
            return i + 15
        self.runner()

    def test_TestParforsDiagnostics(self):
        if False:
            while True:
                i = 10
        self.runner()

    def test_TestParforsLeaks(self):
        if False:
            for i in range(10):
                print('nop')
        self.runner()

    def test_TestParforsMisc(self):
        if False:
            for i in range(10):
                print('nop')
        self.runner()

    def test_TestParforsOptions(self):
        if False:
            while True:
                i = 10
        self.runner()

    def test_TestParforsSlice(self):
        if False:
            i = 10
            return i + 15
        self.runner()

    def test_TestParforsVectorizer(self):
        if False:
            return 10
        self.runner()

    def test_TestPrangeBasic(self):
        if False:
            for i in range(10):
                print('nop')
        self.runner()

    def test_TestPrangeSpecific(self):
        if False:
            return 10
        self.runner()
x86_only = unittest.skipIf(platform.machine() not in ('i386', 'x86_64'), 'x86 only test')
_GLOBAL_INT_FOR_TESTING1 = 17
_GLOBAL_INT_FOR_TESTING2 = 5
TestNamedTuple = namedtuple('TestNamedTuple', ('part0', 'part1'))

def null_comparer(a, b):
    if False:
        i = 10
        return i + 15
    '\n    Used with check_arq_equality to indicate that we do not care\n    whether the value of the parameter at the end of the function\n    has a particular value.\n    '
    pass

@needs_subprocess
class TestParforsBase(TestCase):
    """
    Base class for testing parfors.
    Provides functions for compilation and three way comparison between
    python functions, njit'd functions and parfor njit'd functions.
    """
    _numba_parallel_test_ = False

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        self.cflags = Flags()
        self.cflags.nrt = True
        self.pflags = Flags()
        self.pflags.auto_parallel = cpu.ParallelOptions(True)
        self.pflags.nrt = True
        self.fast_pflags = Flags()
        self.fast_pflags.auto_parallel = cpu.ParallelOptions(True)
        self.fast_pflags.nrt = True
        self.fast_pflags.fastmath = cpu.FastMathOptions(True)
        super(TestParforsBase, self).__init__(*args)

    def _compile_this(self, func, sig, flags):
        if False:
            i = 10
            return i + 15
        return compile_isolated(func, sig, flags=flags)

    def compile_parallel(self, func, sig):
        if False:
            i = 10
            return i + 15
        return self._compile_this(func, sig, flags=self.pflags)

    def compile_parallel_fastmath(self, func, sig):
        if False:
            return 10
        return self._compile_this(func, sig, flags=self.fast_pflags)

    def compile_njit(self, func, sig):
        if False:
            return 10
        return self._compile_this(func, sig, flags=self.cflags)

    def compile_all(self, pyfunc, *args, **kwargs):
        if False:
            while True:
                i = 10
        sig = tuple([numba.typeof(x) for x in args])
        cpfunc = self.compile_parallel(pyfunc, sig)
        cfunc = self.compile_njit(pyfunc, sig)
        return (cfunc, cpfunc)

    def check_parfors_vs_others(self, pyfunc, cfunc, cpfunc, *args, **kwargs):
        if False:
            return 10
        "\n        Checks python, njit and parfor impls produce the same result.\n\n        Arguments:\n            pyfunc - the python function to test\n            cfunc - CompilerResult from njit of pyfunc\n            cpfunc - CompilerResult from njit(parallel=True) of pyfunc\n            args - arguments for the function being tested\n        Keyword Arguments:\n            scheduler_type - 'signed', 'unsigned' or None, default is None.\n                           Supply in cases where the presence of a specific\n                           scheduler is to be asserted.\n            fastmath_pcres - a fastmath parallel compile result, if supplied\n                             will be run to make sure the result is correct\n            check_arg_equality - some functions need to check that a\n                                 parameter is modified rather than a certain\n                                 value returned.  If this keyword argument\n                                 is supplied, it should be a list of\n                                 comparison functions such that the i'th\n                                 function in the list is used to compare the\n                                 i'th parameter of the njit and parallel=True\n                                 functions against the i'th parameter of the\n                                 standard Python function, asserting if they\n                                 differ.  The length of this list must be equal\n                                 to the number of parameters to the function.\n                                 The null comparator is available for use\n                                 when you do not desire to test if some\n                                 particular parameter is changed.\n            Remaining kwargs are passed to np.testing.assert_almost_equal\n        "
        scheduler_type = kwargs.pop('scheduler_type', None)
        check_fastmath = kwargs.pop('check_fastmath', None)
        fastmath_pcres = kwargs.pop('fastmath_pcres', None)
        check_scheduling = kwargs.pop('check_scheduling', True)
        check_args_for_equality = kwargs.pop('check_arg_equality', None)

        def copy_args(*args):
            if False:
                i = 10
                return i + 15
            if not args:
                return tuple()
            new_args = []
            for x in args:
                if isinstance(x, np.ndarray):
                    new_args.append(x.copy('k'))
                elif isinstance(x, np.number):
                    new_args.append(x.copy())
                elif isinstance(x, numbers.Number):
                    new_args.append(x)
                elif x is None:
                    new_args.append(x)
                elif isinstance(x, tuple):
                    new_args.append(copy.deepcopy(x))
                elif isinstance(x, list):
                    new_args.append(x[:])
                else:
                    raise ValueError('Unsupported argument type encountered')
            return tuple(new_args)
        py_args = copy_args(*args)
        py_expected = pyfunc(*py_args)
        njit_args = copy_args(*args)
        njit_output = cfunc.entry_point(*njit_args)
        parfor_args = copy_args(*args)
        parfor_output = cpfunc.entry_point(*parfor_args)
        if check_args_for_equality is None:
            np.testing.assert_almost_equal(njit_output, py_expected, **kwargs)
            np.testing.assert_almost_equal(parfor_output, py_expected, **kwargs)
            self.assertEqual(type(njit_output), type(parfor_output))
        else:
            assert len(py_args) == len(check_args_for_equality)
            for (pyarg, njitarg, parforarg, argcomp) in zip(py_args, njit_args, parfor_args, check_args_for_equality):
                argcomp(njitarg, pyarg, **kwargs)
                argcomp(parforarg, pyarg, **kwargs)
        if check_scheduling:
            self.check_scheduling(cpfunc, scheduler_type)
        if fastmath_pcres is not None:
            parfor_fastmath_output = fastmath_pcres.entry_point(*copy_args(*args))
            np.testing.assert_almost_equal(parfor_fastmath_output, py_expected, **kwargs)

    def check(self, pyfunc, *args, **kwargs):
        if False:
            return 10
        'Checks that pyfunc compiles for *args under parallel=True and njit\n        and asserts that all version execute and produce the same result'
        (cfunc, cpfunc) = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    def check_variants(self, impl, arg_gen, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Run self.check(impl, ...) on array data generated from arg_gen.\n        '
        for args in arg_gen():
            with self.subTest(list(map(typeof, args))):
                self.check(impl, *args, **kwargs)

    def count_parfors_variants(self, impl, arg_gen, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Run self.countParfors(impl, ...) on array types generated from\n        arg_gen.\n        '
        for args in arg_gen():
            with self.subTest(list(map(typeof, args))):
                argtys = tuple(map(typeof, args))
                self.assertGreaterEqual(countParfors(impl, argtys), 1)

    def check_scheduling(self, cres, scheduler_type):
        if False:
            print('Hello World!')
        scheduler_str = '@do_scheduling'
        if scheduler_type is not None:
            if scheduler_type in ['signed', 'unsigned']:
                scheduler_str += '_' + scheduler_type
            else:
                msg = 'Unknown scheduler_type specified: %s'
                raise ValueError(msg % scheduler_type)
        self.assertIn(scheduler_str, cres.library.get_llvm_str())

    def gen_linspace(self, n, ct):
        if False:
            print('Hello World!')
        'Make *ct* sample 1D arrays of length *n* using np.linspace().\n        '

        def gen():
            if False:
                return 10
            yield np.linspace(0, 1, n)
            yield np.linspace(2, 1, n)
            yield np.linspace(1, 2, n)
        src = cycle(gen())
        return [next(src) for i in range(ct)]

    def gen_linspace_variants(self, ct):
        if False:
            print('Hello World!')
        'Make 1D, 2D, 3D variants of the data in C and F orders\n        '
        yield self.gen_linspace(10, ct=ct)
        arr2ds = [x.reshape((2, 3)) for x in self.gen_linspace(n=2 * 3, ct=ct)]
        yield arr2ds
        yield [np.asfortranarray(x) for x in arr2ds]
        arr3ds = [x.reshape((2, 3, 4)) for x in self.gen_linspace(n=2 * 3 * 4, ct=ct)]
        yield arr3ds
        yield [np.asfortranarray(x) for x in arr3ds]

    def _filter_mod(self, mod, magicstr, checkstr=None):
        if False:
            i = 10
            return i + 15
        ' helper function to filter out modules by name'
        filt = [x for x in mod if magicstr in x.name]
        if checkstr is not None:
            for x in filt:
                assert checkstr in str(x)
        return filt

    def _get_gufunc_modules(self, cres, magicstr, checkstr=None):
        if False:
            print('Hello World!')
        ' gets the gufunc LLVM Modules'
        _modules = [x for x in cres.library._codegen._engine._ee._modules]
        return self._filter_mod(_modules, magicstr, checkstr=checkstr)

    def _get_gufunc_info(self, cres, fn):
        if False:
            return 10
        ' helper for gufunc IR/asm generation'
        magicstr = '__numba_parfor_gufunc'
        gufunc_mods = self._get_gufunc_modules(cres, magicstr)
        x = dict()
        for mod in gufunc_mods:
            x[mod.name] = fn(mod)
        return x

    def _get_gufunc_ir(self, cres):
        if False:
            return 10
        '\n        Returns the IR of the gufuncs used as parfor kernels\n        as a dict mapping the gufunc name to its IR.\n\n        Arguments:\n         cres - a CompileResult from `njit(parallel=True, ...)`\n        '
        return self._get_gufunc_info(cres, str)

    def _get_gufunc_asm(self, cres):
        if False:
            return 10
        '\n        Returns the assembly of the gufuncs used as parfor kernels\n        as a dict mapping the gufunc name to its assembly.\n\n        Arguments:\n         cres - a CompileResult from `njit(parallel=True, ...)`\n        '
        tm = cres.library._codegen._tm

        def emit_asm(mod):
            if False:
                print('Hello World!')
            return str(tm.emit_assembly(mod))
        return self._get_gufunc_info(cres, emit_asm)

    def assert_fastmath(self, pyfunc, sig):
        if False:
            return 10
        '\n        Asserts that the fastmath flag has some effect in that suitable\n        instructions are now labelled as `fast`. Whether LLVM can actually do\n        anything to optimise better now the derestrictions are supplied is\n        another matter!\n\n        Arguments:\n         pyfunc - a function that contains operations with parallel semantics\n         sig - the type signature of pyfunc\n        '
        cres = self.compile_parallel_fastmath(pyfunc, sig)
        _ir = self._get_gufunc_ir(cres)

        def _get_fast_instructions(ir):
            if False:
                print('Hello World!')
            splitted = ir.splitlines()
            fast_inst = []
            for x in splitted:
                m = re.search('\\bfast\\b', x)
                if m is not None:
                    fast_inst.append(x)
            return fast_inst

        def _assert_fast(instrs):
            if False:
                i = 10
                return i + 15
            ops = ('fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'fcmp', 'call')
            for inst in instrs:
                count = 0
                for op in ops:
                    match = op + ' fast'
                    if match in inst:
                        count += 1
                self.assertTrue(count > 0)
        for (name, guir) in _ir.items():
            inst = _get_fast_instructions(guir)
            _assert_fast(inst)

def blackscholes_impl(sptprice, strike, rate, volatility, timev):
    if False:
        return 10
    logterm = np.log(sptprice / strike)
    powterm = 0.5 * volatility * volatility
    den = volatility * np.sqrt(timev)
    d1 = ((rate + powterm) * timev + logterm) / den
    d2 = d1 - den
    NofXd1 = 0.5 + 0.5 * 2.0 * d1
    NofXd2 = 0.5 + 0.5 * 2.0 * d2
    futureValue = strike * np.exp(-rate * timev)
    c1 = futureValue * NofXd2
    call = sptprice * NofXd1 - c1
    put = call - futureValue + sptprice
    return put

def lr_impl(Y, X, w, iterations):
    if False:
        print('Hello World!')
    for i in range(iterations):
        w -= np.dot((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y, X)
    return w

def example_kmeans_test(A, numCenter, numIter, init_centroids):
    if False:
        while True:
            i = 10
    centroids = init_centroids
    (N, D) = A.shape
    for l in range(numIter):
        dist = np.array([[math.sqrt(np.sum((A[i, :] - centroids[j, :]) ** 2)) for j in range(numCenter)] for i in range(N)])
        labels = np.array([dist[i, :].argmin() for i in range(N)])
        centroids = np.array([[np.sum(A[labels == i, j]) / np.sum(labels == i) for j in range(D)] for i in range(numCenter)])
    return centroids

def get_optimized_numba_ir(test_func, args, **kws):
    if False:
        while True:
            i = 10
    typingctx = typing.Context()
    targetctx = cpu.CPUContext(typingctx, 'cpu')
    test_ir = compiler.run_frontend(test_func)
    if kws:
        options = cpu.ParallelOptions(kws)
    else:
        options = cpu.ParallelOptions(True)
    tp = TestPipeline(typingctx, targetctx, args, test_ir)
    with cpu_target.nested_context(typingctx, targetctx):
        typingctx.refresh()
        targetctx.refresh()
        inline_pass = inline_closurecall.InlineClosureCallPass(tp.state.func_ir, options, typed=True)
        inline_pass.run()
        rewrites.rewrite_registry.apply('before-inference', tp.state)
        (tp.state.typemap, tp.state.return_type, tp.state.calltypes, _) = typed_passes.type_inference_stage(tp.state.typingctx, tp.state.targetctx, tp.state.func_ir, tp.state.args, None)
        type_annotations.TypeAnnotation(func_ir=tp.state.func_ir, typemap=tp.state.typemap, calltypes=tp.state.calltypes, lifted=(), lifted_from=None, args=tp.state.args, return_type=tp.state.return_type, html_output=config.HTML)
        diagnostics = numba.parfors.parfor.ParforDiagnostics()
        preparfor_pass = numba.parfors.parfor.PreParforPass(tp.state.func_ir, tp.state.typemap, tp.state.calltypes, tp.state.typingctx, tp.state.targetctx, options, swapped=diagnostics.replaced_fns)
        preparfor_pass.run()
        rewrites.rewrite_registry.apply('after-inference', tp.state)
        flags = compiler.Flags()
        parfor_pass = numba.parfors.parfor.ParforPass(tp.state.func_ir, tp.state.typemap, tp.state.calltypes, tp.state.return_type, tp.state.typingctx, tp.state.targetctx, options, flags, tp.state.metadata, diagnostics=diagnostics)
        parfor_pass.run()
        parfor_pass = numba.parfors.parfor.ParforFusionPass(tp.state.func_ir, tp.state.typemap, tp.state.calltypes, tp.state.return_type, tp.state.typingctx, tp.state.targetctx, options, flags, tp.state.metadata, diagnostics=diagnostics)
        parfor_pass.run()
        parfor_pass = numba.parfors.parfor.ParforPreLoweringPass(tp.state.func_ir, tp.state.typemap, tp.state.calltypes, tp.state.return_type, tp.state.typingctx, tp.state.targetctx, options, flags, tp.state.metadata, diagnostics=diagnostics)
        parfor_pass.run()
        test_ir._definitions = build_definitions(test_ir.blocks)
    return (test_ir, tp)

def countParfors(test_func, args, **kws):
    if False:
        while True:
            i = 10
    (test_ir, tp) = get_optimized_numba_ir(test_func, args, **kws)
    ret_count = 0
    for (label, block) in test_ir.blocks.items():
        for (i, inst) in enumerate(block.body):
            if isinstance(inst, numba.parfors.parfor.Parfor):
                ret_count += 1
    return ret_count

def countArrays(test_func, args, **kws):
    if False:
        print('Hello World!')
    (test_ir, tp) = get_optimized_numba_ir(test_func, args, **kws)
    return _count_arrays_inner(test_ir.blocks, tp.state.typemap)

def get_init_block_size(test_func, args, **kws):
    if False:
        print('Hello World!')
    (test_ir, tp) = get_optimized_numba_ir(test_func, args, **kws)
    blocks = test_ir.blocks
    ret_count = 0
    for (label, block) in blocks.items():
        for (i, inst) in enumerate(block.body):
            if isinstance(inst, numba.parfors.parfor.Parfor):
                ret_count += len(inst.init_block.body)
    return ret_count

def _count_arrays_inner(blocks, typemap):
    if False:
        print('Hello World!')
    ret_count = 0
    arr_set = set()
    for (label, block) in blocks.items():
        for (i, inst) in enumerate(block.body):
            if isinstance(inst, numba.parfors.parfor.Parfor):
                parfor_blocks = inst.loop_body.copy()
                parfor_blocks[0] = inst.init_block
                ret_count += _count_arrays_inner(parfor_blocks, typemap)
            if isinstance(inst, ir.Assign) and isinstance(typemap[inst.target.name], types.ArrayCompatible):
                arr_set.add(inst.target.name)
    ret_count += len(arr_set)
    return ret_count

def countArrayAllocs(test_func, args, **kws):
    if False:
        print('Hello World!')
    (test_ir, tp) = get_optimized_numba_ir(test_func, args, **kws)
    ret_count = 0
    for block in test_ir.blocks.values():
        ret_count += _count_array_allocs_inner(test_ir, block)
    return ret_count

def _count_array_allocs_inner(func_ir, block):
    if False:
        print('Hello World!')
    ret_count = 0
    for inst in block.body:
        if isinstance(inst, numba.parfors.parfor.Parfor):
            ret_count += _count_array_allocs_inner(func_ir, inst.init_block)
            for b in inst.loop_body.values():
                ret_count += _count_array_allocs_inner(func_ir, b)
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr) and (inst.value.op == 'call') and (guard(find_callname, func_ir, inst.value) == ('empty', 'numpy') or guard(find_callname, func_ir, inst.value) == ('empty_inferred', 'numba.np.unsafe.ndarray')):
            ret_count += 1
    return ret_count

def countNonParforArrayAccesses(test_func, args, **kws):
    if False:
        i = 10
        return i + 15
    (test_ir, tp) = get_optimized_numba_ir(test_func, args, **kws)
    return _count_non_parfor_array_accesses_inner(test_ir, test_ir.blocks, tp.state.typemap)

def _count_non_parfor_array_accesses_inner(f_ir, blocks, typemap, parfor_indices=None):
    if False:
        for i in range(10):
            print('nop')
    ret_count = 0
    if parfor_indices is None:
        parfor_indices = set()
    for (label, block) in blocks.items():
        for stmt in block.body:
            if isinstance(stmt, numba.parfors.parfor.Parfor):
                parfor_indices.add(stmt.index_var.name)
                parfor_blocks = stmt.loop_body.copy()
                parfor_blocks[0] = stmt.init_block
                ret_count += _count_non_parfor_array_accesses_inner(f_ir, parfor_blocks, typemap, parfor_indices)
            elif is_getitem(stmt) and isinstance(typemap[stmt.value.value.name], types.ArrayCompatible) and (not _uses_indices(f_ir, index_var_of_get_setitem(stmt), parfor_indices)):
                ret_count += 1
            elif is_setitem(stmt) and isinstance(typemap[stmt.target.name], types.ArrayCompatible) and (not _uses_indices(f_ir, index_var_of_get_setitem(stmt), parfor_indices)):
                ret_count += 1
            elif isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var) and (stmt.value.name in parfor_indices):
                parfor_indices.add(stmt.target.name)
    return ret_count

def _uses_indices(f_ir, index, index_set):
    if False:
        return 10
    if index.name in index_set:
        return True
    ind_def = guard(get_definition, f_ir, index)
    if isinstance(ind_def, ir.Expr) and ind_def.op == 'build_tuple':
        varnames = set((v.name for v in ind_def.items))
        return len(varnames & index_set) != 0
    return False

class TestPipeline(object):

    def __init__(self, typingctx, targetctx, args, test_ir):
        if False:
            while True:
                i = 10
        self.state = compiler.StateDict()
        self.state.typingctx = typingctx
        self.state.targetctx = targetctx
        self.state.args = args
        self.state.func_ir = test_ir
        self.state.typemap = None
        self.state.return_type = None
        self.state.calltypes = None
        self.state.metadata = {}

@skip_parfors_unsupported
class TestParforBasic(TestParforsBase):
    """Smoke tests for the parfors transforms. These tests check the most basic
    functionality"""

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        TestParforsBase.__init__(self, *args)
        m = np.reshape(np.arange(12.0), (3, 4))
        self.simple_args = [np.arange(3.0), np.arange(4.0), m, m.T]

    def test_simple01(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                while True:
                    i = 10
            return np.ones(())
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

    def test_simple02(self):
        if False:
            while True:
                i = 10

        def test_impl():
            if False:
                for i in range(10):
                    print('nop')
            return np.ones((1,))
        self.check(test_impl)

    def test_simple03(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                while True:
                    i = 10
            return np.ones((1, 2))
        self.check(test_impl)

    def test_simple04(self):
        if False:
            return 10

        def test_impl():
            if False:
                while True:
                    i = 10
            return np.ones(1)
        self.check(test_impl)

    def test_simple07(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                i = 10
                return i + 15
            return np.ones((1, 2), dtype=np.complex128)
        self.check(test_impl)

    def test_simple08(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                for i in range(10):
                    print('nop')
            return np.ones((1, 2)) + np.ones((1, 2))
        self.check(test_impl)

    def test_simple09(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                print('Hello World!')
            return np.ones((1, 1))
        self.check(test_impl)

    def test_simple10(self):
        if False:
            print('Hello World!')

        def test_impl():
            if False:
                i = 10
                return i + 15
            return np.ones((0, 0))
        self.check(test_impl)

    def test_simple11(self):
        if False:
            while True:
                i = 10

        def test_impl():
            if False:
                return 10
            return np.ones((10, 10)) + 1.0
        self.check(test_impl)

    def test_simple12(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                return 10
            return np.ones((10, 10)) + np.complex128(1.0)
        self.check(test_impl)

    def test_simple13(self):
        if False:
            print('Hello World!')

        def test_impl():
            if False:
                i = 10
                return i + 15
            return np.complex128(1.0)
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

    def test_simple14(self):
        if False:
            while True:
                i = 10

        def test_impl():
            if False:
                print('Hello World!')
            return np.ones((10, 10))[0::20]
        self.check(test_impl)

    def test_simple15(self):
        if False:
            i = 10
            return i + 15

        def test_impl(v1, v2, m1, m2):
            if False:
                return 10
            return v1 + v1
        self.check(test_impl, *self.simple_args)

    def test_simple16(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(v1, v2, m1, m2):
            if False:
                i = 10
                return i + 15
            return m1 + m1
        self.check(test_impl, *self.simple_args)

    def test_simple17(self):
        if False:
            i = 10
            return i + 15

        def test_impl(v1, v2, m1, m2):
            if False:
                print('Hello World!')
            return m2 + v1
        self.check(test_impl, *self.simple_args)

    @needs_lapack
    def test_simple18(self):
        if False:
            return 10

        def test_impl(v1, v2, m1, m2):
            if False:
                print('Hello World!')
            return m1.T + np.linalg.svd(m2)[1]
        self.check(test_impl, *self.simple_args)

    @needs_blas
    def test_simple19(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(v1, v2, m1, m2):
            if False:
                i = 10
                return i + 15
            return np.dot(m1, v2)
        self.check(test_impl, *self.simple_args)

    @needs_blas
    def test_simple20(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(v1, v2, m1, m2):
            if False:
                for i in range(10):
                    print('nop')
            return np.dot(m1, m2)
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl, *self.simple_args)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

    @needs_blas
    def test_simple21(self):
        if False:
            print('Hello World!')

        def test_impl(v1, v2, m1, m2):
            if False:
                print('Hello World!')
            return np.dot(v1, v1)
        self.check(test_impl, *self.simple_args)

    def test_simple22(self):
        if False:
            return 10

        def test_impl(v1, v2, m1, m2):
            if False:
                print('Hello World!')
            return np.sum(v1 + v1)
        self.check(test_impl, *self.simple_args)

    def test_simple23(self):
        if False:
            print('Hello World!')

        def test_impl(v1, v2, m1, m2):
            if False:
                i = 10
                return i + 15
            x = 2 * v1
            y = 2 * v1
            return 4 * np.sum(x ** 2 + y ** 2 < 1) / 10
        self.check(test_impl, *self.simple_args)

    def test_simple24(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                while True:
                    i = 10
            n = 20
            A = np.ones((n, n))
            b = np.arange(n)
            return np.sum(A[:, b])
        self.check(test_impl)

    @disabled_test
    def test_simple_operator_15(self):
        if False:
            return 10
        'same as corresponding test_simple_<n> case but using operator.add'

        def test_impl(v1, v2, m1, m2):
            if False:
                while True:
                    i = 10
            return operator.add(v1, v1)
        self.check(test_impl, *self.simple_args)

    @disabled_test
    def test_simple_operator_16(self):
        if False:
            print('Hello World!')

        def test_impl(v1, v2, m1, m2):
            if False:
                return 10
            return operator.add(m1, m1)
        self.check(test_impl, *self.simple_args)

    @disabled_test
    def test_simple_operator_17(self):
        if False:
            i = 10
            return i + 15

        def test_impl(v1, v2, m1, m2):
            if False:
                for i in range(10):
                    print('nop')
            return operator.add(m2, v1)
        self.check(test_impl, *self.simple_args)

    def test_inplace_alias(self):
        if False:
            print('Hello World!')

        def test_impl(a):
            if False:
                i = 10
                return i + 15
            a += 1
            a[:] = 3

        def comparer(a, b):
            if False:
                print('Hello World!')
            np.testing.assert_equal(a, b)
        x = np.ones(1)
        self.check(test_impl, x, check_arg_equality=[comparer])

@skip_parfors_unsupported
class TestParforNumericalMisc(TestParforsBase):
    """ Miscellaneous 'classical' numerical tests """

    def test_pi(self):
        if False:
            print('Hello World!')

        def test_impl(n):
            if False:
                return 10
            x = 2 * np.random.ranf(n) - 1
            y = 2 * np.random.ranf(n) - 1
            return 4 * np.sum(x ** 2 + y ** 2 < 1) / n
        self.check(test_impl, 100000, decimal=1)
        self.assertEqual(countParfors(test_impl, (types.int64,)), 1)
        self.assertEqual(countArrays(test_impl, (types.intp,)), 0)

    def test_blackscholes(self):
        if False:
            return 10
        args = (numba.float64[:],) * 5
        self.assertEqual(countParfors(blackscholes_impl, args), 1)

    @needs_blas
    def test_logistic_regression(self):
        if False:
            print('Hello World!')
        args = (numba.float64[:], numba.float64[:, :], numba.float64[:], numba.int64)
        self.assertEqual(countParfors(lr_impl, args), 2)
        self.assertEqual(countArrayAllocs(lr_impl, args), 1)

    def test_kmeans(self):
        if False:
            print('Hello World!')
        np.random.seed(0)
        N = 1024
        D = 10
        centers = 3
        A = np.random.ranf((N, D))
        init_centroids = np.random.ranf((centers, D))
        self.check(example_kmeans_test, A, centers, 3, init_centroids, decimal=1)
        arg_typs = (types.Array(types.float64, 2, 'C'), types.intp, types.intp, types.Array(types.float64, 2, 'C'))
        self.assertEqual(countNonParforArrayAccesses(example_kmeans_test, arg_typs), 0)

@skip_parfors_unsupported
class TestParforNumPy(TestParforsBase):
    """Tests NumPy functionality under parfors"""

    @needs_blas
    def test_mvdot(self):
        if False:
            print('Hello World!')

        def test_impl(a, v):
            if False:
                for i in range(10):
                    print('nop')
            return np.dot(a, v)
        A = np.linspace(0, 1, 20).reshape(2, 10)
        v = np.linspace(2, 1, 10)
        self.check(test_impl, A, v)

    def test_fuse_argmin_argmax_max_min(self):
        if False:
            return 10
        for op in [np.argmin, np.argmax, np.min, np.max]:

            def test_impl(n):
                if False:
                    print('Hello World!')
                A = np.ones(n)
                C = op(A)
                B = A.sum()
                return B + C
            self.check(test_impl, 256)
            self.assertEqual(countParfors(test_impl, (types.int64,)), 1)
            self.assertEqual(countArrays(test_impl, (types.intp,)), 0)

    def test_np_random_func_direct_import(self):
        if False:
            while True:
                i = 10

        def test_impl(n):
            if False:
                while True:
                    i = 10
            A = randn(n)
            return A[0]
        self.assertEqual(countParfors(test_impl, (types.int64,)), 1)

    def test_arange(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl1(n):
            if False:
                for i in range(10):
                    print('nop')
            return np.arange(n)

        def test_impl2(s, n):
            if False:
                print('Hello World!')
            return np.arange(s, n)

        def test_impl3(s, n, t):
            if False:
                print('Hello World!')
            return np.arange(s, n, t)
        for arg in [11, 128, 30.0, complex(4, 5), complex(5, 4)]:
            self.check(test_impl1, arg)
            self.check(test_impl2, 2, arg)
            self.check(test_impl3, 2, arg, 2)

    def test_arange_dtype(self):
        if False:
            while True:
                i = 10

        def test_impl1(n):
            if False:
                for i in range(10):
                    print('nop')
            return np.arange(n, dtype=np.float32)

        def test_impl2(s, n):
            if False:
                for i in range(10):
                    print('nop')
            return np.arange(s, n, dtype=np.float32)

        def test_impl3(s, n, t):
            if False:
                while True:
                    i = 10
            return np.arange(s, n, t, dtype=np.float32)
        for arg in [11, 128, 30.0]:
            self.check(test_impl1, arg)
            self.check(test_impl2, 2, arg)
            self.check(test_impl3, 2, arg, 2)

    def test_linspace(self):
        if False:
            i = 10
            return i + 15

        def test_impl1(start, stop):
            if False:
                while True:
                    i = 10
            return np.linspace(start, stop)

        def test_impl2(start, stop, num):
            if False:
                while True:
                    i = 10
            return np.linspace(start, stop, num)
        for arg in [11, 128, 30.0, complex(4, 5), complex(5, 4)]:
            self.check(test_impl1, 2, arg)
            self.check(test_impl2, 2, arg, 30)

    def test_mean(self):
        if False:
            print('Hello World!')

        def test_impl(A):
            if False:
                print('Hello World!')
            return A.mean()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 1, 'C'),)), 1)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 2, 'C'),)), 1)
        data_gen = lambda : self.gen_linspace_variants(1)
        self.check_variants(test_impl, data_gen)
        self.count_parfors_variants(test_impl, data_gen)

    def test_var(self):
        if False:
            return 10

        def test_impl(A):
            if False:
                while True:
                    i = 10
            return A.var()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        C = A + 1j * A
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.check(test_impl, C)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 1, 'C'),)), 2)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 2, 'C'),)), 2)
        data_gen = lambda : self.gen_linspace_variants(1)
        self.check_variants(test_impl, data_gen)
        self.count_parfors_variants(test_impl, data_gen)

    def test_std(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(A):
            if False:
                for i in range(10):
                    print('nop')
            return A.std()
        N = 100
        A = np.random.ranf(N)
        B = np.random.randint(10, size=(N, 3))
        C = A + 1j * A
        self.check(test_impl, A)
        self.check(test_impl, B)
        self.check(test_impl, C)
        argty = (types.Array(types.float64, 1, 'C'),)
        self.assertEqual(countParfors(test_impl, argty), 2)
        self.assertEqual(countParfors(test_impl, argty), 2)
        data_gen = lambda : self.gen_linspace_variants(1)
        self.check_variants(test_impl, data_gen)
        self.count_parfors_variants(test_impl, data_gen)

    def test_random_parfor(self):
        if False:
            return 10
        '\n        Test function with only a random call to make sure a random function\n        like ranf is actually translated to a parfor.\n        '

        def test_impl(n):
            if False:
                for i in range(10):
                    print('nop')
            A = np.random.ranf((n, n))
            return A
        self.assertEqual(countParfors(test_impl, (types.int64,)), 1)

    def test_randoms(self):
        if False:
            while True:
                i = 10

        def test_impl(n):
            if False:
                i = 10
                return i + 15
            A = np.random.standard_normal(size=(n, n))
            B = np.random.randn(n, n)
            C = np.random.normal(0.0, 1.0, (n, n))
            D = np.random.chisquare(1.0, (n, n))
            E = np.random.randint(1, high=3, size=(n, n))
            F = np.random.triangular(1, 2, 3, (n, n))
            return np.sum(A + B + C + D + E + F)
        n = 128
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(n),))
        parfor_output = cpfunc.entry_point(n)
        py_output = test_impl(n)
        np.testing.assert_allclose(parfor_output, py_output, rtol=0.05)
        self.assertEqual(countParfors(test_impl, (types.int64,)), 1)

    def test_dead_randoms(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(n):
            if False:
                return 10
            A = np.random.standard_normal(size=(n, n))
            B = np.random.randn(n, n)
            C = np.random.normal(0.0, 1.0, (n, n))
            D = np.random.chisquare(1.0, (n, n))
            E = np.random.randint(1, high=3, size=(n, n))
            F = np.random.triangular(1, 2, 3, (n, n))
            return 3
        n = 128
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(n),))
        parfor_output = cpfunc.entry_point(n)
        py_output = test_impl(n)
        self.assertEqual(parfor_output, py_output)
        self.assertEqual(countParfors(test_impl, (types.int64,)), 0)

    def test_min(self):
        if False:
            return 10

        def test_impl1(A):
            if False:
                while True:
                    i = 10
            return A.min()

        def test_impl2(A):
            if False:
                return 10
            return np.min(A)
        n = 211
        A = np.random.ranf(n)
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))
        D = np.array([np.inf, np.inf])
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl1, D)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)
        self.check(test_impl2, D)
        msg = 'zero-size array to reduction operation minimum which has no identity'
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))
        data_gen = lambda : self.gen_linspace_variants(1)
        self.check_variants(test_impl1, data_gen)
        self.count_parfors_variants(test_impl1, data_gen)
        self.check_variants(test_impl2, data_gen)
        self.count_parfors_variants(test_impl2, data_gen)

    def test_max(self):
        if False:
            while True:
                i = 10

        def test_impl1(A):
            if False:
                return 10
            return A.max()

        def test_impl2(A):
            if False:
                print('Hello World!')
            return np.max(A)
        n = 211
        A = np.random.ranf(n)
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))
        D = np.array([-np.inf, -np.inf])
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl1, D)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)
        self.check(test_impl2, D)
        msg = 'zero-size array to reduction operation maximum which has no identity'
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))
        data_gen = lambda : self.gen_linspace_variants(1)
        self.check_variants(test_impl1, data_gen)
        self.count_parfors_variants(test_impl1, data_gen)
        self.check_variants(test_impl2, data_gen)
        self.count_parfors_variants(test_impl2, data_gen)

    def test_argmax(self):
        if False:
            print('Hello World!')

        def test_impl1(A):
            if False:
                return 10
            return A.argmax()

        def test_impl2(A):
            if False:
                i = 10
                return i + 15
            return np.argmax(A)
        n = 211
        A = np.array([1.0, 0.0, 3.0, 2.0, 3.0])
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))
        D = np.array([1.0, 0.0, np.nan, 2.0, 3.0])
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl1, D)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)
        self.check(test_impl2, D)
        msg = 'attempt to get argmax of an empty sequence'
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))
        data_gen = lambda : self.gen_linspace_variants(1)
        self.check_variants(test_impl1, data_gen)
        self.count_parfors_variants(test_impl1, data_gen)
        self.check_variants(test_impl2, data_gen)
        self.count_parfors_variants(test_impl2, data_gen)

    def test_argmin(self):
        if False:
            i = 10
            return i + 15

        def test_impl1(A):
            if False:
                while True:
                    i = 10
            return A.argmin()

        def test_impl2(A):
            if False:
                return 10
            return np.argmin(A)
        n = 211
        A = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
        B = np.random.randint(10, size=n).astype(np.int32)
        C = np.random.ranf((n, n))
        D = np.array([1.0, 0.0, np.nan, 0.0, 3.0])
        self.check(test_impl1, A)
        self.check(test_impl1, B)
        self.check(test_impl1, C)
        self.check(test_impl1, D)
        self.check(test_impl2, A)
        self.check(test_impl2, B)
        self.check(test_impl2, C)
        self.check(test_impl2, D)
        msg = 'attempt to get argmin of an empty sequence'
        for impl in (test_impl1, test_impl2):
            pcfunc = self.compile_parallel(impl, (types.int64[:],))
            with self.assertRaises(ValueError) as e:
                pcfunc.entry_point(np.array([], dtype=np.int64))
            self.assertIn(msg, str(e.exception))
        data_gen = lambda : self.gen_linspace_variants(1)
        self.check_variants(test_impl1, data_gen)
        self.count_parfors_variants(test_impl1, data_gen)
        self.check_variants(test_impl2, data_gen)
        self.count_parfors_variants(test_impl2, data_gen)

    def test_ndarray_fill(self):
        if False:
            while True:
                i = 10

        def test_impl(x):
            if False:
                i = 10
                return i + 15
            x.fill(7.0)
            return x
        x = np.zeros(10)
        self.check(test_impl, x)
        argty = (types.Array(types.float64, 1, 'C'),)
        self.assertEqual(countParfors(test_impl, argty), 1)

    def test_ndarray_fill2d(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(x):
            if False:
                return 10
            x.fill(7.0)
            return x
        x = np.zeros((2, 2))
        self.check(test_impl, x)
        argty = (types.Array(types.float64, 2, 'C'),)
        self.assertEqual(countParfors(test_impl, argty), 1)

    def test_reshape_with_neg_one(self):
        if False:
            print('Hello World!')

        def test_impl(a, b):
            if False:
                while True:
                    i = 10
            result_matrix = np.zeros((b, b, 1), dtype=np.float64)
            sub_a = a[0:b]
            a = sub_a.size
            b = a / 1
            z = sub_a.reshape(-1, 1)
            result_data = sub_a / z
            result_matrix[:, :, 0] = result_data
            return result_matrix
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        b = 3
        self.check(test_impl, a, b)

    def test_reshape_with_large_neg(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(a, b):
            if False:
                i = 10
                return i + 15
            result_matrix = np.zeros((b, b, 1), dtype=np.float64)
            sub_a = a[0:b]
            a = sub_a.size
            b = a / 1
            z = sub_a.reshape(-1307, 1)
            result_data = sub_a / z
            result_matrix[:, :, 0] = result_data
            return result_matrix
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        b = 3
        self.check(test_impl, a, b)

    def test_reshape_with_too_many_neg_one(self):
        if False:
            return 10
        with self.assertRaises(errors.UnsupportedRewriteError) as raised:

            @njit(parallel=True)
            def test_impl(a, b):
                if False:
                    return 10
                rm = np.zeros((b, b, 1), dtype=np.float64)
                sub_a = a[0:b]
                a = sub_a.size
                b = a / 1
                z = sub_a.reshape(-1, -1)
                result_data = sub_a / z
                rm[:, :, 0] = result_data
                return rm
            a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            b = 3
            test_impl(a, b)
        msg = 'The reshape API may only include one negative argument.'
        self.assertIn(msg, str(raised.exception))

    def test_0d_array(self):
        if False:
            while True:
                i = 10

        def test_impl(n):
            if False:
                print('Hello World!')
            return np.sum(n) + np.prod(n) + np.min(n) + np.max(n) + np.var(n)
        self.check(test_impl, np.array(7), check_scheduling=False)

    def test_real_imag_attr(self):
        if False:
            while True:
                i = 10

        def test_impl(z):
            if False:
                for i in range(10):
                    print('nop')
            return np.sum(z.real ** 2 + z.imag ** 2)
        z = np.arange(5) * (1 + 1j)
        self.check(test_impl, z)
        self.assertEqual(countParfors(test_impl, (types.complex128[::1],)), 1)

class TestParforsUnsupported(TestCase):
    """Tests for unsupported use of parfors"""

    @unittest.skipIf(not _32bit, 'Only impacts 32 bit hardware')
    @needs_blas
    def test_unsupported_combination_raises(self):
        if False:
            print('Hello World!')
        "\n        This test is in place until issues with the 'parallel'\n        target on 32 bit hardware are fixed.\n        "
        with self.assertRaises(errors.UnsupportedParforsError) as raised:

            @njit(parallel=True)
            def ddot(a, v):
                if False:
                    for i in range(10):
                        print('nop')
                return np.dot(a, v)
            A = np.linspace(0, 1, 20).reshape(2, 10)
            v = np.linspace(2, 1, 10)
            ddot(A, v)
        msg = "The 'parallel' target is not currently supported on 32 bit hardware"
        self.assertIn(msg, str(raised.exception))

@skip_parfors_unsupported
class TestParfors(TestParforsBase):
    """ Tests cpython, reduction and various parfors features"""

    def test_arraymap(self):
        if False:
            i = 10
            return i + 15

        def test_impl(a, x, y):
            if False:
                while True:
                    i = 10
            return a * x + y
        self.check_variants(test_impl, lambda : self.gen_linspace_variants(3))

    def test_0d_broadcast(self):
        if False:
            while True:
                i = 10

        def test_impl():
            if False:
                i = 10
                return i + 15
            X = np.array(1)
            Y = np.ones((10, 12))
            return np.sum(X + Y)
        self.check(test_impl)
        self.assertEqual(countParfors(test_impl, ()), 1)

    def test_2d_parfor(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                return 10
            X = np.ones((10, 12))
            Y = np.zeros((10, 12))
            return np.sum(X + Y)
        self.check(test_impl)
        self.assertEqual(countParfors(test_impl, ()), 1)

    def test_nd_parfor(self):
        if False:
            while True:
                i = 10

        def case1():
            if False:
                i = 10
                return i + 15
            X = np.ones((10, 12))
            Y = np.zeros((10, 12))
            yield (X, Y)
        data_gen = lambda : chain(case1(), self.gen_linspace_variants(2))

        def test_impl(X, Y):
            if False:
                print('Hello World!')
            return np.sum(X + Y)
        self.check_variants(test_impl, data_gen)
        self.count_parfors_variants(test_impl, data_gen)

    def test_np_func_direct_import(self):
        if False:
            return 10
        from numpy import ones

        def test_impl(n):
            if False:
                for i in range(10):
                    print('nop')
            A = ones(n)
            return A[0]
        n = 111
        self.check(test_impl, n)

    def test_size_assertion(self):
        if False:
            i = 10
            return i + 15

        def test_impl(m, n):
            if False:
                while True:
                    i = 10
            A = np.ones(m)
            B = np.ones(n)
            return np.sum(A + B)
        self.check(test_impl, 10, 10)
        with self.assertRaises(AssertionError) as raises:
            cfunc = njit(parallel=True)(test_impl)
            cfunc(10, 9)
        msg = 'Sizes of A, B do not match'
        self.assertIn(msg, str(raises.exception))

    def test_cfg(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(x, is_positive, N):
            if False:
                i = 10
                return i + 15
            for i in numba.prange(2):
                for j in range(i * N // 2, (i + 1) * N // 2):
                    is_positive[j] = 0
                    if x[j] > 0:
                        is_positive[j] = 1
            return is_positive
        N = 100
        x = np.random.rand(N)
        is_positive = np.zeros(N)
        self.check(test_impl, x, is_positive, N)

    def test_reduce(self):
        if False:
            return 10

        def test_impl(A):
            if False:
                for i in range(10):
                    print('nop')
            init_val = 10
            return reduce(lambda a, b: min(a, b), A, init_val)
        n = 211
        A = np.random.ranf(n)
        self.check(test_impl, A)
        A = np.random.randint(10, size=n).astype(np.int32)
        self.check(test_impl, A)

        def test_impl():
            if False:
                print('Hello World!')
            g = lambda x: x ** 2
            return reduce(g, np.array([1, 2, 3, 4, 5]), 2)
        with self.assertTypingError():
            self.check(test_impl)
        n = 160
        A = np.random.randint(10, size=n).astype(np.int32)

        def test_impl(A):
            if False:
                return 10
            return np.sum(A[A >= 3])
        self.check(test_impl, A)

        def test_impl(A):
            if False:
                return 10
            B = A[:, 0]
            return np.sum(A[B >= 3, 1])
        self.check(test_impl, A.reshape((16, 10)))

        def test_impl(A):
            if False:
                print('Hello World!')
            B = A[:, 0]
            return np.sum(A[B >= 3, 1:2])
        self.check(test_impl, A.reshape((16, 10)))
        self.assertEqual(countParfors(test_impl, (numba.float64[:, :],)), 2)

        def test_impl(A):
            if False:
                while True:
                    i = 10
            min_val = np.amin(A)
            return A - min_val
        self.check(test_impl, A)
        self.assertEqual(countParfors(test_impl, (numba.float64[:],)), 2)

    def test_use_of_reduction_var1(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                i = 10
                return i + 15
            acc = 0
            for i in prange(1):
                acc = cmath.sqrt(acc)
            return acc
        msg = 'Use of reduction variable acc in an unsupported reduction function.'
        with self.assertRaises(ValueError) as e:
            pcfunc = self.compile_parallel(test_impl, ())
        self.assertIn(msg, str(e.exception))

    def test_unsupported_floordiv1(self):
        if False:
            print('Hello World!')

        def test_impl():
            if False:
                while True:
                    i = 10
            acc = 100
            for i in prange(2):
                acc //= 2
            return acc
        msg = 'Parallel floordiv reductions are not supported. If all divisors are integers then a floordiv reduction can in some cases be parallelized as a multiply reduction followed by a floordiv of the resulting product.'
        with self.assertRaises(errors.NumbaValueError) as e:
            pcfunc = self.compile_parallel(test_impl, ())
        self.assertIn(msg, str(e.exception))

    def test_unsupported_xor1(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                return 10
            acc = 100
            for i in prange(2):
                acc ^= i + 2
            return acc
        msg = 'Use of reduction variable acc in an unsupported reduction function.'
        with self.assertRaises(ValueError) as e:
            pcfunc = self.compile_parallel(test_impl, ())
        self.assertIn(msg, str(e.exception))

    def test_parfor_array_access1(self):
        if False:
            i = 10
            return i + 15

        def test_impl(n):
            if False:
                i = 10
                return i + 15
            A = np.ones(n)
            return A.sum()
        n = 211
        self.check(test_impl, n)
        self.assertEqual(countArrays(test_impl, (types.intp,)), 0)

    def test_parfor_array_access2(self):
        if False:
            i = 10
            return i + 15

        def test_impl(n):
            if False:
                print('Hello World!')
            A = np.ones(n)
            m = 0
            n = 0
            for i in numba.prange(len(A)):
                m += A[i]
            for i in numba.prange(len(A)):
                if m == n:
                    n += A[i]
            return m + n
        n = 211
        self.check(test_impl, n)
        self.assertEqual(countNonParforArrayAccesses(test_impl, (types.intp,)), 0)

    def test_parfor_array_access3(self):
        if False:
            while True:
                i = 10

        def test_impl(n):
            if False:
                for i in range(10):
                    print('nop')
            A = np.ones(n, np.int64)
            m = 0
            for i in numba.prange(len(A)):
                m += A[i]
                if m == 2:
                    i = m
        n = 211
        with self.assertRaises(errors.UnsupportedRewriteError) as raises:
            self.check(test_impl, n)
        self.assertIn('Overwrite of parallel loop index', str(raises.exception))

    @needs_blas
    def test_parfor_array_access4(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(A, b):
            if False:
                i = 10
                return i + 15
            return np.dot(A, b)
        n = 211
        d = 4
        A = np.random.ranf((n, d))
        b = np.random.ranf(d)
        self.check(test_impl, A, b)
        (test_ir, tp) = get_optimized_numba_ir(test_impl, (types.Array(types.float64, 2, 'C'), types.Array(types.float64, 1, 'C')))
        self.assertTrue(len(test_ir.blocks) == 1 and 0 in test_ir.blocks)
        block = test_ir.blocks[0]
        parfor_found = False
        parfor = None
        for stmt in block.body:
            if isinstance(stmt, numba.parfors.parfor.Parfor):
                parfor_found = True
                parfor = stmt
        self.assertTrue(parfor_found)
        build_tuple_found = False
        for bl in parfor.loop_body.values():
            for stmt in bl.body:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'build_tuple'):
                    build_tuple_found = True
                    self.assertTrue(parfor.index_var in stmt.value.items)
        self.assertTrue(build_tuple_found)

    def test_parfor_dtype_type(self):
        if False:
            return 10

        def test_impl(a):
            if False:
                return 10
            for i in numba.prange(len(a)):
                a[i] = a.dtype.type(0)
            return a[4]
        a = np.ones(10)
        self.check(test_impl, a)

    def test_parfor_array_access5(self):
        if False:
            while True:
                i = 10

        def test_impl(n):
            if False:
                while True:
                    i = 10
            X = np.ones((n, 3))
            y = 0
            for i in numba.prange(n):
                y += X[i, :].sum()
            return y
        n = 211
        self.check(test_impl, n)
        self.assertEqual(countNonParforArrayAccesses(test_impl, (types.intp,)), 0)

    @disabled_test
    def test_parfor_hoist_setitem(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(out):
            if False:
                return 10
            for i in prange(10):
                out[0] = 2 * out[0]
            return out[0]
        out = np.ones(1)
        self.check(test_impl, out)

    @needs_blas
    def test_parfor_generate_fuse(self):
        if False:
            print('Hello World!')

        def test_impl(N, D):
            if False:
                while True:
                    i = 10
            w = np.ones(D)
            X = np.ones((N, D))
            Y = np.ones(N)
            for i in range(3):
                B = -Y * np.dot(X, w)
            return B
        n = 211
        d = 3
        self.check(test_impl, n, d)
        self.assertEqual(countArrayAllocs(test_impl, (types.intp, types.intp)), 4)
        self.assertEqual(countParfors(test_impl, (types.intp, types.intp)), 4)

    def test_ufunc_expr(self):
        if False:
            print('Hello World!')

        def test_impl(A, B):
            if False:
                i = 10
                return i + 15
            return np.bitwise_and(A, B)
        A = np.ones(3, np.uint8)
        B = np.ones(3, np.uint8)
        B[1] = 0
        self.check(test_impl, A, B)

    def test_find_callname_intrinsic(self):
        if False:
            while True:
                i = 10

        def test_impl(n):
            if False:
                for i in range(10):
                    print('nop')
            A = unsafe_empty((n,))
            for i in range(n):
                A[i] = i + 2.0
            return A
        self.assertEqual(countArrayAllocs(test_impl, (types.intp,)), 1)

    def test_reduction_var_reuse(self):
        if False:
            while True:
                i = 10

        def test_impl(n):
            if False:
                return 10
            acc = 0
            for i in prange(n):
                acc += 1
            for i in prange(n):
                acc += 2
            return acc
        self.check(test_impl, 16)

    def test_non_identity_initial(self):
        if False:
            i = 10
            return i + 15

        def test_impl(A, cond):
            if False:
                return 10
            s = 1
            for i in prange(A.shape[0]):
                if cond[i]:
                    s += 1
            return s
        self.check(test_impl, np.ones(10), np.ones(10).astype('bool'))

    def test_if_not_else_reduction(self):
        if False:
            i = 10
            return i + 15

        def test_impl(A, cond):
            if False:
                print('Hello World!')
            s = 1
            t = 10
            for i in prange(A.shape[0]):
                if cond[i]:
                    s += 1
                    t += 1
                else:
                    s += 2
            return s + t
        self.check(test_impl, np.ones(10), np.ones(10).astype('bool'))

    def test_two_d_array_reduction_reuse(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(n):
            if False:
                for i in range(10):
                    print('nop')
            shp = (13, 17)
            size = shp[0] * shp[1]
            result1 = np.zeros(shp, np.int_)
            tmp = np.arange(size).reshape(shp)
            for i in numba.prange(n):
                result1 += tmp
            for i in numba.prange(n):
                result1 += tmp
            return result1
        self.check(test_impl, 100)

    def test_one_d_array_reduction(self):
        if False:
            print('Hello World!')

        def test_impl(n):
            if False:
                i = 10
                return i + 15
            result = np.zeros(1, np.int_)
            for i in numba.prange(n):
                result += np.array([i], np.int_)
            return result
        self.check(test_impl, 100)

    def test_two_d_array_reduction(self):
        if False:
            return 10

        def test_impl(n):
            if False:
                return 10
            shp = (13, 17)
            size = shp[0] * shp[1]
            result1 = np.zeros(shp, np.int_)
            tmp = np.arange(size).reshape(shp)
            for i in numba.prange(n):
                result1 += tmp
            return result1
        self.check(test_impl, 100)

    def test_two_d_array_reduction_with_float_sizes(self):
        if False:
            return 10

        def test_impl(n):
            if False:
                print('Hello World!')
            shp = (2, 3)
            result1 = np.zeros(shp, np.float32)
            tmp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(shp)
            for i in numba.prange(n):
                result1 += tmp
            return result1
        self.check(test_impl, 100)

    def test_two_d_array_reduction_prod(self):
        if False:
            return 10

        def test_impl(n):
            if False:
                print('Hello World!')
            shp = (13, 17)
            result1 = 2 * np.ones(shp, np.int_)
            tmp = 2 * np.ones_like(result1)
            for i in numba.prange(n):
                result1 *= tmp
            return result1
        self.check(test_impl, 100)

    def test_three_d_array_reduction(self):
        if False:
            return 10

        def test_impl(n):
            if False:
                return 10
            shp = (3, 2, 7)
            result1 = np.zeros(shp, np.int_)
            for i in numba.prange(n):
                result1 += np.ones(shp, np.int_)
            return result1
        self.check(test_impl, 100)

    def test_preparfor_canonicalize_kws(self):
        if False:
            return 10

        def test_impl(A):
            if False:
                for i in range(10):
                    print('nop')
            return A.argsort() + 1
        n = 211
        A = np.arange(n)
        self.check(test_impl, A)

    def test_preparfor_datetime64(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(A):
            if False:
                print('Hello World!')
            return A.dtype
        A = np.empty(1, np.dtype('datetime64[ns]'))
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(A),))
        self.assertEqual(cpfunc.entry_point(A), test_impl(A))

    def test_no_hoisting_with_member_function_call(self):
        if False:
            return 10

        def test_impl(X):
            if False:
                while True:
                    i = 10
            n = X.shape[0]
            acc = 0
            for i in prange(n):
                R = {1, 2, 3}
                R.add(i)
                tmp = 0
                for x in R:
                    tmp += x
                acc += tmp
            return acc
        self.check(test_impl, np.random.ranf(128))

    def test_array_compare_scalar(self):
        if False:
            while True:
                i = 10
        ' issue3671: X != 0 becomes an arrayexpr with operator.ne.\n            That is turned into a parfor by devectorizing.  Make sure\n            the return type of the devectorized operator.ne\n            on integer types works properly.\n        '

        def test_impl():
            if False:
                print('Hello World!')
            X = np.zeros(10, dtype=np.int_)
            return X != 0
        self.check(test_impl)

    def test_array_analysis_optional_def(self):
        if False:
            i = 10
            return i + 15

        def test_impl(x, half):
            if False:
                i = 10
                return i + 15
            size = len(x)
            parr = x[0:size]
            if half:
                parr = x[0:size // 2]
            return parr.sum()
        x = np.ones(20)
        self.check(test_impl, x, True, check_scheduling=False)

    def test_prange_side_effects(self):
        if False:
            i = 10
            return i + 15

        def test_impl(a, b):
            if False:
                i = 10
                return i + 15
            data = np.empty(len(a), dtype=np.float64)
            size = len(data)
            for i in numba.prange(size):
                data[i] = a[i]
            for i in numba.prange(size):
                data[i] = data[i] + b[i]
            return data
        x = np.arange(10 ** 2, dtype=float)
        y = np.arange(10 ** 2, dtype=float)
        self.check(test_impl, x, y)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 1, 'C'), types.Array(types.float64, 1, 'C'))), 1)

    def test_tuple1(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(a):
            if False:
                return 10
            atup = (3, 4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += atup[0] + atup[1] + b
            return a
        x = np.arange(10)
        self.check(test_impl, x)

    def test_tuple2(self):
        if False:
            while True:
                i = 10

        def test_impl(a):
            if False:
                for i in range(10):
                    print('nop')
            atup = a.shape
            b = 7
            for i in numba.prange(len(a)):
                a[i] += atup[0] + b
            return a
        x = np.arange(10)
        self.check(test_impl, x)

    def test_tuple3(self):
        if False:
            print('Hello World!')

        def test_impl(a):
            if False:
                return 10
            atup = (np.arange(10), 4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += atup[0][5] + atup[1] + b
            return a
        x = np.arange(10)
        self.check(test_impl, x)

    def test_namedtuple1(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(a):
            if False:
                return 10
            antup = TestNamedTuple(part0=3, part1=4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += antup.part0 + antup.part1 + b
            return a
        x = np.arange(10)
        self.check(test_impl, x)

    def test_namedtuple2(self):
        if False:
            while True:
                i = 10
        TestNamedTuple2 = namedtuple('TestNamedTuple2', ('part0', 'part1'))

        def test_impl(a):
            if False:
                i = 10
                return i + 15
            antup = TestNamedTuple2(part0=3, part1=4)
            b = 7
            for i in numba.prange(len(a)):
                a[i] += antup.part0 + antup.part1 + b
            return a
        x = np.arange(10)
        self.check(test_impl, x)

    def test_namedtuple3(self):
        if False:
            i = 10
            return i + 15
        TestNamedTuple3 = namedtuple(f'TestNamedTuple3', ['y'])

        def test_impl(a):
            if False:
                for i in range(10):
                    print('nop')
            a.y[:] = 5

        def comparer(a, b):
            if False:
                print('Hello World!')
            np.testing.assert_almost_equal(a.y, b.y)
        x = TestNamedTuple3(y=np.zeros(10))
        self.check(test_impl, x, check_arg_equality=[comparer])

    def test_inplace_binop(self):
        if False:
            return 10

        def test_impl(a, b):
            if False:
                for i in range(10):
                    print('nop')
            b += a
            return b
        X = np.arange(10) + 10
        Y = np.arange(10) + 100
        self.check(test_impl, X, Y)
        self.assertEqual(countParfors(test_impl, (types.Array(types.float64, 1, 'C'), types.Array(types.float64, 1, 'C'))), 1)

    def test_tuple_concat(self):
        if False:
            return 10

        def test_impl(a):
            if False:
                return 10
            n = len(a)
            array_shape = (n, n)
            indices = np.zeros((1,) + array_shape + (1,), dtype=np.uint64)
            k_list = indices[0, :]
            for (i, g) in enumerate(a):
                k_list[i, i] = i
            return k_list
        x = np.array([1, 1])
        self.check(test_impl, x)

    def test_tuple_concat_with_reverse_slice(self):
        if False:
            return 10

        def test_impl(a):
            if False:
                while True:
                    i = 10
            n = len(a)
            array_shape = (n, n)
            indices = np.zeros(((1,) + array_shape + (1,))[:-1], dtype=np.uint64)
            k_list = indices[0, :]
            for (i, g) in enumerate(a):
                k_list[i, i] = i
            return k_list
        x = np.array([1, 1])
        self.check(test_impl, x)

    def test_array_tuple_concat(self):
        if False:
            print('Hello World!')

        def test_impl(a):
            if False:
                while True:
                    i = 10
            S = (a,) + (a, a)
            return S[0].sum()
        x = np.ones((3, 3))
        self.check(test_impl, x)

    def test_high_dimension1(self):
        if False:
            print('Hello World!')

        def test_impl(x):
            if False:
                i = 10
                return i + 15
            return x * 5.0
        x = np.ones((2, 2, 2, 2, 2, 15))
        self.check(test_impl, x)

    def test_tuple_arg(self):
        if False:
            return 10

        def test_impl(x, sz):
            if False:
                i = 10
                return i + 15
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10, 5)
        self.check(test_impl, np.empty(sz), sz)

    def test_tuple_arg_not_whole_array(self):
        if False:
            print('Hello World!')

        def test_impl(x, sz):
            if False:
                while True:
                    i = 10
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10, 5)
        self.check(test_impl, np.zeros(sz), (10, 3))

    def test_tuple_for_pndindex(self):
        if False:
            print('Hello World!')

        def test_impl(x):
            if False:
                print('Hello World!')
            sz = (10, 5)
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10, 5)
        self.check(test_impl, np.zeros(sz))

    def test_tuple_arg_literal(self):
        if False:
            print('Hello World!')

        def test_impl(x, first):
            if False:
                while True:
                    i = 10
            sz = (first, 5)
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10, 5)
        self.check(test_impl, np.zeros(sz), 10)

    def test_tuple_of_literal_nonliteral(self):
        if False:
            print('Hello World!')

        def test_impl(x, sz):
            if False:
                return 10
            for i in numba.pndindex(sz):
                x[i] = 1
            return x

        def call(x, fn):
            if False:
                print('Hello World!')
            return fn(x, (10, 3))
        get_input = lambda : np.zeros((10, 10))
        expected = call(get_input(), test_impl)

        def check(dec):
            if False:
                for i in range(10):
                    print('nop')
            f1 = dec(test_impl)
            f2 = njit(call)
            got = f2(get_input(), f1)
            self.assertPreciseEqual(expected, got)
        for d in (njit, njit(parallel=True)):
            check(d)

    def test_tuple_arg_1d(self):
        if False:
            i = 10
            return i + 15

        def test_impl(x, sz):
            if False:
                print('Hello World!')
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10,)
        self.check(test_impl, np.zeros(sz), sz)

    def test_tuple_arg_1d_literal(self):
        if False:
            print('Hello World!')

        def test_impl(x):
            if False:
                for i in range(10):
                    print('nop')
            sz = (10,)
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        sz = (10,)
        self.check(test_impl, np.zeros(sz))

    def test_int_arg_pndindex(self):
        if False:
            print('Hello World!')

        def test_impl(x, sz):
            if False:
                print('Hello World!')
            for i in numba.pndindex(sz):
                x[i] = 1
            return x
        self.check(test_impl, np.zeros((10, 10)), 3)

    def test_prange_unknown_call1(self):
        if False:
            for i in range(10):
                print('nop')

        @register_jitable
        def issue7854_proc(u, i, even, size):
            if False:
                for i in range(10):
                    print('nop')
            for j in range((even + i + 1) % 2 + 1, size - 1, 2):
                u[i, j] = u[i + 1, j] + 1

        def test_impl(u, size):
            if False:
                i = 10
                return i + 15
            for i in numba.prange(1, size - 1):
                issue7854_proc(u, i, 0, size)
            for i in numba.prange(1, size - 1):
                issue7854_proc(u, i, 1, size)
            return u
        size = 4
        u = np.zeros((size, size))
        cptypes = (numba.float64[:, ::1], types.int64)
        self.assertEqual(countParfors(test_impl, cptypes), 2)
        self.check(test_impl, u, size)

    def test_prange_index_calc1(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(u, size):
            if False:
                return 10
            for i in numba.prange(1, size - 1):
                for j in range((i + 1) % 2 + 1, size - 1, 2):
                    u[i, j] = u[i + 1, j] + 1
            for i in numba.prange(1, size - 1):
                for j in range(i % 2 + 1, size - 1, 2):
                    u[i, j] = u[i + 1, j] + 1
            return u
        size = 4
        u = np.zeros((size, size))
        cptypes = (numba.float64[:, ::1], types.int64)
        self.assertEqual(countParfors(test_impl, cptypes), 2)
        self.check(test_impl, u, size)

    def test_prange_reverse_order1(self):
        if False:
            while True:
                i = 10

        def test_impl(a, b, size):
            if False:
                return 10
            for i in numba.prange(size):
                for j in range(size):
                    a[i, j] = b[i, j] + 1
            for i in numba.prange(size):
                for j in range(size):
                    b[j, i] = 3
            return a[0, 0] + b[0, 0]
        size = 10
        a = np.zeros((size, size))
        b = np.zeros((size, size))
        cptypes = (numba.float64[:, ::1], numba.float64[:, ::1], types.int64)
        self.assertEqual(countParfors(test_impl, cptypes), 2)
        self.check(test_impl, a, b, size)

    def test_prange_parfor_index_then_not(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(a, size):
            if False:
                for i in range(10):
                    print('nop')
            b = 0
            for i in numba.prange(size):
                a[i] = i
            for i in numba.prange(size):
                b += a[5]
            return b
        size = 10
        a = np.zeros(size)
        cptypes = (numba.float64[:], types.int64)
        self.assertEqual(countParfors(test_impl, cptypes), 2)
        self.check(test_impl, a, size)

    def test_prange_parfor_index_const_tuple_fusion(self):
        if False:
            while True:
                i = 10

        def test_impl(a, tup, size):
            if False:
                print('Hello World!')
            acc = 0
            for i in numba.prange(size):
                a[i] = i + tup[i]
            for i in numba.prange(size):
                acc += a[i] + tup[1]
            return acc
        size = 10
        a = np.zeros(size)
        b = tuple(a)
        cptypes = (numba.float64[:], types.containers.UniTuple(types.float64, size), types.intp)
        self.assertEqual(countParfors(test_impl, cptypes), 1)
        self.check(test_impl, a, b, size)

    def test_prange_non_parfor_index_then_opposite(self):
        if False:
            print('Hello World!')

        def test_impl(a, b, size):
            if False:
                return 10
            for i in numba.prange(size):
                b[i] = a[5]
            for i in numba.prange(size):
                a[i] = i
            b[0] += a[0]
            return b
        size = 10
        a = np.zeros(size)
        b = np.zeros(size)
        cptypes = (numba.float64[:], numba.float64[:], types.int64)
        self.assertEqual(countParfors(test_impl, cptypes), 2)
        self.check(test_impl, a, b, size)

    def test_prange_optional(self):
        if False:
            while True:
                i = 10

        def test_impl(arr, pred=None):
            if False:
                for i in range(10):
                    print('nop')
            for i in prange(1):
                if pred is not None:
                    arr[i] = 0.0
        arr = np.ones(10)
        self.check(test_impl, arr, None, check_arg_equality=[np.testing.assert_almost_equal, lambda x, y: x == y])
        self.assertEqual(arr.sum(), 10.0)

    def test_untraced_value_tuple(self):
        if False:
            while True:
                i = 10

        def test_impl():
            if False:
                while True:
                    i = 10
            a = (1.2, 1.3)
            return a[0]
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

    def test_recursive_untraced_value_tuple(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                while True:
                    i = 10
            a = ((1.2, 1.3),)
            return a[0][0]
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

    def test_untraced_value_parfor(self):
        if False:
            while True:
                i = 10

        def test_impl(arr):
            if False:
                for i in range(10):
                    print('nop')
            a = (1.2, 1.3)
            n1 = len(arr)
            arr2 = np.empty(n1, np.float64)
            for i in prange(n1):
                arr2[i] = arr[i] * a[0]
            n2 = len(arr2)
            arr3 = np.empty(n2, np.float64)
            for j in prange(n2):
                arr3[j] = arr2[j] - a[1]
            total = 0.0
            n3 = len(arr3)
            for k in prange(n3):
                total += arr3[k]
            return total + a[0]
        arg = (types.Array(types.int64, 1, 'C'),)
        self.assertEqual(countParfors(test_impl, arg), 1)
        arr = np.arange(10, dtype=np.int64)
        self.check(test_impl, arr)

    def test_setitem_2d_one_replaced(self):
        if False:
            while True:
                i = 10

        def test_impl(x):
            if False:
                print('Hello World!')
            count = 0
            for n in range(x.shape[0]):
                if n:
                    n
                x[count, :] = 1
                count += 1
            return x
        self.check(test_impl, np.zeros((3, 1)))

    def test_1array_control_flow(self):
        if False:
            print('Hello World!')

        def test_impl(arr, flag1, flag2):
            if False:
                for i in range(10):
                    print('nop')
            inv = np.arange(arr.size)
            if flag1:
                return inv.astype(np.float64)
            if flag2:
                ret = inv[inv]
            else:
                ret = inv[inv - 1]
            return ret / arr.size
        arr = np.arange(100)
        self.check(test_impl, arr, True, False)
        self.check(test_impl, arr, True, True)
        self.check(test_impl, arr, False, False)

    def test_2array_1_control_flow(self):
        if False:
            i = 10
            return i + 15

        def test_impl(arr, l, flag):
            if False:
                while True:
                    i = 10
            inv1 = np.arange(arr.size)
            inv2 = np.arange(l, arr.size + l)
            if flag:
                ret = inv1[inv1]
            else:
                ret = inv1[inv1 - 1]
            return ret / inv2
        arr = np.arange(100)
        self.check(test_impl, arr, 10, True)
        self.check(test_impl, arr, 10, False)

    def test_2array_2_control_flow(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(arr, l, flag):
            if False:
                for i in range(10):
                    print('nop')
            inv1 = np.arange(arr.size)
            inv2 = np.arange(l, arr.size + l)
            if flag:
                ret1 = inv1[inv1]
                ret2 = inv2[inv1]
            else:
                ret1 = inv1[inv1 - 1]
                ret2 = inv2[inv1 - 1]
            return ret1 / ret2
        arr = np.arange(100)
        self.check(test_impl, arr, 10, True)
        self.check(test_impl, arr, 10, False)

    def test_issue8515(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(n):
            if False:
                return 10
            r = np.zeros(n, dtype=np.intp)
            c = np.zeros(n, dtype=np.intp)
            for i in prange(n):
                for j in range(i):
                    c[i] += 1
            for i in prange(n):
                if i == 0:
                    continue
                r[i] = c[i] - c[i - 1]
            return r[1:]
        self.check(test_impl, 15)
        self.assertEqual(countParfors(test_impl, (types.int64,)), 2)

    def test_fusion_no_side_effects(self):
        if False:
            print('Hello World!')

        def test_impl(a, b):
            if False:
                for i in range(10):
                    print('nop')
            X = np.ones(100)
            b = math.ceil(b)
            Y = np.ones(100)
            c = int(max(a, b))
            return X + Y + c
        self.check(test_impl, 3.7, 4.3)
        self.assertEqual(countParfors(test_impl, (types.float64, types.float64)), 1)

@skip_parfors_unsupported
class TestParforsLeaks(MemoryLeakMixin, TestParforsBase):

    def check(self, pyfunc, *args, **kwargs):
        if False:
            while True:
                i = 10
        (cfunc, cpfunc) = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    def test_reduction(self):
        if False:
            return 10

        @njit(parallel=True)
        def test_impl(arr):
            if False:
                i = 10
                return i + 15
            return arr.sum()
        arr = np.arange(10).astype(np.float64)
        self.check(test_impl, arr)

    def test_multiple_reduction_vars(self):
        if False:
            for i in range(10):
                print('nop')

        @njit(parallel=True)
        def test_impl(arr):
            if False:
                for i in range(10):
                    print('nop')
            a = 0.0
            b = 1.0
            for i in prange(arr.size):
                a += arr[i]
                b += 1.0 / (arr[i] + 1)
            return a * b
        arr = np.arange(10).astype(np.float64)
        self.check(test_impl, arr)

@skip_parfors_unsupported
class TestParforsSlice(TestParforsBase):

    def test_parfor_slice1(self):
        if False:
            i = 10
            return i + 15

        def test_impl(a):
            if False:
                i = 10
                return i + 15
            (n,) = a.shape
            b = a[0:n - 2] + a[1:n - 1]
            return b
        self.check(test_impl, np.ones(10))

    def test_parfor_slice2(self):
        if False:
            return 10

        def test_impl(a, m):
            if False:
                i = 10
                return i + 15
            (n,) = a.shape
            b = a[0:n - 2] + a[1:m]
            return b
        self.check(test_impl, np.ones(10), 9)
        with self.assertRaises(AssertionError) as raises:
            njit(parallel=True)(test_impl)(np.ones(10), 10)
        self.assertIn('do not match', str(raises.exception))

    def test_parfor_slice3(self):
        if False:
            i = 10
            return i + 15

        def test_impl(a):
            if False:
                print('Hello World!')
            (m, n) = a.shape
            b = a[0:m - 1, 0:n - 1] + a[1:m, 1:n]
            return b
        self.check(test_impl, np.ones((4, 3)))

    def test_parfor_slice4(self):
        if False:
            print('Hello World!')

        def test_impl(a):
            if False:
                i = 10
                return i + 15
            (m, n) = a.shape
            b = a[:, 0:n - 1] + a[:, 1:n]
            return b
        self.check(test_impl, np.ones((4, 3)))

    def test_parfor_slice5(self):
        if False:
            while True:
                i = 10

        def test_impl(a):
            if False:
                print('Hello World!')
            (m, n) = a.shape
            b = a[0:m - 1, :] + a[1:m, :]
            return b
        self.check(test_impl, np.ones((4, 3)))

    def test_parfor_slice6(self):
        if False:
            while True:
                i = 10

        def test_impl(a):
            if False:
                return 10
            b = a.transpose()
            c = a[1, :] + b[:, 1]
            return c
        self.check(test_impl, np.ones((4, 3)))

    def test_parfor_slice7(self):
        if False:
            i = 10
            return i + 15

        def test_impl(a):
            if False:
                print('Hello World!')
            b = a.transpose()
            c = a[1, :] + b[1, :]
            return c
        self.check(test_impl, np.ones((3, 3)))
        with self.assertRaises(AssertionError) as raises:
            njit(parallel=True)(test_impl)(np.ones((3, 4)))
        self.assertIn('do not match', str(raises.exception))

    @disabled_test
    def test_parfor_slice8(self):
        if False:
            i = 10
            return i + 15

        def test_impl(a):
            if False:
                i = 10
                return i + 15
            (m, n) = a.shape
            b = a.transpose()
            b[1:m, 1:n] = a[1:m, 1:n]
            return b
        self.check(test_impl, np.arange(9).reshape((3, 3)))

    @disabled_test
    def test_parfor_slice9(self):
        if False:
            while True:
                i = 10

        def test_impl(a):
            if False:
                i = 10
                return i + 15
            (m, n) = a.shape
            b = a.transpose()
            b[1:n, 1:m] = a[:, 1:m]
            return b
        self.check(test_impl, np.arange(12).reshape((3, 4)))

    @disabled_test
    def test_parfor_slice10(self):
        if False:
            while True:
                i = 10

        def test_impl(a):
            if False:
                while True:
                    i = 10
            (m, n) = a.shape
            b = a.transpose()
            b[2, 1:m] = a[2, 1:m]
            return b
        self.check(test_impl, np.arange(9).reshape((3, 3)))

    def test_parfor_slice11(self):
        if False:
            while True:
                i = 10

        def test_impl(a):
            if False:
                i = 10
                return i + 15
            (m, n, l) = a.shape
            b = a.copy()
            b[:, 1, 1:l] = a[:, 2, 1:l]
            return b
        self.check(test_impl, np.arange(27).reshape((3, 3, 3)))

    def test_parfor_slice12(self):
        if False:
            while True:
                i = 10

        def test_impl(a):
            if False:
                print('Hello World!')
            (m, n) = a.shape
            b = a.copy()
            b[1, 1:-1] = a[0, :-2]
            return b
        self.check(test_impl, np.arange(12).reshape((3, 4)))

    def test_parfor_slice13(self):
        if False:
            i = 10
            return i + 15

        def test_impl(a):
            if False:
                return 10
            (m, n) = a.shape
            b = a.copy()
            c = -1
            b[1, 1:c] = a[0, -n:c - 1]
            return b
        self.check(test_impl, np.arange(12).reshape((3, 4)))

    def test_parfor_slice14(self):
        if False:
            return 10

        def test_impl(a):
            if False:
                print('Hello World!')
            (m, n) = a.shape
            b = a.copy()
            b[1, :-1] = a[0, -3:4]
            return b
        self.check(test_impl, np.arange(12).reshape((3, 4)))

    def test_parfor_slice15(self):
        if False:
            print('Hello World!')

        def test_impl(a):
            if False:
                print('Hello World!')
            (m, n) = a.shape
            b = a.copy()
            b[1, -(n - 1):] = a[0, -3:4]
            return b
        self.check(test_impl, np.arange(12).reshape((3, 4)))

    @disabled_test
    def test_parfor_slice16(self):
        if False:
            print('Hello World!')
        ' This test is disabled because if n is larger than the array size\n            then n and n-1 will both be the end of the array and thus the\n            slices will in fact be of different sizes and unable to fuse.\n        '

        def test_impl(a, b, n):
            if False:
                for i in range(10):
                    print('nop')
            assert a.shape == b.shape
            a[1:n] = 10
            b[0:n - 1] = 10
            return a * b
        self.check(test_impl, np.ones(10), np.zeros(10), 8)
        args = (numba.float64[:], numba.float64[:], numba.int64)
        self.assertEqual(countParfors(test_impl, args), 2)

    def test_parfor_slice17(self):
        if False:
            while True:
                i = 10

        def test_impl(m, A):
            if False:
                i = 10
                return i + 15
            B = np.zeros(m)
            n = len(A)
            B[-n:] = A
            return B
        self.check(test_impl, 10, np.ones(10))

    def test_parfor_slice18(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                while True:
                    i = 10
            a = np.zeros(10)
            a[1:8] = np.arange(0, 7)
            y = a[3]
            return y
        self.check(test_impl)

    def test_parfor_slice19(self):
        if False:
            print('Hello World!')

        def test_impl(X):
            if False:
                i = 10
                return i + 15
            X[:0] += 1
            return X
        self.check(test_impl, np.ones(10))

    def test_parfor_slice20(self):
        if False:
            while True:
                i = 10

        def test_impl():
            if False:
                return 10
            a = np.ones(10)
            c = a[1:]
            s = len(c)
            return s
        self.check(test_impl, check_scheduling=False)

    def test_parfor_slice21(self):
        if False:
            return 10

        def test_impl(x1, x2):
            if False:
                print('Hello World!')
            x1 = x1.reshape(x1.size, 1)
            x2 = x2.reshape(x2.size, 1)
            return x1 >= x2[:-1, :]
        x1 = np.random.rand(5)
        x2 = np.random.rand(6)
        self.check(test_impl, x1, x2)

    def test_parfor_slice22(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(x1, x2):
            if False:
                i = 10
                return i + 15
            b = np.zeros((10,))
            for i in prange(1):
                b += x1[:, x2]
            return b
        x1 = np.zeros((10, 7))
        x2 = np.array(4)
        self.check(test_impl, x1, x2)

    def test_parfor_slice23(self):
        if False:
            return 10

        def test_impl(x):
            if False:
                print('Hello World!')
            x[:0] = 2
            return x
        self.check(test_impl, np.ones(10))

    def test_parfor_slice24(self):
        if False:
            i = 10
            return i + 15

        def test_impl(m, A, n):
            if False:
                while True:
                    i = 10
            B = np.zeros(m)
            C = B[n:]
            C = A[:len(C)]
            return B
        for i in range(-15, 15):
            self.check(test_impl, 10, np.ones(10), i)

    def test_parfor_slice25(self):
        if False:
            return 10

        def test_impl(m, A, n):
            if False:
                print('Hello World!')
            B = np.zeros(m)
            C = B[:n]
            C = A[:len(C)]
            return B
        for i in range(-15, 15):
            self.check(test_impl, 10, np.ones(10), i)

    def test_parfor_slice26(self):
        if False:
            return 10

        def test_impl(a):
            if False:
                return 10
            (n,) = a.shape
            b = a.copy()
            b[-(n - 1):] = a[-3:4]
            return b
        self.check(test_impl, np.arange(4))

    def test_parfor_slice27(self):
        if False:
            print('Hello World!')

        def test_impl(a):
            if False:
                i = 10
                return i + 15
            n_valid_vals = 0
            for i in prange(a.shape[0]):
                if a[i] != 0:
                    n_valid_vals += 1
                if n_valid_vals:
                    unused = a[:n_valid_vals]
            return 0
        self.check(test_impl, np.arange(3))

    def test_parfor_array_access_lower_slice(self):
        if False:
            return 10
        for ts in [slice(1, 3, None), slice(2, None, None), slice(None, 2, -1), slice(None, None, None), slice(None, None, -2)]:

            def test_impl(n):
                if False:
                    for i in range(10):
                        print('nop')
                X = np.arange(n * 4).reshape((n, 4))
                y = 0
                for i in numba.prange(n):
                    y += X[i, ts].sum()
                return y
            n = 10
            self.check(test_impl, n)
            X = np.arange(n * 4).reshape((n, 4))

            def test_impl(X):
                if False:
                    return 10
                y = 0
                for i in numba.prange(X.shape[0]):
                    y += X[i, ts].sum()
                return y
            self.check(test_impl, X)

@skip_parfors_unsupported
class TestParforsOptions(TestParforsBase):

    def test_parfor_options(self):
        if False:
            print('Hello World!')

        def test_impl(a):
            if False:
                i = 10
                return i + 15
            n = a.shape[0]
            b = np.ones(n)
            c = np.array([i for i in range(n)])
            b[:n] = a + b * c
            for i in prange(n):
                c[i] = b[i] * a[i]
            return reduce(lambda x, y: x + y, c, 0)
        self.check(test_impl, np.ones(10))
        args = (numba.float64[:],)
        self.assertEqual(countParfors(test_impl, args), 1)
        self.assertEqual(countParfors(test_impl, args, fusion=False), 6)
        self.assertEqual(countParfors(test_impl, args, fusion=False, comprehension=False), 5)
        self.assertEqual(countParfors(test_impl, args, fusion=False, comprehension=False, setitem=False), 4)
        self.assertEqual(countParfors(test_impl, args, fusion=False, comprehension=False, setitem=False, prange=False), 3)
        self.assertEqual(countParfors(test_impl, args, fusion=False, comprehension=False, setitem=False, prange=False, reduction=False), 2)
        self.assertEqual(countParfors(test_impl, args, fusion=False, comprehension=False, setitem=False, prange=False, reduction=False, numpy=False), 0)

@skip_parfors_unsupported
class TestParforsBitMask(TestParforsBase):

    def test_parfor_bitmask1(self):
        if False:
            return 10

        def test_impl(a, n):
            if False:
                while True:
                    i = 10
            b = a > n
            a[b] = 0
            return a
        self.check(test_impl, np.arange(10), 5)

    def test_parfor_bitmask2(self):
        if False:
            i = 10
            return i + 15

        def test_impl(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a[b] = 0
            return a
        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    def test_parfor_bitmask3(self):
        if False:
            while True:
                i = 10

        def test_impl(a, b):
            if False:
                while True:
                    i = 10
            a[b] = a[b]
            return a
        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    def test_parfor_bitmask4(self):
        if False:
            i = 10
            return i + 15

        def test_impl(a, b):
            if False:
                print('Hello World!')
            a[b] = (2 * a)[b]
            return a
        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    def test_parfor_bitmask5(self):
        if False:
            while True:
                i = 10

        def test_impl(a, b):
            if False:
                print('Hello World!')
            a[b] = a[b] * a[b]
            return a
        a = np.arange(10)
        b = a > 5
        self.check(test_impl, a, b)

    def test_parfor_bitmask6(self):
        if False:
            i = 10
            return i + 15

        def test_impl(a, b, c):
            if False:
                print('Hello World!')
            a[b] = c
            return a
        a = np.arange(10)
        b = a > 5
        c = np.zeros(sum(b))
        with self.assertRaises(AssertionError) as raises:
            self.check(test_impl, a, b, c)
        self.assertIn("'@do_scheduling' not found", str(raises.exception))

@skip_parfors_unsupported
class TestParforsMisc(TestParforsBase):
    """
    Tests miscellaneous parts of ParallelAccelerator use.
    """

    def test_no_warn_if_cache_set(self):
        if False:
            i = 10
            return i + 15

        def pyfunc():
            if False:
                return 10
            arr = np.ones(100)
            for i in prange(arr.size):
                arr[i] += i
            return arr
        cfunc = njit(parallel=True, cache=True)(pyfunc)
        with warnings.catch_warnings(record=True) as raised_warnings:
            warnings.simplefilter('always')
            warnings.filterwarnings(action='ignore', module='typeguard')
            warnings.filterwarnings(action='ignore', message='.*TBB_INTERFACE_VERSION.*', category=numba.errors.NumbaWarning, module='numba\\.np\\.ufunc\\.parallel.*')
            cfunc()
        self.assertEqual(len(raised_warnings), 0)
        has_dynamic_globals = [cres.library.has_dynamic_globals for cres in cfunc.overloads.values()]
        self.assertEqual(has_dynamic_globals, [False])

    def test_statement_reordering_respects_aliasing(self):
        if False:
            while True:
                i = 10

        def impl():
            if False:
                print('Hello World!')
            a = np.zeros(10)
            a[1:8] = np.arange(0, 7)
            print('a[3]:', a[3])
            print('a[3]:', a[3])
            return a
        cres = self.compile_parallel(impl, ())
        with captured_stdout() as stdout:
            cres.entry_point()
        for line in stdout.getvalue().splitlines():
            self.assertEqual('a[3]: 2.0', line)

    def test_parfor_ufunc_typing(self):
        if False:
            i = 10
            return i + 15

        def test_impl(A):
            if False:
                return 10
            return np.isinf(A)
        A = np.array([np.inf, 0.0])
        cfunc = njit(parallel=True)(test_impl)
        old_seq_flag = numba.parfors.parfor.sequential_parfor_lowering
        try:
            numba.parfors.parfor.sequential_parfor_lowering = True
            np.testing.assert_array_equal(test_impl(A), cfunc(A))
        finally:
            numba.parfors.parfor.sequential_parfor_lowering = old_seq_flag

    def test_init_block_dce(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                while True:
                    i = 10
            res = 0
            arr = [1, 2, 3, 4, 5]
            numba.parfors.parfor.init_prange()
            dummy = arr
            for i in numba.prange(5):
                res += arr[i]
            return res + dummy[2]
        self.assertEqual(get_init_block_size(test_impl, ()), 0)

    def test_alias_analysis_for_parfor1(self):
        if False:
            print('Hello World!')

        def test_impl():
            if False:
                for i in range(10):
                    print('nop')
            acc = 0
            for _ in range(4):
                acc += 1
            data = np.zeros((acc,))
            return data
        self.check(test_impl)

    def test_no_state_change_in_gufunc_lowering_on_error(self):
        if False:
            for i in range(10):
                print('nop')
        BROKEN_MSG = 'BROKEN_MSG'

        @register_pass(mutates_CFG=True, analysis_only=False)
        class BreakParfors(AnalysisPass):
            _name = 'break_parfors'

            def __init__(self):
                if False:
                    print('Hello World!')
                AnalysisPass.__init__(self)

            def run_pass(self, state):
                if False:
                    for i in range(10):
                        print('nop')
                for blk in state.func_ir.blocks.values():
                    for stmt in blk.body:
                        if isinstance(stmt, numba.parfors.parfor.Parfor):

                            class Broken(list):

                                def difference(self, other):
                                    if False:
                                        while True:
                                            i = 10
                                    raise errors.LoweringError(BROKEN_MSG)
                            stmt.races = Broken()
                    return True

        class BreakParforsCompiler(CompilerBase):

            def define_pipelines(self):
                if False:
                    i = 10
                    return i + 15
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(BreakParfors, IRLegalization)
                pm.finalize()
                return [pm]

        @njit(parallel=True, pipeline_class=BreakParforsCompiler)
        def foo():
            if False:
                while True:
                    i = 10
            x = 1
            for _ in prange(1):
                x += 1
            return x
        self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)
        with self.assertRaises(errors.LoweringError) as raises:
            foo()
        self.assertIn(BROKEN_MSG, str(raises.exception))
        self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)

    def test_issue_5098(self):
        if False:
            print('Hello World!')

        class DummyType(types.Opaque):
            pass
        dummy_type = DummyType('my_dummy')
        register_model(DummyType)(models.OpaqueModel)

        class Dummy(object):
            pass

        @typeof_impl.register(Dummy)
        def typeof_Dummy(val, c):
            if False:
                return 10
            return dummy_type

        @unbox(DummyType)
        def unbox_index(typ, obj, c):
            if False:
                i = 10
                return i + 15
            return NativeValue(c.context.get_dummy_value())

        @overload_method(DummyType, 'method1', jit_options={'parallel': True})
        def _get_method1(obj, arr, func):
            if False:
                i = 10
                return i + 15

            def _foo(obj, arr, func):
                if False:
                    return 10

                def baz(a, f):
                    if False:
                        print('Hello World!')
                    c = a.copy()
                    c[np.isinf(a)] = np.nan
                    return f(c)
                length = len(arr)
                output_arr = np.empty(length, dtype=np.float64)
                for i in prange(length):
                    output_arr[i] = baz(arr[i], func)
                for i in prange(length - 1):
                    output_arr[i] += baz(arr[i], func)
                return output_arr
            return _foo

        @njit
        def bar(v):
            if False:
                return 10
            return v.mean()

        @njit
        def test1(d):
            if False:
                while True:
                    i = 10
            return d.method1(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), bar)
        save_state = numba.parfors.parfor.sequential_parfor_lowering
        self.assertFalse(save_state)
        try:
            test1(Dummy())
            self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)
        finally:
            numba.parfors.parfor.sequential_parfor_lowering = save_state

    def test_oversized_tuple_as_arg_to_kernel(self):
        if False:
            print('Hello World!')

        @njit(parallel=True)
        def oversize_tuple(idx):
            if False:
                for i in range(10):
                    print('nop')
            big_tup = (1, 2, 3, 4)
            z = 0
            for x in prange(10):
                z += big_tup[idx]
            return z
        with override_env_config('NUMBA_PARFOR_MAX_TUPLE_SIZE', '3'):
            with self.assertRaises(errors.UnsupportedParforsError) as raises:
                oversize_tuple(0)
        errstr = str(raises.exception)
        self.assertIn('Use of a tuple', errstr)
        self.assertIn('in a parallel region', errstr)

    def test_issue5167(self):
        if False:
            for i in range(10):
                print('nop')

        def ndvi_njit(img_nir, img_red):
            if False:
                i = 10
                return i + 15
            fillvalue = 0
            out_img = np.full(img_nir.shape, fillvalue, dtype=img_nir.dtype)
            dims = img_nir.shape
            for y in prange(dims[0]):
                for x in prange(dims[1]):
                    out_img[y, x] = (img_nir[y, x] - img_red[y, x]) / (img_nir[y, x] + img_red[y, x])
            return out_img
        tile_shape = (4, 4)
        array1 = np.random.uniform(low=1.0, high=10000.0, size=tile_shape)
        array2 = np.random.uniform(low=1.0, high=10000.0, size=tile_shape)
        self.check(ndvi_njit, array1, array2)

    def test_issue5065(self):
        if False:
            for i in range(10):
                print('nop')

        def reproducer(a, dist, dist_args):
            if False:
                for i in range(10):
                    print('nop')
            result = np.zeros((a.shape[0], a.shape[0]), dtype=np.float32)
            for i in prange(a.shape[0]):
                for j in range(i + 1, a.shape[0]):
                    d = dist(a[i], a[j], *dist_args)
                    result[i, j] = d
                    result[j, i] = d
            return result

        @njit
        def euclidean(x, y):
            if False:
                return 10
            result = 0.0
            for i in range(x.shape[0]):
                result += (x[i] - y[i]) ** 2
            return np.sqrt(result)
        a = np.random.random(size=(5, 2))
        got = njit(parallel=True)(reproducer)(a.copy(), euclidean, ())
        expected = reproducer(a.copy(), euclidean, ())
        np.testing.assert_allclose(got, expected)

    def test_issue5001(self):
        if False:
            print('Hello World!')

        def test_numba_parallel(myarray):
            if False:
                i = 10
                return i + 15
            result = [0] * len(myarray)
            for i in prange(len(myarray)):
                result[i] = len(myarray[i])
            return result
        myarray = (np.empty(100), np.empty(50))
        self.check(test_numba_parallel, myarray)

    def test_issue3169(self):
        if False:
            print('Hello World!')

        @njit
        def foo(grids):
            if False:
                for i in range(10):
                    print('nop')
            pass

        @njit(parallel=True)
        def bar(grids):
            if False:
                print('Hello World!')
            for x in prange(1):
                foo(grids)
        bar(([1],) * 2)

    @disabled_test
    def test_issue4846(self):
        if False:
            return 10
        mytype = namedtuple('mytype', ('a', 'b'))

        def outer(mydata):
            if False:
                print('Hello World!')
            for k in prange(3):
                inner(k, mydata)
            return mydata.a

        @njit(nogil=True)
        def inner(k, mydata):
            if False:
                print('Hello World!')
            f = (k, mydata.a)
            g = (k, mydata.b)
        mydata = mytype(a='a', b='b')
        self.check(outer, mydata)

    def test_issue3748(self):
        if False:
            i = 10
            return i + 15

        def test1b():
            if False:
                for i in range(10):
                    print('nop')
            x = (1, 2, 3, 4, 5)
            a = 0
            for i in prange(len(x)):
                a += x[i]
            return a
        self.check(test1b)

    def test_issue5277(self):
        if False:
            i = 10
            return i + 15

        def parallel_test(size, arr):
            if False:
                print('Hello World!')
            for x in prange(size[0]):
                for y in prange(size[1]):
                    arr[y][x] = x * 4.5 + y
            return arr
        size = (10, 10)
        arr = np.zeros(size, dtype=int)
        self.check(parallel_test, size, arr)

    def test_issue5570_ssa_races(self):
        if False:
            return 10

        @njit(parallel=True)
        def foo(src, method, out):
            if False:
                print('Hello World!')
            for i in prange(1):
                for j in range(1):
                    out[i, j] = 1
            if method:
                out += 1
            return out
        src = np.zeros((5, 5))
        method = 57
        out = np.zeros((2, 2))
        self.assertPreciseEqual(foo(src, method, out), foo.py_func(src, method, out))

    def test_issue6095_numpy_max(self):
        if False:
            while True:
                i = 10

        @njit(parallel=True)
        def find_maxima_3D_jit(args):
            if False:
                print('Hello World!')
            package = args
            for index in range(0, 10):
                z_stack = package[index, :, :]
            return np.max(z_stack)
        np.random.seed(0)
        args = np.random.random((10, 10, 10))
        self.assertPreciseEqual(find_maxima_3D_jit(args), find_maxima_3D_jit.py_func(args))

    def test_issue5942_1(self):
        if False:
            return 10

        def test_impl(gg, gg_next):
            if False:
                print('Hello World!')
            gs = gg.shape
            d = gs[0]
            for i_gg in prange(d):
                gg_next[i_gg, :] = gg[i_gg, :]
                gg_next[i_gg, 0] += 1
            return gg_next
        d = 4
        k = 2
        gg = np.zeros((d, k), dtype=np.int32)
        gg_next = np.zeros((d, k), dtype=np.int32)
        self.check(test_impl, gg, gg_next)

    def test_issue5942_2(self):
        if False:
            return 10

        def test_impl(d, k):
            if False:
                while True:
                    i = 10
            gg = np.zeros((d, k), dtype=np.int32)
            gg_next = np.zeros((d, k), dtype=np.int32)
            for i_gg in prange(d):
                for n in range(k):
                    gg[i_gg, n] = i_gg
                gg_next[i_gg, :] = gg[i_gg, :]
                gg_next[i_gg, 0] += 1
            return gg_next
        d = 4
        k = 2
        self.check(test_impl, d, k)

    @skip_unless_scipy
    def test_issue6102(self):
        if False:
            print('Hello World!')

        @njit(parallel=True)
        def f(r):
            if False:
                return 10
            for ir in prange(r.shape[0]):
                dist = np.inf
                tr = np.array([0, 0, 0], dtype=np.float32)
                for i in [1, 0, -1]:
                    dist_t = np.linalg.norm(r[ir, :] + i)
                    if dist_t < dist:
                        dist = dist_t
                        tr = np.array([i, i, i], dtype=np.float32)
                r[ir, :] += tr
            return r
        r = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertPreciseEqual(f(r), f.py_func(r))

    def test_issue6774(self):
        if False:
            return 10

        @njit(parallel=True)
        def test_impl():
            if False:
                while True:
                    i = 10
            n = 5
            na_mask = np.ones((n,))
            result = np.empty((n - 1,))
            for i in prange(len(result)):
                result[i] = np.sum(na_mask[i:i + 1])
            return result
        self.check(test_impl)

    def test_issue4963_globals(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                while True:
                    i = 10
            buf = np.zeros((_GLOBAL_INT_FOR_TESTING1, _GLOBAL_INT_FOR_TESTING2))
            return buf
        self.check(test_impl)

    def test_issue4963_freevars(self):
        if False:
            i = 10
            return i + 15
        _FREEVAR_INT_FOR_TESTING1 = 17
        _FREEVAR_INT_FOR_TESTING2 = 5

        def test_impl():
            if False:
                i = 10
                return i + 15
            buf = np.zeros((_FREEVAR_INT_FOR_TESTING1, _FREEVAR_INT_FOR_TESTING2))
            return buf
        self.check(test_impl)

    def test_issue_9182_recursion_error(self):
        if False:
            while True:
                i = 10
        from numba.types import ListType, Tuple, intp

        @numba.njit
        def _sink(x):
            if False:
                while True:
                    i = 10
            pass

        @numba.njit(cache=False, parallel=True)
        def _ground_node_rule(clauses, nodes):
            if False:
                i = 10
                return i + 15
            for piter in prange(len(nodes)):
                for clause in clauses:
                    clause_type = clause[0]
                    clause_variables = clause[2]
                    if clause_type == 0:
                        clause_var_1 = clause_variables[0]
                    elif len(clause_variables) == 2:
                        (clause_var_1, clause_var_2) = (clause_variables[0], clause_variables[1])
                    elif len(clause_variables) == 4:
                        pass
                    if clause_type == 1:
                        _sink(clause_var_1)
                        _sink(clause_var_2)
        _ground_node_rule.compile((ListType(Tuple([intp, intp, ListType(intp)])), ListType(intp)))

@skip_parfors_unsupported
class TestParforsDiagnostics(TestParforsBase):

    def check(self, pyfunc, *args, **kwargs):
        if False:
            return 10
        (cfunc, cpfunc) = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    def assert_fusion_equivalence(self, got, expected):
        if False:
            while True:
                i = 10
        a = self._fusion_equivalent(got)
        b = self._fusion_equivalent(expected)
        self.assertEqual(a, b)

    def _fusion_equivalent(self, thing):
        if False:
            return 10
        new = defaultdict(list)
        min_key = min(thing.keys())
        for k in sorted(thing.keys()):
            new[k - min_key] = [x - min_key for x in thing[k]]
        return new

    def assert_diagnostics(self, diagnostics, parfors_count=None, fusion_info=None, nested_fusion_info=None, replaced_fns=None, hoisted_allocations=None):
        if False:
            for i in range(10):
                print('nop')
        if parfors_count is not None:
            self.assertEqual(parfors_count, diagnostics.count_parfors())
        if fusion_info is not None:
            self.assert_fusion_equivalence(fusion_info, diagnostics.fusion_info)
        if nested_fusion_info is not None:
            self.assert_fusion_equivalence(nested_fusion_info, diagnostics.nested_fusion_info)
        if replaced_fns is not None:
            repl = diagnostics.replaced_fns.values()
            for x in replaced_fns:
                for replaced in repl:
                    if replaced[0] == x:
                        break
                else:
                    msg = 'Replacement for %s was not found. Had %s' % (x, repl)
                    raise AssertionError(msg)
        if hoisted_allocations is not None:
            hoisted_allocs = diagnostics.hoisted_allocations()
            self.assertEqual(hoisted_allocations, len(hoisted_allocs))
        with captured_stdout():
            for x in range(1, 5):
                diagnostics.dump(x)

    def test_array_expr(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                for i in range(10):
                    print('nop')
            n = 10
            a = np.ones(n)
            b = np.zeros(n)
            return a + b
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1, fusion_info={3: [4, 5]})

    def test_prange(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                return 10
            n = 10
            a = np.empty(n)
            for i in prange(n):
                a[i] = i * 10
            return a
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1)

    def test_user_varname(self):
        if False:
            for i in range(10):
                print('nop')
        'make sure original user variable name is used in fusion info\n        '

        def test_impl():
            if False:
                for i in range(10):
                    print('nop')
            n = 10
            x = np.ones(n)
            a = np.sin(x)
            b = np.cos(a * a)
            acc = 0
            for i in prange(n - 2):
                for j in prange(n - 1):
                    acc += b[i] + b[j + 1]
            return acc
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assertTrue(any(('slice(0, n, 1)' in r.message for r in diagnostics.fusion_reports)))

    def test_nested_prange(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                print('Hello World!')
            n = 10
            a = np.empty((n, n))
            for i in prange(n):
                for j in prange(n):
                    a[i, j] = i * 10 + j
            return a
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=2, nested_fusion_info={2: [1]})

    def test_function_replacement(self):
        if False:
            print('Hello World!')

        def test_impl():
            if False:
                while True:
                    i = 10
            n = 10
            a = np.ones(n)
            b = np.argmin(a)
            return b
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1, fusion_info={2: [3]}, replaced_fns=[('argmin', 'numpy')])

    def test_reduction(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                for i in range(10):
                    print('nop')
            n = 10
            a = np.ones(n + 1)
            acc = 0
            for i in prange(n):
                acc += a[i]
            return acc
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=2)

    def test_setitem(self):
        if False:
            while True:
                i = 10

        def test_impl():
            if False:
                print('Hello World!')
            n = 10
            a = np.ones(n)
            a[:] = 7
            return a
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, parfors_count=1)

    def test_allocation_hoisting(self):
        if False:
            return 10

        def test_impl():
            if False:
                for i in range(10):
                    print('nop')
            n = 10
            m = 5
            acc = 0
            for i in prange(n):
                temp = np.zeros((m,))
                for j in range(m):
                    temp[j] = i
                acc += temp[-1]
            return acc
        self.check(test_impl)
        cpfunc = self.compile_parallel(test_impl, ())
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        self.assert_diagnostics(diagnostics, hoisted_allocations=1)

class TestPrangeBase(TestParforsBase):

    def generate_prange_func(self, pyfunc, patch_instance):
        if False:
            i = 10
            return i + 15
        '\n        This function does the actual code augmentation to enable the explicit\n        testing of `prange` calls in place of `range`.\n        '
        pyfunc_code = pyfunc.__code__
        prange_names = list(pyfunc_code.co_names)
        if patch_instance is None:
            assert 'range' in pyfunc_code.co_names
            prange_names = tuple([x if x != 'range' else 'prange' for x in pyfunc_code.co_names])
            new_code = bytes(pyfunc_code.co_code)
        else:
            range_idx = pyfunc_code.co_names.index('range')
            range_locations = []
            for instr in dis.Bytecode(pyfunc_code):
                if instr.opname == 'LOAD_GLOBAL':
                    if _fix_LOAD_GLOBAL_arg(instr.arg) == range_idx:
                        range_locations.append(instr.offset + 1)
            prange_names.append('prange')
            prange_names = tuple(prange_names)
            prange_idx = len(prange_names) - 1
            if utils.PYVERSION == (3, 11):
                prange_idx = 1 + (prange_idx << 1)
            new_code = bytearray(pyfunc_code.co_code)
            assert len(patch_instance) <= len(range_locations)
            for i in patch_instance:
                idx = range_locations[i]
                new_code[idx] = prange_idx
            new_code = bytes(new_code)
        prange_code = pyfunc_code.replace(co_code=new_code, co_names=prange_names)
        pfunc = pytypes.FunctionType(prange_code, globals())
        return pfunc

    def prange_tester(self, pyfunc, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        The `prange` tester\n        This is a hack. It basically switches out range calls for prange.\n        It does this by copying the live code object of a function\n        containing 'range' then copying the .co_names and mutating it so\n        that 'range' is replaced with 'prange'. It then creates a new code\n        object containing the mutation and instantiates a function to contain\n        it. At this point three results are created:\n        1. The result of calling the original python function.\n        2. The result of calling a njit compiled version of the original\n            python function.\n        3. The result of calling a njit(parallel=True) version of the mutated\n           function containing `prange`.\n        The three results are then compared and the `prange` based function's\n        llvm_ir is inspected to ensure the scheduler code is present.\n\n        Arguments:\n         pyfunc - the python function to test\n         args - data arguments to pass to the pyfunc under test\n\n        Keyword Arguments:\n         patch_instance - iterable containing which instances of `range` to\n                          replace. If not present all instance of `range` are\n                          replaced.\n         scheduler_type - 'signed', 'unsigned' or None, default is None.\n                           Supply in cases where the presence of a specific\n                           scheduler is to be asserted.\n         check_fastmath - if True then a check will be performed to ensure the\n                          IR contains instructions labelled with 'fast'\n         check_fastmath_result - if True then a check will be performed to\n                                 ensure the result of running with fastmath\n                                 on matches that of the pyfunc\n         Remaining kwargs are passed to np.testing.assert_almost_equal\n\n\n        Example:\n            def foo():\n                acc = 0\n                for x in range(5):\n                    for y in range(10):\n                        acc +=1\n                return acc\n\n            # calling as\n            prange_tester(foo)\n            # will test code equivalent to\n            # def foo():\n            #     acc = 0\n            #     for x in prange(5): # <- changed\n            #         for y in prange(10): # <- changed\n            #             acc +=1\n            #     return acc\n\n            # calling as\n            prange_tester(foo, patch_instance=[1])\n            # will test code equivalent to\n            # def foo():\n            #     acc = 0\n            #     for x in range(5): # <- outer loop (0) unchanged\n            #         for y in prange(10): # <- inner loop (1) changed\n            #             acc +=1\n            #     return acc\n\n        "
        patch_instance = kwargs.pop('patch_instance', None)
        check_fastmath = kwargs.pop('check_fastmath', False)
        check_fastmath_result = kwargs.pop('check_fastmath_result', False)
        pfunc = self.generate_prange_func(pyfunc, patch_instance)
        sig = tuple([numba.typeof(x) for x in args])
        cfunc = self.compile_njit(pyfunc, sig)
        with warnings.catch_warnings(record=True) as raised_warnings:
            warnings.simplefilter('always')
            cpfunc = self.compile_parallel(pfunc, sig)
        if check_fastmath:
            self.assert_fastmath(pfunc, sig)
        if check_fastmath_result:
            fastcpfunc = self.compile_parallel_fastmath(pfunc, sig)
            kwargs = dict({'fastmath_pcres': fastcpfunc}, **kwargs)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)
        return raised_warnings

@skip_parfors_unsupported
class TestPrangeBasic(TestPrangeBase):
    """ Tests Prange """

    def test_prange01(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                print('Hello World!')
            n = 4
            A = np.zeros(n)
            for i in range(n):
                A[i] = 2.0 * i
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange02(self):
        if False:
            print('Hello World!')

        def test_impl():
            if False:
                print('Hello World!')
            n = 4
            A = np.zeros(n - 1)
            for i in range(1, n):
                A[i - 1] = 2.0 * i
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange03(self):
        if False:
            while True:
                i = 10

        def test_impl():
            if False:
                for i in range(10):
                    print('nop')
            s = 10
            for i in range(10):
                s += 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange03mul(self):
        if False:
            return 10

        def test_impl():
            if False:
                print('Hello World!')
            s = 3
            for i in range(10):
                s *= 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange03sub(self):
        if False:
            print('Hello World!')

        def test_impl():
            if False:
                print('Hello World!')
            s = 100
            for i in range(10):
                s -= 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange03div(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                for i in range(10):
                    print('nop')
            s = 10
            for i in range(10):
                s /= 2
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange04(self):
        if False:
            while True:
                i = 10

        def test_impl():
            if False:
                i = 10
                return i + 15
            a = 2
            b = 3
            A = np.empty(4)
            for i in range(4):
                if i == a:
                    A[i] = b
                else:
                    A[i] = 0
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange05(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                i = 10
                return i + 15
            n = 4
            A = np.ones(n, dtype=np.float64)
            s = 0
            for i in range(1, n - 1, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange06(self):
        if False:
            return 10

        def test_impl():
            if False:
                while True:
                    i = 10
            n = 4
            A = np.ones(n, dtype=np.float64)
            s = 0
            for i in range(1, 1, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange07(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                i = 10
                return i + 15
            n = 4
            A = np.ones(n, dtype=np.float64)
            s = 0
            for i in range(n, 1):
                s += A[i]
            return s
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange08(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                while True:
                    i = 10
            n = 4
            A = np.ones(n)
            acc = 0
            for i in range(len(A)):
                for j in range(len(A)):
                    acc += A[i]
            return acc
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange08_1(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl():
            if False:
                print('Hello World!')
            n = 4
            A = np.ones(n)
            acc = 0
            for i in range(4):
                for j in range(4):
                    acc += A[i]
            return acc
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange09(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                print('Hello World!')
            n = 4
            acc = 0
            for i in range(n):
                for j in range(n):
                    acc += 1
            return acc
        self.prange_tester(test_impl, patch_instance=[1], scheduler_type='unsigned', check_fastmath=True)

    def test_prange10(self):
        if False:
            print('Hello World!')

        def test_impl():
            if False:
                for i in range(10):
                    print('nop')
            n = 4
            acc2 = 0
            for j in range(n):
                acc1 = 0
                for i in range(n):
                    acc1 += 1
                acc2 += acc1
            return acc2
        self.prange_tester(test_impl, patch_instance=[0], scheduler_type='unsigned', check_fastmath=True)

    @unittest.skip('list append is not thread-safe yet (#2391, #2408)')
    def test_prange11(self):
        if False:
            return 10

        def test_impl():
            if False:
                return 10
            n = 4
            return [np.sin(j) for j in range(n)]
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange12(self):
        if False:
            print('Hello World!')

        def test_impl():
            if False:
                print('Hello World!')
            acc = 0
            n = 4
            X = np.ones(n)
            for i in range(-len(X)):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)

    def test_prange13(self):
        if False:
            print('Hello World!')

        def test_impl(n):
            if False:
                print('Hello World!')
            acc = 0
            for i in range(n):
                acc += 1
            return acc
        self.prange_tester(test_impl, np.int32(4), scheduler_type='unsigned', check_fastmath=True)

    def test_prange14(self):
        if False:
            while True:
                i = 10

        def test_impl(A):
            if False:
                return 10
            s = 3
            for i in range(len(A)):
                s += A[i] * 2
            return s
        self.prange_tester(test_impl, np.random.ranf(4), scheduler_type='unsigned', check_fastmath=True)

    def test_prange15(self):
        if False:
            while True:
                i = 10

        def test_impl(N):
            if False:
                for i in range(10):
                    print('nop')
            acc = 0
            for i in range(N):
                x = np.ones((1, 1))
                acc += x[0, 0]
            return acc
        self.prange_tester(test_impl, 1024, scheduler_type='unsigned', check_fastmath=True)

    def test_prange16(self):
        if False:
            while True:
                i = 10

        def test_impl(N):
            if False:
                for i in range(10):
                    print('nop')
            acc = 0
            for i in range(-N, N):
                acc += 2
            return acc
        self.prange_tester(test_impl, 1024, scheduler_type='signed', check_fastmath=True)

    def test_prange17(self):
        if False:
            print('Hello World!')

        def test_impl(N):
            if False:
                i = 10
                return i + 15
            acc = 0
            X = np.ones(N)
            for i in range(-N, N):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed', check_fastmath=True)

    def test_prange18(self):
        if False:
            print('Hello World!')

        def test_impl(N):
            if False:
                i = 10
                return i + 15
            acc = 0
            X = np.ones(N)
            for i in range(-N, 5):
                acc += X[i]
                for j in range(-4, N):
                    acc += X[j]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed', check_fastmath=True)

    def test_prange19(self):
        if False:
            return 10

        def test_impl(N):
            if False:
                i = 10
                return i + 15
            acc = 0
            M = N + 4
            X = np.ones((N, M))
            for i in range(-N, N):
                for j in range(-M, M):
                    acc += X[i, j]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed', check_fastmath=True)

    def test_prange20(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(N):
            if False:
                i = 10
                return i + 15
            acc = 0
            X = np.ones(N)
            for i in range(-1, N):
                acc += X[i]
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed', check_fastmath=True)

    def test_prange21(self):
        if False:
            print('Hello World!')

        def test_impl(N):
            if False:
                while True:
                    i = 10
            acc = 0
            for i in range(-3, -1):
                acc += 3
            return acc
        self.prange_tester(test_impl, 9, scheduler_type='signed', check_fastmath=True)

    def test_prange22(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                i = 10
                return i + 15
            a = 0
            b = 3
            A = np.empty(4)
            for i in range(-2, 2):
                if i == a:
                    A[i] = b
                elif i < 1:
                    A[i] = -1
                else:
                    A[i] = 7
            return A
        self.prange_tester(test_impl, scheduler_type='signed', check_fastmath=True, check_fastmath_result=True)

    def test_prange23(self):
        if False:
            i = 10
            return i + 15

        def test_impl(A):
            if False:
                for i in range(10):
                    print('nop')
            for i in range(len(A)):
                A[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='unsigned', check_fastmath=True, check_fastmath_result=True)

    def test_prange24(self):
        if False:
            print('Hello World!')

        def test_impl(A):
            if False:
                return 10
            for i in range(-len(A), 0):
                A[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='signed', check_fastmath=True, check_fastmath_result=True)

    def test_prange25(self):
        if False:
            print('Hello World!')

        def test_impl(A):
            if False:
                return 10
            n = len(A)
            buf = [np.zeros_like(A) for _ in range(n)]
            for i in range(n):
                buf[i] = A + i
            return buf
        A = np.ones((10,))
        self.prange_tester(test_impl, A, patch_instance=[1], scheduler_type='unsigned', check_fastmath=True, check_fastmath_result=True)
        cpfunc = self.compile_parallel(test_impl, (numba.typeof(A),))
        diagnostics = cpfunc.metadata['parfor_diagnostics']
        hoisted_allocs = diagnostics.hoisted_allocations()
        self.assertEqual(len(hoisted_allocs), 0)

    def test_prange26(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(A):
            if False:
                return 10
            B = A[::3]
            for i in range(len(B)):
                B[i] = i
            return A
        A = np.zeros(32)[::2]
        self.prange_tester(test_impl, A, scheduler_type='unsigned', check_fastmath=True, check_fastmath_result=True)

    def test_prange27(self):
        if False:
            return 10

        def test_impl(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            for j in range(b[0] - 1):
                for k in range(2):
                    z = np.abs(a[c - 1:c + 1])
            return 0
        self.prange_tester(test_impl, np.arange(20), np.asarray([4, 4, 4, 4, 4, 4, 4, 4, 4, 4]), 0, patch_instance=[1], scheduler_type='unsigned', check_fastmath=True)

    def test_prange28(self):
        if False:
            return 10

        def test_impl(x, y):
            if False:
                for i in range(10):
                    print('nop')
            out = np.zeros(len(y))
            for idx in range(0, len(y)):
                i0 = y[idx, 0]
                i1 = y[idx, 1]
                Pt1 = x[i0]
                Pt2 = x[i1]
                v = Pt1 - Pt2
                vl2 = v[0] + v[1]
                out[idx] = vl2
            return out
        X = np.array([[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
        Y = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        self.prange_tester(test_impl, X, Y, scheduler_type='unsigned', check_fastmath=True, check_fastmath_result=True)

    def test_prange29(self):
        if False:
            while True:
                i = 10

        def test_impl(flag):
            if False:
                return 10
            result = 0
            if flag:
                for i in range(1):
                    result += 1
            else:
                for i in range(1):
                    result -= 3
            return result
        self.prange_tester(test_impl, True)
        self.prange_tester(test_impl, False)

    def test_prange30(self):
        if False:
            i = 10
            return i + 15

        def test_impl(x, par, numthreads):
            if False:
                print('Hello World!')
            n_par = par.shape[0]
            n_x = len(x)
            result = np.zeros((n_par, n_x), dtype=np.float64)
            chunklen = (len(x) + numthreads - 1) // numthreads
            for i in range(numthreads):
                start = i * chunklen
                stop = (i + 1) * chunklen
                result[:, start:stop] = x[start:stop] * par[:]
            return result
        x = np.array(np.arange(0, 6, 1.0))
        par = np.array([1.0, 2.0, 3.0])
        self.prange_tester(test_impl, x, par, 2)

@skip_parfors_unsupported
class TestPrangeSpecific(TestPrangeBase):
    """ Tests specific features/problems found under prange"""

    def test_prange_two_instances_same_reduction_var(self):
        if False:
            print('Hello World!')

        def test_impl(n):
            if False:
                while True:
                    i = 10
            c = 0
            for i in range(n):
                c += 1
                if i > 10:
                    c += 1
            return c
        self.prange_tester(test_impl, 9)

    def test_prange_conflicting_reduction_ops(self):
        if False:
            print('Hello World!')

        def test_impl(n):
            if False:
                for i in range(10):
                    print('nop')
            c = 0
            for i in range(n):
                c += 1
                if i > 10:
                    c *= 1
            return c
        with self.assertRaises(errors.UnsupportedError) as raises:
            self.prange_tester(test_impl, 9)
        msg = 'Reduction variable c has multiple conflicting reduction operators.'
        self.assertIn(msg, str(raises.exception))

    def test_prange_two_conditional_reductions(self):
        if False:
            i = 10
            return i + 15

        def test_impl():
            if False:
                print('Hello World!')
            A = B = 0
            for k in range(1):
                if k == 2:
                    A += 1
                else:
                    x = np.zeros((1, 1))
                    if x[0, 0]:
                        B += 1
            return (A, B)
        self.prange_tester(test_impl)

    def test_prange_nested_reduction1(self):
        if False:
            return 10

        def test_impl():
            if False:
                i = 10
                return i + 15
            A = 0
            for k in range(1):
                for i in range(1):
                    if i == 0:
                        A += 1
            return A
        self.prange_tester(test_impl)

    @disabled_test
    def test_check_error_model(self):
        if False:
            print('Hello World!')

        def test_impl():
            if False:
                while True:
                    i = 10
            n = 32
            A = np.zeros(n)
            for i in range(n):
                A[i] = 1 / i
            return A
        with self.assertRaises(ZeroDivisionError) as raises:
            test_impl()
        pfunc = self.generate_prange_func(test_impl, None)
        pcres = self.compile_parallel(pfunc, ())
        pfcres = self.compile_parallel_fastmath(pfunc, ())
        with self.assertRaises(ZeroDivisionError) as raises:
            pcres.entry_point()
        result = pfcres.entry_point()
        self.assertEqual(result[0], np.inf)

    def test_check_alias_analysis(self):
        if False:
            return 10

        def test_impl(A):
            if False:
                return 10
            for i in range(len(A)):
                B = A[i]
                B[:] = 1
            return A
        A = np.zeros(32).reshape(4, 8)
        self.prange_tester(test_impl, A, scheduler_type='unsigned', check_fastmath=True, check_fastmath_result=True)
        pfunc = self.generate_prange_func(test_impl, None)
        sig = tuple([numba.typeof(A)])
        cres = self.compile_parallel_fastmath(pfunc, sig)
        _ir = self._get_gufunc_ir(cres)
        for (k, v) in _ir.items():
            for line in v.splitlines():
                if 'define' in line and k in line:
                    self.assertEqual(line.count('noalias'), 2)
                    break

    def test_prange_raises_invalid_step_size(self):
        if False:
            print('Hello World!')

        def test_impl(N):
            if False:
                return 10
            acc = 0
            for i in range(0, N, 2):
                acc += 2
            return acc
        with self.assertRaises(errors.UnsupportedRewriteError) as raises:
            self.prange_tester(test_impl, 1024)
        msg = 'Only constant step size of 1 is supported for prange'
        self.assertIn(msg, str(raises.exception))

    def test_prange_fastmath_check_works(self):
        if False:
            return 10

        def test_impl():
            if False:
                while True:
                    i = 10
            n = 128
            A = 0
            for i in range(n):
                A += i / 2.0
            return A
        self.prange_tester(test_impl, scheduler_type='unsigned', check_fastmath=True)
        pfunc = self.generate_prange_func(test_impl, None)
        cres = self.compile_parallel_fastmath(pfunc, ())
        ir = self._get_gufunc_ir(cres)
        _id = '%[A-Z_0-9]?(.[0-9]+)+[.]?[i]?'
        recipr_str = '\\s+%s = fmul fast double %s, 5.000000e-01'
        reciprocal_inst = re.compile(recipr_str % (_id, _id))
        fadd_inst = re.compile('\\s+%s = fadd fast double %s, %s' % (_id, _id, _id))
        for (name, kernel) in ir.items():
            splitted = kernel.splitlines()
            for (i, x) in enumerate(splitted):
                if reciprocal_inst.match(x):
                    break
            self.assertTrue(fadd_inst.match(splitted[i + 1]))

    def test_parfor_alias1(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(n):
            if False:
                while True:
                    i = 10
            b = np.zeros((n, n))
            a = b[0]
            for j in range(n):
                a[j] = j + 1
            return b.sum()
        self.prange_tester(test_impl, 4)

    def test_parfor_alias2(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(n):
            if False:
                for i in range(10):
                    print('nop')
            b = np.zeros((n, n))
            for i in range(n):
                a = b[i]
                for j in range(n):
                    a[j] = i + j
            return b.sum()
        self.prange_tester(test_impl, 4)

    def test_parfor_alias3(self):
        if False:
            return 10

        def test_impl(n):
            if False:
                for i in range(10):
                    print('nop')
            b = np.zeros((n, n, n))
            for i in range(n):
                a = b[i]
                for j in range(n):
                    c = a[j]
                    for k in range(n):
                        c[k] = i + j + k
            return b.sum()
        self.prange_tester(test_impl, 4)

    def test_parfor_race_1(self):
        if False:
            i = 10
            return i + 15

        def test_impl(x, y):
            if False:
                print('Hello World!')
            for j in range(y):
                k = x
            return k
        raised_warnings = self.prange_tester(test_impl, 10, 20)
        warning_obj = raised_warnings[0]
        expected_msg = 'Variable k used in parallel loop may be written to simultaneously by multiple workers and may result in non-deterministic or unintended results.'
        self.assertIn(expected_msg, str(warning_obj.message))

    def test_nested_parfor_push_call_vars(self):
        if False:
            while True:
                i = 10
        " issue 3686: if a prange has something inside it that causes\n            a nested parfor to be generated and both the inner and outer\n            parfor use the same call variable defined outside the parfors\n            then ensure that when that call variable is pushed into the\n            parfor that the call variable isn't duplicated with the same\n            name resulting in a redundant type lock.\n        "

        def test_impl():
            if False:
                i = 10
                return i + 15
            B = 0
            f = np.negative
            for i in range(1):
                this_matters = f(1.0)
                B += f(np.zeros(1))[0]
            for i in range(2):
                this_matters = f(1.0)
                B += f(np.zeros(1))[0]
            return B
        self.prange_tester(test_impl)

    def test_copy_global_for_parfor(self):
        if False:
            i = 10
            return i + 15
        ' issue4903: a global is copied next to a parfor so that\n            it can be inlined into the parfor and thus not have to be\n            passed to the parfor (i.e., an unsupported function type).\n            This global needs to be renamed in the block into which\n            it is copied.\n        '

        def test_impl(zz, tc):
            if False:
                for i in range(10):
                    print('nop')
            lh = np.zeros(len(tc))
            lc = np.zeros(len(tc))
            for i in range(1):
                nt = tc[i]
                for t in range(nt):
                    lh += np.exp(zz[i, t])
                for t in range(nt):
                    lc += np.exp(zz[i, t])
            return (lh, lc)
        m = 2
        zz = np.ones((m, m, m))
        tc = np.ones(m, dtype=np.int_)
        self.prange_tester(test_impl, zz, tc, patch_instance=[0])

    def test_multiple_call_getattr_object(self):
        if False:
            print('Hello World!')

        def test_impl(n):
            if False:
                i = 10
                return i + 15
            B = 0
            f = np.negative
            for i in range(1):
                this_matters = f(1.0)
                B += f(n)
            return B
        self.prange_tester(test_impl, 1.0)

    def test_argument_alias_recarray_field(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(n):
            if False:
                i = 10
                return i + 15
            for i in range(len(n)):
                n.x[i] = 7.0
            return n
        X1 = np.zeros(10, dtype=[('x', float), ('y', int)])
        X2 = np.zeros(10, dtype=[('x', float), ('y', int)])
        X3 = np.zeros(10, dtype=[('x', float), ('y', int)])
        v1 = X1.view(np.recarray)
        v2 = X2.view(np.recarray)
        v3 = X3.view(np.recarray)
        python_res = list(test_impl(v1))
        njit_res = list(njit(test_impl)(v2))
        pa_func = njit(test_impl, parallel=True)
        pa_res = list(pa_func(v3))
        self.assertEqual(python_res, njit_res)
        self.assertEqual(python_res, pa_res)

    def test_mutable_list_param(self):
        if False:
            for i in range(10):
                print('nop')
        ' issue3699: test that mutable variable to call in loop\n            is not hoisted.  The call in test_impl forces a manual\n            check here rather than using prange_tester.\n        '

        @njit
        def list_check(X):
            if False:
                return 10
            ' If the variable X is hoisted in the test_impl prange\n                then subsequent list_check calls would return increasing\n                values.\n            '
            ret = X[-1]
            a = X[-1] + 1
            X.append(a)
            return ret

        def test_impl(n):
            if False:
                i = 10
                return i + 15
            for i in prange(n):
                X = [100]
                a = list_check(X)
            return a
        python_res = test_impl(10)
        njit_res = njit(test_impl)(10)
        pa_func = njit(test_impl, parallel=True)
        pa_res = pa_func(10)
        self.assertEqual(python_res, njit_res)
        self.assertEqual(python_res, pa_res)

    def test_list_comprehension_prange(self):
        if False:
            print('Hello World!')

        def test_impl(x):
            if False:
                i = 10
                return i + 15
            return np.array([len(x[i]) for i in range(len(x))])
        x = [np.array([1, 2, 3], dtype=int), np.array([1, 2], dtype=int)]
        self.prange_tester(test_impl, x)

    def test_ssa_false_reduction(self):
        if False:
            for i in range(10):
                print('nop')

        def test_impl(image, a, b):
            if False:
                for i in range(10):
                    print('nop')
            empty = np.zeros(image.shape)
            for i in range(image.shape[0]):
                r = image[i][0] / 255.0
                if a == 0:
                    h = 0
                if b == 0:
                    h = 0
                empty[i] = [h, h, h]
            return empty
        image = np.zeros((3, 3), dtype=np.int32)
        self.prange_tester(test_impl, image, 0, 0)

    def test_list_setitem_hoisting(self):
        if False:
            while True:
                i = 10

        def test_impl():
            if False:
                i = 10
                return i + 15
            n = 5
            a = np.empty(n, dtype=np.int64)
            for k in range(5):
                X = [0]
                X[0] = 1
                a[k] = X[0]
            return a
        self.prange_tester(test_impl)

    def test_record_array_setitem(self):
        if False:
            print('Hello World!')
        state_dtype = np.dtype([('var', np.int32)])

        def test_impl(states):
            if False:
                i = 10
                return i + 15
            for i in range(1):
                states[i]['var'] = 1

        def comparer(a, b):
            if False:
                while True:
                    i = 10
            assert a[0]['var'] == b[0]['var']
        self.prange_tester(test_impl, np.zeros(shape=1, dtype=state_dtype), check_arg_equality=[comparer])

    def test_record_array_setitem_yield_array(self):
        if False:
            return 10
        state_dtype = np.dtype([('x', np.intp)])

        def test_impl(states):
            if False:
                print('Hello World!')
            n = states.size
            for i in range(states.size):
                states['x'][i] = 7 + i
            return states
        states = np.zeros(10, dtype=state_dtype)

        def comparer(a, b):
            if False:
                while True:
                    i = 10
            np.testing.assert_equal(a, b)
        self.prange_tester(test_impl, states, check_arg_equality=[comparer])

    def test_issue7501(self):
        if False:
            while True:
                i = 10

        def test_impl(size, case):
            if False:
                for i in range(10):
                    print('nop')
            result = np.zeros((size,))
            if case == 1:
                for i in range(size):
                    result[i] += 1
            else:
                for i in range(size):
                    result[i] += 2
            return result[0]
        self.prange_tester(test_impl, 3, 1)

    def test_kde_example(self):
        if False:
            return 10

        def test_impl(X):
            if False:
                return 10
            b = 0.5
            points = np.array([-1.0, 2.0, 5.0])
            N = points.shape[0]
            n = X.shape[0]
            exps = 0
            for i in range(n):
                p = X[i]
                d = -(p - points) ** 2 / (2 * b ** 2)
                m = np.min(d)
                exps += m - np.log(b * N) + np.log(np.sum(np.exp(d - m)))
            return exps
        n = 128
        X = np.random.ranf(n)
        self.prange_tester(test_impl, X)

    @skip_parfors_unsupported
    def test_issue_due_to_max_label(self):
        if False:
            while True:
                i = 10
        out = subp.check_output([sys.executable, '-m', 'numba.tests.parfors_max_label_error'], timeout=30, stderr=subp.STDOUT)
        self.assertIn('TEST PASSED', out.decode())

    @skip_parfors_unsupported
    def test_issue7578(self):
        if False:
            i = 10
            return i + 15

        def test_impl(x):
            if False:
                for i in range(10):
                    print('nop')
            A = np.zeros_like(x)
            tmp = np.cos(x)
            for i in range(len(x)):
                A[i] = tmp.sum()
            return A
        x = np.arange(10.0)
        self.prange_tester(test_impl, x)

@skip_parfors_unsupported
class TestParforChunksizing(TestCase):
    """
    Tests chunksize handling in ParallelAccelerator.
    """
    _numba_parallel_test_ = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        set_parallel_chunksize(0)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        set_parallel_chunksize(0)

    def test_python_parallel_chunksize_basic(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(get_parallel_chunksize(), 0)
        set_parallel_chunksize(8)
        self.assertEqual(get_parallel_chunksize(), 8)
        set_parallel_chunksize(0)
        self.assertEqual(get_parallel_chunksize(), 0)

    def test_python_with_chunksize(self):
        if False:
            while True:
                i = 10
        self.assertEqual(get_parallel_chunksize(), 0)
        with parallel_chunksize(8):
            self.assertEqual(get_parallel_chunksize(), 8)
        self.assertEqual(get_parallel_chunksize(), 0)

    def test_njit_parallel_chunksize_basic(self):
        if False:
            print('Hello World!')

        @njit
        def get_cs():
            if False:
                print('Hello World!')
            return get_parallel_chunksize()

        @njit
        def set_cs(x):
            if False:
                print('Hello World!')
            return set_parallel_chunksize(x)
        self.assertEqual(get_cs(), 0)
        set_cs(8)
        self.assertEqual(get_cs(), 8)
        set_cs(0)
        self.assertEqual(get_cs(), 0)

    def test_njit_with_chunksize(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def test_impl(x):
            if False:
                for i in range(10):
                    print('nop')
            cs1 = get_parallel_chunksize()
            with parallel_chunksize(8):
                cs2 = get_parallel_chunksize()
            cs3 = get_parallel_chunksize()
            return (cs1, cs2, cs3)
        (cs1, cs2, cs3) = test_impl(8)
        self.assertEqual(cs1, 0)
        self.assertEqual(cs2, 8)
        self.assertEqual(cs3, 0)

    def test_all_iterations_reset_chunksize(self):
        if False:
            while True:
                i = 10
        ' Test that all the iterations get run if you set the\n            chunksize.  Also check that the chunksize that each\n            worker thread sees has been reset to 0. '

        @njit(parallel=True)
        def test_impl(cs, n):
            if False:
                i = 10
                return i + 15
            res = np.zeros(n)
            inner_cs = np.full(n, -13)
            with numba.parallel_chunksize(cs):
                for i in numba.prange(n):
                    inner_cs[i] = numba.get_parallel_chunksize()
                    res[i] = 13
            return (res, inner_cs)
        for j in [1000, 997, 943, 961]:
            for i in range(15):
                (res, inner_cs) = test_impl(i + 1, j)
                self.assertTrue(np.all(res == 13))
                self.assertTrue(np.all(inner_cs == 0))

    def test_njit_parallel_chunksize_negative(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError) as raised:

            @njit
            def neg_test():
                if False:
                    while True:
                        i = 10
                set_parallel_chunksize(-1)
            neg_test()
        msg = 'chunksize must be greater than or equal to zero'
        self.assertIn(msg, str(raised.exception))

    def test_python_parallel_chunksize_negative(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError) as raised:
            set_parallel_chunksize(-1)
        msg = 'chunksize must be greater than or equal to zero'
        self.assertIn(msg, str(raised.exception))

    def test_njit_parallel_chunksize_invalid_type(self):
        if False:
            return 10
        with self.assertRaises(errors.TypingError) as raised:

            @njit
            def impl():
                if False:
                    while True:
                        i = 10
                set_parallel_chunksize('invalid_type')
            impl()
        msg = 'The parallel chunksize must be an integer'
        self.assertIn(msg, str(raised.exception))

    def test_python_parallel_chunksize_invalid_type(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError) as raised:
            set_parallel_chunksize('invalid_type')
        msg = 'The parallel chunksize must be an integer'
        self.assertIn(msg, str(raised.exception))

@skip_parfors_unsupported
@x86_only
class TestParforsVectorizer(TestPrangeBase):
    _numba_parallel_test_ = False

    def get_gufunc_asm(self, func, schedule_type, *args, **kwargs):
        if False:
            while True:
                i = 10
        fastmath = kwargs.pop('fastmath', False)
        cpu_name = kwargs.pop('cpu_name', 'skylake-avx512')
        assertions = kwargs.pop('assertions', True)
        cpu_features = kwargs.pop('cpu_features', '-prefer-256-bit')
        env_opts = {'NUMBA_CPU_NAME': cpu_name, 'NUMBA_CPU_FEATURES': cpu_features}
        overrides = []
        for (k, v) in env_opts.items():
            overrides.append(override_env_config(k, v))
        with overrides[0], overrides[1]:
            sig = tuple([numba.typeof(x) for x in args])
            pfunc_vectorizable = self.generate_prange_func(func, None)
            if fastmath == True:
                cres = self.compile_parallel_fastmath(pfunc_vectorizable, sig)
            else:
                cres = self.compile_parallel(pfunc_vectorizable, sig)
            asm = self._get_gufunc_asm(cres)
            if assertions:
                schedty = re.compile('call\\s+\\w+\\*\\s+@do_scheduling_(\\w+)\\(')
                matches = schedty.findall(cres.library.get_llvm_str())
                self.assertGreaterEqual(len(matches), 1)
                self.assertEqual(matches[0], schedule_type)
                self.assertNotEqual(asm, {})
            return asm

    @linux_only
    def test_vectorizer_fastmath_asm(self):
        if False:
            while True:
                i = 10
        ' This checks that if fastmath is set and the underlying hardware\n        is suitable, and the function supplied is amenable to fastmath based\n        vectorization, that the vectorizer actually runs.\n        '

        def will_vectorize(A):
            if False:
                i = 10
                return i + 15
            n = len(A)
            acc = 0
            for i in range(n):
                acc += np.sqrt(i)
            return acc
        arg = np.zeros(10)
        fast_asm = self.get_gufunc_asm(will_vectorize, 'unsigned', arg, fastmath=True)
        slow_asm = self.get_gufunc_asm(will_vectorize, 'unsigned', arg, fastmath=False)
        for v in fast_asm.values():
            self.assertTrue('vaddpd' in v)
            self.assertTrue('vsqrtpd' in v or '__svml_sqrt' in v)
            self.assertTrue('zmm' in v)
        for v in slow_asm.values():
            self.assertTrue('vaddpd' not in v)
            self.assertTrue('vsqrtpd' not in v)
            self.assertTrue('vsqrtsd' in v and '__svml_sqrt' not in v)
            self.assertTrue('vaddsd' in v)
            self.assertTrue('zmm' not in v)

    @linux_only
    def test_unsigned_refusal_to_vectorize(self):
        if False:
            i = 10
            return i + 15
        ' This checks that if fastmath is set and the underlying hardware\n        is suitable, and the function supplied is amenable to fastmath based\n        vectorization, that the vectorizer actually runs.\n        '

        def will_not_vectorize(A):
            if False:
                print('Hello World!')
            n = len(A)
            for i in range(-n, 0):
                A[i] = np.sqrt(A[i])
            return A

        def will_vectorize(A):
            if False:
                print('Hello World!')
            n = len(A)
            for i in range(n):
                A[i] = np.sqrt(A[i])
            return A
        arg = np.zeros(10)
        with override_env_config('NUMBA_BOUNDSCHECK', '0'):
            novec_asm = self.get_gufunc_asm(will_not_vectorize, 'signed', arg, fastmath=True)
            vec_asm = self.get_gufunc_asm(will_vectorize, 'unsigned', arg, fastmath=True)
        for v in novec_asm.values():
            self.assertTrue('vsqrtpd' not in v)
            self.assertTrue('vsqrtsd' in v)
            self.assertTrue('zmm' not in v)
        for v in vec_asm.values():
            self.assertTrue('vsqrtpd' in v or '__svml_sqrt' in v)
            self.assertTrue('vmovupd' in v)
            self.assertTrue('zmm' in v)

    @linux_only
    def test_signed_vs_unsigned_vec_asm(self):
        if False:
            i = 10
            return i + 15
        ' This checks vectorization for signed vs unsigned variants of a\n        trivial accumulator, the only meaningful difference should be the\n        presence of signed vs. unsigned unpack instructions (for the\n        induction var).\n        '

        def signed_variant():
            if False:
                i = 10
                return i + 15
            n = 4096
            A = 0.0
            for i in range(-n, 0):
                A += i
            return A

        def unsigned_variant():
            if False:
                return 10
            n = 4096
            A = 0.0
            for i in range(n):
                A += i
            return A
        with override_env_config('NUMBA_BOUNDSCHECK', '0'):
            signed_asm = self.get_gufunc_asm(signed_variant, 'signed', fastmath=True)
            unsigned_asm = self.get_gufunc_asm(unsigned_variant, 'unsigned', fastmath=True)

        def strip_instrs(asm):
            if False:
                for i in range(10):
                    print('nop')
            acc = []
            for x in asm.splitlines():
                spd = x.strip()
                if spd != '' and (not (spd.startswith('.') or spd.startswith('_') or spd.startswith('"') or ('__numba_parfor_gufunc' in spd))):
                    acc.append(re.sub('[\t]', '', spd))
            return acc
        for (k, v) in signed_asm.items():
            signed_instr = strip_instrs(v)
            break
        for (k, v) in unsigned_asm.items():
            unsigned_instr = strip_instrs(v)
            break
        from difflib import SequenceMatcher as sm
        self.assertEqual(len(signed_instr), len(unsigned_instr))
        for (a, b) in zip(signed_instr, unsigned_instr):
            if a == b:
                continue
            else:
                s = sm(lambda x: x == '\t', a, b)
                ops = s.get_opcodes()
                for op in ops:
                    if op[0] == 'insert':
                        self.assertEqual(b[op[-2]:op[-1]], 'u')

@skip_parfors_unsupported
class TestParforReductionSetNumThreads(TestCase):
    """Test execution correctness on reductions with set_num_threads.
    """

    def test_add(self):
        if False:
            return 10
        N = config.NUMBA_NUM_THREADS
        M = 2 * N
        mask = N - 1

        @njit(parallel=True)
        def udt(nthreads):
            if False:
                i = 10
                return i + 15
            acc = 0
            set_num_threads(nthreads)
            for i in prange(M):
                local_mask = 1 + i % mask
                set_num_threads(local_mask)
                gnt = get_num_threads()
                acc += gnt
            return acc
        expect = udt.py_func(mask)
        got = udt(mask)
        self.assertPreciseEqual(expect, got)

    def test_mul(self):
        if False:
            while True:
                i = 10
        N = min(4, config.NUMBA_NUM_THREADS)
        M = 2 * N
        mask = N - 1

        @njit(parallel=True)
        def udt(nthreads):
            if False:
                print('Hello World!')
            acc = 1
            set_num_threads(nthreads)
            for i in prange(M):
                local_mask = 1 + i % mask
                set_num_threads(local_mask)
                gnt = get_num_threads()
                acc *= gnt
            return acc
        expect = udt.py_func(mask)
        got = udt(mask)
        self.assertPreciseEqual(expect, got)

    def test_max(self):
        if False:
            print('Hello World!')
        N = config.NUMBA_NUM_THREADS
        M = 2 * N
        mask = N - 1

        @njit(parallel=True)
        def udt(nthreads):
            if False:
                return 10
            acc = 1
            set_num_threads(nthreads)
            for i in prange(M):
                local_mask = 1 + i % mask
                set_num_threads(local_mask)
                gnt = get_num_threads()
                acc = max(acc, gnt)
            return acc
        expect = udt.py_func(mask)
        got = udt(mask)
        self.assertPreciseEqual(expect, got)

@skip_parfors_unsupported
class TestDiagnosticEnvVar(TestCase):

    @TestCase.run_test_in_subprocess()
    def test_diagnostics_env_var1(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ['NUMBA_PARALLEL_DIAGNOSTICS'] = '4'
        with captured_stdout() as stdout:

            @njit(parallel=True)
            def impl():
                if False:
                    i = 10
                    return i + 15
                n = 100
                b = np.zeros(n, dtype=np.float64)
                for i in prange(n):
                    b[i] = 1
                return b
            impl()
        the_output = stdout.getvalue()
        self.assertIn('Parallel Accelerator Optimizing', the_output)
if __name__ == '__main__':
    unittest.main()