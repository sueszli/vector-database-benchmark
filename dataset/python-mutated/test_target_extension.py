"""This tests the target extension API to ensure that rudimentary expected
behaviours are present and correct. It uses a piece of fake hardware as a
target, the Dummy Processing Unit (DPU), to do this. The DPU borrows a lot from
the CPU but is part of the GPU class of target. The DPU target has deliberately
strange implementations of fundamental operations so as to make it identifiable
in testing."""
import unittest
from numba.tests.support import TestCase
import contextlib
import ctypes
import operator
from functools import cached_property
import numpy as np
from numba import njit, types
from numba.extending import overload, intrinsic, overload_classmethod
from numba.core.target_extension import JitDecorator, target_registry, dispatcher_registry, jit_registry, target_override, GPU, resolve_dispatcher_from_str
from numba.core import utils, fastmathpass, errors
from numba.core.dispatcher import Dispatcher
from numba.core.descriptors import TargetDescriptor
from numba.core import cpu, typing, cgutils
from numba.core.base import BaseContext
from numba.core.compiler_lock import global_compiler_lock
from numba.core import callconv
from numba.core.codegen import CPUCodegen, JITCodeLibrary
from numba.core.callwrapper import PyCallWrapper
from numba.core.imputils import RegistryLoader, Registry
from numba import _dynfunc
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.runtime import rtsys
from numba.core import compiler
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.typed_passes import PreLowerStripPhis

class DPU(GPU):
    ...
target_registry['dpu'] = DPU

class JITDPUCodegen(CPUCodegen):
    _library_class = JITCodeLibrary

    def _customize_tm_options(self, options):
        if False:
            return 10
        options['cpu'] = self._get_host_cpu_name()
        arch = ll.Target.from_default_triple().name
        if arch.startswith('x86'):
            reloc_model = 'static'
        elif arch.startswith('ppc'):
            reloc_model = 'pic'
        else:
            reloc_model = 'default'
        options['reloc'] = reloc_model
        options['codemodel'] = 'jitdefault'
        options['features'] = self._tm_features
        sig = utils.pysignature(ll.Target.create_target_machine)
        if 'jit' in sig.parameters:
            options['jit'] = True

    def _customize_tm_features(self):
        if False:
            print('Hello World!')
        return self._get_host_cpu_features()

    def _add_module(self, module):
        if False:
            return 10
        self._engine.add_module(module)

    def set_env(self, env_name, env):
        if False:
            i = 10
            return i + 15
        'Set the environment address.\n\n        Update the GlobalVariable named *env_name* to the address of *env*.\n        '
        gvaddr = self._engine.get_global_value_address(env_name)
        envptr = (ctypes.c_void_p * 1).from_address(gvaddr)
        envptr[0] = ctypes.c_void_p(id(env))
dpu_function_registry = Registry()

class DPUContext(BaseContext):
    allow_dynamic_globals = True

    def create_module(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self._internal_codegen._create_empty_module(name)

    @global_compiler_lock
    def init(self):
        if False:
            print('Hello World!')
        self._internal_codegen = JITDPUCodegen('numba.exec')
        rtsys.initialize(self)
        self.refresh()

    def refresh(self):
        if False:
            while True:
                i = 10
        registry = dpu_function_registry
        try:
            loader = self._registries[registry]
        except KeyError:
            loader = RegistryLoader(registry)
            self._registries[registry] = loader
        self.install_registry(registry)
        self.typing_context.refresh()

    @property
    def target_data(self):
        if False:
            while True:
                i = 10
        return self._internal_codegen.target_data

    def codegen(self):
        if False:
            for i in range(10):
                print('nop')
        return self._internal_codegen

    @cached_property
    def call_conv(self):
        if False:
            while True:
                i = 10
        return callconv.CPUCallConv(self)

    def get_env_body(self, builder, envptr):
        if False:
            return 10
        '\n        From the given *envptr* (a pointer to a _dynfunc.Environment object),\n        get a EnvBody allowing structured access to environment fields.\n        '
        body_ptr = cgutils.pointer_add(builder, envptr, _dynfunc._impl_info['offsetof_env_body'])
        return cpu.EnvBody(self, builder, ref=body_ptr, cast_ref=True)

    def get_env_manager(self, builder):
        if False:
            return 10
        envgv = self.declare_env_global(builder.module, self.get_env_name(self.fndesc))
        envarg = builder.load(envgv)
        pyapi = self.get_python_api(builder)
        pyapi.emit_environment_sentry(envarg, debug_msg=self.fndesc.env_name)
        env_body = self.get_env_body(builder, envarg)
        return pyapi.get_env_manager(self.environment, env_body, envarg)

    def get_generator_state(self, builder, genptr, return_type):
        if False:
            i = 10
            return i + 15
        '\n        From the given *genptr* (a pointer to a _dynfunc.Generator object),\n        get a pointer to its state area.\n        '
        return cgutils.pointer_add(builder, genptr, _dynfunc._impl_info['offsetof_generator_state'], return_type=return_type)

    def post_lowering(self, mod, library):
        if False:
            print('Hello World!')
        if self.fastmath:
            fastmathpass.rewrite_module(mod, self.fastmath)
        library.add_linking_library(rtsys.library)

    def create_cpython_wrapper(self, library, fndesc, env, call_helper, release_gil=False):
        if False:
            for i in range(10):
                print('nop')
        wrapper_module = self.create_module('wrapper')
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        wrapper_callee = llir.Function(wrapper_module, fnty, fndesc.llvm_func_name)
        builder = PyCallWrapper(self, wrapper_module, wrapper_callee, fndesc, env, call_helper=call_helper, release_gil=release_gil)
        builder.build()
        library.add_ir_module(wrapper_module)

    def create_cfunc_wrapper(self, library, fndesc, env, call_helper):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_executable(self, library, fndesc, env):
        if False:
            while True:
                i = 10
        '\n        Returns\n        -------\n        (cfunc, fnptr)\n\n        - cfunc\n            callable function (Can be None)\n        - fnptr\n            callable function address\n        - env\n            an execution environment (from _dynfunc)\n        '
        fnptr = library.get_pointer_to_function(fndesc.llvm_cpython_wrapper_name)
        doc = 'compiled wrapper for %r' % (fndesc.qualname,)
        cfunc = _dynfunc.make_function(fndesc.lookup_module(), fndesc.qualname.split('.')[-1], doc, fnptr, env, (library,))
        library.codegen.set_env(self.get_env_name(fndesc), env)
        return cfunc

class _NestedContext(object):
    _typing_context = None
    _target_context = None

    @contextlib.contextmanager
    def nested(self, typing_context, target_context):
        if False:
            return 10
        old_nested = (self._typing_context, self._target_context)
        try:
            self._typing_context = typing_context
            self._target_context = target_context
            yield
        finally:
            (self._typing_context, self._target_context) = old_nested

class DPUTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions
    _nested = _NestedContext()

    @cached_property
    def _toplevel_target_context(self):
        if False:
            return 10
        return DPUContext(self.typing_context, self._target_name)

    @cached_property
    def _toplevel_typing_context(self):
        if False:
            for i in range(10):
                print('nop')
        return typing.Context()

    @property
    def target_context(self):
        if False:
            print('Hello World!')
        '\n        The target context for DPU targets.\n        '
        nested = self._nested._target_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_target_context

    @property
    def typing_context(self):
        if False:
            print('Hello World!')
        '\n        The typing context for CPU targets.\n        '
        nested = self._nested._typing_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_typing_context

    def nested_context(self, typing_context, target_context):
        if False:
            i = 10
            return i + 15
        '\n        A context manager temporarily replacing the contexts with the\n        given ones, for the current thread of execution.\n        '
        return self._nested.nested(typing_context, target_context)
dpu_target = DPUTarget('dpu')

class DPUDispatcher(Dispatcher):
    targetdescr = dpu_target
dispatcher_registry[target_registry['dpu']] = DPUDispatcher

class djit(JitDecorator):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        assert len(args) < 2
        if args:
            func = args[0]
        else:
            func = self._args[0]
        self.py_func = func
        return self.dispatcher_wrapper()

    def get_dispatcher(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the dispatcher\n        '
        return dispatcher_registry[target_registry['dpu']]

    def dispatcher_wrapper(self):
        if False:
            print('Hello World!')
        disp = self.get_dispatcher()
        topt = {}
        if 'nopython' in self._kwargs:
            topt['nopython'] = True
        pipeline_class = compiler.Compiler
        if 'pipeline_class' in self._kwargs:
            pipeline_class = self._kwargs['pipeline_class']
        return disp(py_func=self.py_func, targetoptions=topt, pipeline_class=pipeline_class)
jit_registry[target_registry['dpu']] = djit

@dpu_function_registry.lower_constant(types.Dummy)
def constant_dummy(context, builder, ty, pyval):
    if False:
        print('Hello World!')
    return context.get_dummy_value()

@dpu_function_registry.lower_cast(types.IntegerLiteral, types.Integer)
def literal_int_to_number(context, builder, fromty, toty, val):
    if False:
        while True:
            i = 10
    lit = context.get_constant_generic(builder, fromty.literal_type, fromty.literal_value)
    return context.cast(builder, lit, fromty.literal_type, toty)

@dpu_function_registry.lower_constant(types.Integer)
def const_int(context, builder, ty, pyval):
    if False:
        i = 10
        return i + 15
    lty = context.get_value_type(ty)
    return lty(pyval)

@dpu_function_registry.lower_constant(types.Float)
def const_float(context, builder, ty, pyval):
    if False:
        i = 10
        return i + 15
    lty = context.get_value_type(ty)
    return lty(pyval)

@intrinsic(target='dpu')
def intrin_add(tyctx, x, y):
    if False:
        for i in range(10):
            print('nop')
    sig = x(x, y)

    def codegen(cgctx, builder, tyargs, llargs):
        if False:
            for i in range(10):
                print('nop')
        return builder.sub(*llargs)
    return (sig, codegen)

@overload(operator.add, target='dpu')
def ol_add(x, y):
    if False:
        i = 10
        return i + 15
    if isinstance(x, types.Integer) and isinstance(y, types.Integer):

        def impl(x, y):
            if False:
                while True:
                    i = 10
            return intrin_add(x, y)
        return impl

class TestTargetHierarchySelection(TestCase):
    """This tests that the target hierarchy is scanned in the right order,
    that appropriate functions are selected based on what's available and that
    the DPU target is distinctly different to the CPU"""

    def test_0_dpu_registry(self):
        if False:
            print('Hello World!')
        'Checks that the DPU registry only contains the things added\n\n        This test must be first to execute among all tests in this file to\n        ensure the no lazily loaded entries are added yet.\n        '
        self.assertFalse(dpu_function_registry.functions)
        self.assertFalse(dpu_function_registry.getattrs)
        self.assertEqual(len(dpu_function_registry.casts), 1)
        self.assertEqual(len(dpu_function_registry.constants), 3)

    def test_specialise_gpu(self):
        if False:
            i = 10
            return i + 15

        def my_func(x):
            if False:
                print('Hello World!')
            pass

        @overload(my_func, target='generic')
        def ol_my_func1(x):
            if False:
                while True:
                    i = 10

            def impl(x):
                if False:
                    print('Hello World!')
                return 1 + x
            return impl

        @overload(my_func, target='gpu')
        def ol_my_func2(x):
            if False:
                i = 10
                return i + 15

            def impl(x):
                if False:
                    print('Hello World!')
                return 10 + x
            return impl

        @djit()
        def dpu_foo():
            if False:
                print('Hello World!')
            return my_func(7)

        @njit()
        def cpu_foo():
            if False:
                i = 10
                return i + 15
            return my_func(7)
        self.assertPreciseEqual(dpu_foo(), 3)
        self.assertPreciseEqual(cpu_foo(), 8)

    def test_specialise_dpu(self):
        if False:
            return 10

        def my_func(x):
            if False:
                i = 10
                return i + 15
            pass

        @overload(my_func, target='generic')
        def ol_my_func1(x):
            if False:
                return 10

            def impl(x):
                if False:
                    print('Hello World!')
                return 1 + x
            return impl

        @overload(my_func, target='gpu')
        def ol_my_func2(x):
            if False:
                while True:
                    i = 10

            def impl(x):
                if False:
                    return 10
                return 10 + x
            return impl

        @overload(my_func, target='dpu')
        def ol_my_func3(x):
            if False:
                i = 10
                return i + 15

            def impl(x):
                if False:
                    while True:
                        i = 10
                return 100 + x
            return impl

        @djit()
        def dpu_foo():
            if False:
                print('Hello World!')
            return my_func(7)

        @njit()
        def cpu_foo():
            if False:
                print('Hello World!')
            return my_func(7)
        self.assertPreciseEqual(dpu_foo(), 93)
        self.assertPreciseEqual(cpu_foo(), 8)

    def test_no_specialisation_found(self):
        if False:
            i = 10
            return i + 15

        def my_func(x):
            if False:
                while True:
                    i = 10
            pass

        @overload(my_func, target='cuda')
        def ol_my_func_cuda(x):
            if False:
                return 10
            return lambda x: None

        @djit(nopython=True)
        def dpu_foo():
            if False:
                i = 10
                return i + 15
            my_func(1)
        accept = (errors.UnsupportedError, errors.TypingError)
        with self.assertRaises(accept) as raises:
            dpu_foo()
        msgs = ['Function resolution cannot find any matches for function', 'test_no_specialisation_found.<locals>.my_func', 'for the current target:', "'numba.tests.test_target_extension.DPU'"]
        for msg in msgs:
            self.assertIn(msg, str(raises.exception))

    def test_invalid_target_jit(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(errors.NumbaValueError) as raises:

            @njit(_target='invalid_silicon')
            def foo():
                if False:
                    while True:
                        i = 10
                pass
            foo()
        msg = "No target is registered against 'invalid_silicon'"
        self.assertIn(msg, str(raises.exception))

    def test_invalid_target_overload(self):
        if False:
            return 10

        def bar():
            if False:
                i = 10
                return i + 15
            pass
        with self.assertRaises(errors.TypingError) as raises:

            @overload(bar, target='invalid_silicon')
            def ol_bar():
                if False:
                    while True:
                        i = 10
                return lambda : None

            @njit
            def foo():
                if False:
                    print('Hello World!')
                bar()
            foo()
        msg = "No target is registered against 'invalid_silicon'"
        self.assertIn(msg, str(raises.exception))

    def test_intrinsic_selection(self):
        if False:
            while True:
                i = 10
        '\n        Test to make sure that targets can share generic implementations and\n        cannot reach implementations that are not in their target hierarchy.\n        '

        @intrinsic(target='generic')
        def intrin_math_generic(tyctx, x, y):
            if False:
                return 10
            sig = x(x, y)

            def codegen(cgctx, builder, tyargs, llargs):
                if False:
                    while True:
                        i = 10
                return builder.mul(*llargs)
            return (sig, codegen)

        @intrinsic(target='dpu')
        def intrin_math_dpu(tyctx, x, y):
            if False:
                print('Hello World!')
            sig = x(x, y)

            def codegen(cgctx, builder, tyargs, llargs):
                if False:
                    return 10
                return builder.sub(*llargs)
            return (sig, codegen)

        @intrinsic(target='cpu')
        def intrin_math_cpu(tyctx, x, y):
            if False:
                print('Hello World!')
            sig = x(x, y)

            def codegen(cgctx, builder, tyargs, llargs):
                if False:
                    while True:
                        i = 10
                return builder.add(*llargs)
            return (sig, codegen)

        @njit
        def cpu_foo_specific():
            if False:
                return 10
            return intrin_math_cpu(3, 4)
        self.assertEqual(cpu_foo_specific(), 7)

        @njit
        def cpu_foo_generic():
            if False:
                i = 10
                return i + 15
            return intrin_math_generic(3, 4)
        self.assertEqual(cpu_foo_generic(), 12)

        @njit
        def cpu_foo_dpu():
            if False:
                for i in range(10):
                    print('nop')
            return intrin_math_dpu(3, 4)
        accept = (errors.UnsupportedError, errors.TypingError)
        with self.assertRaises(accept) as raises:
            cpu_foo_dpu()
        msgs = ['Function resolution cannot find any matches for function', 'intrinsic intrin_math_dpu', 'for the current target']
        for msg in msgs:
            self.assertIn(msg, str(raises.exception))

        @djit(nopython=True)
        def dpu_foo_specific():
            if False:
                while True:
                    i = 10
            return intrin_math_dpu(3, 4)
        self.assertEqual(dpu_foo_specific(), -1)

        @djit(nopython=True)
        def dpu_foo_generic():
            if False:
                i = 10
                return i + 15
            return intrin_math_generic(3, 4)
        self.assertEqual(dpu_foo_generic(), 12)

        @djit(nopython=True)
        def dpu_foo_cpu():
            if False:
                return 10
            return intrin_math_cpu(3, 4)
        accept = (errors.UnsupportedError, errors.TypingError)
        with self.assertRaises(accept) as raises:
            dpu_foo_cpu()
        msgs = ['Function resolution cannot find any matches for function', 'intrinsic intrin_math_cpu', 'for the current target']
        for msg in msgs:
            self.assertIn(msg, str(raises.exception))

    def test_overload_allocation(self):
        if False:
            return 10

        def cast_integer(context, builder, val, fromty, toty):
            if False:
                while True:
                    i = 10
            if toty.bitwidth == fromty.bitwidth:
                return val
            elif toty.bitwidth < fromty.bitwidth:
                return builder.trunc(val, context.get_value_type(toty))
            elif fromty.signed:
                return builder.sext(val, context.get_value_type(toty))
            else:
                return builder.zext(val, context.get_value_type(toty))

        @intrinsic(target='dpu')
        def intrin_alloc(typingctx, allocsize, align):
            if False:
                i = 10
                return i + 15
            'Intrinsic to call into the allocator for Array\n            '

            def codegen(context, builder, signature, args):
                if False:
                    while True:
                        i = 10
                [allocsize, align] = args
                align_u32 = cast_integer(context, builder, align, signature.args[1], types.uint32)
                meminfo = context.nrt.meminfo_alloc_aligned(builder, allocsize, align_u32)
                return meminfo
            from numba.core.typing import signature
            mip = types.MemInfoPointer(types.voidptr)
            sig = signature(mip, allocsize, align)
            return (sig, codegen)

        @overload_classmethod(types.Array, '_allocate', target='dpu', jit_options={'nopython': True})
        def _ol_arr_allocate_dpu(cls, allocsize, align):
            if False:
                return 10

            def impl(cls, allocsize, align):
                if False:
                    print('Hello World!')
                return intrin_alloc(allocsize, align)
            return impl

        @overload(np.empty, target='dpu', jit_options={'nopython': True})
        def ol_empty_impl(n):
            if False:
                while True:
                    i = 10

            def impl(n):
                if False:
                    for i in range(10):
                        print('nop')
                return types.Array._allocate(n, 7)
            return impl

        def buffer_func():
            if False:
                i = 10
                return i + 15
            pass

        @overload(buffer_func, target='dpu', jit_options={'nopython': True})
        def ol_buffer_func_impl():
            if False:
                i = 10
                return i + 15

            def impl():
                if False:
                    return 10
                return np.empty(10)
            return impl
        from numba.core.target_extension import target_override
        with target_override('dpu'):

            @djit(nopython=True)
            def foo():
                if False:
                    while True:
                        i = 10
                return buffer_func()
            r = foo()
        from numba.core.runtime import nrt
        self.assertIsInstance(r, nrt.MemInfo)

class TestTargetOffload(TestCase):
    """In this use case the CPU compilation pipeline is extended with a new
     compilation pass that runs just prior to lowering. The pass looks for
     function calls and when it finds one it sees if there's a DPU function
     available that is a valid overload for the function call. If there is one
     then it swaps the CPU implementation out for a DPU implementation. This
     producing an "offload" effect.
    """

    def test_basic_offload(self):
        if False:
            for i in range(10):
                print('nop')
        _DEBUG = False

        @overload(np.sin, target='dpu')
        def ol_np_sin_DPU(x):
            if False:
                print('Hello World!')

            def dpu_sin_impl(x):
                if False:
                    for i in range(10):
                        print('nop')
                return 314159.0
            return dpu_sin_impl

        @djit(nopython=True)
        def foo(x):
            if False:
                i = 10
                return i + 15
            return np.sin(x)
        self.assertPreciseEqual(foo(5), 314159.0)

        @njit
        def foo(x):
            if False:
                while True:
                    i = 10
            return np.sin(x)
        self.assertPreciseEqual(foo(5), np.sin(5))

        @register_pass(mutates_CFG=False, analysis_only=False)
        class DispatcherSwitcher(FunctionPass):
            _name = 'DispatcherSwitcher'

            def __init__(self):
                if False:
                    while True:
                        i = 10
                FunctionPass.__init__(self)

            def run_pass(self, state):
                if False:
                    i = 10
                    return i + 15
                func_ir = state.func_ir
                mutated = False
                for blk in func_ir.blocks.values():
                    for call in blk.find_exprs('call'):
                        function = state.typemap[call.func.name]
                        tname = 'dpu'
                        with target_override(tname):
                            try:
                                sig = function.get_call_type(state.typingctx, state.calltypes[call].args, {})
                                disp = resolve_dispatcher_from_str(tname)
                                hw_ctx = disp.targetdescr.target_context
                                hw_ctx.get_function(function, sig)
                            except Exception as e:
                                if _DEBUG:
                                    msg = f'Failed to find and compile an overload for {function} for {tname} due to {e}'
                                    print(msg)
                                continue
                            hw_ctx._codelib_stack = state.targetctx._codelib_stack
                            call.target = tname
                            mutated = True
                return mutated

        class DPUOffloadCompiler(CompilerBase):

            def define_pipelines(self):
                if False:
                    i = 10
                    return i + 15
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(DispatcherSwitcher, PreLowerStripPhis)
                pm.finalize()
                return [pm]

        @njit(pipeline_class=DPUOffloadCompiler)
        def foo(x):
            if False:
                print('Hello World!')
            return (np.sin(x), np.cos(x))
        self.assertPreciseEqual(foo(5), (314159.0, np.cos(5)))
if __name__ == '__main__':
    unittest.main()