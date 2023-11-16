import unittest
from contextlib import contextmanager
from llvmlite import ir
from numba.core import types, typing, callconv, cpu, cgutils
from numba.core.registry import cpu_target

class TestCompileCache(unittest.TestCase):
    """
    Tests that the caching in BaseContext.compile_internal() works correctly by
    checking the state of the cache when it is used by the CPUContext.
    """

    @contextmanager
    def _context_builder_sig_args(self):
        if False:
            for i in range(10):
                print('nop')
        typing_context = cpu_target.typing_context
        context = cpu_target.target_context
        lib = context.codegen().create_library('testing')
        with context.push_code_library(lib):
            module = ir.Module('test_module')
            sig = typing.signature(types.int32, types.int32)
            llvm_fnty = context.call_conv.get_function_type(sig.return_type, sig.args)
            function = cgutils.get_or_insert_function(module, llvm_fnty, 'test_fn')
            args = context.call_conv.get_arguments(function)
            assert function.is_declaration
            entry_block = function.append_basic_block('entry')
            builder = ir.IRBuilder(entry_block)
            yield (context, builder, sig, args)

    def test_cache(self):
        if False:
            return 10

        def times2(i):
            if False:
                while True:
                    i = 10
            return 2 * i

        def times3(i):
            if False:
                print('Hello World!')
            return i * 3
        with self._context_builder_sig_args() as (context, builder, sig, args):
            initial_cache_size = len(context.cached_internal_func)
            self.assertEqual(initial_cache_size + 0, len(context.cached_internal_func))
            context.compile_internal(builder, times2, sig, args)
            self.assertEqual(initial_cache_size + 1, len(context.cached_internal_func))
            context.compile_internal(builder, times2, sig, args)
            self.assertEqual(initial_cache_size + 1, len(context.cached_internal_func))
            context.compile_internal(builder, times3, sig, args)
            self.assertEqual(initial_cache_size + 2, len(context.cached_internal_func))
            sig2 = typing.signature(types.float64, types.float64)
            llvm_fnty2 = context.call_conv.get_function_type(sig2.return_type, sig2.args)
            function2 = cgutils.get_or_insert_function(builder.module, llvm_fnty2, 'test_fn_2')
            args2 = context.call_conv.get_arguments(function2)
            assert function2.is_declaration
            entry_block2 = function2.append_basic_block('entry')
            builder2 = ir.IRBuilder(entry_block2)
            context.compile_internal(builder2, times3, sig2, args2)
            self.assertEqual(initial_cache_size + 3, len(context.cached_internal_func))

    def test_closures(self):
        if False:
            i = 10
            return i + 15
        '\n        Caching must not mix up closures reusing the same code object.\n        '

        def make_closure(x, y):
            if False:
                i = 10
                return i + 15

            def f(z):
                if False:
                    for i in range(10):
                        print('nop')
                return y + z
            return f
        with self._context_builder_sig_args() as (context, builder, sig, args):
            clo11 = make_closure(1, 1)
            clo12 = make_closure(1, 2)
            clo22 = make_closure(2, 2)
            initial_cache_size = len(context.cached_internal_func)
            res1 = context.compile_internal(builder, clo11, sig, args)
            self.assertEqual(initial_cache_size + 1, len(context.cached_internal_func))
            res2 = context.compile_internal(builder, clo12, sig, args)
            self.assertEqual(initial_cache_size + 2, len(context.cached_internal_func))
            res3 = context.compile_internal(builder, clo22, sig, args)
            self.assertEqual(initial_cache_size + 2, len(context.cached_internal_func))

    def test_error_model(self):
        if False:
            return 10
        '\n        Caching must not mix up different error models.\n        '

        def inv(x):
            if False:
                i = 10
                return i + 15
            return 1.0 / x
        inv_sig = typing.signature(types.float64, types.float64)

        def compile_inv(context):
            if False:
                i = 10
                return i + 15
            return context.compile_subroutine(builder, inv, inv_sig)
        with self._context_builder_sig_args() as (context, builder, sig, args):
            py_error_model = callconv.create_error_model('python', context)
            np_error_model = callconv.create_error_model('numpy', context)
            py_context1 = context.subtarget(error_model=py_error_model)
            py_context2 = context.subtarget(error_model=py_error_model)
            np_context = context.subtarget(error_model=np_error_model)
            initial_cache_size = len(context.cached_internal_func)
            self.assertEqual(initial_cache_size + 0, len(context.cached_internal_func))
            compile_inv(py_context1)
            self.assertEqual(initial_cache_size + 1, len(context.cached_internal_func))
            compile_inv(py_context2)
            self.assertEqual(initial_cache_size + 1, len(context.cached_internal_func))
            compile_inv(np_context)
            self.assertEqual(initial_cache_size + 2, len(context.cached_internal_func))
if __name__ == '__main__':
    unittest.main()