import _testcapi
import asyncio
import builtins
import cinder
import dis
import faulthandler
import gc
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import traceback
import unittest
import warnings
import weakref
from compiler.consts import CO_FUTURE_BARRY_AS_BDFL, CO_SUPPRESS_JIT
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent
import _testcindercapi
from test import cinder_support
try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from .test_compiler.test_static.common import StaticTestBase
        from .test_compiler.test_strict.test_loader import base_sandbox, sandbox
except ImportError:
    from test_compiler.test_static.common import StaticTestBase
    from test_compiler.test_strict.test_loader import base_sandbox, sandbox
try:
    import cinderjit
    from cinderjit import _deopt_gen, is_jit_compiled, jit_suppress
except:
    cinderjit = None

    def jit_suppress(func):
        if False:
            i = 10
            return i + 15
        return func

    def _deopt_gen(gen):
        if False:
            print('Hello World!')
        return False

    def is_jit_compiled(func):
        if False:
            for i in range(10):
                print('nop')
        return False

def failUnlessHasOpcodes(*required_opnames):
    if False:
        i = 10
        return i + 15
    'Fail a test unless func has all of the opcodes in `required` in its code\n    object.\n    '

    def decorator(func):
        if False:
            return 10
        opnames = {i.opname for i in dis.get_instructions(func)}
        missing = set(required_opnames) - opnames
        if missing:

            def wrapper(*args):
                if False:
                    return 10
                raise AssertionError(f'Function {func.__qualname__} missing required opcodes: {missing}')
            return wrapper
        return func
    return decorator

def firstlineno(func):
    if False:
        print('Hello World!')
    return func.__code__.co_firstlineno

class GetFrameLineNumberTests(unittest.TestCase):

    def assert_code_and_lineno(self, frame, func, line_offset):
        if False:
            while True:
                i = 10
        self.assertEqual(frame.f_code, func.__code__)
        self.assertEqual(frame.f_lineno, firstlineno(func) + line_offset)

    def test_line_numbers(self):
        if False:
            while True:
                i = 10
        'Verify that line numbers are correct'

        @cinder_support.failUnlessJITCompiled
        def g():
            if False:
                print('Hello World!')
            return sys._getframe()
        self.assert_code_and_lineno(g(), g, 2)

    def test_line_numbers_for_running_generators(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify that line numbers are correct for running generator functions'

        @cinder_support.failUnlessJITCompiled
        def g(x, y):
            if False:
                while True:
                    i = 10
            yield sys._getframe()
            z = x + y
            yield sys._getframe()
            yield z
        gen = g(1, 2)
        frame = next(gen)
        self.assert_code_and_lineno(frame, g, 2)
        frame = next(gen)
        self.assert_code_and_lineno(frame, g, 4)
        self.assertEqual(next(gen), 3)

    def test_line_numbers_for_suspended_generators(self):
        if False:
            print('Hello World!')
        'Verify that line numbers are correct for suspended generator functions'

        @cinder_support.failUnlessJITCompiled
        def g(x):
            if False:
                i = 10
                return i + 15
            x = x + 1
            yield x
            z = x + 1
            yield z
        gen = g(0)
        self.assert_code_and_lineno(gen.gi_frame, g, 0)
        v = next(gen)
        self.assertEqual(v, 1)
        self.assert_code_and_lineno(gen.gi_frame, g, 3)
        v = next(gen)
        self.assertEqual(v, 2)
        self.assert_code_and_lineno(gen.gi_frame, g, 5)

    def test_line_numbers_during_gen_throw(self):
        if False:
            while True:
                i = 10
        'Verify that line numbers are correct for suspended generator functions when\n        an exception is thrown into them.\n        '

        @cinder_support.failUnlessJITCompiled
        def f1(g):
            if False:
                while True:
                    i = 10
            yield from g

        @cinder_support.failUnlessJITCompiled
        def f2(g):
            if False:
                print('Hello World!')
            yield from g
        (gen1, gen2) = (None, None)
        (gen1_frame, gen2_frame) = (None, None)

        @cinder_support.failUnlessJITCompiled
        def f3():
            if False:
                i = 10
                return i + 15
            nonlocal gen1_frame, gen2_frame
            try:
                yield 'hello'
            except TestException:
                gen1_frame = gen1.gi_frame
                gen2_frame = gen2.gi_frame
                raise
        gen3 = f3()
        gen2 = f2(gen3)
        gen1 = f1(gen2)
        gen1.send(None)
        with self.assertRaises(TestException):
            gen1.throw(TestException())
        initial_lineno = 126
        self.assert_code_and_lineno(gen1_frame, f1, 2)
        self.assert_code_and_lineno(gen2_frame, f2, 2)

    def test_line_numbers_from_finalizers(self):
        if False:
            i = 10
            return i + 15
        'Make sure we can get accurate line numbers from finalizers'
        stack = None

        class StackGetter:

            def __del__(self):
                if False:
                    i = 10
                    return i + 15
                nonlocal stack
                stack = traceback.extract_stack()

        @cinder_support.failUnlessJITCompiled
        def double(x):
            if False:
                i = 10
                return i + 15
            ret = x
            tmp = StackGetter()
            del tmp
            ret += x
            return ret
        res = double(5)
        self.assertEqual(res, 10)
        line_base = firstlineno(double)
        self.assertEqual(stack[-1].lineno, firstlineno(StackGetter.__del__) + 2)
        self.assertEqual(stack[-2].lineno, firstlineno(double) + 4)

    @cinder_support.skipUnlessJITEnabled('Runs a subprocess with the JIT enabled')
    def test_line_numbers_after_jit_disabled(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('\n            import cinderjit\n            import sys\n\n            def f():\n                frame = sys._getframe(0)\n                print(f"{frame.f_code.co_name}:{frame.f_lineno}")\n                return 1\n\n            f()\n\n            # Depending on which JIT mode is being used, f might not have been\n            # compiled on the first call, but it will be after `force_compile`.\n            cinderjit.force_compile(f)\n            assert cinderjit.is_jit_compiled(f)\n\n            cinderjit.disable()\n            f()\n        ')
        jitlist = '__main__:*\n'
        with tempfile.TemporaryDirectory() as tmp:
            dirpath = Path(tmp)
            codepath = dirpath / 'mod.py'
            jitlistpath = dirpath / 'jitlist.txt'
            codepath.write_text(code)
            jitlistpath.write_text(jitlist)
            proc = subprocess.run([sys.executable, '-X', 'jit', '-X', 'jit-list-file=jitlist.txt', '-X', 'jit-enable-jit-list-wildcards', 'mod.py'], cwd=tmp, stdout=subprocess.PIPE, encoding=sys.stdout.encoding)
        self.assertEqual(proc.returncode, 0, proc)
        expected_stdout = 'f:6\nf:6\n'
        self.assertEqual(proc.stdout, expected_stdout)

@cinder_support.failUnlessJITCompiled
def get_stack():
    if False:
        return 10
    z = 1 + 1
    stack = traceback.extract_stack()
    return stack

@cinder_support.failUnlessJITCompiled
def get_stack_twice():
    if False:
        i = 10
        return i + 15
    stacks = []
    stacks.append(get_stack())
    stacks.append(get_stack())
    return stacks

@cinder_support.failUnlessJITCompiled
def get_stack2():
    if False:
        for i in range(10):
            print('nop')
    z = 2 + 2
    stack = traceback.extract_stack()
    return stack

@cinder_support.failUnlessJITCompiled
def get_stack_siblings():
    if False:
        return 10
    return [get_stack(), get_stack2()]

@cinder_support.failUnlessJITCompiled
def get_stack_multi():
    if False:
        while True:
            i = 10
    stacks = []
    stacks.append(traceback.extract_stack())
    z = 1 + 1
    stacks.append(traceback.extract_stack())
    return stacks

@cinder_support.failUnlessJITCompiled
def call_get_stack_multi():
    if False:
        return 10
    x = 1 + 1
    return get_stack_multi()

@cinder_support.failUnlessJITCompiled
def func_to_be_inlined(x, y):
    if False:
        print('Hello World!')
    return x + y

@cinder_support.failUnlessJITCompiled
def func_with_defaults(x=1, y=2):
    if False:
        for i in range(10):
            print('nop')
    return x + y

@cinder_support.failUnlessJITCompiled
def func_with_varargs(x, *args):
    if False:
        for i in range(10):
            print('nop')
    return x

@cinder_support.failUnlessJITCompiled
def func():
    if False:
        i = 10
        return i + 15
    a = func_to_be_inlined(2, 3)
    b = func_with_defaults()
    c = func_with_varargs(1, 2, 3)
    return a + b + c

@cinder_support.failUnlessJITCompiled
def func_with_defaults_that_will_change(x=1, y=2):
    if False:
        return 10
    return x + y

@cinder_support.failUnlessJITCompiled
def change_defaults():
    if False:
        i = 10
        return i + 15
    func_with_defaults_that_will_change.__defaults__ = (4, 5)

@cinder_support.failUnlessJITCompiled
def func_that_change_defaults():
    if False:
        return 10
    change_defaults()
    return func_with_defaults_that_will_change()

class InlinedFunctionTests(unittest.TestCase):

    @jit_suppress
    @unittest.skipIf(not cinderjit or not cinderjit.is_hir_inliner_enabled(), 'meaningless without HIR inliner enabled')
    def test_deopt_when_func_defaults_change(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(cinderjit.get_num_inlined_functions(func_that_change_defaults), 2)
        self.assertEqual(func_that_change_defaults(), 9)

class InlineCacheStatsTests(unittest.TestCase):

    @jit_suppress
    @unittest.skipIf(not cinderjit or not cinderjit.is_inline_cache_stats_collection_enabled(), 'meaningless without inline cache stats collection enabled')
    def test_load_method_cache_stats(self):
        if False:
            return 10
        cinderjit.get_and_clear_inline_cache_stats()
        import linecache

        class BinOps:

            def instance_mul(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x * y

            @staticmethod
            def mul(x, y):
                if False:
                    print('Hello World!')
                return x * y

        @cinder_support.failUnlessJITCompiled
        def trigger_load_method_with_stats():
            if False:
                print('Hello World!')
            a = BinOps()
            a.instance_mul(100, 1)
            a.mul(100, 1)
            b = linecache.getline('abc', 123)
            return a
        trigger_load_method_with_stats()
        stats = cinderjit.get_and_clear_inline_cache_stats()
        load_method_stats = stats['load_method_stats']
        relevant_load_method_stats = list(filter(lambda stat: 'test_cinderjit' in stat['filename'] and stat['method'] == 'trigger_load_method_with_stats', load_method_stats))
        self.assertTrue(len(relevant_load_method_stats) == 3)
        misses = [cache['cache_misses'] for cache in relevant_load_method_stats]
        load_method_cache_misses = {k: v for miss in misses for (k, v) in miss.items()}
        self.assertEqual(load_method_cache_misses, {'test_cinderx.test_cinderjit:BinOps.mul': {'count': 1, 'reason': 'Uncategorized'}, 'module.getline': {'count': 1, 'reason': 'WrongTpGetAttro'}})

class InlinedFunctionLineNumberTests(unittest.TestCase):

    @jit_suppress
    @unittest.skipIf(not cinderjit or not cinderjit.is_hir_inliner_enabled(), 'meaningless without HIR inliner enabled')
    def test_line_numbers_with_sibling_inlined_functions(self):
        if False:
            print('Hello World!')
        'Verify that line numbers are correct when function calls are inlined in the same\n        expression'
        self.assertEqual(cinderjit.get_num_inlined_functions(get_stack_siblings), 2)
        stacks = get_stack_siblings()
        self.assertEqual(stacks[0][-1].lineno, firstlineno(get_stack) + 3)
        self.assertEqual(stacks[0][-2].lineno, firstlineno(get_stack_siblings) + 2)
        self.assertEqual(stacks[1][-1].lineno, firstlineno(get_stack2) + 3)
        self.assertEqual(stacks[1][-2].lineno, firstlineno(get_stack_siblings) + 2)

    @jit_suppress
    @unittest.skipIf(not cinderjit or not cinderjit.is_hir_inliner_enabled(), 'meaningless without HIR inliner enabled')
    def test_line_numbers_at_multiple_points_in_inlined_functions(self):
        if False:
            while True:
                i = 10
        'Verify that line numbers are are correct at different points in an inlined\n        function'
        self.assertEqual(cinderjit.get_num_inlined_functions(call_get_stack_multi), 1)
        stacks = call_get_stack_multi()
        self.assertEqual(stacks[0][-1].lineno, firstlineno(get_stack_multi) + 3)
        self.assertEqual(stacks[0][-2].lineno, firstlineno(call_get_stack_multi) + 3)
        self.assertEqual(stacks[1][-1].lineno, firstlineno(get_stack_multi) + 5)
        self.assertEqual(stacks[1][-2].lineno, firstlineno(call_get_stack_multi) + 3)

    @jit_suppress
    @unittest.skipIf(not cinderjit or not cinderjit.is_hir_inliner_enabled(), 'meaningless without HIR inliner enabled')
    def test_inline_function_stats(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(cinderjit.get_num_inlined_functions(func), 2)
        stats = cinderjit.get_inlined_functions_stats(func)
        self.assertEqual({'num_inlined_functions': 2, 'failure_stats': {'HasVarargs': {'test_cinderx.test_cinderjit:func_with_varargs'}}}, stats)

    @jit_suppress
    @unittest.skipIf(not cinderjit or not cinderjit.is_hir_inliner_enabled(), 'meaningless without HIR inliner enabled')
    def test_line_numbers_with_multiple_inlined_calls(self):
        if False:
            return 10
        'Verify that line numbers are correct for inlined calls that appear\n        in different statements\n        '
        self.assertEqual(cinderjit.get_num_inlined_functions(get_stack_twice), 2)
        stacks = get_stack_twice()
        self.assertEqual(stacks[0][-1].lineno, firstlineno(get_stack) + 3)
        self.assertEqual(stacks[0][-2].lineno, firstlineno(get_stack_twice) + 3)
        self.assertEqual(stacks[1][-1].lineno, firstlineno(get_stack) + 3)
        self.assertEqual(stacks[1][-2].lineno, firstlineno(get_stack_twice) + 4)

class FaulthandlerTracebackTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def f1(self, fd):
        if False:
            i = 10
            return i + 15
        self.f2(fd)

    @cinder_support.failUnlessJITCompiled
    def f2(self, fd):
        if False:
            while True:
                i = 10
        self.f3(fd)

    @cinder_support.failUnlessJITCompiled
    def f3(self, fd):
        if False:
            for i in range(10):
                print('nop')
        faulthandler.dump_traceback(fd)

    def test_dumptraceback(self):
        if False:
            return 10
        expected = [f'  File "{__file__}", line {firstlineno(self.f3) + 2} in f3', f'  File "{__file__}", line {firstlineno(self.f2) + 2} in f2', f'  File "{__file__}", line {firstlineno(self.f1) + 2} in f1']
        with tempfile.TemporaryFile() as f:
            self.f1(f.fileno())
            f.seek(0)
            output = f.read().decode('ascii')
            lines = output.split('\n')
            self.assertGreaterEqual(len(lines), len(expected) + 1)
            self.assertEqual(lines[1:4], expected)

def with_globals(gbls):
    if False:
        return 10

    def decorator(func):
        if False:
            print('Hello World!')
        new_func = type(func)(func.__code__, gbls, func.__name__, func.__defaults__, func.__closure__)
        new_func.__module__ = func.__module__
        new_func.__kwdefaults__ = func.__kwdefaults__
        return new_func
    return decorator

@cinder_support.failUnlessJITCompiled
def get_meaning_of_life(obj):
    if False:
        return 10
    return obj.meaning_of_life()

def nothing():
    if False:
        return 10
    return 0

def _simpleFunc(a, b):
    if False:
        while True:
            i = 10
    return (a, b)

class _CallableObj:

    def __call__(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        return (self, a, b)

class CallKWArgsTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def test_call_basic_function_pos_and_kw(self):
        if False:
            for i in range(10):
                print('nop')
        r = _simpleFunc(1, b=2)
        self.assertEqual(r, (1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_basic_function_kw_only(self):
        if False:
            return 10
        r = _simpleFunc(b=2, a=1)
        self.assertEqual(r, (1, 2))
        r = _simpleFunc(a=1, b=2)
        self.assertEqual(r, (1, 2))

    @staticmethod
    def _f1(a, b):
        if False:
            i = 10
            return i + 15
        return (a, b)

    @cinder_support.failUnlessJITCompiled
    def test_call_class_static_pos_and_kw(self):
        if False:
            for i in range(10):
                print('nop')
        r = CallKWArgsTests._f1(1, b=2)
        self.assertEqual(r, (1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_class_static_kw_only(self):
        if False:
            return 10
        r = CallKWArgsTests._f1(b=2, a=1)
        self.assertEqual(r, (1, 2))

    def _f2(self, a, b):
        if False:
            print('Hello World!')
        return (self, a, b)

    @cinder_support.failUnlessJITCompiled
    def test_call_method_kw_and_pos(self):
        if False:
            while True:
                i = 10
        r = self._f2(1, b=2)
        self.assertEqual(r, (self, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_method_kw_only(self):
        if False:
            print('Hello World!')
        r = self._f2(b=2, a=1)
        self.assertEqual(r, (self, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_bound_method_kw_and_pos(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._f2
        r = f(1, b=2)
        self.assertEqual(r, (self, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_bound_method_kw_only(self):
        if False:
            print('Hello World!')
        f = self._f2
        r = f(b=2, a=1)
        self.assertEqual(r, (self, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_obj_kw_and_pos(self):
        if False:
            for i in range(10):
                print('nop')
        o = _CallableObj()
        r = o(1, b=2)
        self.assertEqual(r, (o, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_obj_kw_only(self):
        if False:
            return 10
        o = _CallableObj()
        r = o(b=2, a=1)
        self.assertEqual(r, (o, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_c_func(self):
        if False:
            print('Hello World!')
        self.assertEqual(__import__('sys', globals=None), sys)

class CallExTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def test_call_dynamic_kw_dict(self):
        if False:
            return 10
        r = _simpleFunc(**{'b': 2, 'a': 1})
        self.assertEqual(r, (1, 2))

    class _DummyMapping:

        def keys(self):
            if False:
                return 10
            return ('a', 'b')

        def __getitem__(self, k):
            if False:
                print('Hello World!')
            return {'a': 1, 'b': 2}[k]

    @cinder_support.failUnlessJITCompiled
    def test_call_dynamic_kw_dict(self):
        if False:
            return 10
        r = _simpleFunc(**CallExTests._DummyMapping())
        self.assertEqual(r, (1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_dynamic_pos_tuple(self):
        if False:
            print('Hello World!')
        r = _simpleFunc(*(1, 2))
        self.assertEqual(r, (1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_dynamic_pos_list(self):
        if False:
            for i in range(10):
                print('nop')
        r = _simpleFunc(*[1, 2])
        self.assertEqual(r, (1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_dynamic_pos_and_kw(self):
        if False:
            for i in range(10):
                print('nop')
        r = _simpleFunc(*(1,), **{'b': 2})
        self.assertEqual(r, (1, 2))

    @cinder_support.failUnlessJITCompiled
    def _doCall(self, args, kwargs):
        if False:
            return 10
        return _simpleFunc(*args, **kwargs)

    def test_invalid_kw_type(self):
        if False:
            while True:
                i = 10
        err = '_simpleFunc\\(\\) argument after \\*\\* must be a mapping, not int'
        with self.assertRaisesRegex(TypeError, err):
            self._doCall([], 1)

    @cinder_support.skipUnlessJITEnabled('Exposes interpreter reference leak')
    def test_invalid_pos_type(self):
        if False:
            print('Hello World!')
        err = '_simpleFunc\\(\\) argument after \\* must be an iterable, not int'
        with self.assertRaisesRegex(TypeError, err):
            self._doCall(1, {})

    @staticmethod
    def _f1(a, b):
        if False:
            i = 10
            return i + 15
        return (a, b)

    @cinder_support.failUnlessJITCompiled
    def test_call_class_static_pos_and_kw(self):
        if False:
            while True:
                i = 10
        r = CallExTests._f1(*(1,), **{'b': 2})
        self.assertEqual(r, (1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_class_static_kw_only(self):
        if False:
            print('Hello World!')
        r = CallKWArgsTests._f1(**{'b': 2, 'a': 1})
        self.assertEqual(r, (1, 2))

    def _f2(self, a, b):
        if False:
            while True:
                i = 10
        return (self, a, b)

    @cinder_support.failUnlessJITCompiled
    def test_call_method_kw_and_pos(self):
        if False:
            i = 10
            return i + 15
        r = self._f2(*(1,), **{'b': 2})
        self.assertEqual(r, (self, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_method_kw_only(self):
        if False:
            return 10
        r = self._f2(**{'b': 2, 'a': 1})
        self.assertEqual(r, (self, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_bound_method_kw_and_pos(self):
        if False:
            print('Hello World!')
        f = self._f2
        r = f(*(1,), **{'b': 2})
        self.assertEqual(r, (self, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_bound_method_kw_only(self):
        if False:
            i = 10
            return i + 15
        f = self._f2
        r = f(**{'b': 2, 'a': 1})
        self.assertEqual(r, (self, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_obj_kw_and_pos(self):
        if False:
            for i in range(10):
                print('nop')
        o = _CallableObj()
        r = o(*(1,), **{'b': 2})
        self.assertEqual(r, (o, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_obj_kw_only(self):
        if False:
            i = 10
            return i + 15
        o = _CallableObj()
        r = o(**{'b': 2, 'a': 1})
        self.assertEqual(r, (o, 1, 2))

    @cinder_support.failUnlessJITCompiled
    def test_call_c_func_pos_only(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(*([2],)), 1)

    @cinder_support.failUnlessJITCompiled
    def test_call_c_func_pos_and_kw(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(__import__(*('sys',), **{'globals': None}), sys)

class LoadMethodCacheTests(unittest.TestCase):

    def test_type_modified(self):
        if False:
            print('Hello World!')

        class Oracle:

            def meaning_of_life(self):
                if False:
                    while True:
                        i = 10
                return 42
        obj = Oracle()
        self.assertEqual(get_meaning_of_life(obj), 42)
        self.assertEqual(get_meaning_of_life(obj), 42)

        def new_meaning_of_life(x):
            if False:
                print('Hello World!')
            return 0
        Oracle.meaning_of_life = new_meaning_of_life
        self.assertEqual(get_meaning_of_life(obj), 0)

    def test_base_type_modified(self):
        if False:
            print('Hello World!')

        class Base:

            def meaning_of_life(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 42

        class Derived(Base):
            pass
        obj = Derived()
        self.assertEqual(get_meaning_of_life(obj), 42)
        self.assertEqual(get_meaning_of_life(obj), 42)

        def new_meaning_of_life(x):
            if False:
                while True:
                    i = 10
            return 0
        Base.meaning_of_life = new_meaning_of_life
        self.assertEqual(get_meaning_of_life(obj), 0)

    def test_second_base_type_modified(self):
        if False:
            print('Hello World!')

        class Base1:
            pass

        class Base2:

            def meaning_of_life(self):
                if False:
                    i = 10
                    return i + 15
                return 42

        class Derived(Base1, Base2):
            pass
        obj = Derived()
        self.assertEqual(get_meaning_of_life(obj), 42)
        self.assertEqual(get_meaning_of_life(obj), 42)

        def new_meaning_of_life(x):
            if False:
                return 10
            return 0
        Base1.meaning_of_life = new_meaning_of_life
        self.assertEqual(get_meaning_of_life(obj), 0)

    def test_type_dunder_bases_reassigned(self):
        if False:
            for i in range(10):
                print('nop')

        class Base1:
            pass

        class Derived(Base1):
            pass
        obj1 = Derived()
        obj2 = Derived()
        obj2.meaning_of_life = nothing

        class Base2:

            def meaning_of_life(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 42
        Derived.__bases__ = (Base2,)
        self.assertEqual(get_meaning_of_life(obj1), 42)
        self.assertEqual(get_meaning_of_life(obj1), 42)
        self.assertEqual(get_meaning_of_life(obj2), 0)

    def _make_obj(self):
        if False:
            i = 10
            return i + 15

        class Oracle:

            def meaning_of_life(self):
                if False:
                    while True:
                        i = 10
                return 42
        obj = Oracle()
        self.assertEqual(get_meaning_of_life(obj), 42)
        self.assertEqual(get_meaning_of_life(obj), 42)
        return obj

    def test_instance_assignment(self):
        if False:
            return 10
        obj = self._make_obj()
        obj.meaning_of_life = nothing
        self.assertEqual(get_meaning_of_life(obj), 0)

    def test_instance_dict_assignment(self):
        if False:
            return 10
        obj = self._make_obj()
        obj.__dict__['meaning_of_life'] = nothing
        self.assertEqual(get_meaning_of_life(obj), 0)

    def test_instance_dict_replacement(self):
        if False:
            while True:
                i = 10
        obj = self._make_obj()
        obj.__dict__ = {'meaning_of_life': nothing}
        self.assertEqual(get_meaning_of_life(obj), 0)

    def test_instance_dunder_class_assignment(self):
        if False:
            return 10
        obj = self._make_obj()

        class Other:
            pass
        other = Other()
        other.meaning_of_life = nothing
        other.__class__ = obj.__class__
        self.assertEqual(get_meaning_of_life(other), 0)

    def test_shadowcode_setattr(self):
        if False:
            while True:
                i = 10
        'sets attribute via shadow byte code, it should update the\n        type bit for instance shadowing'
        obj = self._make_obj()
        obj.foo = 42
        obj1 = type(obj)()
        obj1.other = 100

        def f(obj, set):
            if False:
                return 10
            if set:
                obj.meaning_of_life = nothing
            yield 42
        for i in range(100):
            list(f(obj, False))
        list(f(obj, True))
        self.assertEqual(get_meaning_of_life(obj), 0)

    def test_shadowcode_setattr_split(self):
        if False:
            while True:
                i = 10
        'sets attribute via shadow byte code on a split dict,\n        it should update the type bit for instance shadowing'
        obj = self._make_obj()

        def f(obj, set):
            if False:
                print('Hello World!')
            if set:
                obj.meaning_of_life = nothing
            yield 42
        for i in range(100):
            list(f(obj, False))
        list(f(obj, True))
        self.assertEqual(get_meaning_of_life(obj), 0)

    def _index_long(self):
        if False:
            i = 10
            return i + 15
        return 6 .__index__()

    def test_call_wrapper_descriptor(self):
        if False:
            return 10
        self.assertEqual(self._index_long(), 6)

@unittest.skipUnless(cinderjit, 'Test meaningless without the JIT enabled')
class LoadModuleMethodCacheTests(unittest.TestCase):

    def test_load_method_from_module(self):
        if False:
            i = 10
            return i + 15
        with cinder_support.temp_sys_path() as tmp:
            (tmp / 'tmp_a.py').write_text(dedent('\n                    a = 1\n                    def get_a():\n                        return 1+2\n                    '), encoding='utf8')
            (tmp / 'tmp_b.py').write_text(dedent('\n                    import tmp_a\n\n                    def test():\n                        return tmp_a.get_a()\n                    '), encoding='utf8')
            import tmp_b
            cinderjit.force_compile(tmp_b.test)
            self.assertEqual(tmp_b.test(), 3)
            self.assertTrue(cinderjit.is_jit_compiled(tmp_b.test))
            self.assertTrue('LoadModuleMethod' in cinderjit.get_function_hir_opcode_counts(tmp_b.test))
            import tmp_a
            tmp_a.get_a = lambda : 10
            self.assertEqual(tmp_b.test(), 10)
            delattr(tmp_a, 'get_a')
            with self.assertRaises(AttributeError):
                tmp_b.test()

    def test_load_method_from_strict_module(self):
        if False:
            print('Hello World!')
        strict_sandbox = base_sandbox.use_cm(sandbox, self)
        code_str = '\n        import __strict__\n        a = 1\n        def get_a():\n            return 1+2\n        '
        strict_sandbox.write_file('tmp_a.py', code_str)
        code_str = '\n        import __strict__\n        import tmp_a\n\n        def test():\n            return tmp_a.get_a()\n        '
        strict_sandbox.write_file('tmp_b.py', code_str)
        tmp_b = strict_sandbox.strict_import('tmp_b')
        cinderjit.force_compile(tmp_b.test)
        self.assertTrue(cinderjit.is_jit_compiled(tmp_b.test))
        self.assertTrue('LoadModuleMethod' in cinderjit.get_function_hir_opcode_counts(tmp_b.test))
        self.assertEqual(tmp_b.test(), 3)
        self.assertEqual(tmp_b.test(), 3)

@cinder_support.failUnlessJITCompiled
@failUnlessHasOpcodes('LOAD_ATTR')
def get_foo(obj):
    if False:
        return 10
    return obj.foo

class LoadAttrCacheTests(unittest.TestCase):

    def test_dict_reassigned(self):
        if False:
            for i in range(10):
                print('nop')

        class Base:

            def __init__(self, x):
                if False:
                    print('Hello World!')
                self.foo = x
        obj1 = Base(100)
        obj2 = Base(200)
        self.assertEqual(get_foo(obj1), 100)
        self.assertEqual(get_foo(obj1), 100)
        self.assertEqual(get_foo(obj2), 200)
        obj1.__dict__ = {'foo': 200}
        self.assertEqual(get_foo(obj1), 200)
        self.assertEqual(get_foo(obj2), 200)

    def test_dict_mutated(self):
        if False:
            while True:
                i = 10

        class Base:

            def __init__(self, foo):
                if False:
                    print('Hello World!')
                self.foo = foo
        obj = Base(100)
        self.assertEqual(get_foo(obj), 100)
        self.assertEqual(get_foo(obj), 100)
        obj.__dict__['foo'] = 200
        self.assertEqual(get_foo(obj), 200)

    def test_dict_resplit(self):
        if False:
            while True:
                i = 10

        class Base:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                (self.foo, self.a, self.b) = (100, 200, 300)
                (self.c, self.d, self.e) = (400, 500, 600)
        obj = Base()
        self.assertEqual(get_foo(obj), 100)
        self.assertEqual(get_foo(obj), 100)
        obj.foo = 800
        self.assertEqual(get_foo(obj), 800)

    def test_dict_combined(self):
        if False:
            print('Hello World!')

        class Base:

            def __init__(self, foo):
                if False:
                    for i in range(10):
                        print('nop')
                self.foo = foo
        obj1 = Base(100)
        self.assertEqual(get_foo(obj1), 100)
        self.assertEqual(get_foo(obj1), 100)
        obj2 = Base(200)
        obj2.bar = 300
        obj3 = Base(400)
        obj3.baz = 500
        obj4 = Base(600)
        self.assertEqual(get_foo(obj1), 100)
        self.assertEqual(get_foo(obj2), 200)
        self.assertEqual(get_foo(obj3), 400)
        self.assertEqual(get_foo(obj4), 600)

class SetNonDataDescrAttrTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('STORE_ATTR')
    def set_foo(self, obj, val):
        if False:
            i = 10
            return i + 15
        obj.foo = val

    def setUp(self):
        if False:
            while True:
                i = 10

        class Descr:

            def __init__(self, name):
                if False:
                    return 10
                self.name = name

            def __get__(self, obj, typ):
                if False:
                    while True:
                        i = 10
                return obj.__dict__[self.name]
        self.descr_type = Descr
        self.descr = Descr('foo')

        class Test:
            foo = self.descr
        self.obj = Test()

    def test_set_when_changed_to_data_descr(self):
        if False:
            print('Hello World!')
        self.set_foo(self.obj, 100)
        self.assertEqual(self.obj.foo, 100)
        self.set_foo(self.obj, 200)
        self.assertEqual(self.obj.foo, 200)

        def setter(self, obj, val):
            if False:
                print('Hello World!')
            self.invoked = True
        self.descr.__class__.__set__ = setter
        self.set_foo(self.obj, 300)
        self.assertEqual(self.obj.foo, 200)
        self.assertTrue(self.descr.invoked)

class GetSetNonDataDescrAttrTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('LOAD_ATTR')
    def get_foo(self, obj):
        if False:
            print('Hello World!')
        return obj.foo

    def setUp(self):
        if False:
            print('Hello World!')

        class NonDataDescr:

            def __init__(self, val):
                if False:
                    print('Hello World!')
                self.val = val
                self.invoked_count = 0
                self.set_dict = True

            def __get__(self, obj, typ):
                if False:
                    print('Hello World!')
                self.invoked_count += 1
                if self.set_dict:
                    obj.__dict__['foo'] = self.val
                return self.val
        self.descr_type = NonDataDescr
        self.descr = NonDataDescr('testing 123')

        class Test:
            foo = self.descr
        self.obj = Test()

    def test_get(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.get_foo(self.obj), 'testing 123')
        self.assertEqual(self.descr.invoked_count, 1)
        self.assertEqual(self.get_foo(self.obj), 'testing 123')
        self.assertEqual(self.descr.invoked_count, 1)
        obj2 = self.obj.__class__()
        self.assertEqual(self.get_foo(obj2), 'testing 123')
        self.assertEqual(self.descr.invoked_count, 2)

    def test_get_when_changed_to_data_descr(self):
        if False:
            return 10
        self.assertEqual(self.get_foo(self.obj), 'testing 123')
        self.assertEqual(self.descr.invoked_count, 1)
        self.assertEqual(self.get_foo(self.obj), 'testing 123')
        self.assertEqual(self.descr.invoked_count, 1)

        def setter(self, obj, val):
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.descr.__class__.__set__ = setter
        self.assertEqual(self.get_foo(self.obj), 'testing 123')
        self.assertEqual(self.descr.invoked_count, 2)

    def test_get_when_changed_to_classvar(self):
        if False:
            while True:
                i = 10
        self.descr.set_dict = False
        self.assertEqual(self.get_foo(self.obj), 'testing 123')
        self.assertEqual(self.descr.invoked_count, 1)
        self.assertEqual(self.get_foo(self.obj), 'testing 123')
        self.assertEqual(self.descr.invoked_count, 2)

        class ClassVar:
            pass
        self.descr.__class__ = ClassVar
        self.assertIs(self.get_foo(self.obj), self.descr)
        self.assertEqual(self.descr.invoked_count, 2)

@cinder_support.failUnlessJITCompiled
@failUnlessHasOpcodes('STORE_ATTR')
def set_foo(x, val):
    if False:
        return 10
    x.foo = val

class DataDescr:

    def __init__(self, val):
        if False:
            return 10
        self.val = val
        self.invoked = False

    def __get__(self, obj, typ):
        if False:
            print('Hello World!')
        return self.val

    def __set__(self, obj, val):
        if False:
            print('Hello World!')
        self.invoked = True

class StoreAttrCacheTests(unittest.TestCase):

    def test_data_descr_attached(self):
        if False:
            return 10

        class Base:

            def __init__(self, x):
                if False:
                    while True:
                        i = 10
                self.foo = x
        obj = Base(100)
        set_foo(obj, 200)
        set_foo(obj, 200)
        self.assertEqual(obj.foo, 200)
        descr = DataDescr(300)
        Base.foo = descr
        set_foo(obj, 200)
        self.assertEqual(obj.foo, 300)
        self.assertTrue(descr.invoked)
        descr.invoked = False
        set_foo(obj, 400)
        self.assertEqual(obj.foo, 300)
        self.assertTrue(descr.invoked)

    def test_swap_split_dict_with_combined(self):
        if False:
            while True:
                i = 10

        class Base:

            def __init__(self, x):
                if False:
                    while True:
                        i = 10
                self.foo = x
        obj = Base(100)
        set_foo(obj, 200)
        set_foo(obj, 200)
        self.assertEqual(obj.foo, 200)
        d = {'foo': 300}
        obj.__dict__ = d
        set_foo(obj, 400)
        self.assertEqual(obj.foo, 400)
        self.assertEqual(d['foo'], 400)

    def test_swap_combined_dict_with_split(self):
        if False:
            for i in range(10):
                print('nop')

        class Base:

            def __init__(self, x):
                if False:
                    return 10
                self.foo = x
        obj = Base(100)
        obj.__dict__ = {'foo': 100}
        set_foo(obj, 200)
        set_foo(obj, 200)
        self.assertEqual(obj.foo, 200)
        obj2 = Base(300)
        set_foo(obj2, 400)
        self.assertEqual(obj2.foo, 400)

    def test_split_dict_no_slot(self):
        if False:
            print('Hello World!')

        class Base:
            pass
        obj = Base()
        obj.quox = 42
        obj1 = Base()
        obj1.__dict__['other'] = 100
        set_foo(obj1, 300)
        self.assertEqual(obj1.foo, 300)
        set_foo(obj, 400)
        self.assertEqual(obj1.foo, 300)

class LoadGlobalCacheTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        global license, a_global
        try:
            del license
        except NameError:
            pass
        try:
            del a_global
        except NameError:
            pass

    @staticmethod
    def set_global(value):
        if False:
            while True:
                i = 10
        global a_global
        a_global = value

    @staticmethod
    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('LOAD_GLOBAL')
    def get_global():
        if False:
            i = 10
            return i + 15
        return a_global

    @staticmethod
    def del_global():
        if False:
            return 10
        global a_global
        del a_global

    @staticmethod
    def set_license(value):
        if False:
            return 10
        global license
        license = value

    @staticmethod
    def del_license():
        if False:
            i = 10
            return i + 15
        global license
        del license

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('LOAD_GLOBAL')
    def test_simple(self):
        if False:
            return 10
        global a_global
        self.set_global(123)
        self.assertEqual(a_global, 123)
        self.set_global(456)
        self.assertEqual(a_global, 456)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('LOAD_GLOBAL')
    def test_shadow_builtin(self):
        if False:
            return 10
        self.assertIs(license, builtins.license)
        self.set_license(3735928559)
        self.assertIs(license, 3735928559)
        self.del_license()
        self.assertIs(license, builtins.license)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('LOAD_GLOBAL')
    def test_shadow_fake_builtin(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(NameError, self.get_global)
        builtins.a_global = 'poke'
        self.assertEqual(a_global, 'poke')
        self.set_global('override poke')
        self.assertEqual(a_global, 'override poke')
        self.del_global()
        self.assertEqual(a_global, 'poke')
        delattr(builtins, 'a_global')
        self.assertRaises(NameError, self.get_global)

    class prefix_str(str):

        def __new__(ty, prefix, value):
            if False:
                for i in range(10):
                    print('nop')
            s = super().__new__(ty, value)
            s.prefix = prefix
            return s

        def __hash__(self):
            if False:
                return 10
            return hash(self.prefix + self)

        def __eq__(self, other):
            if False:
                return 10
            return self.prefix + self == other

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('LOAD_GLOBAL')
    def test_weird_key_in_globals(self):
        if False:
            print('Hello World!')
        global a_global
        self.assertRaises(NameError, self.get_global)
        globals()[self.prefix_str('a_glo', 'bal')] = 'a value'
        self.assertEqual(a_global, 'a value')
        self.assertEqual(self.get_global(), 'a value')

    class MyGlobals(dict):

        def __getitem__(self, key):
            if False:
                return 10
            if key == 'knock_knock':
                return "who's there?"
            return super().__getitem__(key)

    @with_globals(MyGlobals())
    def return_knock_knock(self):
        if False:
            while True:
                i = 10
        return knock_knock

    def test_dict_subclass_globals(self):
        if False:
            return 10
        self.assertEqual(self.return_knock_knock(), "who's there?")

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('LOAD_GLOBAL')
    def _test_unwatch_builtins(self):
        if False:
            i = 10
            return i + 15
        self.set_global('hey')
        self.assertEqual(self.get_global(), 'hey')
        builtins.__dict__[42] = 42

    @cinder_support.runInSubprocess
    def test_unwatch_builtins(self):
        if False:
            i = 10
            return i + 15
        try:
            self._test_unwatch_builtins()
        finally:
            del builtins.__dict__[42]

    @failUnlessHasOpcodes('LOAD_GLOBAL')
    @cinder_support.runInSubprocess
    def test_preload_side_effect_modifies_globals(self):
        if False:
            for i in range(10):
                print('nop')
        with cinder_support.temp_sys_path() as tmp:
            (tmp / 'tmp_a.py').write_text(dedent('\n                    import importlib\n                    importlib.set_lazy_imports(True)\n                    from tmp_b import B\n\n                    A = 1\n\n                    def get_a():\n                        return A + B\n\n                    '), encoding='utf8')
            (tmp / 'tmp_b.py').write_text(dedent('\n                    import tmp_a\n\n                    tmp_a.A = 2\n\n                    B = 3\n                    '), encoding='utf8')
            if cinderjit:
                cinderjit.clear_runtime_stats()
            import tmp_a
            if cinderjit:
                cinderjit.force_compile(tmp_a.get_a)
            tmp_a.get_a()
            self.assertEqual(tmp_a.get_a(), 5)
            if cinderjit:
                self.assertTrue(cinderjit.is_jit_compiled(tmp_a.get_a))
                stats = cinderjit.get_and_clear_runtime_stats()
                relevant_deopts = [d for d in stats['deopt'] if d['normal']['func_qualname'] == 'get_a']
                self.assertEqual(relevant_deopts, [])

    @failUnlessHasOpcodes('LOAD_GLOBAL')
    @cinder_support.runInSubprocess
    def test_preload_side_effect_makes_globals_unwatchable(self):
        if False:
            while True:
                i = 10
        with cinder_support.temp_sys_path() as tmp:
            (tmp / 'tmp_a.py').write_text(dedent('\n                    import importlib\n                    importlib.set_lazy_imports(True)\n                    from tmp_b import B\n\n                    A = 1\n\n                    def get_a():\n                        return A + B\n\n                    '), encoding='utf8')
            (tmp / 'tmp_b.py').write_text(dedent('\n                    import tmp_a\n\n                    tmp_a.__dict__[42] = 1\n                    tmp_a.A = 2\n\n                    B = 3\n                    '), encoding='utf8')
            if cinderjit:
                cinderjit.clear_runtime_stats()
            import tmp_a
            if cinderjit:
                cinderjit.force_compile(tmp_a.get_a)
            tmp_a.get_a()
            self.assertEqual(tmp_a.get_a(), 5)
            if cinderjit:
                self.assertTrue(cinderjit.is_jit_compiled(tmp_a.get_a))

    @failUnlessHasOpcodes('LOAD_GLOBAL')
    @cinder_support.runInSubprocess
    def test_preload_side_effect_makes_builtins_unwatchable(self):
        if False:
            i = 10
            return i + 15
        with cinder_support.temp_sys_path() as tmp:
            (tmp / 'tmp_a.py').write_text(dedent('\n                    import importlib\n                    importlib.set_lazy_imports(True)\n                    from tmp_b import B\n\n                    def get_a():\n                        return max(1, 2) + B\n\n                    '), encoding='utf8')
            (tmp / 'tmp_b.py').write_text(dedent('\n                    __builtins__[42] = 2\n\n                    B = 3\n                    '), encoding='utf8')
            if cinderjit:
                cinderjit.clear_runtime_stats()
            import tmp_a
            if cinderjit:
                cinderjit.force_compile(tmp_a.get_a)
            tmp_a.get_a()
            self.assertEqual(tmp_a.get_a(), 5)
            if cinderjit:
                self.assertTrue(cinderjit.is_jit_compiled(tmp_a.get_a))

    @cinder_support.runInSubprocess
    def test_lazy_import_after_global_cached(self):
        if False:
            print('Hello World!')
        with cinder_support.temp_sys_path() as tmp:
            (tmp / 'tmp_a.py').write_text(dedent('\n                    import importlib\n                    importlib.set_lazy_imports(True)\n                    from tmp_b import B\n\n                    def f():\n                        return B\n\n                    for _ in range(51):\n                        f()\n\n                    from tmp_b import B\n                    '))
            (tmp / 'tmp_b.py').write_text(dedent('\n                    B = 3\n                    '))
            import tmp_a
            self.assertEqual(tmp_a.f(), 3)

class ClosureTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def test_cellvar(self):
        if False:
            return 10
        a = 1

        def foo():
            if False:
                return 10
            return a
        self.assertEqual(foo(), 1)

    @cinder_support.failUnlessJITCompiled
    def test_two_cellvars(self):
        if False:
            while True:
                i = 10
        a = 1
        b = 2

        def g():
            if False:
                i = 10
                return i + 15
            return a + b
        self.assertEqual(g(), 3)

    @cinder_support.failUnlessJITCompiled
    def test_cellvar_argument(self):
        if False:
            while True:
                i = 10

        def foo():
            if False:
                return 10
            self.assertEqual(1, 1)
        foo()

    @cinder_support.failUnlessJITCompiled
    def test_cellvar_argument_modified(self):
        if False:
            while True:
                i = 10
        self_ = self

        def foo():
            if False:
                print('Hello World!')
            nonlocal self
            self = 1
        self_.assertIs(self, self_)
        foo()
        self_.assertEqual(self, 1)

    @cinder_support.failUnlessJITCompiled
    def _cellvar_unbound(self):
        if False:
            print('Hello World!')
        b = a
        a = 1

        def g():
            if False:
                i = 10
                return i + 15
            return a

    def test_cellvar_unbound(self):
        if False:
            return 10
        with self.assertRaises(UnboundLocalError) as ctx:
            self._cellvar_unbound()
        self.assertEqual(str(ctx.exception), "local variable 'a' referenced before assignment")

    def test_freevars(self):
        if False:
            i = 10
            return i + 15
        x = 1

        @cinder_support.failUnlessJITCompiled
        def nested():
            if False:
                while True:
                    i = 10
            return x
        x = 2
        self.assertEqual(nested(), 2)

    def test_freevars_multiple_closures(self):
        if False:
            i = 10
            return i + 15

        def get_func(a):
            if False:
                for i in range(10):
                    print('nop')

            @cinder_support.failUnlessJITCompiled
            def f():
                if False:
                    for i in range(10):
                        print('nop')
                return a
            return f
        f1 = get_func(1)
        f2 = get_func(2)
        self.assertEqual(f1(), 1)
        self.assertEqual(f2(), 2)

    def test_nested_func(self):
        if False:
            while True:
                i = 10

        @cinder_support.failUnlessJITCompiled
        def add(a, b):
            if False:
                return 10
            return a + b
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add('eh', 'bee'), 'ehbee')

    @staticmethod
    def make_adder(a):
        if False:
            print('Hello World!')

        @cinder_support.failUnlessJITCompiled
        def add(b):
            if False:
                while True:
                    i = 10
            return a + b
        return add

    def test_nested_func_with_closure(self):
        if False:
            i = 10
            return i + 15
        add_3 = self.make_adder(3)
        add_7 = self.make_adder(7)
        self.assertEqual(add_3(10), 13)
        self.assertEqual(add_7(12), 19)
        self.assertEqual(add_3(add_7(-100)), -90)
        with self.assertRaises(TypeError):
            add_3('ok')

    def test_nested_func_with_different_globals(self):
        if False:
            i = 10
            return i + 15

        @cinder_support.failUnlessJITCompiled
        @with_globals({'A_GLOBAL_CONSTANT': 3735928559})
        def return_global():
            if False:
                return 10
            return A_GLOBAL_CONSTANT
        self.assertEqual(return_global(), 3735928559)
        return_other_global = with_globals({'A_GLOBAL_CONSTANT': 4207849484})(return_global)
        self.assertEqual(return_other_global(), 4207849484)
        self.assertEqual(return_global(), 3735928559)
        self.assertEqual(return_other_global(), 4207849484)

    def test_nested_func_outlives_parent(self):
        if False:
            return 10

        @cinder_support.failUnlessJITCompiled
        def nested(x):
            if False:
                i = 10
                return i + 15

            @cinder_support.failUnlessJITCompiled
            def inner(y):
                if False:
                    print('Hello World!')
                return x + y
            return inner
        nested_ref = weakref.ref(nested)
        add_5 = nested(5)
        nested = None
        self.assertIsNone(nested_ref())
        self.assertEqual(add_5(10), 15)

class TempNameTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def _tmp_name(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        tmp1 = 'hello'
        c = a + b
        return tmp1

    def test_tmp_name(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self._tmp_name(1, 2), 'hello')

    @cinder_support.failUnlessJITCompiled
    def test_tmp_name2(self):
        if False:
            i = 10
            return i + 15
        v0 = 5
        self.assertEqual(v0, 5)

class DummyContainer:

    def __len__(self):
        if False:
            while True:
                i = 10
        raise Exception('hello!')

class ExceptionInConditional(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def doit(self, x):
        if False:
            i = 10
            return i + 15
        if x:
            return 1
        return 2

    def test_exception_thrown_in_conditional(self):
        if False:
            return 10
        with self.assertRaisesRegex(Exception, 'hello!'):
            self.doit(DummyContainer())

class JITCompileCrasherRegressionTests(StaticTestBase):

    @cinder_support.failUnlessJITCompiled
    def _fstring(self, flag, it1, it2):
        if False:
            while True:
                i = 10
        for a in it1:
            for b in it2:
                if flag:
                    return f'{a}'

    def test_fstring_no_fmt_spec_in_nested_loops_and_if(self):
        if False:
            print('Hello World!')
        self.assertEqual(self._fstring(True, [1], [1]), '1')

    @cinder_support.failUnlessJITCompiled
    async def _sharedAwait(self, x, y, z):
        return await (x() if y else z())

    def test_shared_await(self):
        if False:
            i = 10
            return i + 15

        async def zero():
            return 0

        async def one():
            return 1
        with self.assertRaises(StopIteration) as exc:
            self._sharedAwait(zero, True, one).send(None)
        self.assertEqual(exc.exception.value, 0)
        with self.assertRaises(StopIteration) as exc:
            self._sharedAwait(zero, False, one).send(None)
        self.assertEqual(exc.exception.value, 1)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('LOAD_METHOD')
    def load_method_on_maybe_defined_value(self):
        if False:
            while True:
                i = 10
        try:
            pass
        except:
            x = 1
        return x.__index__()

    def test_load_method_on_maybe_defined_value(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(NameError):
            self.load_method_on_maybe_defined_value()

    @cinder_support.runInSubprocess
    def test_condbranch_codegen(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = f'\n            from __static__ import cbool\n            from typing import Optional\n\n\n            class Foo:\n                def __init__(self, x: bool) -> None:\n                    y = cbool(x)\n                    self.start_offset_us: float = 0.0\n                    self.y: cbool = y\n        '
        with self.in_module(codestr) as mod:
            gc.immortalize_heap()
            if cinderjit:
                cinderjit.force_compile(mod.Foo.__init__)
            foo = mod.Foo(True)

    def test_restore_materialized_parent_pyframe_in_gen_throw(self):
        if False:
            return 10
        from __static__ import ContextDecorator

        async def a(child_fut, main_fut, box):
            return await b(child_fut, main_fut, box)

        async def b(child_fut, main_fut, box):
            return await c(child_fut, main_fut, box)

        @ContextDecorator()
        async def c(child_fut, main_fut, box):
            return await d(child_fut, main_fut, box)

        async def d(child_fut, main_fut, box):
            main_fut.set_result(True)
            try:
                await child_fut
            except:
                box[0].cr_frame
                raise

        async def main():
            child_fut = asyncio.Future()
            main_fut = asyncio.Future()
            box = [None]
            coro = a(child_fut, main_fut, box)
            box[0] = coro
            t = asyncio.create_task(coro)
            await main_fut
            t.cancel()
            await t
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(main())
        if cinderjit and cinderjit.auto_jit_threshold() <= 1:
            self.assertTrue(cinderjit.is_jit_compiled(a))
            self.assertTrue(cinderjit.is_jit_compiled(b))
            self.assertTrue(cinderjit.is_jit_compiled(c.__wrapped__))
            self.assertTrue(cinderjit.is_jit_compiled(d))

class DelObserver:

    def __init__(self, id, cb):
        if False:
            print('Hello World!')
        self.id = id
        self.cb = cb

    def __del__(self):
        if False:
            i = 10
            return i + 15
        self.cb(self.id)

class UnwindStateTests(unittest.TestCase):
    DELETED = []

    def setUp(self):
        if False:
            print('Hello World!')
        self.DELETED.clear()
        self.addCleanup(lambda : self.DELETED.clear())

    def get_del_observer(self, id):
        if False:
            i = 10
            return i + 15
        return DelObserver(id, lambda i: self.DELETED.append(i))

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('RAISE_VARARGS')
    def _copied_locals(self, a):
        if False:
            while True:
                i = 10
        b = c = a
        raise RuntimeError()

    def test_copied_locals_in_frame(self):
        if False:
            return 10
        try:
            self._copied_locals('hello')
        except RuntimeError as re:
            f_locals = re.__traceback__.tb_next.tb_frame.f_locals
            self.assertEqual(f_locals, {'self': self, 'a': 'hello', 'b': 'hello', 'c': 'hello'})

    @cinder_support.failUnlessJITCompiled
    def _raise_with_del_observer_on_stack(self):
        if False:
            return 10
        for x in (1 for i in [self.get_del_observer(1)]):
            raise RuntimeError()

    def test_decref_stack_objects(self):
        if False:
            i = 10
            return i + 15
        'Items on stack should be decrefed on unwind.'
        try:
            self._raise_with_del_observer_on_stack()
        except RuntimeError:
            deleted = list(self.DELETED)
        else:
            self.fail('should have raised RuntimeError')
        self.assertEqual(deleted, [1])

    @cinder_support.failUnlessJITCompiled
    def _raise_with_del_observer_on_stack_and_cell_arg(self):
        if False:
            return 10
        for x in (self for i in [self.get_del_observer(1)]):
            raise RuntimeError()

    def test_decref_stack_objs_with_cell_args(self):
        if False:
            return 10
        try:
            self._raise_with_del_observer_on_stack_and_cell_arg()
        except RuntimeError:
            deleted = list(self.DELETED)
        else:
            self.fail('should have raised RuntimeError')
        self.assertEqual(deleted, [1])

class ImportTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('IMPORT_NAME')
    def test_import_name(self):
        if False:
            while True:
                i = 10
        import math
        self.assertEqual(int(math.pow(1, 2)), 1)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('IMPORT_NAME')
    def _fail_to_import_name(self):
        if False:
            i = 10
            return i + 15
        import non_existent_module

    def test_import_name_failure(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ModuleNotFoundError):
            self._fail_to_import_name()

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('IMPORT_NAME', 'IMPORT_FROM')
    def test_import_from(self):
        if False:
            i = 10
            return i + 15
        from math import pow as math_pow
        self.assertEqual(int(math_pow(1, 2)), 1)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('IMPORT_NAME', 'IMPORT_FROM')
    def _fail_to_import_from(self):
        if False:
            i = 10
            return i + 15
        from math import non_existent_attr

    def test_import_from_failure(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ImportError):
            self._fail_to_import_from()

class RaiseTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('RAISE_VARARGS')
    def _jitRaise(self, exc):
        if False:
            for i in range(10):
                print('nop')
        raise exc

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('RAISE_VARARGS')
    def _jitRaiseCause(self, exc, cause):
        if False:
            i = 10
            return i + 15
        raise exc from cause

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('RAISE_VARARGS')
    def _jitReraise(self):
        if False:
            i = 10
            return i + 15
        raise

    def test_raise_type(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            self._jitRaise(ValueError)

    def test_raise_value(self):
        if False:
            return 10
        with self.assertRaises(ValueError) as exc:
            self._jitRaise(ValueError(1))
        self.assertEqual(exc.exception.args, (1,))

    def test_raise_with_cause(self):
        if False:
            print('Hello World!')
        cause = ValueError(2)
        cause_tb_str = f'{cause.__traceback__}'
        with self.assertRaises(ValueError) as exc:
            self._jitRaiseCause(ValueError(1), cause)
        self.assertIs(exc.exception.__cause__, cause)
        self.assertEqual(f'{exc.exception.__cause__.__traceback__}', cause_tb_str)

    def test_reraise(self):
        if False:
            while True:
                i = 10
        original_raise = ValueError(1)
        with self.assertRaises(ValueError) as exc:
            try:
                raise original_raise
            except ValueError:
                self._jitReraise()
        self.assertIs(exc.exception, original_raise)

    def test_reraise_of_nothing(self):
        if False:
            print('Hello World!')
        with self.assertRaises(RuntimeError) as exc:
            self._jitReraise()
        self.assertEqual(exc.exception.args, ('No active exception to reraise',))

class GeneratorsTest(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def _f1(self):
        if False:
            i = 10
            return i + 15
        yield 1

    def test_basic_operation(self):
        if False:
            while True:
                i = 10
        g = self._f1()
        self.assertEqual(g.send(None), 1)
        with self.assertRaises(StopIteration) as exc:
            g.send(None)
        self.assertIsNone(exc.exception.value)

    @cinder_support.failUnlessJITCompiled
    def _f2(self):
        if False:
            return 10
        yield 1
        yield 2
        return 3

    def test_multi_yield_and_return(self):
        if False:
            while True:
                i = 10
        g = self._f2()
        self.assertEqual(g.send(None), 1)
        self.assertEqual(g.send(None), 2)
        with self.assertRaises(StopIteration) as exc:
            g.send(None)
        self.assertEqual(exc.exception.value, 3)

    @cinder_support.failUnlessJITCompiled
    def _f3(self):
        if False:
            i = 10
            return i + 15
        a = (yield 1)
        b = (yield 2)
        return a + b

    def test_receive_values(self):
        if False:
            while True:
                i = 10
        g = self._f3()
        self.assertEqual(g.send(None), 1)
        self.assertEqual(g.send(100), 2)
        with self.assertRaises(StopIteration) as exc:
            g.send(1000)
        self.assertEqual(exc.exception.value, 1100)

    @cinder_support.failUnlessJITCompiled
    def _f4(self, a):
        if False:
            return 10
        yield a
        yield a
        return a

    def test_one_arg(self):
        if False:
            return 10
        g = self._f4(10)
        self.assertEqual(g.send(None), 10)
        self.assertEqual(g.send(None), 10)
        with self.assertRaises(StopIteration) as exc:
            g.send(None)
        self.assertEqual(exc.exception.value, 10)

    @cinder_support.failUnlessJITCompiled
    def _f5(self, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16):
        if False:
            print('Hello World!')
        v = (yield (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16))
        a1 <<= v
        a2 <<= v
        a3 <<= v
        a4 <<= v
        a5 <<= v
        a6 <<= v
        a7 <<= v
        a8 <<= v
        a9 <<= v
        a10 <<= v
        a11 <<= v
        a12 <<= v
        a13 <<= v
        a14 <<= v
        a15 <<= v
        a16 <<= v
        v = (yield (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16))
        a1 <<= v
        a2 <<= v
        a3 <<= v
        a4 <<= v
        a5 <<= v
        a6 <<= v
        a7 <<= v
        a8 <<= v
        a9 <<= v
        a10 <<= v
        a11 <<= v
        a12 <<= v
        a13 <<= v
        a14 <<= v
        a15 <<= v
        a16 <<= v
        return a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16

    def test_save_all_registers_and_spill(self):
        if False:
            return 10
        g = self._f5(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
        self.assertEqual(g.send(None), 65535)
        self.assertEqual(g.send(1), 65535 << 1)
        with self.assertRaises(StopIteration) as exc:
            g.send(2)
        self.assertEqual(exc.exception.value, 65535 << 3)

    def test_for_loop_driven(self):
        if False:
            return 10
        l = []
        for x in self._f2():
            l.append(x)
        self.assertEqual(l, [1, 2])

    @cinder_support.failUnlessJITCompiled
    def _f6(self):
        if False:
            return 10
        i = 0
        while i < 1000:
            i = (yield i)

    def test_many_iterations(self):
        if False:
            while True:
                i = 10
        g = self._f6()
        self.assertEqual(g.send(None), 0)
        for i in range(1, 1000):
            self.assertEqual(g.send(i), i)
        with self.assertRaises(StopIteration) as exc:
            g.send(1000)
        self.assertIsNone(exc.exception.value)

    def _f_raises(self):
        if False:
            while True:
                i = 10
        raise ValueError

    @cinder_support.failUnlessJITCompiled
    def _f7(self):
        if False:
            return 10
        self._f_raises()
        yield 1

    def test_raise(self):
        if False:
            print('Hello World!')
        g = self._f7()
        with self.assertRaises(ValueError):
            g.send(None)

    def test_throw_into_initial_yield(self):
        if False:
            return 10
        g = self._f1()
        with self.assertRaises(ValueError):
            g.throw(ValueError)

    def test_throw_into_yield(self):
        if False:
            return 10
        g = self._f2()
        self.assertEqual(g.send(None), 1)
        with self.assertRaises(ValueError):
            g.throw(ValueError)

    def test_close_on_initial_yield(self):
        if False:
            i = 10
            return i + 15
        g = self._f1()
        g.close()

    def test_close_on_yield(self):
        if False:
            i = 10
            return i + 15
        g = self._f2()
        self.assertEqual(g.send(None), 1)
        g.close()

    @cinder_support.failUnlessJITCompiled
    def _f8(self, a):
        if False:
            for i in range(10):
                print('nop')
        x += (yield a)

    def test_do_not_deopt_before_initial_yield(self):
        if False:
            i = 10
            return i + 15
        g = self._f8(1)
        with self.assertRaises(UnboundLocalError):
            g.send(None)

    @cinder_support.failUnlessJITCompiled
    def _f9(self, a):
        if False:
            while True:
                i = 10
        yield
        return a

    def test_incref_args(self):
        if False:
            return 10

        class X:
            pass
        g = self._f9(X())
        g.send(None)
        with self.assertRaises(StopIteration) as exc:
            g.send(None)
        self.assertIsInstance(exc.exception.value, X)

    @cinder_support.failUnlessJITCompiled
    def _f10(self, X):
        if False:
            while True:
                i = 10
        x = X()
        yield weakref.ref(x)
        return x

    def test_gc_traversal(self):
        if False:
            while True:
                i = 10

        class X:
            pass
        g = self._f10(X)
        weak_ref_x = g.send(None)
        self.assertIn(weak_ref_x(), gc.get_objects())
        referrers = gc.get_referrers(weak_ref_x())
        self.assertEqual(len(referrers), 1)
        if cinder_support.CINDERJIT_ENABLED:
            self.assertIs(referrers[0], g)
        else:
            self.assertIs(referrers[0], g.gi_frame)
        with self.assertRaises(StopIteration):
            g.send(None)

    def test_resuming_in_another_thread(self):
        if False:
            while True:
                i = 10
        g = self._f1()

        def thread_function(g):
            if False:
                return 10
            self.assertEqual(g.send(None), 1)
            with self.assertRaises(StopIteration):
                g.send(None)
        t = threading.Thread(target=thread_function, args=(g,))
        t.start()
        t.join()

    def test_release_data_on_discard(self):
        if False:
            i = 10
            return i + 15
        o = object()
        base_count = sys.getrefcount(o)
        g = self._f9(o)
        self.assertEqual(sys.getrefcount(o), base_count + 1)
        del g
        self.assertEqual(sys.getrefcount(o), base_count)

    @cinder_support.failUnlessJITCompiled
    def _f12(self, g):
        if False:
            print('Hello World!')
        a = (yield from g)
        return a

    def test_yield_from_generator(self):
        if False:
            return 10
        g = self._f12(self._f2())
        self.assertEqual(g.send(None), 1)
        self.assertEqual(g.send(None), 2)
        with self.assertRaises(StopIteration) as exc:
            g.send(None)
        self.assertEqual(exc.exception.value, 3)

    def test_yield_from_iterator(self):
        if False:
            for i in range(10):
                print('nop')
        g = self._f12([1, 2])
        self.assertEqual(g.send(None), 1)
        self.assertEqual(g.send(None), 2)
        with self.assertRaises(StopIteration):
            g.send(None)

    def test_yield_from_forwards_raise_down(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                i = 10
                return i + 15
            try:
                yield 1
            except ValueError:
                return 2
            return 3
        g = self._f12(f())
        self.assertEqual(g.send(None), 1)
        with self.assertRaises(StopIteration) as exc:
            g.throw(ValueError)
        self.assertEqual(exc.exception.value, 2)

    def test_yield_from_forwards_raise_up(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                return 10
            raise ValueError
            yield 1
        g = self._f12(f())
        with self.assertRaises(ValueError):
            g.send(None)

    def test_yield_from_passes_raise_through(self):
        if False:
            return 10
        g = self._f12(self._f2())
        self.assertEqual(g.send(None), 1)
        with self.assertRaises(ValueError):
            g.throw(ValueError)

    def test_yield_from_forwards_close_down(self):
        if False:
            for i in range(10):
                print('nop')
        saw_close = False

        def f():
            if False:
                return 10
            nonlocal saw_close
            try:
                yield 1
            except GeneratorExit:
                saw_close = True
                return 2
        g = self._f12(f())
        self.assertEqual(g.send(None), 1)
        g.close()
        self.assertTrue(saw_close)

    def test_yield_from_passes_close_through(self):
        if False:
            print('Hello World!')
        g = self._f12(self._f2())
        self.assertEqual(g.send(None), 1)
        g.close()

    def test_assert_on_yield_from_coro(self):
        if False:
            for i in range(10):
                print('nop')

        async def coro():
            pass
        c = coro()
        with self.assertRaises(TypeError) as exc:
            self._f12(c).send(None)
        self.assertEqual(str(exc.exception), "cannot 'yield from' a coroutine object in a non-coroutine generator")
        c.close()

    def test_gen_freelist(self):
        if False:
            while True:
                i = 10
        'Exercise making a JITted generator with gen_data memory off the freelist.'
        sc = self.small_coro()
        with self.assertRaises(StopIteration):
            sc.send(None)
        del sc
        sc2 = self.small_coro()
        with self.assertRaises(StopIteration):
            sc2.send(None)
        del sc2
        bc = self.big_coro()
        with self.assertRaises(StopIteration):
            bc.send(None)
        del bc

    @cinder_support.failUnlessJITCompiled
    async def big_coro(self):
        return dict(a=dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9), b=dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9), c=dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9), d=dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9), e=dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9), f=dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9), g=dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9), h=dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9))

    @cinder_support.failUnlessJITCompiled
    async def small_coro(self):
        return 1

    def test_generator_globals(self):
        if False:
            i = 10
            return i + 15
        val1 = 'a value'
        val2 = 'another value'
        gbls = {'A_GLOBAL': val1}

        @with_globals(gbls)
        def gen():
            if False:
                print('Hello World!')
            yield A_GLOBAL
            yield A_GLOBAL
        g = gen()
        self.assertIs(g.__next__(), val1)
        gbls['A_GLOBAL'] = val2
        del gbls
        self.assertIs(g.__next__(), val2)
        with self.assertRaises(StopIteration):
            g.__next__()

    def test_deopt_at_initial_yield(self):
        if False:
            return 10

        @cinder_support.failUnlessJITCompiled
        def gen(a, b):
            if False:
                i = 10
                return i + 15
            yield a
            return a + b
        g = gen(3, 8)
        self.assertEqual(_deopt_gen(g), is_jit_compiled(gen))
        self.assertEqual(next(g), 3)
        with self.assertRaises(StopIteration) as cm:
            next(g)
        self.assertEqual(cm.exception.value, 11)

    def test_deopt_at_yield(self):
        if False:
            i = 10
            return i + 15

        @cinder_support.failUnlessJITCompiled
        def gen(a, b):
            if False:
                while True:
                    i = 10
            yield a
            return a * b
        g = gen(5, 9)
        self.assertEqual(next(g), 5)
        self.assertEqual(_deopt_gen(g), is_jit_compiled(gen))
        with self.assertRaises(StopIteration) as cm:
            next(g)
        self.assertEqual(cm.exception.value, 45)

    def test_deopt_at_yield_from(self):
        if False:
            print('Hello World!')

        @cinder_support.failUnlessJITCompiled
        def gen(l):
            if False:
                for i in range(10):
                    print('nop')
            yield from iter(l)
        g = gen([2, 4, 6])
        self.assertEqual(next(g), 2)
        self.assertEqual(_deopt_gen(g), is_jit_compiled(gen))
        self.assertEqual(next(g), 4)
        self.assertEqual(next(g), 6)
        with self.assertRaises(StopIteration) as cm:
            next(g)
        self.assertEqual(cm.exception.value, None)

    def test_deopt_at_yield_from_handle_stop_async_iteration(self):
        if False:
            i = 10
            return i + 15

        class BusyWait:

            def __await__(self):
                if False:
                    i = 10
                    return i + 15
                return iter(['one', 'two'])

        class AsyncIter:

            def __init__(self, l):
                if False:
                    while True:
                        i = 10
                self._iter = iter(l)

            async def __anext__(self):
                try:
                    item = next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration
                await BusyWait()
                return item

        class AsyncList:

            def __init__(self, l):
                if False:
                    return 10
                self._list = l

            def __aiter__(self):
                if False:
                    return 10
                return AsyncIter(self._list)

        @cinder_support.failUnlessJITCompiled
        async def coro(l1, l2):
            async for i in AsyncList(l1):
                l2.append(i * 2)
            return l2
        l = []
        c = coro([7, 8], l)
        it = iter(c.__await__())
        self.assertEqual(next(it), 'one')
        self.assertEqual(l, [])
        self.assertEqual(_deopt_gen(c), is_jit_compiled(coro))
        self.assertEqual(next(it), 'two')
        self.assertEqual(l, [])
        self.assertEqual(next(it), 'one')
        self.assertEqual(l, [14])
        self.assertEqual(next(it), 'two')
        self.assertEqual(l, [14])
        with self.assertRaises(StopIteration) as cm:
            next(it)
        self.assertIs(cm.exception.value, l)
        self.assertEqual(l, [14, 16])

class GeneratorFrameTest(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def gen1(self):
        if False:
            return 10
        a = 1
        yield a
        a = 2
        yield a

    def test_access_before_send(self):
        if False:
            for i in range(10):
                print('nop')
        g = self.gen1()
        f = g.gi_frame
        self.assertEqual(next(g), 1)
        self.assertEqual(g.gi_frame, f)
        self.assertEqual(next(g), 2)
        self.assertEqual(g.gi_frame, f)

    def test_access_after_send(self):
        if False:
            return 10
        g = self.gen1()
        self.assertEqual(next(g), 1)
        f = g.gi_frame
        self.assertEqual(next(g), 2)
        self.assertEqual(g.gi_frame, f)

    @cinder_support.failUnlessJITCompiled
    def gen2(self):
        if False:
            i = 10
            return i + 15
        me = (yield)
        f = me.gi_frame
        yield f
        yield 10

    def test_access_while_running(self):
        if False:
            for i in range(10):
                print('nop')
        g = self.gen2()
        next(g)
        f = g.send(g)
        self.assertEqual(f, g.gi_frame)
        next(g)

class CoroutinesTest(unittest.TestCase):

    def tearDown(self):
        if False:
            print('Hello World!')
        asyncio.set_event_loop_policy(None)

    @cinder_support.failUnlessJITCompiled
    async def _f1(self):
        return 1

    @cinder_support.failUnlessJITCompiled
    async def _f1(self):
        return 1

    @cinder_support.failUnlessJITCompiled
    async def _f2(self, await_target):
        return await await_target

    def test_basic_coroutine(self):
        if False:
            while True:
                i = 10
        c = self._f2(self._f1())
        with self.assertRaises(StopIteration) as exc:
            c.send(None)
        self.assertEqual(exc.exception.value, 1)

    def test_cannot_await_coro_already_awaiting_on_a_sub_iterator(self):
        if False:
            return 10

        class DummyAwaitable:

            def __await__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return iter([1])
        c = self._f2(DummyAwaitable())
        self.assertEqual(c.send(None), 1)
        with self.assertRaises(RuntimeError) as exc:
            self._f2(c).send(None)
        self.assertEqual(str(exc.exception), 'coroutine is being awaited already')

    def test_works_with_asyncio(self):
        if False:
            return 10
        asyncio.run(self._f2(asyncio.sleep(0.1)))

    @cinder_support.failUnlessJITCompiled
    @asyncio.coroutine
    def _f3(self):
        if False:
            while True:
                i = 10
        yield 1
        return 2

    def test_pre_async_coroutine(self):
        if False:
            i = 10
            return i + 15
        c = self._f3()
        self.assertEqual(c.send(None), 1)
        with self.assertRaises(StopIteration) as exc:
            c.send(None)
        self.assertEqual(exc.exception.value, 2)

    @staticmethod
    @cinder_support.failUnlessJITCompiled
    async def _use_async_with(mgr_type):
        async with mgr_type():
            pass

    def test_bad_awaitable_in_with(self):
        if False:
            return 10

        class BadAEnter:

            def __aenter__(self):
                if False:
                    return 10
                pass

            async def __aexit__(self, exc, ty, tb):
                pass

        class BadAExit:

            async def __aenter__(self):
                pass

            def __aexit__(self, exc, ty, tb):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        with self.assertRaisesRegex(TypeError, "'async with' received an object from __aenter__ that does not implement __await__: NoneType"):
            asyncio.run(self._use_async_with(BadAEnter))
        with self.assertRaisesRegex(TypeError, "'async with' received an object from __aexit__ that does not implement __await__: NoneType"):
            asyncio.run(self._use_async_with(BadAExit))

    class FakeFuture:

        def __init__(self, obj):
            if False:
                i = 10
                return i + 15
            self._obj = obj

        def __await__(self):
            if False:
                while True:
                    i = 10
            i = iter([self._obj])
            self._obj = None
            return i

    @cinder_support.skipUnlessJITEnabled('Exercises JIT-specific bug')
    def test_jit_coro_awaits_interp_coro(self):
        if False:
            print('Hello World!')

        @cinderjit.jit_suppress
        async def eager_suspend(suffix):
            await self.FakeFuture('hello, ' + suffix)

        @cinder_support.failUnlessJITCompiled
        async def jit_coro():
            await eager_suspend('bob')
        coro = jit_coro()
        v1 = coro.send(None)
        with self.assertRaises(StopIteration):
            coro.send(None)
        self.assertEqual(v1, 'hello, bob')

    def assert_already_awaited(self, coro):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'coroutine is being awaited already'):
            asyncio.run(coro)

    def test_already_awaited_coroutine_in_try_except(self):
        if False:
            while True:
                i = 10
        'Except blocks should execute when a coroutine is already awaited'

        async def f():
            await asyncio.sleep(0.1)
        executed_except_block = False

        async def runner():
            nonlocal executed_except_block
            coro = f()
            loop = asyncio.get_running_loop()
            t = loop.create_task(coro)
            try:
                await asyncio.sleep(0)
                await coro
            except RuntimeError:
                executed_except_block = True
                t.cancel()
                raise
        self.assert_already_awaited(runner())
        self.assertTrue(executed_except_block)

    def test_already_awaited_coroutine_in_try_finally(self):
        if False:
            i = 10
            return i + 15
        'Finally blocks should execute when a coroutine is already awaited'

        async def f():
            await asyncio.sleep(0.1)
        executed_finally_block = False

        async def runner():
            nonlocal executed_finally_block
            coro = f()
            loop = asyncio.get_running_loop()
            t = loop.create_task(coro)
            try:
                await asyncio.sleep(0)
                await coro
            finally:
                executed_finally_block = True
                t.cancel()
        self.assert_already_awaited(runner())
        self.assertTrue(executed_finally_block)

    def test_already_awaited_coroutine_in_try_except_finally(self):
        if False:
            return 10
        'Except and finally blocks should execute when a coroutine is already\n        awaited.\n        '

        async def f():
            await asyncio.sleep(0.1)
        executed_except_block = False
        executed_finally_block = False

        async def runner():
            nonlocal executed_except_block, executed_finally_block
            coro = f()
            loop = asyncio.get_running_loop()
            t = loop.create_task(coro)
            try:
                await asyncio.sleep(0)
                await coro
            except RuntimeError:
                executed_except_block = True
                raise
            finally:
                executed_finally_block = True
                t.cancel()
        self.assert_already_awaited(runner())
        self.assertTrue(executed_except_block)
        self.assertTrue(executed_finally_block)

class EagerCoroutineDispatch(StaticTestBase):

    def tearDown(self):
        if False:
            print('Hello World!')
        asyncio.set_event_loop_policy(None)

    def _assert_awaited_flag_seen(self, async_f_under_test):
        if False:
            print('Hello World!')
        awaited_capturer = _testcapi.TestAwaitedCall()
        self.assertIsNone(awaited_capturer.last_awaited())
        coro = async_f_under_test(awaited_capturer)
        with self.assertRaisesRegex(TypeError, ".*can't be used in 'await' expression"):
            coro.send(None)
        coro.close()
        self.assertTrue(awaited_capturer.last_awaited())
        self.assertIsNone(awaited_capturer.last_awaited())

    def _assert_awaited_flag_not_seen(self, async_f_under_test):
        if False:
            return 10
        awaited_capturer = _testcapi.TestAwaitedCall()
        self.assertIsNone(awaited_capturer.last_awaited())
        coro = async_f_under_test(awaited_capturer)
        with self.assertRaises(StopIteration):
            coro.send(None)
        coro.close()
        self.assertFalse(awaited_capturer.last_awaited())
        self.assertIsNone(awaited_capturer.last_awaited())

    @cinder_support.failUnlessJITCompiled
    async def _call_ex(self, t):
        t(*[1])

    @cinder_support.failUnlessJITCompiled
    async def _call_ex_awaited(self, t):
        await t(*[1])

    @cinder_support.failUnlessJITCompiled
    async def _call_ex_kw(self, t):
        t(*[1], **{'2': 3})

    @cinder_support.failUnlessJITCompiled
    async def _call_ex_kw_awaited(self, t):
        await t(*[1], **{'2': 3})

    @cinder_support.failUnlessJITCompiled
    async def _call_method(self, t):
        o = type('', (), {})()
        o.t = t
        o.t()

    @cinder_support.failUnlessJITCompiled
    async def _call_method_awaited(self, t):
        o = type('', (), {})()
        o.t = t
        await o.t()

    @cinder_support.failUnlessJITCompiled
    async def _vector_call(self, t):
        t()

    @cinder_support.failUnlessJITCompiled
    async def _vector_call_awaited(self, t):
        await t()

    @cinder_support.failUnlessJITCompiled
    async def _vector_call_kw(self, t):
        t(a=1)

    @cinder_support.failUnlessJITCompiled
    async def _vector_call_kw_awaited(self, t):
        await t(a=1)

    def test_call_ex(self):
        if False:
            while True:
                i = 10
        self._assert_awaited_flag_not_seen(self._call_ex)

    def test_call_ex_awaited(self):
        if False:
            return 10
        self._assert_awaited_flag_seen(self._call_ex_awaited)

    def test_call_ex_kw(self):
        if False:
            print('Hello World!')
        self._assert_awaited_flag_not_seen(self._call_ex_kw)

    def test_call_ex_kw_awaited(self):
        if False:
            print('Hello World!')
        self._assert_awaited_flag_seen(self._call_ex_kw_awaited)

    def test_call_method(self):
        if False:
            while True:
                i = 10
        self._assert_awaited_flag_not_seen(self._call_method)

    def test_call_method_awaited(self):
        if False:
            return 10
        self._assert_awaited_flag_seen(self._call_method_awaited)

    def test_vector_call(self):
        if False:
            while True:
                i = 10
        self._assert_awaited_flag_not_seen(self._vector_call)

    def test_vector_call_awaited(self):
        if False:
            print('Hello World!')
        self._assert_awaited_flag_seen(self._vector_call_awaited)

    def test_vector_call_kw(self):
        if False:
            return 10
        self._assert_awaited_flag_not_seen(self._vector_call_kw)

    def test_vector_call_kw_awaited(self):
        if False:
            while True:
                i = 10
        self._assert_awaited_flag_seen(self._vector_call_kw_awaited)

    def test_invoke_function(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = f'\n        async def x() -> None:\n            pass\n\n        async def await_x() -> None:\n            await x()\n\n        # Exercise call path through Ci_PyFunction_CallStatic\n        async def await_await_x() -> None:\n            await await_x()\n\n        async def call_x() -> None:\n            c = x()\n        '
        with self.in_module(codestr, name='test_invoke_function') as mod:
            self.assertInBytecode(mod.await_x, 'INVOKE_FUNCTION', (('test_invoke_function', 'x'), 0))
            self.assertInBytecode(mod.await_await_x, 'INVOKE_FUNCTION', (('test_invoke_function', 'await_x'), 0))
            self.assertInBytecode(mod.call_x, 'INVOKE_FUNCTION', (('test_invoke_function', 'x'), 0))
            mod.x = _testcapi.TestAwaitedCall()
            self.assertIsInstance(mod.x, _testcapi.TestAwaitedCall)
            self.assertIsNone(mod.x.last_awaited())
            coro = mod.await_await_x()
            with self.assertRaisesRegex(TypeError, ".*can't be used in 'await' expression"):
                coro.send(None)
            coro.close()
            self.assertTrue(mod.x.last_awaited())
            self.assertIsNone(mod.x.last_awaited())
            coro = mod.call_x()
            with self.assertRaises(StopIteration):
                coro.send(None)
            coro.close()
            self.assertFalse(mod.x.last_awaited())
            if cinderjit and cinderjit.auto_jit_threshold() <= 1:
                self.assertTrue(cinderjit.is_jit_compiled(mod.await_x))
                self.assertTrue(cinderjit.is_jit_compiled(mod.call_x))

    def test_invoke_method(self):
        if False:
            while True:
                i = 10
        codestr = f'\n        class X:\n            async def x(self) -> None:\n                pass\n\n        async def await_x(x: X) -> None:\n            await x.x()\n\n        async def call_x(x: X) -> None:\n            x.x()\n        '
        with self.in_module(codestr, name='test_invoke_method') as mod:
            self.assertInBytecode(mod.await_x, 'INVOKE_METHOD', (('test_invoke_method', 'X', 'x'), 0))
            self.assertInBytecode(mod.call_x, 'INVOKE_METHOD', (('test_invoke_method', 'X', 'x'), 0))
            awaited_capturer = mod.X.x = _testcapi.TestAwaitedCall()
            self.assertIsNone(awaited_capturer.last_awaited())
            coro = mod.await_x(mod.X())
            with self.assertRaisesRegex(TypeError, ".*can't be used in 'await' expression"):
                coro.send(None)
            coro.close()
            self.assertTrue(awaited_capturer.last_awaited())
            self.assertIsNone(awaited_capturer.last_awaited())
            coro = mod.call_x(mod.X())
            with self.assertRaises(StopIteration):
                coro.send(None)
            coro.close()
            self.assertFalse(awaited_capturer.last_awaited())
            if cinderjit and cinderjit.auto_jit_threshold() <= 1:
                self.assertTrue(cinderjit.is_jit_compiled(mod.await_x))
                self.assertTrue(cinderjit.is_jit_compiled(mod.call_x))

        async def y():
            await DummyAwaitable()

    def test_async_yielding(self):
        if False:
            return 10

        class DummyAwaitable:

            def __await__(self):
                if False:
                    return 10
                return iter([1, 2])
        coro = self._vector_call_awaited(DummyAwaitable)
        self.assertEqual(coro.send(None), 1)
        self.assertEqual(coro.send(None), 2)

    @cinder_support.failUnlessJITCompiled
    async def _f4(self):
        """Function must have a docstring, so None is not first constant."""
        return await self._f5(k000=1, k001=1, k002=1, k003=1, k004=1, k005=1, k006=1, k007=1, k008=1, k009=1, k010=1, k011=1, k012=1, k013=1, k014=1, k015=1, k016=1, k017=1, k018=1, k019=1, k020=1, k021=1, k022=1, k023=1, k024=1, k025=1, k026=1, k027=1, k028=1, k029=1, k030=1, k031=1, k032=1, k033=1, k034=1, k035=1, k036=1, k037=1, k038=1, k039=1, k040=1, k041=1, k042=1, k043=1, k044=1, k045=1, k046=1, k047=1, k048=1, k049=1, k050=1, k051=1, k052=1, k053=1, k054=1, k055=1, k056=1, k057=1, k058=1, k059=1, k060=1, k061=1, k062=1, k063=1, k064=1, k065=1, k066=1, k067=1, k068=1, k069=1, k070=1, k071=1, k072=1, k073=1, k074=1, k075=1, k076=1, k077=1, k078=1, k079=1, k080=1, k081=1, k082=1, k083=1, k084=1, k085=1, k086=1, k087=1, k088=1, k089=1, k090=1, k091=1, k092=1, k093=1, k094=1, k095=1, k096=1, k097=1, k098=1, k099=1, k100=1, k101=1, k102=1, k103=1, k104=1, k105=1, k106=1, k107=1, k108=1, k109=1, k110=1, k111=1, k112=1, k113=1, k114=1, k115=1, k116=1, k117=1, k118=1, k119=1, k120=1, k121=1, k122=1, k123=1, k124=1, k125=1, k126=1, k127=1, k128=1, k129=1, k130=1, k131=1, k132=1, k133=1, k134=1, k135=1, k136=1, k137=1, k138=1, k139=1, k140=1, k141=1, k142=1, k143=1, k144=1, k145=1, k146=1, k147=1, k148=1, k149=1, k150=1, k151=1, k152=1, k153=1, k154=1, k155=1, k156=1, k157=1, k158=1, k159=1, k160=1, k161=1, k162=1, k163=1, k164=1, k165=1, k166=1, k167=1, k168=1, k169=1, k170=1, k171=1, k172=1, k173=1, k174=1, k175=1, k176=1, k177=1, k178=1, k179=1, k180=1, k181=1, k182=1, k183=1, k184=1, k185=1, k186=1, k187=1, k188=1, k189=1, k190=1, k191=1, k192=1, k193=1, k194=1, k195=1, k196=1, k197=1, k198=1, k199=1, k200=1, k201=1, k202=1, k203=1, k204=1, k205=1, k206=1, k207=1, k208=1, k209=1, k210=1, k211=1, k212=1, k213=1, k214=1, k215=1, k216=1, k217=1, k218=1, k219=1, k220=1, k221=1, k222=1, k223=1, k224=1, k225=1, k226=1, k227=1, k228=1, k229=1, k230=1, k231=1, k232=1, k233=1, k234=1, k235=1, k236=1, k237=1, k238=1, k239=1, k240=1, k241=1, k242=1, k243=1, k244=1, k245=1, k246=1, k247=1, k248=1, k249=1, k250=1, k251=1, k252=1, k253=1, k254=1)

    async def _f5(self, **kw):
        return kw

    def test_awaited_call_extended_arg(self):
        if False:
            print('Hello World!')
        instrs = dis.get_instructions(self._f4)
        expected_instrs = ['CALL_FUNCTION_EX', 'GET_AWAITABLE', 'EXTENDED_ARG', 'LOAD_CONST', 'YIELD_FROM', 'RETURN_VALUE']
        self.assertEqual([i.opname for i in list(instrs)[-6:]], expected_instrs)
        self.assertEqual(len(asyncio.run(self._f4())), 255)

class AsyncGeneratorsTest(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    async def _f1(self, awaitable):
        x = (yield 1)
        yield x
        await awaitable

    def test_basic_coroutine(self):
        if False:
            for i in range(10):
                print('nop')

        class DummyAwaitable:

            def __await__(self):
                if False:
                    return 10
                return iter([3])
        async_gen = self._f1(DummyAwaitable())
        async_itt1 = async_gen.asend(None)
        with self.assertRaises(StopIteration) as exc:
            async_itt1.send(None)
        self.assertEqual(exc.exception.value, 1)
        async_itt2 = async_gen.asend(2)
        with self.assertRaises(StopIteration) as exc:
            async_itt2.send(None)
        self.assertEqual(exc.exception.value, 2)
        async_itt3 = async_gen.asend(None)
        self.assertEqual(async_itt3.send(None), 3)
        with self.assertRaises(StopAsyncIteration):
            async_itt3.send(None)

    @cinder_support.failUnlessJITCompiled
    async def _f2(self, asyncgen):
        res = []
        async for x in asyncgen:
            res.append(x)
        return res

    def test_for_iteration(self):
        if False:
            print('Hello World!')

        async def asyncgen():
            yield 1
            yield 2
        self.assertEqual(asyncio.run(self._f2(asyncgen())), [1, 2])

    def _assertExceptionFlowsThroughYieldFrom(self, exc):
        if False:
            return 10
        tb_prev = None
        tb = exc.__traceback__
        while tb.tb_next:
            tb_prev = tb
            tb = tb.tb_next
        instrs = [x for x in dis.get_instructions(tb_prev.tb_frame.f_code)]
        self.assertEqual(instrs[tb_prev.tb_lasti // 2].opname, 'YIELD_FROM')

    def test_for_exception(self):
        if False:
            return 10

        async def asyncgen():
            yield 1
            raise ValueError
        try:
            asyncio.run(self._f2(asyncgen()))
        except ValueError as e:
            self._assertExceptionFlowsThroughYieldFrom(e)
        else:
            self.fail('Expected ValueError to be raised')

    @cinder_support.failUnlessJITCompiled
    async def _f3(self, asyncgen):
        return [x async for x in asyncgen]

    def test_comprehension(self):
        if False:
            while True:
                i = 10

        async def asyncgen():
            yield 1
            yield 2
        self.assertEqual(asyncio.run(self._f3(asyncgen())), [1, 2])

    def test_comprehension_exception(self):
        if False:
            return 10

        async def asyncgen():
            yield 1
            raise ValueError
        try:
            asyncio.run(self._f3(asyncgen()))
        except ValueError as e:
            self._assertExceptionFlowsThroughYieldFrom(e)
        else:
            self.fail('Expected ValueError to be raised')

class Err1(Exception):
    pass

class Err2(Exception):
    pass

class ExceptionHandlingTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def try_except(self, func):
        if False:
            for i in range(10):
                print('nop')
        try:
            func()
        except:
            return True
        return False

    def test_raise_and_catch(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                while True:
                    i = 10
            raise Exception('hello')
        self.assertTrue(self.try_except(f))

        def g():
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.assertFalse(self.try_except(g))

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY', 'JUMP_IF_NOT_EXC_MATCH')
    def catch_multiple(self, func):
        if False:
            i = 10
            return i + 15
        try:
            func()
        except Err1:
            return 1
        except Err2:
            return 2

    def test_multiple_except_blocks(self):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                return 10
            raise Err1('err1')
        self.assertEqual(self.catch_multiple(f), 1)

        def g():
            if False:
                while True:
                    i = 10
            raise Err2('err2')
        self.assertEqual(self.catch_multiple(g), 2)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def reraise(self, func):
        if False:
            for i in range(10):
                print('nop')
        try:
            func()
        except:
            raise

    def test_reraise(self):
        if False:
            print('Hello World!')

        def f():
            if False:
                while True:
                    i = 10
            raise Exception('hello')
        with self.assertRaisesRegex(Exception, 'hello'):
            self.reraise(f)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def try_except_in_loop(self, niters, f):
        if False:
            print('Hello World!')
        for i in range(niters):
            try:
                try:
                    f(i)
                except Err2:
                    pass
            except Err1:
                break
        return i

    def test_try_except_in_loop(self):
        if False:
            print('Hello World!')

        def f(i):
            if False:
                while True:
                    i = 10
            if i == 10:
                raise Err1('hello')
        self.assertEqual(self.try_except_in_loop(20, f), 10)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def nested_try_except(self, f):
        if False:
            for i in range(10):
                print('nop')
        try:
            try:
                try:
                    f()
                except:
                    raise
            except:
                raise
        except:
            return 100

    def test_nested_try_except(self):
        if False:
            return 10

        def f():
            if False:
                return 10
            raise Exception('hello')
        self.assertEqual(self.nested_try_except(f), 100)

    @cinder_support.failUnlessJITCompiled
    def try_except_in_generator(self, f):
        if False:
            for i in range(10):
                print('nop')
        try:
            yield f(0)
            yield f(1)
            yield f(2)
        except:
            yield 123

    def test_except_in_generator(self):
        if False:
            print('Hello World!')

        def f(i):
            if False:
                for i in range(10):
                    print('nop')
            if i == 1:
                raise Exception('hello')
            return
        g = self.try_except_in_generator(f)
        next(g)
        self.assertEqual(next(g), 123)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY', 'RERAISE')
    def try_finally(self, should_raise):
        if False:
            while True:
                i = 10
        result = None
        try:
            if should_raise:
                raise Exception('testing 123')
        finally:
            result = 100
        return result

    def test_try_finally(self):
        if False:
            return 10
        self.assertEqual(self.try_finally(False), 100)
        with self.assertRaisesRegex(Exception, 'testing 123'):
            self.try_finally(True)

    @cinder_support.failUnlessJITCompiled
    def try_except_finally(self, should_raise):
        if False:
            i = 10
            return i + 15
        result = None
        try:
            if should_raise:
                raise Exception('testing 123')
        except Exception:
            result = 200
        finally:
            if result is None:
                result = 100
        return result

    def test_try_except_finally(self):
        if False:
            return 10
        self.assertEqual(self.try_except_finally(False), 100)
        self.assertEqual(self.try_except_finally(True), 200)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def return_in_finally(self, v):
        if False:
            i = 10
            return i + 15
        try:
            pass
        finally:
            return v

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def return_in_finally2(self, v):
        if False:
            print('Hello World!')
        try:
            return v
        finally:
            return 100

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def return_in_finally3(self, v):
        if False:
            print('Hello World!')
        try:
            1 / 0
        finally:
            return v

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def return_in_finally4(self, v):
        if False:
            print('Hello World!')
        try:
            return 100
        finally:
            try:
                1 / 0
            finally:
                return v

    def test_return_in_finally(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.return_in_finally(100), 100)
        self.assertEqual(self.return_in_finally2(200), 100)
        self.assertEqual(self.return_in_finally3(300), 300)
        self.assertEqual(self.return_in_finally4(400), 400)

    @cinder_support.failUnlessJITCompiled
    def break_in_finally_after_return(self, x):
        if False:
            return 10
        for count in [0, 1]:
            count2 = 0
            while count2 < 20:
                count2 += 10
                try:
                    return count + count2
                finally:
                    if x:
                        break
        return ('end', count, count2)

    @cinder_support.failUnlessJITCompiled
    def break_in_finally_after_return2(self, x):
        if False:
            return 10
        for count in [0, 1]:
            for count2 in [10, 20]:
                try:
                    return count + count2
                finally:
                    if x:
                        break
        return ('end', count, count2)

    def test_break_in_finally_after_return(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.break_in_finally_after_return(False), 10)
        self.assertEqual(self.break_in_finally_after_return(True), ('end', 1, 10))
        self.assertEqual(self.break_in_finally_after_return2(False), 10)
        self.assertEqual(self.break_in_finally_after_return2(True), ('end', 1, 10))

    @cinder_support.failUnlessJITCompiled
    def continue_in_finally_after_return(self, x):
        if False:
            return 10
        count = 0
        while count < 100:
            count += 1
            try:
                return count
            finally:
                if x:
                    continue
        return ('end', count)

    @cinder_support.failUnlessJITCompiled
    def continue_in_finally_after_return2(self, x):
        if False:
            for i in range(10):
                print('nop')
        for count in [0, 1]:
            try:
                return count
            finally:
                if x:
                    continue
        return ('end', count)

    def test_continue_in_finally_after_return(self):
        if False:
            return 10
        self.assertEqual(self.continue_in_finally_after_return(False), 1)
        self.assertEqual(self.continue_in_finally_after_return(True), ('end', 100))
        self.assertEqual(self.continue_in_finally_after_return2(False), 0)
        self.assertEqual(self.continue_in_finally_after_return2(True), ('end', 1))

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def return_in_loop_in_finally(self, x):
        if False:
            return 10
        try:
            for _ in [1, 2, 3]:
                if x:
                    return x
        finally:
            pass
        return 100

    def test_return_in_loop_in_finally(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.return_in_loop_in_finally(True), True)
        self.assertEqual(self.return_in_loop_in_finally(False), 100)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def conditional_return_in_finally(self, x, y, z):
        if False:
            while True:
                i = 10
        try:
            if x:
                return x
            if y:
                return y
        finally:
            pass
        return z

    def test_conditional_return_in_finally(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.conditional_return_in_finally(100, False, False), 100)
        self.assertEqual(self.conditional_return_in_finally(False, 200, False), 200)
        self.assertEqual(self.conditional_return_in_finally(False, False, 300), 300)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_FINALLY')
    def nested_finally(self, x):
        if False:
            print('Hello World!')
        try:
            if x:
                return x
        finally:
            try:
                y = 10
            finally:
                z = y
        return z

    def test_nested_finally(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.nested_finally(100), 100)
        self.assertEqual(self.nested_finally(False), 10)

class SpecializeCCallTests(unittest.TestCase):
    """
    The JIT performs direct calls of C functions with CallStatic when possible.
    """

    @cinder_support.failUnlessJITCompiled
    def _c_func_that_sets_pyerr(self):
        if False:
            while True:
                i = 10
        s = 'abc'
        return s.removeprefix(1)

    def test_c_call_error_raised(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            self._c_func_that_sets_pyerr()

class UnpackSequenceTests(unittest.TestCase):

    @failUnlessHasOpcodes('UNPACK_SEQUENCE')
    @cinder_support.failUnlessJITCompiled
    def _unpack_arg(self, seq, which):
        if False:
            while True:
                i = 10
        (a, b, c, d) = seq
        if which == 'a':
            return a
        if which == 'b':
            return b
        if which == 'c':
            return c
        return d

    @failUnlessHasOpcodes('UNPACK_EX')
    @cinder_support.failUnlessJITCompiled
    def _unpack_ex_arg(self, seq, which):
        if False:
            return 10
        (a, b, *c, d) = seq
        if which == 'a':
            return a
        if which == 'b':
            return b
        if which == 'c':
            return c
        return d

    def test_unpack_tuple(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self._unpack_arg(('eh', 'bee', 'see', 'dee'), 'b'), 'bee')
        self.assertEqual(self._unpack_arg((3, 2, 1, 0), 'c'), 1)

    def test_unpack_tuple_wrong_size(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            self._unpack_arg((1, 2, 3, 4, 5), 'a')

    def test_unpack_list(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self._unpack_arg(['one', 'two', 'three', 'four'], 'a'), 'one')

    def test_unpack_gen(self):
        if False:
            return 10

        def gen():
            if False:
                while True:
                    i = 10
            yield 'first'
            yield 'second'
            yield 'third'
            yield 'fourth'
        self.assertEqual(self._unpack_arg(gen(), 'd'), 'fourth')

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('UNPACK_EX')
    def _unpack_not_iterable(self):
        if False:
            return 10
        (a, b, *c) = 1

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('UNPACK_EX')
    def _unpack_insufficient_values(self):
        if False:
            return 10
        (a, b, *c) = [1]

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('UNPACK_EX')
    def _unpack_insufficient_values_after(self):
        if False:
            while True:
                i = 10
        (a, *b, c, d) = [1, 2]

    def test_unpack_ex(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            self._unpack_not_iterable()
        with self.assertRaises(ValueError):
            self._unpack_insufficient_values()
        with self.assertRaises(ValueError):
            self._unpack_insufficient_values_after()
        seq = [1, 2, 3, 4, 5, 6]
        self.assertEqual(self._unpack_ex_arg(seq, 'a'), 1)
        self.assertEqual(self._unpack_ex_arg(seq, 'b'), 2)
        self.assertEqual(self._unpack_ex_arg(seq, 'c'), [3, 4, 5])
        self.assertEqual(self._unpack_ex_arg(seq, 'd'), 6)

    def test_unpack_sequence_with_iterable(self):
        if False:
            return 10

        class C:

            def __init__(self, value):
                if False:
                    while True:
                        i = 10
                self.value = value

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                return iter(self.value)
        seq = (1, 2, 3, 4)
        self.assertEqual(self._unpack_arg(C(seq), 'a'), 1)
        self.assertEqual(self._unpack_arg(C(seq), 'b'), 2)
        self.assertEqual(self._unpack_arg(C(seq), 'c'), 3)
        self.assertEqual(self._unpack_arg(C(seq), 'd'), 4)
        with self.assertRaisesRegex(ValueError, 'not enough values to unpack'):
            self._unpack_arg(C(()), 'a')

    def test_unpack_ex_with_iterable(self):
        if False:
            while True:
                i = 10

        class C:

            def __init__(self, value):
                if False:
                    while True:
                        i = 10
                self.value = value

            def __iter__(self):
                if False:
                    return 10
                return iter(self.value)
        seq = (1, 2, 3, 4, 5, 6)
        self.assertEqual(self._unpack_ex_arg(C(seq), 'a'), 1)
        self.assertEqual(self._unpack_ex_arg(C(seq), 'b'), 2)
        self.assertEqual(self._unpack_ex_arg(C(seq), 'c'), [3, 4, 5])
        self.assertEqual(self._unpack_ex_arg(C(seq), 'd'), 6)
        with self.assertRaisesRegex(ValueError, 'not enough values to unpack'):
            self._unpack_ex_arg(C(()), 'a')

class DeleteSubscrTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('DELETE_SUBSCR')
    def _delit(self, container, key):
        if False:
            for i in range(10):
                print('nop')
        del container[key]

    def test_builtin_types(self):
        if False:
            while True:
                i = 10
        l = [1, 2, 3]
        self._delit(l, 1)
        self.assertEqual(l, [1, 3])
        d = {'foo': 1, 'bar': 2}
        self._delit(d, 'foo')
        self.assertEqual(d, {'bar': 2})

    def test_custom_type(self):
        if False:
            return 10

        class CustomContainer:

            def __init__(self):
                if False:
                    return 10
                self.item = None

            def __delitem__(self, item):
                if False:
                    return 10
                self.item = item
        c = CustomContainer()
        self._delit(c, 'foo')
        self.assertEqual(c.item, 'foo')

    def test_missing_key(self):
        if False:
            i = 10
            return i + 15
        d = {'foo': 1}
        with self.assertRaises(KeyError):
            self._delit(d, 'bar')

    def test_custom_error(self):
        if False:
            return 10

        class CustomContainer:

            def __delitem__(self, item):
                if False:
                    print('Hello World!')
                raise Exception('testing 123')
        c = CustomContainer()
        with self.assertRaisesRegex(Exception, 'testing 123'):
            self._delit(c, 'foo')

class DeleteFastTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('DELETE_FAST')
    def _del(self):
        if False:
            i = 10
            return i + 15
        x = 2
        del x

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('DELETE_FAST')
    def _del_arg(self, a):
        if False:
            print('Hello World!')
        del a

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('DELETE_FAST')
    def _del_and_raise(self):
        if False:
            i = 10
            return i + 15
        x = 2
        del x
        return x

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('DELETE_FAST')
    def _del_arg_and_raise(self, a):
        if False:
            for i in range(10):
                print('nop')
        del a
        return a

    @failUnlessHasOpcodes('DELETE_FAST')
    @cinder_support.failUnlessJITCompiled
    def _del_ex_no_raise(self):
        if False:
            i = 10
            return i + 15
        try:
            return min(1, 2)
        except Exception as e:
            pass

    @failUnlessHasOpcodes('DELETE_FAST')
    @cinder_support.failUnlessJITCompiled
    def _del_ex_raise(self):
        if False:
            return 10
        try:
            raise Exception()
        except Exception as e:
            pass
        return e

    def test_del_local(self):
        if False:
            print('Hello World!')
        self.assertEqual(self._del(), None)

    def test_del_arg(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self._del_arg(42), None)

    def test_del_and_raise(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(NameError):
            self._del_and_raise()

    def test_del_arg_and_raise(self):
        if False:
            print('Hello World!')
        with self.assertRaises(NameError):
            self.assertEqual(self._del_arg_and_raise(42), None)

    def test_del_ex_no_raise(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self._del_ex_no_raise(), 1)

    def test_del_ex_raise(self):
        if False:
            return 10
        with self.assertRaises(NameError):
            self.assertEqual(self._del_ex_raise(), 42)

class DictSubscrTests(unittest.TestCase):

    def test_int_custom_class(self):
        if False:
            return 10

        class C:

            def __init__(self, value):
                if False:
                    return 10
                self.value = value

            def __eq__(self, other):
                if False:
                    while True:
                        i = 10
                raise RuntimeError('no way!!')

            def __hash__(self):
                if False:
                    while True:
                        i = 10
                return hash(self.value)
        c = C(333)
        d = {}
        d[c] = 1
        with self.assertRaises(RuntimeError):
            d[333]

    def test_unicode_custom_class(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __init__(self, value):
                if False:
                    while True:
                        i = 10
                self.value = value

            def __eq__(self, other):
                if False:
                    return 10
                raise RuntimeError('no way!!')

            def __hash__(self):
                if False:
                    print('Hello World!')
                return hash(self.value)
        c = C('x')
        d = {}
        d[c] = 1
        with self.assertRaises(RuntimeError):
            d['x']

class KeywordOnlyArgTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def f1(self, *, val=10):
        if False:
            return 10
        return val

    @cinder_support.failUnlessJITCompiled
    def f2(self, which, *, y=10, z=20):
        if False:
            while True:
                i = 10
        if which == 0:
            return y
        elif which == 1:
            return z
        return which

    @cinder_support.failUnlessJITCompiled
    def f3(self, which, *, y, z=20):
        if False:
            for i in range(10):
                print('nop')
        if which == 0:
            return y
        elif which == 1:
            return z
        return which

    @cinder_support.failUnlessJITCompiled
    def f4(self, which, *, y, z=20, **kwargs):
        if False:
            while True:
                i = 10
        if which == 0:
            return y
        elif which == 1:
            return z
        elif which == 2:
            return kwargs
        return which

    def test_kwonly_arg_passed_as_positional(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'takes 1 positional argument but 2 were given'
        with self.assertRaisesRegex(TypeError, msg):
            self.f1(100)
        msg = 'takes 2 positional arguments but 3 were given'
        with self.assertRaisesRegex(TypeError, msg):
            self.f3(0, 1)

    def test_kwonly_args_with_kwdefaults(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.f1(), 10)
        self.assertEqual(self.f1(val=20), 20)
        self.assertEqual(self.f2(0), 10)
        self.assertEqual(self.f2(0, y=20), 20)
        self.assertEqual(self.f2(1), 20)
        self.assertEqual(self.f2(1, z=30), 30)

    def test_kwonly_args_without_kwdefaults(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.f3(0, y=10), 10)
        self.assertEqual(self.f3(1, y=10), 20)
        self.assertEqual(self.f3(1, y=10, z=30), 30)

    def test_kwonly_args_and_varkwargs(self):
        if False:
            return 10
        self.assertEqual(self.f4(0, y=10), 10)
        self.assertEqual(self.f4(1, y=10), 20)
        self.assertEqual(self.f4(1, y=10, z=30, a=40), 30)
        self.assertEqual(self.f4(2, y=10, z=30, a=40, b=50), {'a': 40, 'b': 50})

class ClassA:
    z = 100
    x = 41

    def g(self, a):
        if False:
            print('Hello World!')
        return 42 + a

    @classmethod
    def cls_g(cls, a):
        if False:
            print('Hello World!')
        return 100 + a

class ClassB(ClassA):

    def f(self, a):
        if False:
            i = 10
            return i + 15
        return super().g(a=a)

    def f_2arg(self, a):
        if False:
            for i in range(10):
                print('nop')
        return super(ClassB, self).g(a=a)

    @classmethod
    def cls_f(cls, a):
        if False:
            i = 10
            return i + 15
        return super().cls_g(a=a)

    @classmethod
    def cls_f_2arg(cls, a):
        if False:
            print('Hello World!')
        return super(ClassB, cls).cls_g(a=a)

    @property
    def x(self):
        if False:
            for i in range(10):
                print('nop')
        return super().x + 1

    @property
    def x_2arg(self):
        if False:
            while True:
                i = 10
        return super(ClassB, self).x + 1

class SuperAccessTest(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def test_super_method(self):
        if False:
            print('Hello World!')
        self.assertEqual(ClassB().f(1), 43)
        self.assertEqual(ClassB().f_2arg(1), 43)
        self.assertEqual(ClassB.cls_f(99), 199)
        self.assertEqual(ClassB.cls_f_2arg(99), 199)

    @cinder_support.failUnlessJITCompiled
    def test_super_method_kwarg(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(ClassB().f(1), 43)
        self.assertEqual(ClassB().f_2arg(1), 43)
        self.assertEqual(ClassB.cls_f(1), 101)
        self.assertEqual(ClassB.cls_f_2arg(1), 101)

    @cinder_support.failUnlessJITCompiled
    def test_super_attr(self):
        if False:
            return 10
        self.assertEqual(ClassB().x, 42)
        self.assertEqual(ClassB().x_2arg, 42)

class RegressionTests(StaticTestBase):

    def test_store_of_64bit_immediates(self):
        if False:
            print('Hello World!')
        codestr = f'\n            from __static__ import int64, box\n            class Cint64:\n                def __init__(self):\n                    self.a: int64 = 0x5555555555555555\n\n            def testfunc():\n                c = Cint64()\n                c.a = 2\n                return box(c.a) == 2\n        '
        with self.in_module(codestr) as mod:
            testfunc = mod.testfunc
            self.assertTrue(testfunc())
            if cinderjit and cinderjit.auto_jit_threshold() <= 1:
                self.assertTrue(cinderjit.is_jit_compiled(testfunc))

@cinder_support.skipUnlessJITEnabled('Requires cinderjit module')
class CinderJitModuleTests(StaticTestBase):

    def test_bad_disable(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TypeError):
            cinderjit.disable(1, 2)
        with self.assertRaises(TypeError):
            cinderjit.disable(None)

    def test_jit_suppress(self):
        if False:
            print('Hello World!')

        @cinderjit.jit_suppress
        def x():
            if False:
                print('Hello World!')
            pass
        self.assertEqual(x.__code__.co_flags & CO_SUPPRESS_JIT, CO_SUPPRESS_JIT)

    def test_jit_suppress_static(self):
        if False:
            i = 10
            return i + 15
        codestr = f'\n            import cinderjit\n\n            @cinderjit.jit_suppress\n            def f():\n                return True\n\n            def g():\n                return True\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            g = mod.g
            self.assertTrue(f())
            self.assertTrue(g())
            self.assertFalse(cinderjit.is_jit_compiled(f))
            if cinderjit.auto_jit_threshold() <= 1:
                self.assertTrue(cinderjit.is_jit_compiled(g))

    @unittest.skipIf(not cinderjit or not cinderjit.is_hir_inliner_enabled(), 'meaningless without HIR inliner enabled')
    def test_num_inlined_functions(self):
        if False:
            print('Hello World!')
        codestr = f'\n            import cinderjit\n\n            @cinderjit.jit_suppress\n            def f():\n                return True\n\n            def g():\n                return f()\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            g = mod.g
            self.assertTrue(g())
            self.assertFalse(cinderjit.is_jit_compiled(f))
            if cinderjit.auto_jit_threshold() <= 1:
                self.assertTrue(cinderjit.is_jit_compiled(g))
            self.assertEqual(cinderjit.get_num_inlined_functions(g), 1)

@cinder_support.failUnlessJITCompiled
def _outer(inner):
    if False:
        for i in range(10):
            print('nop')
    return inner()

class GetFrameInFinalizer:

    def __del__(self):
        if False:
            print('Hello World!')
        sys._getframe()

def _create_getframe_cycle():
    if False:
        while True:
            i = 10
    a = {'fg': GetFrameInFinalizer()}
    b = {'a': a}
    a['b'] = b
    return a

class TestException(Exception):
    pass

class GetFrameTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    def f1(self, leaf):
        if False:
            print('Hello World!')
        return self.f2(leaf)

    @cinder_support.failUnlessJITCompiled
    def f2(self, leaf):
        if False:
            for i in range(10):
                print('nop')
        return self.f3(leaf)

    @cinder_support.failUnlessJITCompiled
    def f3(self, leaf):
        if False:
            i = 10
            return i + 15
        return leaf()

    def assert_frames(self, frame, names):
        if False:
            for i in range(10):
                print('nop')
        for name in names:
            self.assertEqual(frame.f_code.co_name, name)
            frame = frame.f_back

    @cinder_support.failUnlessJITCompiled
    def simple_getframe(self):
        if False:
            i = 10
            return i + 15
        return sys._getframe()

    def test_simple_getframe(self):
        if False:
            for i in range(10):
                print('nop')
        stack = ['simple_getframe', 'f3', 'f2', 'f1', 'test_simple_getframe']
        frame = self.f1(self.simple_getframe)
        self.assert_frames(frame, stack)

    @cinder_support.failUnlessJITCompiled
    def consecutive_getframe(self):
        if False:
            for i in range(10):
                print('nop')
        f1 = sys._getframe()
        f2 = sys._getframe()
        return (f1, f2)

    @cinder_support.failUnlessJITCompiled
    def test_consecutive_getframe(self):
        if False:
            for i in range(10):
                print('nop')
        stack = ['consecutive_getframe', 'f3', 'f2', 'f1', 'test_consecutive_getframe']
        (frame1, frame2) = self.f1(self.consecutive_getframe)
        self.assert_frames(frame1, stack)
        for _ in range(4):
            self.assertTrue(frame1 is frame2)
            frame1 = frame1.f_back
            frame2 = frame2.f_back

    @cinder_support.failUnlessJITCompiled
    def getframe_then_deopt(self):
        if False:
            return 10
        f = sys._getframe()
        try:
            raise Exception('testing 123')
        except:
            return f

    def test_getframe_then_deopt(self):
        if False:
            for i in range(10):
                print('nop')
        stack = ['getframe_then_deopt', 'f3', 'f2', 'f1', 'test_getframe_then_deopt']
        frame = self.f1(self.getframe_then_deopt)
        self.assert_frames(frame, stack)

    @cinder_support.failUnlessJITCompiled
    def getframe_in_except(self):
        if False:
            while True:
                i = 10
        try:
            raise Exception('testing 123')
        except:
            return sys._getframe()

    def test_getframe_after_deopt(self):
        if False:
            return 10
        stack = ['getframe_in_except', 'f3', 'f2', 'f1', 'test_getframe_after_deopt']
        frame = self.f1(self.getframe_in_except)
        self.assert_frames(frame, stack)

    class FrameGetter:

        def __init__(self, box):
            if False:
                while True:
                    i = 10
            self.box = box

        def __del__(self):
            if False:
                return 10
            self.box[0] = sys._getframe()

    def do_raise(self, x):
        if False:
            i = 10
            return i + 15
        del x
        raise Exception('testing 123')

    @cinder_support.failUnlessJITCompiled
    def getframe_in_dtor_during_deopt(self):
        if False:
            i = 10
            return i + 15
        ref = ['notaframe']
        try:
            self.do_raise(self.FrameGetter(ref))
        except:
            return ref[0]

    def test_getframe_in_dtor_during_deopt(self):
        if False:
            i = 10
            return i + 15
        frame = self.f1(self.getframe_in_dtor_during_deopt)
        stack = ['__del__', 'getframe_in_dtor_during_deopt', 'f3', 'f2', 'f1', 'test_getframe_in_dtor_during_deopt']
        self.assert_frames(frame, stack)

    @cinder_support.failUnlessJITCompiled
    def getframe_in_dtor_after_deopt(self):
        if False:
            for i in range(10):
                print('nop')
        ref = ['notaframe']
        frame_getter = self.FrameGetter(ref)
        try:
            raise Exception('testing 123')
        except:
            return ref

    def test_getframe_in_dtor_after_deopt(self):
        if False:
            while True:
                i = 10
        frame = self.f1(self.getframe_in_dtor_after_deopt)[0]
        stack = ['__del__', 'f3', 'f2', 'f1', 'test_getframe_in_dtor_after_deopt']
        self.assert_frames(frame, stack)

    @jit_suppress
    def test_frame_allocation_race(self):
        if False:
            for i in range(10):
                print('nop')
        thresholds = gc.get_threshold()

        @jit_suppress
        def inner():
            if False:
                while True:
                    i = 10
            w = 1
            x = 2
            y = 3
            z = 4

            def f():
                if False:
                    return 10
                return w + x + y + z
            return 100
        _outer(inner)
        gc.collect()
        _create_getframe_cycle()
        try:
            gc.set_threshold(1)
            _outer(inner)
        finally:
            gc.set_threshold(*thresholds)

class GetGenFrameDuringThrowTest(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop

    def tearDown(self):
        if False:
            print('Hello World!')
        self.loop.close()
        asyncio.set_event_loop_policy(None)

    @cinder_support.failUnlessJITCompiled
    async def outer_propagates_exc(self, inner):
        return await inner

    @cinder_support.failUnlessJITCompiled
    async def outer_handles_exc(self, inner):
        try:
            await inner
        except TestException:
            return 123

    async def inner(self, fut, outer_box):
        try:
            await fut
        except TestException:
            outer_coro = outer_box[0]
            outer_coro.cr_frame
            raise

    def run_test(self, outer_func):
        if False:
            print('Hello World!')
        box = [None]
        fut = asyncio.Future()
        inner = self.inner(fut, box)
        outer = outer_func(inner)
        box[0] = outer
        outer.send(None)
        return outer.throw(TestException())

    def test_unhandled_exc(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(TestException):
            self.run_test(self.outer_propagates_exc)

    def test_handled_exc(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(StopIteration) as cm:
            self.run_test(self.outer_handles_exc)
        self.assertEqual(cm.exception.value, 123)

class DeleteAttrTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('DELETE_ATTR')
    def del_foo(self, obj):
        if False:
            for i in range(10):
                print('nop')
        del obj.foo

    def test_delete_attr(self):
        if False:
            i = 10
            return i + 15

        class C:
            pass
        c = C()
        c.foo = 'bar'
        self.assertEqual(c.foo, 'bar')
        self.del_foo(c)
        with self.assertRaises(AttributeError):
            c.foo

    def test_delete_attr_raises(self):
        if False:
            i = 10
            return i + 15

        class C:

            @property
            def foo(self):
                if False:
                    return 10
                return 'hi'
        c = C()
        self.assertEqual(c.foo, 'hi')
        with self.assertRaises(AttributeError):
            self.del_foo(c)

class OtherTests(unittest.TestCase):

    @unittest.skipIf(not cinderjit, 'meaningless without JIT enabled')
    def test_mlock_profiler_dependencies(self):
        if False:
            for i in range(10):
                print('nop')
        cinderjit.mlock_profiler_dependencies()

    @unittest.skipIf(cinderjit is None, 'not jitting')
    def test_page_in_profiler_dependencies(self):
        if False:
            while True:
                i = 10
        qualnames = cinderjit.page_in_profiler_dependencies()
        self.assertTrue(len(qualnames) > 0)

class GetIterForIterTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('FOR_ITER', 'GET_ITER')
    def doit(self, iterable):
        if False:
            print('Hello World!')
        for _ in iterable:
            pass
        return 42

    def test_iterate_through_builtin(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.doit([1, 2, 3]), 42)

    def test_custom_iterable(self):
        if False:
            print('Hello World!')

        class MyIterable:

            def __init__(self, limit):
                if False:
                    i = 10
                    return i + 15
                self.idx = 0
                self.limit = limit

            def __iter__(self):
                if False:
                    print('Hello World!')
                return self

            def __next__(self):
                if False:
                    while True:
                        i = 10
                if self.idx == self.limit:
                    raise StopIteration
                retval = self.idx
                self.idx += 1
                return retval
        it = MyIterable(5)
        self.assertEqual(self.doit(it), 42)
        self.assertEqual(it.idx, it.limit)

    def test_iteration_raises_error(self):
        if False:
            print('Hello World!')

        class MyException(Exception):
            pass

        class MyIterable:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.idx = 0

            def __iter__(self):
                if False:
                    i = 10
                    return i + 15
                return self

            def __next__(self):
                if False:
                    print('Hello World!')
                if self.idx == 3:
                    raise MyException(f'raised error on idx {self.idx}')
                self.idx += 1
                return 1
        with self.assertRaisesRegex(MyException, 'raised error on idx 3'):
            self.doit(MyIterable())

    def test_iterate_generator(self):
        if False:
            return 10
        x = None

        def gen():
            if False:
                return 10
            nonlocal x
            yield 1
            yield 2
            yield 3
            x = 42
        self.doit(gen())
        self.assertEqual(x, 42)

class SetUpdateTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('BUILD_SET', 'SET_UPDATE')
    def doit_unchecked(self, iterable):
        if False:
            i = 10
            return i + 15
        return {*iterable}

    def doit(self, iterable):
        if False:
            return 10
        result = self.doit_unchecked(iterable)
        self.assertIs(type(result), set)
        return result

    def test_iterate_non_iterable_raises_type_error(self):
        if False:
            return 10
        with self.assertRaisesRegex(TypeError, "'int' object is not iterable"):
            self.doit(42)

    def test_iterate_set_builds_set(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.doit({1, 2, 3}), {1, 2, 3})

    def test_iterate_dict_builds_set(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.doit({'hello': 'world', 'goodbye': 'world'}), {'hello', 'goodbye'})

    def test_iterate_getitem_iterable_builds_set(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __getitem__(self, index):
                if False:
                    i = 10
                    return i + 15
                if index < 4:
                    return index
                raise IndexError
        self.assertEqual(self.doit(C()), {0, 1, 2, 3})

    def test_iterate_iter_iterable_builds_set(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                return iter([1, 2, 3])
        self.assertEqual(self.doit(C()), {1, 2, 3})

class UnpackSequenceTestsWithoutCompare(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('UNPACK_SEQUENCE')
    def doit(self, iterable):
        if False:
            print('Hello World!')
        (x, y) = iterable
        return x

    def test_unpack_sequence_with_tuple(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.doit((1, 2)), 1)
        with self.assertRaisesRegex(ValueError, 'not enough values to unpack'):
            self.doit(())

    def test_unpack_sequence_with_list(self):
        if False:
            return 10
        self.assertEqual(self.doit([1, 2]), 1)
        with self.assertRaisesRegex(ValueError, 'not enough values to unpack'):
            self.doit([])

    def test_unpack_sequence_with_iterable(self):
        if False:
            return 10

        class C:

            def __init__(self, value):
                if False:
                    i = 10
                    return i + 15
                self.value = value

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                return iter(self.value)
        self.assertEqual(self.doit(C((1, 2))), 1)
        with self.assertRaisesRegex(ValueError, 'not enough values to unpack'):
            self.doit(C(()))

class UnpackExTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('UNPACK_EX')
    def doit(self, iterable):
        if False:
            while True:
                i = 10
        (x, *y) = iterable
        return x

    def test_unpack_ex_with_tuple(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.doit((1, 2)), 1)
        with self.assertRaisesRegex(ValueError, 'not enough values to unpack'):
            self.doit(())

    def test_unpack_ex_with_list(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.doit([1, 2]), 1)
        with self.assertRaisesRegex(ValueError, 'not enough values to unpack'):
            self.doit([])

    def test_unpack_ex_with_iterable(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __init__(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                self.value = value

            def __iter__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return iter(self.value)
        self.assertEqual(self.doit(C((1, 2))), 1)
        with self.assertRaisesRegex(ValueError, 'not enough values to unpack'):
            self.doit(C(()))

class StoreSubscrTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('STORE_SUBSCR')
    def doit(self, obj, key, value):
        if False:
            i = 10
            return i + 15
        obj[key] = value

    def test_store_subscr_with_list_sets_item(self):
        if False:
            while True:
                i = 10
        obj = [1, 2, 3]
        self.doit(obj, 1, 'hello')
        self.assertEqual(obj, [1, 'hello', 3])

    def test_store_subscr_with_dict_sets_item(self):
        if False:
            i = 10
            return i + 15
        obj = {'hello': 'cinder'}
        self.doit(obj, 'hello', 'world')
        self.assertEqual(obj, {'hello': 'world'})

    def test_store_subscr_calls_setitem(self):
        if False:
            while True:
                i = 10

        class C:

            def __init__(self):
                if False:
                    return 10
                self.called = None

            def __setitem__(self, key, value):
                if False:
                    return 10
                self.called = (key, value)
        obj = C()
        self.doit(obj, 'hello', 'world')
        self.assertEqual(obj.called, ('hello', 'world'))

    def test_store_subscr_deopts_on_exception(self):
        if False:
            print('Hello World!')

        class C:

            def __setitem__(self, key, value):
                if False:
                    i = 10
                    return i + 15
                raise TestException('hello')
        obj = C()
        with self.assertRaisesRegex(TestException, 'hello'):
            self.doit(obj, 1, 2)

class FormatValueTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('BUILD_STRING', 'FORMAT_VALUE')
    def doit(self, obj):
        if False:
            i = 10
            return i + 15
        return f'hello{obj}world'

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('BUILD_STRING', 'FORMAT_VALUE')
    def doit_repr(self, obj):
        if False:
            i = 10
            return i + 15
        return f'hello{obj!r}world'

    def test_format_value_calls_str(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __str__(self):
                if False:
                    i = 10
                    return i + 15
                return 'foo'
        self.assertEqual(self.doit(C()), 'hellofooworld')

    def test_format_value_calls_str_with_exception(self):
        if False:
            return 10

        class C:

            def __str__(self):
                if False:
                    i = 10
                    return i + 15
                raise TestException('no')
        with self.assertRaisesRegex(TestException, 'no'):
            self.assertEqual(self.doit(C()))

    def test_format_value_calls_repr(self):
        if False:
            for i in range(10):
                print('nop')

        class C:

            def __repr__(self):
                if False:
                    print('Hello World!')
                return 'bar'
        self.assertEqual(self.doit_repr(C()), 'hellobarworld')

    def test_format_value_calls_repr_with_exception(self):
        if False:
            for i in range(10):
                print('nop')

        class C:

            def __repr__(self):
                if False:
                    print('Hello World!')
                raise TestException('no')
        with self.assertRaisesRegex(TestException, 'no'):
            self.assertEqual(self.doit_repr(C()))

class ListExtendTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('LIST_EXTEND')
    def extend_list(self, it):
        if False:
            i = 10
            return i + 15
        return [1, *it]

    def test_list_extend_with_list(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.extend_list([2, 3, 4]), [1, 2, 3, 4])

    def test_list_extend_with_iterable(self):
        if False:
            return 10

        class A:

            def __init__(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                self.value = value

            def __iter__(self):
                if False:
                    print('Hello World!')
                return iter(self.value)
        extended_list = self.extend_list(A([2, 3]))
        self.assertEqual(type(extended_list), list)
        self.assertEqual(extended_list, [1, 2, 3])

    def test_list_extend_with_non_iterable_raises_type_error(self):
        if False:
            for i in range(10):
                print('nop')
        err_msg = 'Value after \\* must be an iterable, not int'
        with self.assertRaisesRegex(TypeError, err_msg):
            self.extend_list(1)

class SetupWithException(Exception):
    pass

class SetupWithTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_WITH', 'WITH_EXCEPT_START')
    def with_returns_value(self, mgr):
        if False:
            print('Hello World!')
        with mgr as x:
            return x

    def test_with_calls_enter_and_exit(self):
        if False:
            for i in range(10):
                print('nop')

        class MyCtxMgr:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.enter_called = False
                self.exit_args = None

            def __enter__(self):
                if False:
                    print('Hello World!')
                self.enter_called = True
                return self

            def __exit__(self, typ, val, tb):
                if False:
                    i = 10
                    return i + 15
                self.exit_args = (typ, val, tb)
                return False
        mgr = MyCtxMgr()
        self.assertEqual(self.with_returns_value(mgr), mgr)
        self.assertTrue(mgr.enter_called)
        self.assertEqual(mgr.exit_args, (None, None, None))

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('SETUP_WITH', 'WITH_EXCEPT_START')
    def with_raises(self, mgr):
        if False:
            i = 10
            return i + 15
        with mgr:
            raise SetupWithException('foo')
        return 100

    def test_with_calls_enter_and_exit(self):
        if False:
            while True:
                i = 10

        class MyCtxMgr:

            def __init__(self, should_suppress_exc):
                if False:
                    while True:
                        i = 10
                self.exit_args = None
                self.should_suppress_exc = should_suppress_exc

            def __enter__(self):
                if False:
                    print('Hello World!')
                return self

            def __exit__(self, typ, val, tb):
                if False:
                    while True:
                        i = 10
                self.exit_args = (typ, val, tb)
                return self.should_suppress_exc
        mgr = MyCtxMgr(should_suppress_exc=False)
        with self.assertRaisesRegex(SetupWithException, 'foo'):
            self.with_raises(mgr)
        self.assertEqual(mgr.exit_args[0], SetupWithException)
        self.assertTrue(isinstance(mgr.exit_args[1], SetupWithException))
        self.assertNotEqual(mgr.exit_args[2], None)
        mgr = MyCtxMgr(should_suppress_exc=True)
        self.assertEqual(self.with_raises(mgr), 100)

class ListToTupleTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('LIST_TO_TUPLE')
    def it_to_tup(self, it):
        if False:
            return 10
        return (*it,)

    def test_list_to_tuple_returns_tuple(self):
        if False:
            i = 10
            return i + 15
        new_tup = self.it_to_tup([1, 2, 3, 4])
        self.assertEqual(type(new_tup), tuple)
        self.assertEqual(new_tup, (1, 2, 3, 4))

class CompareTests(unittest.TestCase):

    class Incomparable:

        def __lt__(self, other):
            if False:
                print('Hello World!')
            raise TestException('no lt')

    class NonIterable:

        def __iter__(self):
            if False:
                print('Hello World!')
            raise TestException('no iter')

    class NonIndexable:

        def __getitem__(self, idx):
            if False:
                return 10
            raise TestException('no getitem')

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('COMPARE_OP')
    def compare_op(self, left, right):
        if False:
            while True:
                i = 10
        return left < right

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('CONTAINS_OP')
    def compare_in(self, left, right):
        if False:
            return 10
        return left in right

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('CONTAINS_OP')
    def compare_not_in(self, left, right):
        if False:
            return 10
        return left not in right

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('IS_OP')
    def compare_is(self, left, right):
        if False:
            i = 10
            return i + 15
        return left is right

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('IS_OP')
    def compare_is_not(self, left, right):
        if False:
            print('Hello World!')
        return left is not right

    def test_compare_op(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.compare_op(3, 4))
        self.assertFalse(self.compare_op(3, 3))
        with self.assertRaisesRegex(TestException, 'no lt'):
            self.compare_op(self.Incomparable(), 123)

    def test_contains_op(self):
        if False:
            return 10
        self.assertTrue(self.compare_in(3, [1, 2, 3]))
        self.assertFalse(self.compare_in(4, [1, 2, 3]))
        with self.assertRaisesRegex(TestException, 'no iter'):
            self.compare_in(123, self.NonIterable())
        with self.assertRaisesRegex(TestException, 'no getitem'):
            self.compare_in(123, self.NonIndexable())
        self.assertTrue(self.compare_not_in(4, [1, 2, 3]))
        self.assertFalse(self.compare_not_in(3, [1, 2, 3]))
        with self.assertRaisesRegex(TestException, 'no iter'):
            self.compare_not_in(123, self.NonIterable())
        with self.assertRaisesRegex(TestException, 'no getitem'):
            self.compare_not_in(123, self.NonIndexable())

    def test_is_op(self):
        if False:
            while True:
                i = 10
        obj = object()
        self.assertTrue(self.compare_is(obj, obj))
        self.assertFalse(self.compare_is(obj, 1))
        self.assertTrue(self.compare_is_not(obj, 1))
        self.assertFalse(self.compare_is_not(obj, obj))

class MatchTests(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('MATCH_SEQUENCE', 'ROT_N')
    def match_sequence(self, s: tuple) -> bool:
        if False:
            i = 10
            return i + 15
        match s:
            case [*b, 8, 9, 4, 5]:
                return True
            case _:
                return False

    def test_match_sequence(self):
        if False:
            return 10
        self.assertTrue(self.match_sequence((1, 2, 3, 7, 8, 9, 4, 5)))
        self.assertFalse(self.match_sequence((1, 2, 3, 4, 5, 6, 7, 8)))

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('MATCH_KEYS', 'MATCH_MAPPING')
    def match_keys(self, m: dict) -> bool:
        if False:
            for i in range(10):
                print('nop')
        match m:
            case {'id': 1}:
                return True
            case _:
                return False

    def test_match_keys(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.match_keys({'id': 1}))
        self.assertFalse(self.match_keys({'id': 2}))

    class A:
        __match_args__ = 'id'

        def __init__(self, id):
            if False:
                return 10
            self.id = id

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('MATCH_CLASS')
    def match_class(self, a: A) -> bool:
        if False:
            return 10
        match a:
            case self.A(id=2):
                return True
            case _:
                return False

    def test_match_class(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.match_class(self.A(2)))
        self.assertFalse(self.match_class(self.A(3)))

    class Point:
        __match_args__ = 123

        def __init__(self, x, y):
            if False:
                while True:
                    i = 10
            self.x = x
            self.y = y

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('MATCH_CLASS')
    def match_class_exc():
        if False:
            i = 10
            return i + 15
        (x, y) = (5, 5)
        point = Point(x, y)
        match point:
            case Point(x, y):
                pass

    def test_match_class_exc(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            self.match_class_exc()

class CopyDictWithoutKeysTest(unittest.TestCase):

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('COPY_DICT_WITHOUT_KEYS')
    def match_rest(self, obj):
        if False:
            return 10
        match obj:
            case {**rest}:
                return rest

    def test_rest_with_empty_dict_returns_empty_dict(self):
        if False:
            return 10
        obj = {}
        result = self.match_rest(obj)
        self.assertIs(type(result), dict)
        self.assertEqual(result, {})
        self.assertIsNot(result, obj)

    def test_rest_with_nonempty_dict_returns_dict_copy(self):
        if False:
            i = 10
            return i + 15
        obj = {'x': 1}
        result = self.match_rest(obj)
        self.assertIs(type(result), dict)
        self.assertEqual(result, {'x': 1})
        self.assertIsNot(result, obj)

    @cinder_support.failUnlessJITCompiled
    @failUnlessHasOpcodes('COPY_DICT_WITHOUT_KEYS')
    def match_keys_and_rest(self, obj):
        if False:
            for i in range(10):
                print('nop')
        match obj:
            case {'x': 1, **rest}:
                return rest

    def test_keys_and_rest_with_empty_dict_does_not_match(self):
        if False:
            print('Hello World!')
        result = self.match_keys_and_rest({})
        self.assertIs(result, None)

    def test_keys_and_rest_with_matching_dict_returns_rest(self):
        if False:
            i = 10
            return i + 15
        obj = {'x': 1, 'y': 2}
        result = self.match_keys_and_rest(obj)
        self.assertIs(type(result), dict)
        self.assertEqual(result, {'y': 2})

    def test_with_mappingproxy_returns_dict(self):
        if False:
            return 10

        class C:
            x = 1
            y = 2
        obj = C.__dict__
        self.assertEqual(obj.__class__.__name__, 'mappingproxy')
        result = self.match_keys_and_rest(obj)
        self.assertIs(type(result), dict)
        self.assertEqual(result['y'], 2)

    def test_with_abstract_mapping(self):
        if False:
            for i in range(10):
                print('nop')
        import collections.abc

        class C(collections.abc.Mapping):

            def __iter__(self):
                if False:
                    i = 10
                    return i + 15
                return iter(('x', 'y'))

            def __len__(self):
                if False:
                    while True:
                        i = 10
                return 2

            def __getitem__(self, key):
                if False:
                    i = 10
                    return i + 15
                if key == 'x':
                    return 1
                if key == 'y':
                    return 2
                raise RuntimeError('getitem', key)
        obj = C()
        result = self.match_keys_and_rest(obj)
        self.assertIs(type(result), dict)
        self.assertEqual(result, {'y': 2})

    def test_raising_exception_propagates(self):
        if False:
            for i in range(10):
                print('nop')
        import collections.abc

        class C(collections.abc.Mapping):

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                return iter(('x', 'y'))

            def __len__(self):
                if False:
                    while True:
                        i = 10
                return 2

            def __getitem__(self, key):
                if False:
                    for i in range(10):
                        print('nop')
                raise RuntimeError(f'__getitem__ called with {key}')
        obj = C()
        with self.assertRaisesRegex(RuntimeError, '__getitem__ called with x'):
            self.match_keys_and_rest(obj)

def builtins_getter():
    if False:
        for i in range(10):
            print('nop')
    return _testcindercapi._pyeval_get_builtins()

class GetBuiltinsTests(unittest.TestCase):

    def test_get_builtins(self):
        if False:
            return 10
        new_builtins = {}
        new_globals = {'_testcindercapi': _testcindercapi, '__builtins__': new_builtins}
        func = with_globals(new_globals)(builtins_getter)
        if cinderjit is not None:
            cinderjit.force_compile(func)
        self.assertIs(func(), new_builtins)

def globals_getter():
    if False:
        while True:
            i = 10
    return globals()

class GetGlobalsTests(unittest.TestCase):

    def test_get_globals(self):
        if False:
            i = 10
            return i + 15
        new_globals = dict(globals())
        func = with_globals(new_globals)(globals_getter)
        if cinderjit is not None:
            cinderjit.force_compile(func)
        self.assertIs(func(), new_globals)

class MergeCompilerFlagTests(unittest.TestCase):

    def make_func(self, src, compile_flags=0):
        if False:
            i = 10
            return i + 15
        code = compile(src, '<string>', 'exec', compile_flags)
        glbls = {'_testcindercapi': _testcindercapi}
        exec(code, glbls)
        return glbls['func']

    def run_test(self, callee_src):
        if False:
            i = 10
            return i + 15
        flag = CO_FUTURE_BARRY_AS_BDFL
        caller_src = '\ndef func(callee):\n  return callee()\n'
        caller = self.make_func(caller_src)
        caller = jit_suppress(caller)
        self.assertEqual(caller.__code__.co_flags & flag, 0)
        callee = self.make_func(callee_src, CO_FUTURE_BARRY_AS_BDFL)
        self.assertEqual(callee.__code__.co_flags & flag, flag)
        if cinderjit is not None:
            cinderjit.force_compile(callee)
        flags = caller(callee)
        self.assertEqual(flags & flag, flag)

    def test_merge_compiler_flags(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that PyEval_MergeCompilerFlags retrieves the compiler flags of the\n        calling function.'
        src = '\ndef func():\n  return _testcindercapi._pyeval_merge_compiler_flags()\n'
        self.run_test(src)

    def test_compile_inherits_compiler_flags(self):
        if False:
            return 10
        'Test that compile inherits the compiler flags of the calling function.'
        src = "\ndef func():\n  code = compile('1 + 1', '<string>', 'eval')\n  return code.co_flags\n"
        self.run_test(src)

class PerfMapTests(unittest.TestCase):
    HELPER_FILE = os.path.join(os.path.dirname(__file__), 'perf_fork_helper.py')

    @cinder_support.skipUnlessJITEnabled('Runs a subprocess with the JIT enabled')
    def test_forked_pid_map(self):
        if False:
            for i in range(10):
                print('nop')
        proc = subprocess.run([sys.executable, '-X', 'jit', '-X', 'jit-perfmap', self.HELPER_FILE], stdout=subprocess.PIPE, encoding=sys.stdout.encoding)
        self.assertEqual(proc.returncode, 0)

        def find_mapped_funcs(which):
            if False:
                i = 10
                return i + 15
            pattern = f'{which}\\(([0-9]+)\\) computed '
            m = re.search(pattern, proc.stdout)
            self.assertIsNotNone(m, f"Couldn't find /{pattern}/ in stdout:\n\n{proc.stdout}")
            pid = int(m[1])
            try:
                with open(f'/tmp/perf-{pid}.map') as f:
                    map_contents = f.read()
            except FileNotFoundError:
                self.fail(f'{which} process (pid {pid}) did not generate a map')
            funcs = set(re.findall('__CINDER_JIT:__main__:(.+)', map_contents))
            return funcs
        self.assertEqual(find_mapped_funcs('parent'), {'main', 'parent', 'compute'})
        self.assertEqual(find_mapped_funcs('child1'), {'main', 'child1', 'compute'})
        self.assertEqual(find_mapped_funcs('child2'), {'main', 'child2', 'compute'})

class PreloadTests(unittest.TestCase):
    SCRIPT_FILE = 'cinder_preload_helper_main.py'

    @cinder_support.skipUnlessJITEnabled('Runs a subprocess with the JIT enabled')
    def test_func_destroyed_during_preload(self):
        if False:
            while True:
                i = 10
        proc = subprocess.run([sys.executable, '-X', 'jit', '-X', 'jit-batch-compile-workers=4', '-L', '-mcompiler', '--static', self.SCRIPT_FILE], cwd=os.path.dirname(__file__), stdout=subprocess.PIPE, encoding=sys.stdout.encoding)
        self.assertEqual(proc.returncode, 0)
        expected_stdout = "resolving a_func\nloading helper_a\ndefining main_func()\ndisabling jit\nloading helper_b\njit disabled\n<class 'NoneType'>\nhello from b_func!\n"
        self.assertEqual(proc.stdout, expected_stdout)

class LoadMethodEliminationTests(unittest.TestCase):

    def lme_test_func(self, flag=False):
        if False:
            return 10
        return '{}{}'.format(1, '' if not flag else ' flag')

    def test_multiple_call_method_same_load_method(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.lme_test_func(), '1')
        self.assertEqual(self.lme_test_func(True), '1 flag')
        if cinderjit and cinderjit.auto_jit_threshold() <= 1:
            self.assertTrue(is_jit_compiled(LoadMethodEliminationTests.lme_test_func))

@unittest.skipIf(not cinderjit, 'Tests functionality on cinderjit module')
class HIROpcodeCountTests(unittest.TestCase):

    def test_hir_opcode_count(self):
        if False:
            return 10

        def f1():
            if False:
                for i in range(10):
                    print('nop')
            return 5

        def func():
            if False:
                return 10
            return f1() + f1()
        cinderjit.force_compile(func)
        self.assertEqual(func(), 10)
        ops = cinderjit.get_function_hir_opcode_counts(func)
        self.assertIsInstance(ops, dict)
        self.assertEqual(ops.get('Return'), 1)
        self.assertEqual(ops.get('BinaryOp'), 1)
        self.assertGreaterEqual(ops.get('Decref'), 2)
if __name__ == '__main__':
    unittest.main()