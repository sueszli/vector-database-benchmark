import collections
import sys
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.controlflow import CFGraph, Loop
from numba.core.compiler import compile_extra, compile_isolated, Flags, CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
enable_pyobj_flags = Flags()
enable_pyobj_flags.enable_pyobject = True
forceobj_flags = Flags()
forceobj_flags.force_pyobject = True
no_pyobj_flags = Flags()

class _Dummy(object):

    def __init__(self, recorder, name):
        if False:
            return 10
        self.recorder = recorder
        self.name = name
        recorder._add_dummy(self)

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(other, _Dummy)
        return _Dummy(self.recorder, '%s + %s' % (self.name, other.name))

    def __iter__(self):
        if False:
            while True:
                i = 10
        return _DummyIterator(self.recorder, 'iter(%s)' % self.name)

class _DummyIterator(_Dummy):
    count = 0

    def __next__(self):
        if False:
            i = 10
            return i + 15
        if self.count >= 3:
            raise StopIteration
        self.count += 1
        return _Dummy(self.recorder, '%s#%s' % (self.name, self.count))
    next = __next__

class RefRecorder(object):
    """
    An object which records events when instances created through it
    are deleted.  Custom events can also be recorded to aid in
    diagnosis.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._counts = collections.defaultdict(int)
        self._events = []
        self._wrs = {}

    def make_dummy(self, name):
        if False:
            print('Hello World!')
        '\n        Make an object whose deletion will be recorded as *name*.\n        '
        return _Dummy(self, name)

    def _add_dummy(self, dummy):
        if False:
            return 10
        wr = weakref.ref(dummy, self._on_disposal)
        self._wrs[wr] = dummy.name
    __call__ = make_dummy

    def mark(self, event):
        if False:
            print('Hello World!')
        '\n        Manually append *event* to the recorded events.\n        *event* can be formatted using format().\n        '
        count = self._counts[event] + 1
        self._counts[event] = count
        self._events.append(event.format(count=count))

    def _on_disposal(self, wr):
        if False:
            while True:
                i = 10
        name = self._wrs.pop(wr)
        self._events.append(name)

    @property
    def alive(self):
        if False:
            while True:
                i = 10
        "\n        A list of objects which haven't been deleted yet.\n        "
        return [wr() for wr in self._wrs]

    @property
    def recorded(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A list of recorded events.\n        '
        return self._events

def simple_usecase1(rec):
    if False:
        while True:
            i = 10
    a = rec('a')
    b = rec('b')
    c = rec('c')
    a = b + c
    rec.mark('--1--')
    d = a + a
    rec.mark('--2--')
    return d

def simple_usecase2(rec):
    if False:
        i = 10
        return i + 15
    a = rec('a')
    b = rec('b')
    rec.mark('--1--')
    x = a
    y = x
    a = None
    return y

def looping_usecase1(rec):
    if False:
        print('Hello World!')
    a = rec('a')
    b = rec('b')
    c = rec('c')
    x = b
    for y in a:
        x = x + y
        rec.mark('--loop bottom--')
    rec.mark('--loop exit--')
    x = x + c
    return x

def looping_usecase2(rec):
    if False:
        while True:
            i = 10
    a = rec('a')
    b = rec('b')
    cum = rec('cum')
    for x in a:
        rec.mark('--outer loop top--')
        cum = cum + x
        z = x + x
        rec.mark('--inner loop entry #{count}--')
        for y in b:
            rec.mark('--inner loop top #{count}--')
            cum = cum + y
            rec.mark('--inner loop bottom #{count}--')
        rec.mark('--inner loop exit #{count}--')
        if cum:
            cum = y + z
        else:
            break
        rec.mark('--outer loop bottom #{count}--')
    else:
        rec.mark('--outer loop else--')
    rec.mark('--outer loop exit--')
    return cum

def generator_usecase1(rec):
    if False:
        i = 10
        return i + 15
    a = rec('a')
    b = rec('b')
    yield a
    yield b

def generator_usecase2(rec):
    if False:
        i = 10
        return i + 15
    a = rec('a')
    b = rec('b')
    for x in a:
        yield x
    yield b

class MyError(RuntimeError):
    pass

def do_raise(x):
    if False:
        return 10
    raise MyError(x)

def raising_usecase1(rec):
    if False:
        print('Hello World!')
    a = rec('a')
    b = rec('b')
    d = rec('d')
    if a:
        do_raise('foo')
        c = rec('c')
        c + a
    c + b

def raising_usecase2(rec):
    if False:
        i = 10
        return i + 15
    a = rec('a')
    b = rec('b')
    if a:
        c = rec('c')
        do_raise(b)
    a + c

def raising_usecase3(rec):
    if False:
        i = 10
        return i + 15
    a = rec('a')
    b = rec('b')
    if a:
        raise MyError(b)

def del_before_definition(rec):
    if False:
        for i in range(10):
            print('nop')
    '\n    This test reveal a bug that there is a del on uninitialized variable\n    '
    n = 5
    for i in range(n):
        rec.mark(str(i))
        n = 0
        for j in range(n):
            return 0
        else:
            if i < 2:
                continue
            elif i == 2:
                for j in range(i):
                    return i
                rec.mark('FAILED')
            rec.mark('FAILED')
        rec.mark('FAILED')
    rec.mark('OK')
    return -1

def inf_loop_multiple_back_edge(rec):
    if False:
        i = 10
        return i + 15
    '\n    test to reveal bug of invalid liveness when infinite loop has multiple\n    backedge.\n    '
    while True:
        rec.mark('yield')
        yield
        p = rec('p')
        if p:
            rec.mark('bra')
            pass

class TestObjLifetime(TestCase):
    """
    Test lifetime of Python objects inside jit-compiled functions.
    """

    def compile(self, pyfunc):
        if False:
            return 10
        cfunc = jit((types.pyobject,), forceobj=True, looplift=False)(pyfunc)
        return cfunc

    def compile_and_record(self, pyfunc, raises=None):
        if False:
            for i in range(10):
                print('nop')
        rec = RefRecorder()
        cfunc = self.compile(pyfunc)
        if raises is not None:
            with self.assertRaises(raises):
                cfunc(rec)
        else:
            cfunc(rec)
        return rec

    def assertRecordOrder(self, rec, expected):
        if False:
            return 10
        "\n        Check that the *expected* markers occur in that order in *rec*'s\n        recorded events.\n        "
        actual = []
        recorded = rec.recorded
        remaining = list(expected)
        for d in recorded:
            if d in remaining:
                actual.append(d)
                remaining.remove(d)
        self.assertEqual(actual, expected, 'the full list of recorded events is: %r' % (recorded,))

    def test_simple1(self):
        if False:
            print('Hello World!')
        rec = self.compile_and_record(simple_usecase1)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['a', 'b', '--1--'])
        self.assertRecordOrder(rec, ['a', 'c', '--1--'])
        self.assertRecordOrder(rec, ['--1--', 'b + c', '--2--'])

    def test_simple2(self):
        if False:
            i = 10
            return i + 15
        rec = self.compile_and_record(simple_usecase2)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['b', '--1--', 'a'])

    def test_looping1(self):
        if False:
            for i in range(10):
                print('nop')
        rec = self.compile_and_record(looping_usecase1)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['a', 'b', '--loop exit--', 'c'])
        self.assertRecordOrder(rec, ['iter(a)#1', '--loop bottom--', 'iter(a)#2', '--loop bottom--', 'iter(a)#3', '--loop bottom--', 'iter(a)', '--loop exit--'])

    def test_looping2(self):
        if False:
            print('Hello World!')
        rec = self.compile_and_record(looping_usecase2)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['a', '--outer loop top--'])
        self.assertRecordOrder(rec, ['iter(a)', '--outer loop else--', '--outer loop exit--'])
        self.assertRecordOrder(rec, ['iter(b)', '--inner loop exit #1--', 'iter(b)', '--inner loop exit #2--', 'iter(b)', '--inner loop exit #3--'])
        self.assertRecordOrder(rec, ['iter(a)#1', '--inner loop entry #1--', 'iter(a)#2', '--inner loop entry #2--', 'iter(a)#3', '--inner loop entry #3--'])
        self.assertRecordOrder(rec, ['iter(a)#1 + iter(a)#1', '--outer loop bottom #1--'])

    def exercise_generator(self, genfunc):
        if False:
            while True:
                i = 10
        cfunc = self.compile(genfunc)
        rec = RefRecorder()
        with self.assertRefCount(rec):
            gen = cfunc(rec)
            next(gen)
            self.assertTrue(rec.alive)
            list(gen)
            self.assertFalse(rec.alive)
        rec = RefRecorder()
        with self.assertRefCount(rec):
            gen = cfunc(rec)
            del gen
            gc.collect()
            self.assertFalse(rec.alive)
        rec = RefRecorder()
        with self.assertRefCount(rec):
            gen = cfunc(rec)
            next(gen)
            self.assertTrue(rec.alive)
            del gen
            gc.collect()
            self.assertFalse(rec.alive)

    def test_generator1(self):
        if False:
            for i in range(10):
                print('nop')
        self.exercise_generator(generator_usecase1)

    def test_generator2(self):
        if False:
            i = 10
            return i + 15
        self.exercise_generator(generator_usecase2)

    def test_del_before_definition(self):
        if False:
            return 10
        rec = self.compile_and_record(del_before_definition)
        self.assertEqual(rec.recorded, ['0', '1', '2'])

    def test_raising1(self):
        if False:
            print('Hello World!')
        with self.assertRefCount(do_raise):
            rec = self.compile_and_record(raising_usecase1, raises=MyError)
            self.assertFalse(rec.alive)

    def test_raising2(self):
        if False:
            i = 10
            return i + 15
        with self.assertRefCount(do_raise):
            rec = self.compile_and_record(raising_usecase2, raises=MyError)
            self.assertFalse(rec.alive)

    def test_raising3(self):
        if False:
            i = 10
            return i + 15
        with self.assertRefCount(MyError):
            rec = self.compile_and_record(raising_usecase3, raises=MyError)
            self.assertFalse(rec.alive)

    def test_inf_loop_multiple_back_edge(self):
        if False:
            return 10
        cfunc = self.compile(inf_loop_multiple_back_edge)
        rec = RefRecorder()
        iterator = iter(cfunc(rec))
        next(iterator)
        self.assertEqual(rec.alive, [])
        next(iterator)
        self.assertEqual(rec.alive, [])
        next(iterator)
        self.assertEqual(rec.alive, [])
        self.assertEqual(rec.recorded, ['yield', 'p', 'bra', 'yield', 'p', 'bra', 'yield'])

class TestExtendingVariableLifetimes(SerialMixin, TestCase):

    def test_lifetime_basic(self):
        if False:
            print('Hello World!')

        def get_ir(extend_lifetimes):
            if False:
                i = 10
                return i + 15

            class IRPreservingCompiler(CompilerBase):

                def define_pipelines(self):
                    if False:
                        print('Hello World!')
                    pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                    pm.add_pass_after(PreserveIR, IRLegalization)
                    pm.finalize()
                    return [pm]

            @njit(pipeline_class=IRPreservingCompiler)
            def foo():
                if False:
                    return 10
                a = 10
                b = 20
                c = a + b
                d = c / c
                return d
            with override_config('EXTEND_VARIABLE_LIFETIMES', extend_lifetimes):
                foo()
                cres = foo.overloads[foo.signatures[0]]
                func_ir = cres.metadata['preserved_ir']
            return func_ir

        def check(func_ir, expect):
            if False:
                while True:
                    i = 10
            self.assertEqual(len(func_ir.blocks), 1)
            blk = next(iter(func_ir.blocks.values()))
            for (expect_class, got_stmt) in zip(expect, blk.body):
                self.assertIsInstance(got_stmt, expect_class)
        del_after_use_ir = get_ir(False)
        expect = [*(ir.Assign,) * 3, ir.Del, ir.Del, ir.Assign, ir.Del, ir.Assign, ir.Del, ir.Return]
        check(del_after_use_ir, expect)
        del_at_block_end_ir = get_ir(True)
        expect = [*(ir.Assign,) * 4, ir.Assign, *(ir.Del,) * 4, ir.Return]
        check(del_at_block_end_ir, expect)

    def test_dbg_extend_lifetimes(self):
        if False:
            while True:
                i = 10

        def get_ir(**options):
            if False:
                return 10

            class IRPreservingCompiler(CompilerBase):

                def define_pipelines(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                    pm.add_pass_after(PreserveIR, IRLegalization)
                    pm.finalize()
                    return [pm]

            @njit(pipeline_class=IRPreservingCompiler, **options)
            def foo():
                if False:
                    while True:
                        i = 10
                a = 10
                b = 20
                c = a + b
                d = c / c
                return d
            foo()
            cres = foo.overloads[foo.signatures[0]]
            func_ir = cres.metadata['preserved_ir']
            return func_ir
        ir_debug = get_ir(debug=True)
        ir_debug_ext = get_ir(debug=True, _dbg_extend_lifetimes=True)
        ir_debug_no_ext = get_ir(debug=True, _dbg_extend_lifetimes=False)

        def is_del_grouped_at_the_end(fir):
            if False:
                i = 10
                return i + 15
            [blk] = fir.blocks.values()
            inst_is_del = [isinstance(stmt, ir.Del) for stmt in blk.body]
            not_dels = list(takewhile(operator.not_, inst_is_del))
            begin = len(not_dels)
            all_dels = list(takewhile(operator.truth, inst_is_del[begin:]))
            end = begin + len(all_dels)
            return end == len(inst_is_del) - 1
        self.assertTrue(is_del_grouped_at_the_end(ir_debug))
        self.assertTrue(is_del_grouped_at_the_end(ir_debug_ext))
        self.assertFalse(is_del_grouped_at_the_end(ir_debug_no_ext))
if __name__ == '__main__':
    unittest.main()