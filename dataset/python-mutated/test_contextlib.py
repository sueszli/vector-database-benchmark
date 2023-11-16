"""Unit tests for contextlib.py, and other context managers."""
import io
import sys
import tempfile
import threading
import traceback
import unittest
from contextlib import *
from test import support
from test.support import os_helper
import weakref

class TestAbstractContextManager(unittest.TestCase):

    def test_enter(self):
        if False:
            i = 10
            return i + 15

        class DefaultEnter(AbstractContextManager):

            def __exit__(self, *args):
                if False:
                    for i in range(10):
                        print('nop')
                super().__exit__(*args)
        manager = DefaultEnter()
        self.assertIs(manager.__enter__(), manager)

    def test_exit_is_abstract(self):
        if False:
            return 10

        class MissingExit(AbstractContextManager):
            pass
        with self.assertRaises(TypeError):
            MissingExit()

    def test_structural_subclassing(self):
        if False:
            return 10

        class ManagerFromScratch:

            def __enter__(self):
                if False:
                    return 10
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                if False:
                    print('Hello World!')
                return None
        self.assertTrue(issubclass(ManagerFromScratch, AbstractContextManager))

        class DefaultEnter(AbstractContextManager):

            def __exit__(self, *args):
                if False:
                    for i in range(10):
                        print('nop')
                super().__exit__(*args)
        self.assertTrue(issubclass(DefaultEnter, AbstractContextManager))

        class NoEnter(ManagerFromScratch):
            __enter__ = None
        self.assertFalse(issubclass(NoEnter, AbstractContextManager))

        class NoExit(ManagerFromScratch):
            __exit__ = None
        self.assertFalse(issubclass(NoExit, AbstractContextManager))

class ContextManagerTestCase(unittest.TestCase):

    def test_contextmanager_plain(self):
        if False:
            return 10
        state = []

        @contextmanager
        def woohoo():
            if False:
                i = 10
                return i + 15
            state.append(1)
            yield 42
            state.append(999)
        with woohoo() as x:
            self.assertEqual(state, [1])
            self.assertEqual(x, 42)
            state.append(x)
        self.assertEqual(state, [1, 42, 999])

    def test_contextmanager_finally(self):
        if False:
            i = 10
            return i + 15
        state = []

        @contextmanager
        def woohoo():
            if False:
                return 10
            state.append(1)
            try:
                yield 42
            finally:
                state.append(999)
        with self.assertRaises(ZeroDivisionError):
            with woohoo() as x:
                self.assertEqual(state, [1])
                self.assertEqual(x, 42)
                state.append(x)
                raise ZeroDivisionError()
        self.assertEqual(state, [1, 42, 999])

    def test_contextmanager_no_reraise(self):
        if False:
            while True:
                i = 10

        @contextmanager
        def whee():
            if False:
                print('Hello World!')
            yield
        ctx = whee()
        ctx.__enter__()
        self.assertFalse(ctx.__exit__(TypeError, TypeError('foo'), None))

    def test_contextmanager_trap_yield_after_throw(self):
        if False:
            for i in range(10):
                print('nop')

        @contextmanager
        def whoo():
            if False:
                i = 10
                return i + 15
            try:
                yield
            except:
                yield
        ctx = whoo()
        ctx.__enter__()
        self.assertRaises(RuntimeError, ctx.__exit__, TypeError, TypeError('foo'), None)

    def test_contextmanager_except(self):
        if False:
            for i in range(10):
                print('nop')
        state = []

        @contextmanager
        def woohoo():
            if False:
                while True:
                    i = 10
            state.append(1)
            try:
                yield 42
            except ZeroDivisionError as e:
                state.append(e.args[0])
                self.assertEqual(state, [1, 42, 999])
        with woohoo() as x:
            self.assertEqual(state, [1])
            self.assertEqual(x, 42)
            state.append(x)
            raise ZeroDivisionError(999)
        self.assertEqual(state, [1, 42, 999])

    def test_contextmanager_except_stopiter(self):
        if False:
            i = 10
            return i + 15

        @contextmanager
        def woohoo():
            if False:
                for i in range(10):
                    print('nop')
            yield

        class StopIterationSubclass(StopIteration):
            pass
        for stop_exc in (StopIteration('spam'), StopIterationSubclass('spam')):
            with self.subTest(type=type(stop_exc)):
                try:
                    with woohoo():
                        raise stop_exc
                except Exception as ex:
                    self.assertIs(ex, stop_exc)
                else:
                    self.fail(f'{stop_exc} was suppressed')

    def test_contextmanager_except_pep479(self):
        if False:
            for i in range(10):
                print('nop')
        code = 'from __future__ import generator_stop\nfrom contextlib import contextmanager\n@contextmanager\ndef woohoo():\n    yield\n'
        locals = {}
        exec(code, locals, locals)
        woohoo = locals['woohoo']
        stop_exc = StopIteration('spam')
        try:
            with woohoo():
                raise stop_exc
        except Exception as ex:
            self.assertIs(ex, stop_exc)
        else:
            self.fail('StopIteration was suppressed')

    def test_contextmanager_do_not_unchain_non_stopiteration_exceptions(self):
        if False:
            return 10

        @contextmanager
        def test_issue29692():
            if False:
                for i in range(10):
                    print('nop')
            try:
                yield
            except Exception as exc:
                raise RuntimeError('issue29692:Chained') from exc
        try:
            with test_issue29692():
                raise ZeroDivisionError
        except Exception as ex:
            self.assertIs(type(ex), RuntimeError)
            self.assertEqual(ex.args[0], 'issue29692:Chained')
            self.assertIsInstance(ex.__cause__, ZeroDivisionError)
        try:
            with test_issue29692():
                raise StopIteration('issue29692:Unchained')
        except Exception as ex:
            self.assertIs(type(ex), StopIteration)
            self.assertEqual(ex.args[0], 'issue29692:Unchained')
            self.assertIsNone(ex.__cause__)

    def _create_contextmanager_attribs(self):
        if False:
            for i in range(10):
                print('nop')

        def attribs(**kw):
            if False:
                for i in range(10):
                    print('nop')

            def decorate(func):
                if False:
                    for i in range(10):
                        print('nop')
                for (k, v) in kw.items():
                    setattr(func, k, v)
                return func
            return decorate

        @contextmanager
        @attribs(foo='bar')
        def baz(spam):
            if False:
                print('Hello World!')
            'Whee!'
        return baz

    def test_contextmanager_attribs(self):
        if False:
            print('Hello World!')
        baz = self._create_contextmanager_attribs()
        self.assertEqual(baz.__name__, 'baz')
        self.assertEqual(baz.foo, 'bar')

    @support.requires_docstrings
    def test_contextmanager_doc_attrib(self):
        if False:
            for i in range(10):
                print('nop')
        baz = self._create_contextmanager_attribs()
        self.assertEqual(baz.__doc__, 'Whee!')

    @support.requires_docstrings
    def test_instance_docstring_given_cm_docstring(self):
        if False:
            return 10
        baz = self._create_contextmanager_attribs()(None)
        self.assertEqual(baz.__doc__, 'Whee!')

    def test_keywords(self):
        if False:
            i = 10
            return i + 15

        @contextmanager
        def woohoo(self, func, args, kwds):
            if False:
                return 10
            yield (self, func, args, kwds)
        with woohoo(self=11, func=22, args=33, kwds=44) as target:
            self.assertEqual(target, (11, 22, 33, 44))

    def test_nokeepref(self):
        if False:
            i = 10
            return i + 15

        class A:
            pass

        @contextmanager
        def woohoo(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a = weakref.ref(a)
            b = weakref.ref(b)
            support.gc_collect()
            self.assertIsNone(a())
            self.assertIsNone(b())
            yield
        with woohoo(A(), b=A()):
            pass

    def test_param_errors(self):
        if False:
            for i in range(10):
                print('nop')

        @contextmanager
        def woohoo(a, *, b):
            if False:
                while True:
                    i = 10
            yield
        with self.assertRaises(TypeError):
            woohoo()
        with self.assertRaises(TypeError):
            woohoo(3, 5)
        with self.assertRaises(TypeError):
            woohoo(b=3)

    def test_recursive(self):
        if False:
            i = 10
            return i + 15
        depth = 0

        @contextmanager
        def woohoo():
            if False:
                while True:
                    i = 10
            nonlocal depth
            before = depth
            depth += 1
            yield
            depth -= 1
            self.assertEqual(depth, before)

        @woohoo()
        def recursive():
            if False:
                return 10
            if depth < 10:
                recursive()
        recursive()
        self.assertEqual(depth, 0)

class ClosingTestCase(unittest.TestCase):

    @support.requires_docstrings
    def test_instance_docs(self):
        if False:
            i = 10
            return i + 15
        cm_docstring = closing.__doc__
        obj = closing(None)
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_closing(self):
        if False:
            while True:
                i = 10
        state = []

        class C:

            def close(self):
                if False:
                    for i in range(10):
                        print('nop')
                state.append(1)
        x = C()
        self.assertEqual(state, [])
        with closing(x) as y:
            self.assertEqual(x, y)
        self.assertEqual(state, [1])

    def test_closing_error(self):
        if False:
            return 10
        state = []

        class C:

            def close(self):
                if False:
                    while True:
                        i = 10
                state.append(1)
        x = C()
        self.assertEqual(state, [])
        with self.assertRaises(ZeroDivisionError):
            with closing(x) as y:
                self.assertEqual(x, y)
                1 / 0
        self.assertEqual(state, [1])

class NullcontextTestCase(unittest.TestCase):

    def test_nullcontext(self):
        if False:
            print('Hello World!')

        class C:
            pass
        c = C()
        with nullcontext(c) as c_in:
            self.assertIs(c_in, c)

class FileContextTestCase(unittest.TestCase):

    def testWithOpen(self):
        if False:
            while True:
                i = 10
        tfn = tempfile.mktemp()
        try:
            f = None
            with open(tfn, 'w', encoding='utf-8') as f:
                self.assertFalse(f.closed)
                f.write('Booh\n')
            self.assertTrue(f.closed)
            f = None
            with self.assertRaises(ZeroDivisionError):
                with open(tfn, 'r', encoding='utf-8') as f:
                    self.assertFalse(f.closed)
                    self.assertEqual(f.read(), 'Booh\n')
                    1 / 0
            self.assertTrue(f.closed)
        finally:
            os_helper.unlink(tfn)

class LockContextTestCase(unittest.TestCase):

    def boilerPlate(self, lock, locked):
        if False:
            i = 10
            return i + 15
        self.assertFalse(locked())
        with lock:
            self.assertTrue(locked())
        self.assertFalse(locked())
        with self.assertRaises(ZeroDivisionError):
            with lock:
                self.assertTrue(locked())
                1 / 0
        self.assertFalse(locked())

    def testWithLock(self):
        if False:
            i = 10
            return i + 15
        lock = threading.Lock()
        self.boilerPlate(lock, lock.locked)

    def testWithRLock(self):
        if False:
            for i in range(10):
                print('nop')
        lock = threading.RLock()
        self.boilerPlate(lock, lock._is_owned)

    def testWithCondition(self):
        if False:
            for i in range(10):
                print('nop')
        lock = threading.Condition()

        def locked():
            if False:
                i = 10
                return i + 15
            return lock._is_owned()
        self.boilerPlate(lock, locked)

    def testWithSemaphore(self):
        if False:
            return 10
        lock = threading.Semaphore()

        def locked():
            if False:
                while True:
                    i = 10
            if lock.acquire(False):
                lock.release()
                return False
            else:
                return True
        self.boilerPlate(lock, locked)

    def testWithBoundedSemaphore(self):
        if False:
            for i in range(10):
                print('nop')
        lock = threading.BoundedSemaphore()

        def locked():
            if False:
                for i in range(10):
                    print('nop')
            if lock.acquire(False):
                lock.release()
                return False
            else:
                return True
        self.boilerPlate(lock, locked)

class mycontext(ContextDecorator):
    """Example decoration-compatible context manager for testing"""
    started = False
    exc = None
    catch = False

    def __enter__(self):
        if False:
            print('Hello World!')
        self.started = True
        return self

    def __exit__(self, *exc):
        if False:
            while True:
                i = 10
        self.exc = exc
        return self.catch

class TestContextDecorator(unittest.TestCase):

    @support.requires_docstrings
    def test_instance_docs(self):
        if False:
            print('Hello World!')
        cm_docstring = mycontext.__doc__
        obj = mycontext()
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_contextdecorator(self):
        if False:
            for i in range(10):
                print('nop')
        context = mycontext()
        with context as result:
            self.assertIs(result, context)
            self.assertTrue(context.started)
        self.assertEqual(context.exc, (None, None, None))

    def test_contextdecorator_with_exception(self):
        if False:
            i = 10
            return i + 15
        context = mycontext()
        with self.assertRaisesRegex(NameError, 'foo'):
            with context:
                raise NameError('foo')
        self.assertIsNotNone(context.exc)
        self.assertIs(context.exc[0], NameError)
        context = mycontext()
        context.catch = True
        with context:
            raise NameError('foo')
        self.assertIsNotNone(context.exc)
        self.assertIs(context.exc[0], NameError)

    def test_decorator(self):
        if False:
            return 10
        context = mycontext()

        @context
        def test():
            if False:
                while True:
                    i = 10
            self.assertIsNone(context.exc)
            self.assertTrue(context.started)
        test()
        self.assertEqual(context.exc, (None, None, None))

    def test_decorator_with_exception(self):
        if False:
            return 10
        context = mycontext()

        @context
        def test():
            if False:
                print('Hello World!')
            self.assertIsNone(context.exc)
            self.assertTrue(context.started)
            raise NameError('foo')
        with self.assertRaisesRegex(NameError, 'foo'):
            test()
        self.assertIsNotNone(context.exc)
        self.assertIs(context.exc[0], NameError)

    def test_decorating_method(self):
        if False:
            while True:
                i = 10
        context = mycontext()

        class Test(object):

            @context
            def method(self, a, b, c=None):
                if False:
                    for i in range(10):
                        print('nop')
                self.a = a
                self.b = b
                self.c = c
        test = Test()
        test.method(1, 2)
        self.assertEqual(test.a, 1)
        self.assertEqual(test.b, 2)
        self.assertEqual(test.c, None)
        test = Test()
        test.method('a', 'b', 'c')
        self.assertEqual(test.a, 'a')
        self.assertEqual(test.b, 'b')
        self.assertEqual(test.c, 'c')
        test = Test()
        test.method(a=1, b=2)
        self.assertEqual(test.a, 1)
        self.assertEqual(test.b, 2)

    def test_typo_enter(self):
        if False:
            return 10

        class mycontext(ContextDecorator):

            def __unter__(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def __exit__(self, *exc):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        with self.assertRaises(AttributeError):
            with mycontext():
                pass

    def test_typo_exit(self):
        if False:
            return 10

        class mycontext(ContextDecorator):

            def __enter__(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def __uxit__(self, *exc):
                if False:
                    while True:
                        i = 10
                pass
        with self.assertRaises(AttributeError):
            with mycontext():
                pass

    def test_contextdecorator_as_mixin(self):
        if False:
            i = 10
            return i + 15

        class somecontext(object):
            started = False
            exc = None

            def __enter__(self):
                if False:
                    while True:
                        i = 10
                self.started = True
                return self

            def __exit__(self, *exc):
                if False:
                    i = 10
                    return i + 15
                self.exc = exc

        class mycontext(somecontext, ContextDecorator):
            pass
        context = mycontext()

        @context
        def test():
            if False:
                print('Hello World!')
            self.assertIsNone(context.exc)
            self.assertTrue(context.started)
        test()
        self.assertEqual(context.exc, (None, None, None))

    def test_contextmanager_as_decorator(self):
        if False:
            print('Hello World!')

        @contextmanager
        def woohoo(y):
            if False:
                print('Hello World!')
            state.append(y)
            yield
            state.append(999)
        state = []

        @woohoo(1)
        def test(x):
            if False:
                return 10
            self.assertEqual(state, [1])
            state.append(x)
        test('something')
        self.assertEqual(state, [1, 'something', 999])
        state = []
        test('something else')
        self.assertEqual(state, [1, 'something else', 999])

class TestBaseExitStack:
    exit_stack = None

    @support.requires_docstrings
    def test_instance_docs(self):
        if False:
            while True:
                i = 10
        cm_docstring = self.exit_stack.__doc__
        obj = self.exit_stack()
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_no_resources(self):
        if False:
            for i in range(10):
                print('nop')
        with self.exit_stack():
            pass

    def test_callback(self):
        if False:
            for i in range(10):
                print('nop')
        expected = [((), {}), ((1,), {}), ((1, 2), {}), ((), dict(example=1)), ((1,), dict(example=1)), ((1, 2), dict(example=1)), ((1, 2), dict(self=3, callback=4))]
        result = []

        def _exit(*args, **kwds):
            if False:
                return 10
            'Test metadata propagation'
            result.append((args, kwds))
        with self.exit_stack() as stack:
            for (args, kwds) in reversed(expected):
                if args and kwds:
                    f = stack.callback(_exit, *args, **kwds)
                elif args:
                    f = stack.callback(_exit, *args)
                elif kwds:
                    f = stack.callback(_exit, **kwds)
                else:
                    f = stack.callback(_exit)
                self.assertIs(f, _exit)
            for wrapper in stack._exit_callbacks:
                self.assertIs(wrapper[1].__wrapped__, _exit)
                self.assertNotEqual(wrapper[1].__name__, _exit.__name__)
                self.assertIsNone(wrapper[1].__doc__, _exit.__doc__)
        self.assertEqual(result, expected)
        result = []
        with self.exit_stack() as stack:
            with self.assertRaises(TypeError):
                stack.callback(arg=1)
            with self.assertRaises(TypeError):
                self.exit_stack.callback(arg=2)
            with self.assertRaises(TypeError):
                stack.callback(callback=_exit, arg=3)
        self.assertEqual(result, [])

    def test_push(self):
        if False:
            for i in range(10):
                print('nop')
        exc_raised = ZeroDivisionError

        def _expect_exc(exc_type, exc, exc_tb):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIs(exc_type, exc_raised)

        def _suppress_exc(*exc_details):
            if False:
                for i in range(10):
                    print('nop')
            return True

        def _expect_ok(exc_type, exc, exc_tb):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsNone(exc_type)
            self.assertIsNone(exc)
            self.assertIsNone(exc_tb)

        class ExitCM(object):

            def __init__(self, check_exc):
                if False:
                    i = 10
                    return i + 15
                self.check_exc = check_exc

            def __enter__(self):
                if False:
                    print('Hello World!')
                self.fail('Should not be called!')

            def __exit__(self, *exc_details):
                if False:
                    for i in range(10):
                        print('nop')
                self.check_exc(*exc_details)
        with self.exit_stack() as stack:
            stack.push(_expect_ok)
            self.assertIs(stack._exit_callbacks[-1][1], _expect_ok)
            cm = ExitCM(_expect_ok)
            stack.push(cm)
            self.assertIs(stack._exit_callbacks[-1][1].__self__, cm)
            stack.push(_suppress_exc)
            self.assertIs(stack._exit_callbacks[-1][1], _suppress_exc)
            cm = ExitCM(_expect_exc)
            stack.push(cm)
            self.assertIs(stack._exit_callbacks[-1][1].__self__, cm)
            stack.push(_expect_exc)
            self.assertIs(stack._exit_callbacks[-1][1], _expect_exc)
            stack.push(_expect_exc)
            self.assertIs(stack._exit_callbacks[-1][1], _expect_exc)
            1 / 0

    def test_enter_context(self):
        if False:
            print('Hello World!')

        class TestCM(object):

            def __enter__(self):
                if False:
                    while True:
                        i = 10
                result.append(1)

            def __exit__(self, *exc_details):
                if False:
                    for i in range(10):
                        print('nop')
                result.append(3)
        result = []
        cm = TestCM()
        with self.exit_stack() as stack:

            @stack.callback
            def _exit():
                if False:
                    for i in range(10):
                        print('nop')
                result.append(4)
            self.assertIsNotNone(_exit)
            stack.enter_context(cm)
            self.assertIs(stack._exit_callbacks[-1][1].__self__, cm)
            result.append(2)
        self.assertEqual(result, [1, 2, 3, 4])

    def test_close(self):
        if False:
            print('Hello World!')
        result = []
        with self.exit_stack() as stack:

            @stack.callback
            def _exit():
                if False:
                    i = 10
                    return i + 15
                result.append(1)
            self.assertIsNotNone(_exit)
            stack.close()
            result.append(2)
        self.assertEqual(result, [1, 2])

    def test_pop_all(self):
        if False:
            while True:
                i = 10
        result = []
        with self.exit_stack() as stack:

            @stack.callback
            def _exit():
                if False:
                    print('Hello World!')
                result.append(3)
            self.assertIsNotNone(_exit)
            new_stack = stack.pop_all()
            result.append(1)
        result.append(2)
        new_stack.close()
        self.assertEqual(result, [1, 2, 3])

    def test_exit_raise(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ZeroDivisionError):
            with self.exit_stack() as stack:
                stack.push(lambda *exc: False)
                1 / 0

    def test_exit_suppress(self):
        if False:
            i = 10
            return i + 15
        with self.exit_stack() as stack:
            stack.push(lambda *exc: True)
            1 / 0

    def test_exit_exception_traceback(self):
        if False:
            print('Hello World!')

        def raise_exc(exc):
            if False:
                return 10
            raise exc
        try:
            with self.exit_stack() as stack:
                stack.callback(raise_exc, ValueError)
                1 / 0
        except ValueError as e:
            exc = e
        self.assertIsInstance(exc, ValueError)
        ve_frames = traceback.extract_tb(exc.__traceback__)
        expected = [('test_exit_exception_traceback', 'with self.exit_stack() as stack:')] + self.callback_error_internal_frames + [('_exit_wrapper', 'callback(*args, **kwds)'), ('raise_exc', 'raise exc')]
        self.assertEqual([(f.name, f.line) for f in ve_frames], expected)
        self.assertIsInstance(exc.__context__, ZeroDivisionError)
        zde_frames = traceback.extract_tb(exc.__context__.__traceback__)
        self.assertEqual([(f.name, f.line) for f in zde_frames], [('test_exit_exception_traceback', '1/0')])

    def test_exit_exception_chaining_reference(self):
        if False:
            while True:
                i = 10

        class RaiseExc:

            def __init__(self, exc):
                if False:
                    i = 10
                    return i + 15
                self.exc = exc

            def __enter__(self):
                if False:
                    i = 10
                    return i + 15
                return self

            def __exit__(self, *exc_details):
                if False:
                    while True:
                        i = 10
                raise self.exc

        class RaiseExcWithContext:

            def __init__(self, outer, inner):
                if False:
                    while True:
                        i = 10
                self.outer = outer
                self.inner = inner

            def __enter__(self):
                if False:
                    while True:
                        i = 10
                return self

            def __exit__(self, *exc_details):
                if False:
                    print('Hello World!')
                try:
                    raise self.inner
                except:
                    raise self.outer

        class SuppressExc:

            def __enter__(self):
                if False:
                    while True:
                        i = 10
                return self

            def __exit__(self, *exc_details):
                if False:
                    while True:
                        i = 10
                type(self).saved_details = exc_details
                return True
        try:
            with RaiseExc(IndexError):
                with RaiseExcWithContext(KeyError, AttributeError):
                    with SuppressExc():
                        with RaiseExc(ValueError):
                            1 / 0
        except IndexError as exc:
            self.assertIsInstance(exc.__context__, KeyError)
            self.assertIsInstance(exc.__context__.__context__, AttributeError)
            self.assertIsNone(exc.__context__.__context__.__context__)
        else:
            self.fail('Expected IndexError, but no exception was raised')
        inner_exc = SuppressExc.saved_details[1]
        self.assertIsInstance(inner_exc, ValueError)
        self.assertIsInstance(inner_exc.__context__, ZeroDivisionError)

    def test_exit_exception_chaining(self):
        if False:
            print('Hello World!')

        def raise_exc(exc):
            if False:
                i = 10
                return i + 15
            raise exc
        saved_details = None

        def suppress_exc(*exc_details):
            if False:
                while True:
                    i = 10
            nonlocal saved_details
            saved_details = exc_details
            return True
        try:
            with self.exit_stack() as stack:
                stack.callback(raise_exc, IndexError)
                stack.callback(raise_exc, KeyError)
                stack.callback(raise_exc, AttributeError)
                stack.push(suppress_exc)
                stack.callback(raise_exc, ValueError)
                1 / 0
        except IndexError as exc:
            self.assertIsInstance(exc.__context__, KeyError)
            self.assertIsInstance(exc.__context__.__context__, AttributeError)
            self.assertIsNone(exc.__context__.__context__.__context__)
        else:
            self.fail('Expected IndexError, but no exception was raised')
        inner_exc = saved_details[1]
        self.assertIsInstance(inner_exc, ValueError)
        self.assertIsInstance(inner_exc.__context__, ZeroDivisionError)

    def test_exit_exception_explicit_none_context(self):
        if False:
            for i in range(10):
                print('nop')

        class MyException(Exception):
            pass

        @contextmanager
        def my_cm():
            if False:
                return 10
            try:
                yield
            except BaseException:
                exc = MyException()
                try:
                    raise exc
                finally:
                    exc.__context__ = None

        @contextmanager
        def my_cm_with_exit_stack():
            if False:
                i = 10
                return i + 15
            with self.exit_stack() as stack:
                stack.enter_context(my_cm())
                yield stack
        for cm in (my_cm, my_cm_with_exit_stack):
            with self.subTest():
                try:
                    with cm():
                        raise IndexError()
                except MyException as exc:
                    self.assertIsNone(exc.__context__)
                else:
                    self.fail('Expected IndexError, but no exception was raised')

    def test_exit_exception_non_suppressing(self):
        if False:
            print('Hello World!')

        def raise_exc(exc):
            if False:
                for i in range(10):
                    print('nop')
            raise exc

        def suppress_exc(*exc_details):
            if False:
                i = 10
                return i + 15
            return True
        try:
            with self.exit_stack() as stack:
                stack.callback(lambda : None)
                stack.callback(raise_exc, IndexError)
        except Exception as exc:
            self.assertIsInstance(exc, IndexError)
        else:
            self.fail('Expected IndexError, but no exception was raised')
        try:
            with self.exit_stack() as stack:
                stack.callback(raise_exc, KeyError)
                stack.push(suppress_exc)
                stack.callback(raise_exc, IndexError)
        except Exception as exc:
            self.assertIsInstance(exc, KeyError)
        else:
            self.fail('Expected KeyError, but no exception was raised')

    def test_exit_exception_with_correct_context(self):
        if False:
            print('Hello World!')

        @contextmanager
        def gets_the_context_right(exc):
            if False:
                i = 10
                return i + 15
            try:
                yield
            finally:
                raise exc
        exc1 = Exception(1)
        exc2 = Exception(2)
        exc3 = Exception(3)
        exc4 = Exception(4)
        try:
            with self.exit_stack() as stack:
                stack.enter_context(gets_the_context_right(exc4))
                stack.enter_context(gets_the_context_right(exc3))
                stack.enter_context(gets_the_context_right(exc2))
                raise exc1
        except Exception as exc:
            self.assertIs(exc, exc4)
            self.assertIs(exc.__context__, exc3)
            self.assertIs(exc.__context__.__context__, exc2)
            self.assertIs(exc.__context__.__context__.__context__, exc1)
            self.assertIsNone(exc.__context__.__context__.__context__.__context__)

    def test_exit_exception_with_existing_context(self):
        if False:
            return 10

        def raise_nested(inner_exc, outer_exc):
            if False:
                return 10
            try:
                raise inner_exc
            finally:
                raise outer_exc
        exc1 = Exception(1)
        exc2 = Exception(2)
        exc3 = Exception(3)
        exc4 = Exception(4)
        exc5 = Exception(5)
        try:
            with self.exit_stack() as stack:
                stack.callback(raise_nested, exc4, exc5)
                stack.callback(raise_nested, exc2, exc3)
                raise exc1
        except Exception as exc:
            self.assertIs(exc, exc5)
            self.assertIs(exc.__context__, exc4)
            self.assertIs(exc.__context__.__context__, exc3)
            self.assertIs(exc.__context__.__context__.__context__, exc2)
            self.assertIs(exc.__context__.__context__.__context__.__context__, exc1)
            self.assertIsNone(exc.__context__.__context__.__context__.__context__.__context__)

    def test_body_exception_suppress(self):
        if False:
            return 10

        def suppress_exc(*exc_details):
            if False:
                while True:
                    i = 10
            return True
        try:
            with self.exit_stack() as stack:
                stack.push(suppress_exc)
                1 / 0
        except IndexError as exc:
            self.fail('Expected no exception, got IndexError')

    def test_exit_exception_chaining_suppress(self):
        if False:
            i = 10
            return i + 15
        with self.exit_stack() as stack:
            stack.push(lambda *exc: True)
            stack.push(lambda *exc: 1 / 0)
            stack.push(lambda *exc: {}[1])

    def test_excessive_nesting(self):
        if False:
            i = 10
            return i + 15
        with self.exit_stack() as stack:
            for i in range(10000):
                stack.callback(int)

    def test_instance_bypass(self):
        if False:
            for i in range(10):
                print('nop')

        class Example(object):
            pass
        cm = Example()
        cm.__exit__ = object()
        stack = self.exit_stack()
        self.assertRaises(AttributeError, stack.enter_context, cm)
        stack.push(cm)
        self.assertIs(stack._exit_callbacks[-1][1], cm)

    def test_dont_reraise_RuntimeError(self):
        if False:
            for i in range(10):
                print('nop')

        class UniqueException(Exception):
            pass

        class UniqueRuntimeError(RuntimeError):
            pass

        @contextmanager
        def second():
            if False:
                i = 10
                return i + 15
            try:
                yield 1
            except Exception as exc:
                raise UniqueException('new exception') from exc

        @contextmanager
        def first():
            if False:
                while True:
                    i = 10
            try:
                yield 1
            except Exception as exc:
                raise exc
        with self.assertRaises(UniqueException) as err_ctx:
            with self.exit_stack() as es_ctx:
                es_ctx.enter_context(second())
                es_ctx.enter_context(first())
                raise UniqueRuntimeError('please no infinite loop.')
        exc = err_ctx.exception
        self.assertIsInstance(exc, UniqueException)
        self.assertIsInstance(exc.__context__, UniqueRuntimeError)
        self.assertIsNone(exc.__context__.__context__)
        self.assertIsNone(exc.__context__.__cause__)
        self.assertIs(exc.__cause__, exc.__context__)

class TestExitStack(TestBaseExitStack, unittest.TestCase):
    exit_stack = ExitStack
    callback_error_internal_frames = [('__exit__', 'raise exc_details[1]'), ('__exit__', 'if cb(*exc_details):')]

class TestRedirectStream:
    redirect_stream = None
    orig_stream = None

    @support.requires_docstrings
    def test_instance_docs(self):
        if False:
            while True:
                i = 10
        cm_docstring = self.redirect_stream.__doc__
        obj = self.redirect_stream(None)
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_no_redirect_in_init(self):
        if False:
            while True:
                i = 10
        orig_stdout = getattr(sys, self.orig_stream)
        self.redirect_stream(None)
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)

    def test_redirect_to_string_io(self):
        if False:
            while True:
                i = 10
        f = io.StringIO()
        msg = 'Consider an API like help(), which prints directly to stdout'
        orig_stdout = getattr(sys, self.orig_stream)
        with self.redirect_stream(f):
            print(msg, file=getattr(sys, self.orig_stream))
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)
        s = f.getvalue().strip()
        self.assertEqual(s, msg)

    def test_enter_result_is_target(self):
        if False:
            for i in range(10):
                print('nop')
        f = io.StringIO()
        with self.redirect_stream(f) as enter_result:
            self.assertIs(enter_result, f)

    def test_cm_is_reusable(self):
        if False:
            i = 10
            return i + 15
        f = io.StringIO()
        write_to_f = self.redirect_stream(f)
        orig_stdout = getattr(sys, self.orig_stream)
        with write_to_f:
            print('Hello', end=' ', file=getattr(sys, self.orig_stream))
        with write_to_f:
            print('World!', file=getattr(sys, self.orig_stream))
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)
        s = f.getvalue()
        self.assertEqual(s, 'Hello World!\n')

    def test_cm_is_reentrant(self):
        if False:
            for i in range(10):
                print('nop')
        f = io.StringIO()
        write_to_f = self.redirect_stream(f)
        orig_stdout = getattr(sys, self.orig_stream)
        with write_to_f:
            print('Hello', end=' ', file=getattr(sys, self.orig_stream))
            with write_to_f:
                print('World!', file=getattr(sys, self.orig_stream))
        self.assertIs(getattr(sys, self.orig_stream), orig_stdout)
        s = f.getvalue()
        self.assertEqual(s, 'Hello World!\n')

class TestRedirectStdout(TestRedirectStream, unittest.TestCase):
    redirect_stream = redirect_stdout
    orig_stream = 'stdout'

class TestRedirectStderr(TestRedirectStream, unittest.TestCase):
    redirect_stream = redirect_stderr
    orig_stream = 'stderr'

class TestSuppress(unittest.TestCase):

    @support.requires_docstrings
    def test_instance_docs(self):
        if False:
            for i in range(10):
                print('nop')
        cm_docstring = suppress.__doc__
        obj = suppress()
        self.assertEqual(obj.__doc__, cm_docstring)

    def test_no_result_from_enter(self):
        if False:
            print('Hello World!')
        with suppress(ValueError) as enter_result:
            self.assertIsNone(enter_result)

    def test_no_exception(self):
        if False:
            while True:
                i = 10
        with suppress(ValueError):
            self.assertEqual(pow(2, 5), 32)

    def test_exact_exception(self):
        if False:
            for i in range(10):
                print('nop')
        with suppress(TypeError):
            len(5)

    def test_exception_hierarchy(self):
        if False:
            return 10
        with suppress(LookupError):
            'Hello'[50]

    def test_other_exception(self):
        if False:
            return 10
        with self.assertRaises(ZeroDivisionError):
            with suppress(TypeError):
                1 / 0

    def test_no_args(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ZeroDivisionError):
            with suppress():
                1 / 0

    def test_multiple_exception_args(self):
        if False:
            i = 10
            return i + 15
        with suppress(ZeroDivisionError, TypeError):
            1 / 0
        with suppress(ZeroDivisionError, TypeError):
            len(5)

    def test_cm_is_reentrant(self):
        if False:
            for i in range(10):
                print('nop')
        ignore_exceptions = suppress(Exception)
        with ignore_exceptions:
            pass
        with ignore_exceptions:
            len(5)
        with ignore_exceptions:
            with ignore_exceptions:
                len(5)
            outer_continued = True
            1 / 0
        self.assertTrue(outer_continued)
if __name__ == '__main__':
    unittest.main()