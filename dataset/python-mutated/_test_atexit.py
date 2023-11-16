"""
Tests run by test_atexit in a subprocess since it clears atexit callbacks.
"""
import atexit
import sys
import unittest
from test import support

class GeneralTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        atexit._clear()

    def tearDown(self):
        if False:
            while True:
                i = 10
        atexit._clear()

    def assert_raises_unraisable(self, exc_type, func, *args):
        if False:
            i = 10
            return i + 15
        with support.catch_unraisable_exception() as cm:
            atexit.register(func, *args)
            atexit._run_exitfuncs()
            self.assertEqual(cm.unraisable.object, func)
            self.assertEqual(cm.unraisable.exc_type, exc_type)
            self.assertEqual(type(cm.unraisable.exc_value), exc_type)

    def test_order(self):
        if False:
            return 10
        calls = []

        def func1(*args, **kwargs):
            if False:
                print('Hello World!')
            calls.append(('func1', args, kwargs))

        def func2(*args, **kwargs):
            if False:
                while True:
                    i = 10
            calls.append(('func2', args, kwargs))
        atexit.register(func1, 1, 2)
        atexit.register(func2)
        atexit.register(func2, 3, key='value')
        atexit._run_exitfuncs()
        self.assertEqual(calls, [('func2', (3,), {'key': 'value'}), ('func2', (), {}), ('func1', (1, 2), {})])

    def test_badargs(self):
        if False:
            while True:
                i = 10

        def func():
            if False:
                return 10
            pass
        self.assert_raises_unraisable(TypeError, func, 1, 2)

    def test_raise(self):
        if False:
            i = 10
            return i + 15

        def raise_type_error():
            if False:
                while True:
                    i = 10
            raise TypeError
        self.assert_raises_unraisable(TypeError, raise_type_error)

    def test_raise_unnormalized(self):
        if False:
            i = 10
            return i + 15

        def div_zero():
            if False:
                print('Hello World!')
            1 / 0
        self.assert_raises_unraisable(ZeroDivisionError, div_zero)

    def test_exit(self):
        if False:
            print('Hello World!')
        self.assert_raises_unraisable(SystemExit, sys.exit)

    def test_stress(self):
        if False:
            for i in range(10):
                print('nop')
        a = [0]

        def inc():
            if False:
                i = 10
                return i + 15
            a[0] += 1
        for i in range(128):
            atexit.register(inc)
        atexit._run_exitfuncs()
        self.assertEqual(a[0], 128)

    def test_clear(self):
        if False:
            return 10
        a = [0]

        def inc():
            if False:
                i = 10
                return i + 15
            a[0] += 1
        atexit.register(inc)
        atexit._clear()
        atexit._run_exitfuncs()
        self.assertEqual(a[0], 0)

    def test_unregister(self):
        if False:
            return 10
        a = [0]

        def inc():
            if False:
                while True:
                    i = 10
            a[0] += 1

        def dec():
            if False:
                for i in range(10):
                    print('nop')
            a[0] -= 1
        for i in range(4):
            atexit.register(inc)
        atexit.register(dec)
        atexit.unregister(inc)
        atexit._run_exitfuncs()
        self.assertEqual(a[0], -1)

    def test_bound_methods(self):
        if False:
            while True:
                i = 10
        l = []
        atexit.register(l.append, 5)
        atexit._run_exitfuncs()
        self.assertEqual(l, [5])
        atexit.unregister(l.append)
        atexit._run_exitfuncs()
        self.assertEqual(l, [5])

    def test_atexit_with_unregistered_function(self):
        if False:
            while True:
                i = 10

        def func():
            if False:
                while True:
                    i = 10
            atexit.unregister(func)
            1 / 0
        atexit.register(func)
        try:
            with support.catch_unraisable_exception() as cm:
                atexit._run_exitfuncs()
                self.assertEqual(cm.unraisable.object, func)
                self.assertEqual(cm.unraisable.exc_type, ZeroDivisionError)
                self.assertEqual(type(cm.unraisable.exc_value), ZeroDivisionError)
        finally:
            atexit.unregister(func)
if __name__ == '__main__':
    unittest.main()