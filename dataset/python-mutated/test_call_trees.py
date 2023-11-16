"""Tests for call_trees module."""
import imp
from nvidia.dali._autograph.converters import call_trees
from nvidia.dali._autograph.converters import functions
from nvidia.dali._autograph.core import converter_testing

class MockConvertedCall(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.calls = []

    def __call__(self, f, args, kwargs, caller_fn_scope=None, options=None):
        if False:
            return 10
        del caller_fn_scope, options
        self.calls.append((args, kwargs))
        kwargs = kwargs or {}
        return f(*args, **kwargs)

class CallTreesTest(converter_testing.TestCase):

    def _transform_with_mock(self, f):
        if False:
            print('Hello World!')
        mock = MockConvertedCall()
        tr = self.transform(f, (functions, call_trees), ag_overrides={'converted_call': mock})
        return (tr, mock)

    def test_function_no_args(self):
        if False:
            while True:
                i = 10

        def f(f):
            if False:
                print('Hello World!')
            return f() + 20
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(lambda : 1), 21)
        self.assertListEqual(mock.calls, [((), None)])

    def test_function_with_expression_in_argument(self):
        if False:
            for i in range(10):
                print('nop')

        def f(f, g):
            if False:
                return 10
            return f(g() + 20) + 4000
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(lambda x: x + 300, lambda : 1), 4321)
        self.assertListEqual(mock.calls, [((), None), ((21,), None)])

    def test_function_with_call_in_argument(self):
        if False:
            for i in range(10):
                print('nop')

        def f(f, g):
            if False:
                return 10
            return f(g()) + 300
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(lambda x: x + 20, lambda : 1), 321)
        self.assertListEqual(mock.calls, [((), None), ((1,), None)])

    def test_function_chaining(self):
        if False:
            i = 10
            return i + 15

        def get_one():
            if False:
                return 10
            return 1

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return get_one().__add__(20)
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(), 21)
        self.assertListEqual(mock.calls, [((), None), ((20,), None)])

    def test_function_with_single_arg(self):
        if False:
            return 10

        def f(f, a):
            if False:
                return 10
            return f(a) + 20
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(lambda a: a, 1), 21)
        self.assertListEqual(mock.calls, [((1,), None)])

    def test_function_with_args_only(self):
        if False:
            while True:
                i = 10

        def f(f, a, b):
            if False:
                print('Hello World!')
            return f(a, b) + 300
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(lambda a, b: a + b, 1, 20), 321)
        self.assertListEqual(mock.calls, [((1, 20), None)])

    def test_function_with_kwarg(self):
        if False:
            return 10

        def f(f, a, b):
            if False:
                for i in range(10):
                    print('nop')
            return f(a, c=b) + 300
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(lambda a, c: a + c, 1, 20), 321)
        self.assertListEqual(mock.calls, [((1,), {'c': 20})])

    def test_function_with_kwargs_starargs(self):
        if False:
            return 10

        def f(f, a, *args, **kwargs):
            if False:
                while True:
                    i = 10
            return f(a, *args, **kwargs) + 5
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(lambda *args, **kwargs: 7, 1, *[2, 3], **{'b': 4, 'c': 5}), 12)
        self.assertListEqual(mock.calls, [((1, 2, 3), {'b': 4, 'c': 5})])

    def test_function_with_starargs_only(self):
        if False:
            return 10

        def g(*args):
            if False:
                while True:
                    i = 10
            return sum(args)

        def f():
            if False:
                return 10
            args = [1, 20, 300]
            return g(*args) + 4000
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(), 4321)
        self.assertListEqual(mock.calls, [((1, 20, 300), None)])

    def test_function_with_starargs_mixed(self):
        if False:
            print('Hello World!')

        def g(a, b, c, d):
            if False:
                for i in range(10):
                    print('nop')
            return a * 1000 + b * 100 + c * 10 + d

        def f():
            if False:
                i = 10
                return i + 15
            args1 = (1,)
            args2 = [3]
            return g(*args1, 2, *args2, 4)
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(), 1234)
        self.assertListEqual(mock.calls, [((1, 2, 3, 4), None)])

    def test_function_with_kwargs_keywords(self):
        if False:
            return 10

        def f(f, a, b, **kwargs):
            if False:
                return 10
            return f(a, b=b, **kwargs) + 5
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(lambda *args, **kwargs: 7, 1, 2, **{'c': 3}), 12)
        self.assertListEqual(mock.calls, [((1,), {'b': 2, 'c': 3})])

    def test_function_with_multiple_kwargs(self):
        if False:
            print('Hello World!')

        def f(f, a, b, c, kwargs1, kwargs2):
            if False:
                while True:
                    i = 10
            return f(a, b=b, **kwargs1, c=c, **kwargs2) + 5
        (tr, mock) = self._transform_with_mock(f)
        self.assertEqual(tr(lambda *args, **kwargs: 7, 1, 2, 3, {'d': 4}, {'e': 5}), 12)
        self.assertListEqual(mock.calls, [((1,), {'b': 2, 'c': 3, 'd': 4, 'e': 5})])

    def test_function_with_call_in_lambda_argument(self):
        if False:
            while True:
                i = 10

        def h(l, a):
            if False:
                while True:
                    i = 10
            return l(a) + 4000

        def g(a, *args):
            if False:
                for i in range(10):
                    print('nop')
            return a + sum(args)

        def f(h, g, a, *args):
            if False:
                print('Hello World!')
            return h(lambda x: g(x, *args), a)
        (tr, _) = self._transform_with_mock(f)
        self.assertEqual(tr(h, g, 1, *(20, 300)), 4321)

    def test_debugger_set_trace(self):
        if False:
            for i in range(10):
                print('nop')
        tracking_list = []
        pdb = imp.new_module('fake_pdb')
        pdb.set_trace = lambda : tracking_list.append(1)

        def f():
            if False:
                while True:
                    i = 10
            return pdb.set_trace()
        (tr, _) = self._transform_with_mock(f)
        tr()
        self.assertListEqual(tracking_list, [1])

    def test_class_method(self):
        if False:
            i = 10
            return i + 15

        class TestClass(object):

            def other_method(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + 20

            def test_method(self, a):
                if False:
                    while True:
                        i = 10
                return self.other_method(a) + 300
        tc = TestClass()
        (tr, mock) = self._transform_with_mock(TestClass.test_method)
        self.assertEqual(321, tr(tc, 1))
        self.assertListEqual(mock.calls, [((1,), None)])

    def test_object_method(self):
        if False:
            i = 10
            return i + 15

        class TestClass(object):

            def other_method(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + 20

            def test_method(self, a):
                if False:
                    while True:
                        i = 10
                return self.other_method(a) + 300
        tc = TestClass()
        (tr, mock) = self._transform_with_mock(tc.test_method)
        self.assertEqual(321, tr(tc, 1))
        self.assertListEqual(mock.calls, [((1,), None)])