"""Tests for inspect_utils module."""
import abc
import collections
import functools
import imp
import textwrap
import unittest
import six
from nvidia.dali._autograph.pyct import inspect_utils
from nvidia.dali._autograph.pyct.testing import basic_definitions
from nvidia.dali._autograph.pyct.testing import decorators

def decorator(f):
    if False:
        while True:
            i = 10
    return f

def function_decorator():
    if False:
        i = 10
        return i + 15

    def dec(f):
        if False:
            while True:
                i = 10
        return f
    return dec

def wrapping_decorator():
    if False:
        i = 10
        return i + 15

    def dec(f):
        if False:
            print('Hello World!')

        def replacement(*_):
            if False:
                print('Hello World!')
            return None

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            return replacement(*args, **kwargs)
        return wrapper
    return dec

class TestClass(object):

    def member_function(self):
        if False:
            print('Hello World!')
        pass

    @decorator
    def decorated_member(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @function_decorator()
    def fn_decorated_member(self):
        if False:
            while True:
                i = 10
        pass

    @wrapping_decorator()
    def wrap_decorated_member(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def static_method():
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    def class_method(cls):
        if False:
            i = 10
            return i + 15
        pass

def free_function():
    if False:
        while True:
            i = 10
    pass

def factory():
    if False:
        return 10
    return free_function

def free_factory():
    if False:
        while True:
            i = 10

    def local_function():
        if False:
            return 10
        pass
    return local_function

class InspectUtilsTest(unittest.TestCase):

    def test_islambda(self):
        if False:
            while True:
                i = 10

        def test_fn():
            if False:
                while True:
                    i = 10
            pass
        self.assertTrue(inspect_utils.islambda(lambda x: x))
        self.assertFalse(inspect_utils.islambda(test_fn))

    def test_islambda_renamed_lambda(self):
        if False:
            for i in range(10):
                print('nop')
        l = lambda x: 1
        l.__name__ = 'f'
        self.assertTrue(inspect_utils.islambda(l))

    def test_isnamedtuple(self):
        if False:
            print('Hello World!')
        nt = collections.namedtuple('TestNamedTuple', ['a', 'b'])

        class NotANamedTuple(tuple):
            pass
        self.assertTrue(inspect_utils.isnamedtuple(nt))
        self.assertFalse(inspect_utils.isnamedtuple(NotANamedTuple))

    def test_isnamedtuple_confounder(self):
        if False:
            while True:
                i = 10
        'This test highlights false positives when detecting named tuples.'

        class NamedTupleLike(tuple):
            _fields = ('a', 'b')
        self.assertTrue(inspect_utils.isnamedtuple(NamedTupleLike))

    def test_isnamedtuple_subclass(self):
        if False:
            for i in range(10):
                print('nop')
        'This test highlights false positives when detecting named tuples.'

        class NamedTupleSubclass(collections.namedtuple('Test', ['a', 'b'])):
            pass
        self.assertTrue(inspect_utils.isnamedtuple(NamedTupleSubclass))

    def assertSourceIdentical(self, actual, expected):
        if False:
            i = 10
            return i + 15
        self.assertEqual(textwrap.dedent(actual).strip(), textwrap.dedent(expected).strip())

    def test_getimmediatesource_basic(self):
        if False:
            return 10

        def test_decorator(f):
            if False:
                while True:
                    i = 10

            def f_wrapper(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                return f(*args, **kwargs)
            return f_wrapper
        expected = '\n      def f_wrapper(*args, **kwargs):\n        return f(*args, **kwargs)\n    '

        @test_decorator
        def test_fn(a):
            if False:
                while True:
                    i = 10
            'Test docstring.'
            return [a]
        self.assertSourceIdentical(inspect_utils.getimmediatesource(test_fn), expected)

    def test_getimmediatesource_noop_decorator(self):
        if False:
            return 10

        def test_decorator(f):
            if False:
                i = 10
                return i + 15
            return f
        expected = '\n      @test_decorator\n      def test_fn(a):\n        """Test docstring."""\n        return [a]\n    '

        @test_decorator
        def test_fn(a):
            if False:
                print('Hello World!')
            'Test docstring.'
            return [a]
        self.assertSourceIdentical(inspect_utils.getimmediatesource(test_fn), expected)

    def test_getimmediatesource_functools_wrapper(self):
        if False:
            while True:
                i = 10

        def wrapper_decorator(f):
            if False:
                i = 10
                return i + 15

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                if False:
                    print('Hello World!')
                return f(*args, **kwargs)
            return wrapper
        expected = textwrap.dedent('\n      @functools.wraps(f)\n      def wrapper(*args, **kwargs):\n        return f(*args, **kwargs)\n    ')

        @wrapper_decorator
        def test_fn(a):
            if False:
                return 10
            'Test docstring.'
            return [a]
        self.assertSourceIdentical(inspect_utils.getimmediatesource(test_fn), expected)

    def test_getimmediatesource_functools_wrapper_different_module(self):
        if False:
            for i in range(10):
                print('nop')
        expected = textwrap.dedent('\n      @functools.wraps(f)\n      def wrapper(*args, **kwargs):\n        return f(*args, **kwargs)\n    ')

        @decorators.wrapping_decorator
        def test_fn(a):
            if False:
                for i in range(10):
                    print('nop')
            'Test docstring.'
            return [a]
        self.assertSourceIdentical(inspect_utils.getimmediatesource(test_fn), expected)

    def test_getimmediatesource_normal_decorator_different_module(self):
        if False:
            for i in range(10):
                print('nop')
        expected = textwrap.dedent('\n      def standalone_wrapper(*args, **kwargs):\n        return f(*args, **kwargs)\n    ')

        @decorators.standalone_decorator
        def test_fn(a):
            if False:
                while True:
                    i = 10
            'Test docstring.'
            return [a]
        self.assertSourceIdentical(inspect_utils.getimmediatesource(test_fn), expected)

    def test_getimmediatesource_normal_functional_decorator_different_module(self):
        if False:
            for i in range(10):
                print('nop')
        expected = textwrap.dedent('\n      def functional_wrapper(*args, **kwargs):\n        return f(*args, **kwargs)\n    ')

        @decorators.functional_decorator()
        def test_fn(a):
            if False:
                while True:
                    i = 10
            'Test docstring.'
            return [a]
        self.assertSourceIdentical(inspect_utils.getimmediatesource(test_fn), expected)

    def test_getnamespace_globals(self):
        if False:
            i = 10
            return i + 15
        ns = inspect_utils.getnamespace(factory)
        self.assertEqual(ns['free_function'], free_function)

    def test_getnamespace_closure_with_undefined_var(self):
        if False:
            print('Hello World!')
        if False:
            a = 1

        def test_fn():
            if False:
                i = 10
                return i + 15
            return a
        ns = inspect_utils.getnamespace(test_fn)
        self.assertNotIn('a', ns)
        a = 2
        ns = inspect_utils.getnamespace(test_fn)
        self.assertEqual(ns['a'], 2)

    def test_getnamespace_hermetic(self):
        if False:
            return 10
        free_function = object()

        def test_fn():
            if False:
                i = 10
                return i + 15
            return free_function
        ns = inspect_utils.getnamespace(test_fn)
        globs = six.get_function_globals(test_fn)
        self.assertTrue(ns['free_function'] is free_function)
        self.assertFalse(globs['free_function'] is free_function)

    def test_getnamespace_locals(self):
        if False:
            while True:
                i = 10

        def called_fn():
            if False:
                return 10
            return 0
        closed_over_list = []
        closed_over_primitive = 1

        def local_fn():
            if False:
                print('Hello World!')
            closed_over_list.append(1)
            local_var = 1
            return called_fn() + local_var + closed_over_primitive
        ns = inspect_utils.getnamespace(local_fn)
        self.assertEqual(ns['called_fn'], called_fn)
        self.assertEqual(ns['closed_over_list'], closed_over_list)
        self.assertEqual(ns['closed_over_primitive'], closed_over_primitive)
        self.assertTrue('local_var' not in ns)

    def test_getqualifiedname(self):
        if False:
            for i in range(10):
                print('nop')
        foo = object()
        qux = imp.new_module('quxmodule')
        bar = imp.new_module('barmodule')
        baz = object()
        bar.baz = baz
        ns = {'foo': foo, 'bar': bar, 'qux': qux}
        self.assertIsNone(inspect_utils.getqualifiedname(ns, inspect_utils))
        self.assertEqual(inspect_utils.getqualifiedname(ns, foo), 'foo')
        self.assertEqual(inspect_utils.getqualifiedname(ns, bar), 'bar')
        self.assertEqual(inspect_utils.getqualifiedname(ns, baz), 'bar.baz')

    def test_getqualifiedname_efficiency(self):
        if False:
            for i in range(10):
                print('nop')
        foo = object()
        ns = {}
        prev_level = []
        for i in range(10):
            current_level = []
            for j in range(10):
                mod_name = 'mod_{}_{}'.format(i, j)
                mod = imp.new_module(mod_name)
                current_level.append(mod)
                if i == 9 and j == 9:
                    mod.foo = foo
            if prev_level:
                for prev in prev_level:
                    for mod in current_level:
                        prev.__dict__[mod.__name__] = mod
            else:
                for mod in current_level:
                    ns[mod.__name__] = mod
            prev_level = current_level
        self.assertIsNone(inspect_utils.getqualifiedname(ns, inspect_utils))
        self.assertIsNotNone(inspect_utils.getqualifiedname(ns, foo, max_depth=10000000000))

    def test_getqualifiedname_cycles(self):
        if False:
            while True:
                i = 10
        foo = object()
        ns = {}
        mods = []
        for i in range(10):
            mod = imp.new_module('mod_{}'.format(i))
            if i == 9:
                mod.foo = foo
            if mods:
                mods[-1].__dict__[mod.__name__] = mod
            else:
                ns[mod.__name__] = mod
            for prev in mods:
                mod.__dict__[prev.__name__] = prev
            mods.append(mod)
        self.assertIsNone(inspect_utils.getqualifiedname(ns, inspect_utils))
        self.assertIsNotNone(inspect_utils.getqualifiedname(ns, foo, max_depth=10000000000))

    def test_getmethodclass(self):
        if False:
            return 10
        self.assertEqual(inspect_utils.getmethodclass(free_function), None)
        self.assertEqual(inspect_utils.getmethodclass(free_factory()), None)
        self.assertEqual(inspect_utils.getmethodclass(TestClass.member_function), TestClass)
        self.assertEqual(inspect_utils.getmethodclass(TestClass.decorated_member), TestClass)
        self.assertEqual(inspect_utils.getmethodclass(TestClass.fn_decorated_member), TestClass)
        self.assertEqual(inspect_utils.getmethodclass(TestClass.wrap_decorated_member), TestClass)
        self.assertEqual(inspect_utils.getmethodclass(TestClass.static_method), TestClass)
        self.assertEqual(inspect_utils.getmethodclass(TestClass.class_method), TestClass)
        test_obj = TestClass()
        self.assertEqual(inspect_utils.getmethodclass(test_obj.member_function), TestClass)
        self.assertEqual(inspect_utils.getmethodclass(test_obj.decorated_member), TestClass)
        self.assertEqual(inspect_utils.getmethodclass(test_obj.fn_decorated_member), TestClass)
        self.assertEqual(inspect_utils.getmethodclass(test_obj.wrap_decorated_member), TestClass)
        self.assertEqual(inspect_utils.getmethodclass(test_obj.static_method), TestClass)
        self.assertEqual(inspect_utils.getmethodclass(test_obj.class_method), TestClass)

    def test_getmethodclass_locals(self):
        if False:
            while True:
                i = 10

        def local_function():
            if False:
                i = 10
                return i + 15
            pass

        class LocalClass(object):

            def member_function(self):
                if False:
                    print('Hello World!')
                pass

            @decorator
            def decorated_member(self):
                if False:
                    return 10
                pass

            @function_decorator()
            def fn_decorated_member(self):
                if False:
                    while True:
                        i = 10
                pass

            @wrapping_decorator()
            def wrap_decorated_member(self):
                if False:
                    while True:
                        i = 10
                pass
        self.assertEqual(inspect_utils.getmethodclass(local_function), None)
        self.assertEqual(inspect_utils.getmethodclass(LocalClass.member_function), LocalClass)
        self.assertEqual(inspect_utils.getmethodclass(LocalClass.decorated_member), LocalClass)
        self.assertEqual(inspect_utils.getmethodclass(LocalClass.fn_decorated_member), LocalClass)
        self.assertEqual(inspect_utils.getmethodclass(LocalClass.wrap_decorated_member), LocalClass)
        test_obj = LocalClass()
        self.assertEqual(inspect_utils.getmethodclass(test_obj.member_function), LocalClass)
        self.assertEqual(inspect_utils.getmethodclass(test_obj.decorated_member), LocalClass)
        self.assertEqual(inspect_utils.getmethodclass(test_obj.fn_decorated_member), LocalClass)
        self.assertEqual(inspect_utils.getmethodclass(test_obj.wrap_decorated_member), LocalClass)

    def test_getmethodclass_callables(self):
        if False:
            print('Hello World!')

        class TestCallable(object):

            def __call__(self):
                if False:
                    return 10
                pass
        c = TestCallable()
        self.assertEqual(inspect_utils.getmethodclass(c), TestCallable)

    def test_getdefiningclass(self):
        if False:
            return 10

        class Superclass(object):

            def foo(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def bar(self):
                if False:
                    while True:
                        i = 10
                pass

            @classmethod
            def class_method(cls):
                if False:
                    print('Hello World!')
                pass

        class Subclass(Superclass):

            def foo(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def baz(self):
                if False:
                    i = 10
                    return i + 15
                pass
        self.assertIs(inspect_utils.getdefiningclass(Subclass.foo, Subclass), Subclass)
        self.assertIs(inspect_utils.getdefiningclass(Subclass.bar, Subclass), Superclass)
        self.assertIs(inspect_utils.getdefiningclass(Subclass.baz, Subclass), Subclass)
        self.assertIs(inspect_utils.getdefiningclass(Subclass.class_method, Subclass), Superclass)

    def test_isbuiltin(self):
        if False:
            print('Hello World!')
        self.assertTrue(inspect_utils.isbuiltin(enumerate))
        self.assertTrue(inspect_utils.isbuiltin(eval))
        self.assertTrue(inspect_utils.isbuiltin(float))
        self.assertTrue(inspect_utils.isbuiltin(int))
        self.assertTrue(inspect_utils.isbuiltin(len))
        self.assertTrue(inspect_utils.isbuiltin(range))
        self.assertTrue(inspect_utils.isbuiltin(zip))
        self.assertFalse(inspect_utils.isbuiltin(function_decorator))

    def test_isconstructor(self):
        if False:
            print('Hello World!')

        class OrdinaryClass(object):
            pass

        class OrdinaryCallableClass(object):

            def __call__(self):
                if False:
                    return 10
                pass

        class Metaclass(type):
            pass

        class CallableMetaclass(type):

            def __call__(cls):
                if False:
                    return 10
                pass
        self.assertTrue(inspect_utils.isconstructor(OrdinaryClass))
        self.assertTrue(inspect_utils.isconstructor(OrdinaryCallableClass))
        self.assertTrue(inspect_utils.isconstructor(Metaclass))
        self.assertTrue(inspect_utils.isconstructor(Metaclass('TestClass', (), {})))
        self.assertTrue(inspect_utils.isconstructor(CallableMetaclass))
        self.assertFalse(inspect_utils.isconstructor(CallableMetaclass('TestClass', (), {})))

    def test_isconstructor_abc_callable(self):
        if False:
            return 10

        @six.add_metaclass(abc.ABCMeta)
        class AbcBase(object):

            @abc.abstractmethod
            def __call__(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

        class AbcSubclass(AbcBase):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                pass

            def __call__(self):
                if False:
                    return 10
                pass
        self.assertTrue(inspect_utils.isconstructor(AbcBase))
        self.assertTrue(inspect_utils.isconstructor(AbcSubclass))

    def test_getfutureimports_functions(self):
        if False:
            i = 10
            return i + 15
        imps = inspect_utils.getfutureimports(basic_definitions.function_with_print)
        self.assertNotIn('absolute_import', imps)
        self.assertNotIn('division', imps)
        self.assertNotIn('print_function', imps)
        self.assertNotIn('generators', imps)

    def test_getfutureimports_lambdas(self):
        if False:
            while True:
                i = 10
        imps = inspect_utils.getfutureimports(basic_definitions.simple_lambda)
        self.assertNotIn('absolute_import', imps)
        self.assertNotIn('division', imps)
        self.assertNotIn('print_function', imps)
        self.assertNotIn('generators', imps)

    def test_getfutureimports_methods(self):
        if False:
            return 10
        imps = inspect_utils.getfutureimports(basic_definitions.SimpleClass.method_with_print)
        self.assertNotIn('absolute_import', imps)
        self.assertNotIn('division', imps)
        self.assertNotIn('print_function', imps)
        self.assertNotIn('generators', imps)