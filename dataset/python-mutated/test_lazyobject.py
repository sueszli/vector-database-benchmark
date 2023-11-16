import copy
import pickle
import sys
import unittest
import warnings
from django.test import TestCase
from django.utils.functional import LazyObject, SimpleLazyObject, empty
from .models import Category, CategoryInfo

class Foo:
    """
    A simple class with just one attribute.
    """
    foo = 'bar'

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.foo == other.foo

class LazyObjectTestCase(unittest.TestCase):

    def lazy_wrap(self, wrapped_object):
        if False:
            while True:
                i = 10
        '\n        Wrap the given object into a LazyObject\n        '

        class AdHocLazyObject(LazyObject):

            def _setup(self):
                if False:
                    print('Hello World!')
                self._wrapped = wrapped_object
        return AdHocLazyObject()

    def test_getattribute(self):
        if False:
            i = 10
            return i + 15
        "\n        Proxy methods don't exist on wrapped objects unless they're set.\n        "
        attrs = ['__getitem__', '__setitem__', '__delitem__', '__iter__', '__len__', '__contains__']
        foo = Foo()
        obj = self.lazy_wrap(foo)
        for attr in attrs:
            with self.subTest(attr):
                self.assertFalse(hasattr(obj, attr))
                setattr(foo, attr, attr)
                obj_with_attr = self.lazy_wrap(foo)
                self.assertTrue(hasattr(obj_with_attr, attr))
                self.assertEqual(getattr(obj_with_attr, attr), attr)

    def test_getattr(self):
        if False:
            i = 10
            return i + 15
        obj = self.lazy_wrap(Foo())
        self.assertEqual(obj.foo, 'bar')

    def test_getattr_falsey(self):
        if False:
            while True:
                i = 10

        class Thing:

            def __getattr__(self, key):
                if False:
                    while True:
                        i = 10
                return []
        obj = self.lazy_wrap(Thing())
        self.assertEqual(obj.main, [])

    def test_setattr(self):
        if False:
            return 10
        obj = self.lazy_wrap(Foo())
        obj.foo = 'BAR'
        obj.bar = 'baz'
        self.assertEqual(obj.foo, 'BAR')
        self.assertEqual(obj.bar, 'baz')

    def test_setattr2(self):
        if False:
            print('Hello World!')
        obj = self.lazy_wrap(Foo())
        obj.bar = 'baz'
        obj.foo = 'BAR'
        self.assertEqual(obj.foo, 'BAR')
        self.assertEqual(obj.bar, 'baz')

    def test_delattr(self):
        if False:
            return 10
        obj = self.lazy_wrap(Foo())
        obj.bar = 'baz'
        self.assertEqual(obj.bar, 'baz')
        del obj.bar
        with self.assertRaises(AttributeError):
            obj.bar

    def test_cmp(self):
        if False:
            print('Hello World!')
        obj1 = self.lazy_wrap('foo')
        obj2 = self.lazy_wrap('bar')
        obj3 = self.lazy_wrap('foo')
        self.assertEqual(obj1, 'foo')
        self.assertEqual(obj1, obj3)
        self.assertNotEqual(obj1, obj2)
        self.assertNotEqual(obj1, 'bar')

    def test_lt(self):
        if False:
            i = 10
            return i + 15
        obj1 = self.lazy_wrap(1)
        obj2 = self.lazy_wrap(2)
        self.assertLess(obj1, obj2)

    def test_gt(self):
        if False:
            print('Hello World!')
        obj1 = self.lazy_wrap(1)
        obj2 = self.lazy_wrap(2)
        self.assertGreater(obj2, obj1)

    def test_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        obj = self.lazy_wrap(b'foo')
        self.assertEqual(bytes(obj), b'foo')

    def test_text(self):
        if False:
            return 10
        obj = self.lazy_wrap('foo')
        self.assertEqual(str(obj), 'foo')

    def test_bool(self):
        if False:
            print('Hello World!')
        for f in [False, 0, (), {}, [], None, set()]:
            self.assertFalse(self.lazy_wrap(f))
        for t in [True, 1, (1,), {1: 2}, [1], object(), {1}]:
            self.assertTrue(t)

    def test_dir(self):
        if False:
            for i in range(10):
                print('nop')
        obj = self.lazy_wrap('foo')
        self.assertEqual(dir(obj), dir('foo'))

    def test_len(self):
        if False:
            print('Hello World!')
        for seq in ['asd', [1, 2, 3], {'a': 1, 'b': 2, 'c': 3}]:
            obj = self.lazy_wrap(seq)
            self.assertEqual(len(obj), 3)

    def test_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(self.lazy_wrap(42), int)

        class Bar(Foo):
            pass
        self.assertIsInstance(self.lazy_wrap(Bar()), Foo)

    def test_hash(self):
        if False:
            while True:
                i = 10
        obj = self.lazy_wrap('foo')
        d = {obj: 'bar'}
        self.assertIn('foo', d)
        self.assertEqual(d['foo'], 'bar')

    def test_contains(self):
        if False:
            return 10
        test_data = [('c', 'abcde'), (2, [1, 2, 3]), ('a', {'a': 1, 'b': 2, 'c': 3}), (2, {1, 2, 3})]
        for (needle, haystack) in test_data:
            self.assertIn(needle, self.lazy_wrap(haystack))
        for needle_haystack in test_data[1:]:
            self.assertIn(self.lazy_wrap(needle), haystack)
            self.assertIn(self.lazy_wrap(needle), self.lazy_wrap(haystack))

    def test_getitem(self):
        if False:
            print('Hello World!')
        obj_list = self.lazy_wrap([1, 2, 3])
        obj_dict = self.lazy_wrap({'a': 1, 'b': 2, 'c': 3})
        self.assertEqual(obj_list[0], 1)
        self.assertEqual(obj_list[-1], 3)
        self.assertEqual(obj_list[1:2], [2])
        self.assertEqual(obj_dict['b'], 2)
        with self.assertRaises(IndexError):
            obj_list[3]
        with self.assertRaises(KeyError):
            obj_dict['f']

    def test_setitem(self):
        if False:
            while True:
                i = 10
        obj_list = self.lazy_wrap([1, 2, 3])
        obj_dict = self.lazy_wrap({'a': 1, 'b': 2, 'c': 3})
        obj_list[0] = 100
        self.assertEqual(obj_list, [100, 2, 3])
        obj_list[1:2] = [200, 300, 400]
        self.assertEqual(obj_list, [100, 200, 300, 400, 3])
        obj_dict['a'] = 100
        obj_dict['d'] = 400
        self.assertEqual(obj_dict, {'a': 100, 'b': 2, 'c': 3, 'd': 400})

    def test_delitem(self):
        if False:
            while True:
                i = 10
        obj_list = self.lazy_wrap([1, 2, 3])
        obj_dict = self.lazy_wrap({'a': 1, 'b': 2, 'c': 3})
        del obj_list[-1]
        del obj_dict['c']
        self.assertEqual(obj_list, [1, 2])
        self.assertEqual(obj_dict, {'a': 1, 'b': 2})
        with self.assertRaises(IndexError):
            del obj_list[3]
        with self.assertRaises(KeyError):
            del obj_dict['f']

    def test_iter(self):
        if False:
            for i in range(10):
                print('nop')

        class IterObject:

            def __init__(self, values):
                if False:
                    while True:
                        i = 10
                self.values = values

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                return iter(self.values)
        original_list = ['test', '123']
        self.assertEqual(list(self.lazy_wrap(IterObject(original_list))), original_list)

    def test_pickle(self):
        if False:
            return 10
        obj = self.lazy_wrap(Foo())
        obj.bar = 'baz'
        pickled = pickle.dumps(obj)
        unpickled = pickle.loads(pickled)
        self.assertIsInstance(unpickled, Foo)
        self.assertEqual(unpickled, obj)
        self.assertEqual(unpickled.foo, obj.foo)
        self.assertEqual(unpickled.bar, obj.bar)

    def test_copy_list(self):
        if False:
            for i in range(10):
                print('nop')
        lst = [1, 2, 3]
        obj = self.lazy_wrap(lst)
        len(lst)
        obj2 = copy.copy(obj)
        self.assertIsNot(obj, obj2)
        self.assertIsInstance(obj2, list)
        self.assertEqual(obj2, [1, 2, 3])

    def test_copy_list_no_evaluation(self):
        if False:
            print('Hello World!')
        lst = [1, 2, 3]
        obj = self.lazy_wrap(lst)
        obj2 = copy.copy(obj)
        self.assertIsNot(obj, obj2)
        self.assertIs(obj._wrapped, empty)
        self.assertIs(obj2._wrapped, empty)

    def test_copy_class(self):
        if False:
            while True:
                i = 10
        foo = Foo()
        obj = self.lazy_wrap(foo)
        str(foo)
        obj2 = copy.copy(obj)
        self.assertIsNot(obj, obj2)
        self.assertIsInstance(obj2, Foo)
        self.assertEqual(obj2, Foo())

    def test_copy_class_no_evaluation(self):
        if False:
            i = 10
            return i + 15
        foo = Foo()
        obj = self.lazy_wrap(foo)
        obj2 = copy.copy(obj)
        self.assertIsNot(obj, obj2)
        self.assertIs(obj._wrapped, empty)
        self.assertIs(obj2._wrapped, empty)

    def test_deepcopy_list(self):
        if False:
            while True:
                i = 10
        lst = [1, 2, 3]
        obj = self.lazy_wrap(lst)
        len(lst)
        obj2 = copy.deepcopy(obj)
        self.assertIsNot(obj, obj2)
        self.assertIsInstance(obj2, list)
        self.assertEqual(obj2, [1, 2, 3])

    def test_deepcopy_list_no_evaluation(self):
        if False:
            while True:
                i = 10
        lst = [1, 2, 3]
        obj = self.lazy_wrap(lst)
        obj2 = copy.deepcopy(obj)
        self.assertIsNot(obj, obj2)
        self.assertIs(obj._wrapped, empty)
        self.assertIs(obj2._wrapped, empty)

    def test_deepcopy_class(self):
        if False:
            print('Hello World!')
        foo = Foo()
        obj = self.lazy_wrap(foo)
        str(foo)
        obj2 = copy.deepcopy(obj)
        self.assertIsNot(obj, obj2)
        self.assertIsInstance(obj2, Foo)
        self.assertEqual(obj2, Foo())

    def test_deepcopy_class_no_evaluation(self):
        if False:
            i = 10
            return i + 15
        foo = Foo()
        obj = self.lazy_wrap(foo)
        obj2 = copy.deepcopy(obj)
        self.assertIsNot(obj, obj2)
        self.assertIs(obj._wrapped, empty)
        self.assertIs(obj2._wrapped, empty)

class SimpleLazyObjectTestCase(LazyObjectTestCase):

    def lazy_wrap(self, wrapped_object):
        if False:
            i = 10
            return i + 15
        return SimpleLazyObject(lambda : wrapped_object)

    def test_repr(self):
        if False:
            print('Hello World!')
        obj = self.lazy_wrap(42)
        self.assertRegex(repr(obj), '^<SimpleLazyObject:')
        self.assertIs(obj._wrapped, empty)
        self.assertEqual(obj, 42)
        self.assertIsInstance(obj._wrapped, int)
        self.assertEqual(repr(obj), '<SimpleLazyObject: 42>')

    def test_add(self):
        if False:
            i = 10
            return i + 15
        obj1 = self.lazy_wrap(1)
        self.assertEqual(obj1 + 1, 2)
        obj2 = self.lazy_wrap(2)
        self.assertEqual(obj2 + obj1, 3)
        self.assertEqual(obj1 + obj2, 3)

    def test_radd(self):
        if False:
            print('Hello World!')
        obj1 = self.lazy_wrap(1)
        self.assertEqual(1 + obj1, 2)

    def test_trace(self):
        if False:
            for i in range(10):
                print('nop')
        old_trace_func = sys.gettrace()
        try:

            def trace_func(frame, event, arg):
                if False:
                    return 10
                frame.f_locals['self'].__class__
                if old_trace_func is not None:
                    old_trace_func(frame, event, arg)
            sys.settrace(trace_func)
            self.lazy_wrap(None)
        finally:
            sys.settrace(old_trace_func)

    def test_none(self):
        if False:
            for i in range(10):
                print('nop')
        i = [0]

        def f():
            if False:
                for i in range(10):
                    print('nop')
            i[0] += 1
            return None
        x = SimpleLazyObject(f)
        self.assertEqual(str(x), 'None')
        self.assertEqual(i, [1])
        self.assertEqual(str(x), 'None')
        self.assertEqual(i, [1])

    def test_dict(self):
        if False:
            for i in range(10):
                print('nop')
        lazydict = SimpleLazyObject(lambda : {'one': 1})
        self.assertEqual(lazydict['one'], 1)
        lazydict['one'] = -1
        self.assertEqual(lazydict['one'], -1)
        self.assertIn('one', lazydict)
        self.assertNotIn('two', lazydict)
        self.assertEqual(len(lazydict), 1)
        del lazydict['one']
        with self.assertRaises(KeyError):
            lazydict['one']

    def test_list_set(self):
        if False:
            return 10
        lazy_list = SimpleLazyObject(lambda : [1, 2, 3, 4, 5])
        lazy_set = SimpleLazyObject(lambda : {1, 2, 3, 4})
        self.assertIn(1, lazy_list)
        self.assertIn(1, lazy_set)
        self.assertNotIn(6, lazy_list)
        self.assertNotIn(6, lazy_set)
        self.assertEqual(len(lazy_list), 5)
        self.assertEqual(len(lazy_set), 4)

class BaseBaz:
    """
    A base class with a funky __reduce__ method, meant to simulate the
    __reduce__ method of Model, which sets self._django_version.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.baz = 'wrong'

    def __reduce__(self):
        if False:
            while True:
                i = 10
        self.baz = 'right'
        return super().__reduce__()

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if self.__class__ != other.__class__:
            return False
        for attr in ['bar', 'baz', 'quux']:
            if hasattr(self, attr) != hasattr(other, attr):
                return False
            elif getattr(self, attr, None) != getattr(other, attr, None):
                return False
        return True

class Baz(BaseBaz):
    """
    A class that inherits from BaseBaz and has its own __reduce_ex__ method.
    """

    def __init__(self, bar):
        if False:
            while True:
                i = 10
        self.bar = bar
        super().__init__()

    def __reduce_ex__(self, proto):
        if False:
            while True:
                i = 10
        self.quux = 'quux'
        return super().__reduce_ex__(proto)

class BazProxy(Baz):
    """
    A class that acts as a proxy for Baz. It does some scary mucking about with
    dicts, which simulates some crazy things that people might do with
    e.g. proxy models.
    """

    def __init__(self, baz):
        if False:
            i = 10
            return i + 15
        self.__dict__ = baz.__dict__
        self._baz = baz
        super(BaseBaz, self).__init__()

class SimpleLazyObjectPickleTestCase(TestCase):
    """
    Regression test for pickling a SimpleLazyObject wrapping a model (#25389).
    Also covers other classes with a custom __reduce__ method.
    """

    def test_pickle_with_reduce(self):
        if False:
            while True:
                i = 10
        '\n        Test in a fairly synthetic setting.\n        '
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            lazy_objs = [SimpleLazyObject(lambda : BaseBaz()), SimpleLazyObject(lambda : Baz(1)), SimpleLazyObject(lambda : BazProxy(Baz(2)))]
            for obj in lazy_objs:
                pickled = pickle.dumps(obj, protocol)
                unpickled = pickle.loads(pickled)
                self.assertEqual(unpickled, obj)
                self.assertEqual(unpickled.baz, 'right')

    def test_pickle_model(self):
        if False:
            return 10
        '\n        Test on an actual model, based on the report in #25426.\n        '
        category = Category.objects.create(name='thing1')
        CategoryInfo.objects.create(category=category)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            lazy_category = SimpleLazyObject(lambda : category)
            lazy_category.categoryinfo
            lazy_category_2 = SimpleLazyObject(lambda : category)
            with warnings.catch_warnings(record=True) as recorded:
                self.assertEqual(pickle.loads(pickle.dumps(lazy_category, protocol)), category)
                self.assertEqual(pickle.loads(pickle.dumps(lazy_category_2, protocol)), category)
                self.assertEqual(len(recorded), 0)