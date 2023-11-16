import os
import sys
from collections import OrderedDict
import unittest
from unittest.test.testmock import support
from unittest.test.testmock.support import SomeClass, is_instance
from test.test_importlib.util import uncache
from unittest.mock import NonCallableMock, CallableMixin, sentinel, MagicMock, Mock, NonCallableMagicMock, patch, _patch, DEFAULT, call, _get_target
builtin_string = 'builtins'
PTModule = sys.modules[__name__]
MODNAME = '%s.PTModule' % __name__

def _get_proxy(obj, get_only=True):
    if False:
        i = 10
        return i + 15

    class Proxy(object):

        def __getattr__(self, name):
            if False:
                return 10
            return getattr(obj, name)
    if not get_only:

        def __setattr__(self, name, value):
            if False:
                i = 10
                return i + 15
            setattr(obj, name, value)

        def __delattr__(self, name):
            if False:
                i = 10
                return i + 15
            delattr(obj, name)
        Proxy.__setattr__ = __setattr__
        Proxy.__delattr__ = __delattr__
    return Proxy()
something = sentinel.Something
something_else = sentinel.SomethingElse

class Foo(object):

    def __init__(self, a):
        if False:
            print('Hello World!')
        pass

    def f(self, a):
        if False:
            return 10
        pass

    def g(self):
        if False:
            i = 10
            return i + 15
        pass
    foo = 'bar'

    @staticmethod
    def static_method():
        if False:
            while True:
                i = 10
        pass

    @classmethod
    def class_method(cls):
        if False:
            for i in range(10):
                print('nop')
        pass

    class Bar(object):

        def a(self):
            if False:
                for i in range(10):
                    print('nop')
            pass
foo_name = '%s.Foo' % __name__

def function(a, b=Foo):
    if False:
        while True:
            i = 10
    pass

class Container(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.values = {}

    def __getitem__(self, name):
        if False:
            while True:
                i = 10
        return self.values[name]

    def __setitem__(self, name, value):
        if False:
            print('Hello World!')
        self.values[name] = value

    def __delitem__(self, name):
        if False:
            for i in range(10):
                print('nop')
        del self.values[name]

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.values)

class PatchTest(unittest.TestCase):

    def assertNotCallable(self, obj, magic=True):
        if False:
            print('Hello World!')
        MockClass = NonCallableMagicMock
        if not magic:
            MockClass = NonCallableMock
        self.assertRaises(TypeError, obj)
        self.assertTrue(is_instance(obj, MockClass))
        self.assertFalse(is_instance(obj, CallableMixin))

    def test_single_patchobject(self):
        if False:
            return 10

        class Something(object):
            attribute = sentinel.Original

        @patch.object(Something, 'attribute', sentinel.Patched)
        def test():
            if False:
                i = 10
                return i + 15
            self.assertEqual(Something.attribute, sentinel.Patched, 'unpatched')
        test()
        self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')

    def test_patchobject_with_string_as_target(self):
        if False:
            for i in range(10):
                print('nop')
        msg = "'Something' must be the actual object to be patched, not a str"
        with self.assertRaisesRegex(TypeError, msg):
            patch.object('Something', 'do_something')

    def test_patchobject_with_none(self):
        if False:
            return 10

        class Something(object):
            attribute = sentinel.Original

        @patch.object(Something, 'attribute', None)
        def test():
            if False:
                i = 10
                return i + 15
            self.assertIsNone(Something.attribute, 'unpatched')
        test()
        self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')

    def test_multiple_patchobject(self):
        if False:
            for i in range(10):
                print('nop')

        class Something(object):
            attribute = sentinel.Original
            next_attribute = sentinel.Original2

        @patch.object(Something, 'attribute', sentinel.Patched)
        @patch.object(Something, 'next_attribute', sentinel.Patched2)
        def test():
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(Something.attribute, sentinel.Patched, 'unpatched')
            self.assertEqual(Something.next_attribute, sentinel.Patched2, 'unpatched')
        test()
        self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')
        self.assertEqual(Something.next_attribute, sentinel.Original2, 'patch not restored')

    def test_object_lookup_is_quite_lazy(self):
        if False:
            while True:
                i = 10
        global something
        original = something

        @patch('%s.something' % __name__, sentinel.Something2)
        def test():
            if False:
                return 10
            pass
        try:
            something = sentinel.replacement_value
            test()
            self.assertEqual(something, sentinel.replacement_value)
        finally:
            something = original

    def test_patch(self):
        if False:
            print('Hello World!')

        @patch('%s.something' % __name__, sentinel.Something2)
        def test():
            if False:
                i = 10
                return i + 15
            self.assertEqual(PTModule.something, sentinel.Something2, 'unpatched')
        test()
        self.assertEqual(PTModule.something, sentinel.Something, 'patch not restored')

        @patch('%s.something' % __name__, sentinel.Something2)
        @patch('%s.something_else' % __name__, sentinel.SomethingElse)
        def test():
            if False:
                return 10
            self.assertEqual(PTModule.something, sentinel.Something2, 'unpatched')
            self.assertEqual(PTModule.something_else, sentinel.SomethingElse, 'unpatched')
        self.assertEqual(PTModule.something, sentinel.Something, 'patch not restored')
        self.assertEqual(PTModule.something_else, sentinel.SomethingElse, 'patch not restored')
        test()
        self.assertEqual(PTModule.something, sentinel.Something, 'patch not restored')
        self.assertEqual(PTModule.something_else, sentinel.SomethingElse, 'patch not restored')
        mock = Mock()
        mock.return_value = sentinel.Handle

        @patch('%s.open' % builtin_string, mock)
        def test():
            if False:
                return 10
            self.assertEqual(open('filename', 'r'), sentinel.Handle, 'open not patched')
        test()
        test()
        self.assertNotEqual(open, mock, 'patch not restored')

    def test_patch_class_attribute(self):
        if False:
            i = 10
            return i + 15

        @patch('%s.SomeClass.class_attribute' % __name__, sentinel.ClassAttribute)
        def test():
            if False:
                print('Hello World!')
            self.assertEqual(PTModule.SomeClass.class_attribute, sentinel.ClassAttribute, 'unpatched')
        test()
        self.assertIsNone(PTModule.SomeClass.class_attribute, 'patch not restored')

    def test_patchobject_with_default_mock(self):
        if False:
            i = 10
            return i + 15

        class Test(object):
            something = sentinel.Original
            something2 = sentinel.Original2

        @patch.object(Test, 'something')
        def test(mock):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(mock, Test.something, 'Mock not passed into test function')
            self.assertIsInstance(mock, MagicMock, 'patch with two arguments did not create a mock')
        test()

        @patch.object(Test, 'something')
        @patch.object(Test, 'something2')
        def test(this1, this2, mock1, mock2):
            if False:
                while True:
                    i = 10
            self.assertEqual(this1, sentinel.this1, "Patched function didn't receive initial argument")
            self.assertEqual(this2, sentinel.this2, "Patched function didn't receive second argument")
            self.assertEqual(mock1, Test.something2, 'Mock not passed into test function')
            self.assertEqual(mock2, Test.something, 'Second Mock not passed into test function')
            self.assertIsInstance(mock2, MagicMock, 'patch with two arguments did not create a mock')
            self.assertIsInstance(mock2, MagicMock, 'patch with two arguments did not create a mock')
            self.assertNotEqual(outerMock1, mock1, 'unexpected value for mock1')
            self.assertNotEqual(outerMock2, mock2, 'unexpected value for mock1')
            return (mock1, mock2)
        outerMock1 = outerMock2 = None
        (outerMock1, outerMock2) = test(sentinel.this1, sentinel.this2)
        test(sentinel.this1, sentinel.this2)

    def test_patch_with_spec(self):
        if False:
            print('Hello World!')

        @patch('%s.SomeClass' % __name__, spec=SomeClass)
        def test(MockSomeClass):
            if False:
                while True:
                    i = 10
            self.assertEqual(SomeClass, MockSomeClass)
            self.assertTrue(is_instance(SomeClass.wibble, MagicMock))
            self.assertRaises(AttributeError, lambda : SomeClass.not_wibble)
        test()

    def test_patchobject_with_spec(self):
        if False:
            i = 10
            return i + 15

        @patch.object(SomeClass, 'class_attribute', spec=SomeClass)
        def test(MockAttribute):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(SomeClass.class_attribute, MockAttribute)
            self.assertTrue(is_instance(SomeClass.class_attribute.wibble, MagicMock))
            self.assertRaises(AttributeError, lambda : SomeClass.class_attribute.not_wibble)
        test()

    def test_patch_with_spec_as_list(self):
        if False:
            for i in range(10):
                print('nop')

        @patch('%s.SomeClass' % __name__, spec=['wibble'])
        def test(MockSomeClass):
            if False:
                return 10
            self.assertEqual(SomeClass, MockSomeClass)
            self.assertTrue(is_instance(SomeClass.wibble, MagicMock))
            self.assertRaises(AttributeError, lambda : SomeClass.not_wibble)
        test()

    def test_patchobject_with_spec_as_list(self):
        if False:
            return 10

        @patch.object(SomeClass, 'class_attribute', spec=['wibble'])
        def test(MockAttribute):
            if False:
                print('Hello World!')
            self.assertEqual(SomeClass.class_attribute, MockAttribute)
            self.assertTrue(is_instance(SomeClass.class_attribute.wibble, MagicMock))
            self.assertRaises(AttributeError, lambda : SomeClass.class_attribute.not_wibble)
        test()

    def test_nested_patch_with_spec_as_list(self):
        if False:
            print('Hello World!')

        @patch('%s.open' % builtin_string)
        @patch('%s.SomeClass' % __name__, spec=['wibble'])
        def test(MockSomeClass, MockOpen):
            if False:
                print('Hello World!')
            self.assertEqual(SomeClass, MockSomeClass)
            self.assertTrue(is_instance(SomeClass.wibble, MagicMock))
            self.assertRaises(AttributeError, lambda : SomeClass.not_wibble)
        test()

    def test_patch_with_spec_as_boolean(self):
        if False:
            for i in range(10):
                print('nop')

        @patch('%s.SomeClass' % __name__, spec=True)
        def test(MockSomeClass):
            if False:
                return 10
            self.assertEqual(SomeClass, MockSomeClass)
            MockSomeClass.wibble
            self.assertRaises(AttributeError, lambda : MockSomeClass.not_wibble)
        test()

    def test_patch_object_with_spec_as_boolean(self):
        if False:
            return 10

        @patch.object(PTModule, 'SomeClass', spec=True)
        def test(MockSomeClass):
            if False:
                print('Hello World!')
            self.assertEqual(SomeClass, MockSomeClass)
            MockSomeClass.wibble
            self.assertRaises(AttributeError, lambda : MockSomeClass.not_wibble)
        test()

    def test_patch_class_acts_with_spec_is_inherited(self):
        if False:
            for i in range(10):
                print('nop')

        @patch('%s.SomeClass' % __name__, spec=True)
        def test(MockSomeClass):
            if False:
                return 10
            self.assertTrue(is_instance(MockSomeClass, MagicMock))
            instance = MockSomeClass()
            self.assertNotCallable(instance)
            instance.wibble
            self.assertRaises(AttributeError, lambda : instance.not_wibble)
        test()

    def test_patch_with_create_mocks_non_existent_attributes(self):
        if False:
            print('Hello World!')

        @patch('%s.frooble' % builtin_string, sentinel.Frooble, create=True)
        def test():
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(frooble, sentinel.Frooble)
        test()
        self.assertRaises(NameError, lambda : frooble)

    def test_patchobject_with_create_mocks_non_existent_attributes(self):
        if False:
            for i in range(10):
                print('nop')

        @patch.object(SomeClass, 'frooble', sentinel.Frooble, create=True)
        def test():
            if False:
                while True:
                    i = 10
            self.assertEqual(SomeClass.frooble, sentinel.Frooble)
        test()
        self.assertFalse(hasattr(SomeClass, 'frooble'))

    def test_patch_wont_create_by_default(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AttributeError):

            @patch('%s.frooble' % builtin_string, sentinel.Frooble)
            def test():
                if False:
                    i = 10
                    return i + 15
                pass
            test()
        self.assertRaises(NameError, lambda : frooble)

    def test_patchobject_wont_create_by_default(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AttributeError):

            @patch.object(SomeClass, 'ord', sentinel.Frooble)
            def test():
                if False:
                    print('Hello World!')
                pass
            test()
        self.assertFalse(hasattr(SomeClass, 'ord'))

    def test_patch_builtins_without_create(self):
        if False:
            print('Hello World!')

        @patch(__name__ + '.ord')
        def test_ord(mock_ord):
            if False:
                return 10
            mock_ord.return_value = 101
            return ord('c')

        @patch(__name__ + '.open')
        def test_open(mock_open):
            if False:
                for i in range(10):
                    print('nop')
            m = mock_open.return_value
            m.read.return_value = 'abcd'
            fobj = open('doesnotexists.txt')
            data = fobj.read()
            fobj.close()
            return data
        self.assertEqual(test_ord(), 101)
        self.assertEqual(test_open(), 'abcd')

    def test_patch_with_static_methods(self):
        if False:
            while True:
                i = 10

        class Foo(object):

            @staticmethod
            def woot():
                if False:
                    i = 10
                    return i + 15
                return sentinel.Static

        @patch.object(Foo, 'woot', staticmethod(lambda : sentinel.Patched))
        def anonymous():
            if False:
                while True:
                    i = 10
            self.assertEqual(Foo.woot(), sentinel.Patched)
        anonymous()
        self.assertEqual(Foo.woot(), sentinel.Static)

    def test_patch_local(self):
        if False:
            while True:
                i = 10
        foo = sentinel.Foo

        @patch.object(sentinel, 'Foo', 'Foo')
        def anonymous():
            if False:
                return 10
            self.assertEqual(sentinel.Foo, 'Foo')
        anonymous()
        self.assertEqual(sentinel.Foo, foo)

    def test_patch_slots(self):
        if False:
            print('Hello World!')

        class Foo(object):
            __slots__ = ('Foo',)
        foo = Foo()
        foo.Foo = sentinel.Foo

        @patch.object(foo, 'Foo', 'Foo')
        def anonymous():
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(foo.Foo, 'Foo')
        anonymous()
        self.assertEqual(foo.Foo, sentinel.Foo)

    def test_patchobject_class_decorator(self):
        if False:
            while True:
                i = 10

        class Something(object):
            attribute = sentinel.Original

        class Foo(object):

            def test_method(other_self):
                if False:
                    return 10
                self.assertEqual(Something.attribute, sentinel.Patched, 'unpatched')

            def not_test_method(other_self):
                if False:
                    return 10
                self.assertEqual(Something.attribute, sentinel.Original, 'non-test method patched')
        Foo = patch.object(Something, 'attribute', sentinel.Patched)(Foo)
        f = Foo()
        f.test_method()
        f.not_test_method()
        self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')

    def test_patch_class_decorator(self):
        if False:
            i = 10
            return i + 15

        class Something(object):
            attribute = sentinel.Original

        class Foo(object):
            test_class_attr = 'whatever'

            def test_method(other_self, mock_something):
                if False:
                    return 10
                self.assertEqual(PTModule.something, mock_something, 'unpatched')

            def not_test_method(other_self):
                if False:
                    return 10
                self.assertEqual(PTModule.something, sentinel.Something, 'non-test method patched')
        Foo = patch('%s.something' % __name__)(Foo)
        f = Foo()
        f.test_method()
        f.not_test_method()
        self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')
        self.assertEqual(PTModule.something, sentinel.Something, 'patch not restored')

    def test_patchobject_twice(self):
        if False:
            return 10

        class Something(object):
            attribute = sentinel.Original
            next_attribute = sentinel.Original2

        @patch.object(Something, 'attribute', sentinel.Patched)
        @patch.object(Something, 'attribute', sentinel.Patched)
        def test():
            if False:
                print('Hello World!')
            self.assertEqual(Something.attribute, sentinel.Patched, 'unpatched')
        test()
        self.assertEqual(Something.attribute, sentinel.Original, 'patch not restored')

    def test_patch_dict(self):
        if False:
            while True:
                i = 10
        foo = {'initial': object(), 'other': 'something'}
        original = foo.copy()

        @patch.dict(foo)
        def test():
            if False:
                for i in range(10):
                    print('nop')
            foo['a'] = 3
            del foo['initial']
            foo['other'] = 'something else'
        test()
        self.assertEqual(foo, original)

        @patch.dict(foo, {'a': 'b'})
        def test():
            if False:
                print('Hello World!')
            self.assertEqual(len(foo), 3)
            self.assertEqual(foo['a'], 'b')
        test()
        self.assertEqual(foo, original)

        @patch.dict(foo, [('a', 'b')])
        def test():
            if False:
                i = 10
                return i + 15
            self.assertEqual(len(foo), 3)
            self.assertEqual(foo['a'], 'b')
        test()
        self.assertEqual(foo, original)

    def test_patch_dict_with_container_object(self):
        if False:
            for i in range(10):
                print('nop')
        foo = Container()
        foo['initial'] = object()
        foo['other'] = 'something'
        original = foo.values.copy()

        @patch.dict(foo)
        def test():
            if False:
                return 10
            foo['a'] = 3
            del foo['initial']
            foo['other'] = 'something else'
        test()
        self.assertEqual(foo.values, original)

        @patch.dict(foo, {'a': 'b'})
        def test():
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(len(foo.values), 3)
            self.assertEqual(foo['a'], 'b')
        test()
        self.assertEqual(foo.values, original)

    def test_patch_dict_with_clear(self):
        if False:
            return 10
        foo = {'initial': object(), 'other': 'something'}
        original = foo.copy()

        @patch.dict(foo, clear=True)
        def test():
            if False:
                i = 10
                return i + 15
            self.assertEqual(foo, {})
            foo['a'] = 3
            foo['other'] = 'something else'
        test()
        self.assertEqual(foo, original)

        @patch.dict(foo, {'a': 'b'}, clear=True)
        def test():
            if False:
                return 10
            self.assertEqual(foo, {'a': 'b'})
        test()
        self.assertEqual(foo, original)

        @patch.dict(foo, [('a', 'b')], clear=True)
        def test():
            if False:
                i = 10
                return i + 15
            self.assertEqual(foo, {'a': 'b'})
        test()
        self.assertEqual(foo, original)

    def test_patch_dict_with_container_object_and_clear(self):
        if False:
            for i in range(10):
                print('nop')
        foo = Container()
        foo['initial'] = object()
        foo['other'] = 'something'
        original = foo.values.copy()

        @patch.dict(foo, clear=True)
        def test():
            if False:
                return 10
            self.assertEqual(foo.values, {})
            foo['a'] = 3
            foo['other'] = 'something else'
        test()
        self.assertEqual(foo.values, original)

        @patch.dict(foo, {'a': 'b'}, clear=True)
        def test():
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(foo.values, {'a': 'b'})
        test()
        self.assertEqual(foo.values, original)

    def test_patch_dict_as_context_manager(self):
        if False:
            while True:
                i = 10
        foo = {'a': 'b'}
        with patch.dict(foo, a='c') as patched:
            self.assertEqual(patched, {'a': 'c'})
        self.assertEqual(foo, {'a': 'b'})

    def test_name_preserved(self):
        if False:
            i = 10
            return i + 15
        foo = {}

        @patch('%s.SomeClass' % __name__, object())
        @patch('%s.SomeClass' % __name__, object(), autospec=True)
        @patch.object(SomeClass, object())
        @patch.dict(foo)
        def some_name():
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.assertEqual(some_name.__name__, 'some_name')

    def test_patch_with_exception(self):
        if False:
            return 10
        foo = {}

        @patch.dict(foo, {'a': 'b'})
        def test():
            if False:
                print('Hello World!')
            raise NameError('Konrad')
        with self.assertRaises(NameError):
            test()
        self.assertEqual(foo, {})

    def test_patch_dict_with_string(self):
        if False:
            i = 10
            return i + 15

        @patch.dict('os.environ', {'konrad_delong': 'some value'})
        def test():
            if False:
                while True:
                    i = 10
            self.assertIn('konrad_delong', os.environ)
        test()

    def test_patch_dict_decorator_resolution(self):
        if False:
            while True:
                i = 10
        original = support.target.copy()

        @patch.dict('unittest.test.testmock.support.target', {'bar': 'BAR'})
        def test():
            if False:
                print('Hello World!')
            self.assertEqual(support.target, {'foo': 'BAZ', 'bar': 'BAR'})
        try:
            support.target = {'foo': 'BAZ'}
            test()
            self.assertEqual(support.target, {'foo': 'BAZ'})
        finally:
            support.target = original

    def test_patch_spec_set(self):
        if False:
            print('Hello World!')

        @patch('%s.SomeClass' % __name__, spec=SomeClass, spec_set=True)
        def test(MockClass):
            if False:
                i = 10
                return i + 15
            MockClass.z = 'foo'
        self.assertRaises(AttributeError, test)

        @patch.object(support, 'SomeClass', spec=SomeClass, spec_set=True)
        def test(MockClass):
            if False:
                return 10
            MockClass.z = 'foo'
        self.assertRaises(AttributeError, test)

        @patch('%s.SomeClass' % __name__, spec_set=True)
        def test(MockClass):
            if False:
                for i in range(10):
                    print('nop')
            MockClass.z = 'foo'
        self.assertRaises(AttributeError, test)

        @patch.object(support, 'SomeClass', spec_set=True)
        def test(MockClass):
            if False:
                i = 10
                return i + 15
            MockClass.z = 'foo'
        self.assertRaises(AttributeError, test)

    def test_spec_set_inherit(self):
        if False:
            print('Hello World!')

        @patch('%s.SomeClass' % __name__, spec_set=True)
        def test(MockClass):
            if False:
                i = 10
                return i + 15
            instance = MockClass()
            instance.z = 'foo'
        self.assertRaises(AttributeError, test)

    def test_patch_start_stop(self):
        if False:
            i = 10
            return i + 15
        original = something
        patcher = patch('%s.something' % __name__)
        self.assertIs(something, original)
        mock = patcher.start()
        try:
            self.assertIsNot(mock, original)
            self.assertIs(something, mock)
        finally:
            patcher.stop()
        self.assertIs(something, original)

    def test_stop_without_start(self):
        if False:
            print('Hello World!')
        patcher = patch(foo_name, 'bar', 3)
        self.assertIsNone(patcher.stop())

    def test_stop_idempotent(self):
        if False:
            while True:
                i = 10
        patcher = patch(foo_name, 'bar', 3)
        patcher.start()
        patcher.stop()
        self.assertIsNone(patcher.stop())

    def test_patchobject_start_stop(self):
        if False:
            print('Hello World!')
        original = something
        patcher = patch.object(PTModule, 'something', 'foo')
        self.assertIs(something, original)
        replaced = patcher.start()
        try:
            self.assertEqual(replaced, 'foo')
            self.assertIs(something, replaced)
        finally:
            patcher.stop()
        self.assertIs(something, original)

    def test_patch_dict_start_stop(self):
        if False:
            for i in range(10):
                print('nop')
        d = {'foo': 'bar'}
        original = d.copy()
        patcher = patch.dict(d, [('spam', 'eggs')], clear=True)
        self.assertEqual(d, original)
        patcher.start()
        try:
            self.assertEqual(d, {'spam': 'eggs'})
        finally:
            patcher.stop()
        self.assertEqual(d, original)

    def test_patch_dict_stop_without_start(self):
        if False:
            while True:
                i = 10
        d = {'foo': 'bar'}
        original = d.copy()
        patcher = patch.dict(d, [('spam', 'eggs')], clear=True)
        self.assertFalse(patcher.stop())
        self.assertEqual(d, original)

    def test_patch_dict_class_decorator(self):
        if False:
            return 10
        this = self
        d = {'spam': 'eggs'}
        original = d.copy()

        class Test(object):

            def test_first(self):
                if False:
                    i = 10
                    return i + 15
                this.assertEqual(d, {'foo': 'bar'})

            def test_second(self):
                if False:
                    for i in range(10):
                        print('nop')
                this.assertEqual(d, {'foo': 'bar'})
        Test = patch.dict(d, {'foo': 'bar'}, clear=True)(Test)
        self.assertEqual(d, original)
        test = Test()
        test.test_first()
        self.assertEqual(d, original)
        test.test_second()
        self.assertEqual(d, original)
        test = Test()
        test.test_first()
        self.assertEqual(d, original)
        test.test_second()
        self.assertEqual(d, original)

    def test_get_only_proxy(self):
        if False:
            for i in range(10):
                print('nop')

        class Something(object):
            foo = 'foo'

        class SomethingElse:
            foo = 'foo'
        for thing in (Something, SomethingElse, Something(), SomethingElse):
            proxy = _get_proxy(thing)

            @patch.object(proxy, 'foo', 'bar')
            def test():
                if False:
                    return 10
                self.assertEqual(proxy.foo, 'bar')
            test()
            self.assertEqual(proxy.foo, 'foo')
            self.assertEqual(thing.foo, 'foo')
            self.assertNotIn('foo', proxy.__dict__)

    def test_get_set_delete_proxy(self):
        if False:
            return 10

        class Something(object):
            foo = 'foo'

        class SomethingElse:
            foo = 'foo'
        for thing in (Something, SomethingElse, Something(), SomethingElse):
            proxy = _get_proxy(Something, get_only=False)

            @patch.object(proxy, 'foo', 'bar')
            def test():
                if False:
                    while True:
                        i = 10
                self.assertEqual(proxy.foo, 'bar')
            test()
            self.assertEqual(proxy.foo, 'foo')
            self.assertEqual(thing.foo, 'foo')
            self.assertNotIn('foo', proxy.__dict__)

    def test_patch_keyword_args(self):
        if False:
            print('Hello World!')
        kwargs = {'side_effect': KeyError, 'foo.bar.return_value': 33, 'foo': MagicMock()}
        patcher = patch(foo_name, **kwargs)
        mock = patcher.start()
        patcher.stop()
        self.assertRaises(KeyError, mock)
        self.assertEqual(mock.foo.bar(), 33)
        self.assertIsInstance(mock.foo, MagicMock)

    def test_patch_object_keyword_args(self):
        if False:
            return 10
        kwargs = {'side_effect': KeyError, 'foo.bar.return_value': 33, 'foo': MagicMock()}
        patcher = patch.object(Foo, 'f', **kwargs)
        mock = patcher.start()
        patcher.stop()
        self.assertRaises(KeyError, mock)
        self.assertEqual(mock.foo.bar(), 33)
        self.assertIsInstance(mock.foo, MagicMock)

    def test_patch_dict_keyword_args(self):
        if False:
            i = 10
            return i + 15
        original = {'foo': 'bar'}
        copy = original.copy()
        patcher = patch.dict(original, foo=3, bar=4, baz=5)
        patcher.start()
        try:
            self.assertEqual(original, dict(foo=3, bar=4, baz=5))
        finally:
            patcher.stop()
        self.assertEqual(original, copy)

    def test_autospec(self):
        if False:
            while True:
                i = 10

        class Boo(object):

            def __init__(self, a):
                if False:
                    while True:
                        i = 10
                pass

            def f(self, a):
                if False:
                    print('Hello World!')
                pass

            def g(self):
                if False:
                    i = 10
                    return i + 15
                pass
            foo = 'bar'

            class Bar(object):

                def a(self):
                    if False:
                        while True:
                            i = 10
                    pass

        def _test(mock):
            if False:
                return 10
            mock(1)
            mock.assert_called_with(1)
            self.assertRaises(TypeError, mock)

        def _test2(mock):
            if False:
                return 10
            mock.f(1)
            mock.f.assert_called_with(1)
            self.assertRaises(TypeError, mock.f)
            mock.g()
            mock.g.assert_called_with()
            self.assertRaises(TypeError, mock.g, 1)
            self.assertRaises(AttributeError, getattr, mock, 'h')
            mock.foo.lower()
            mock.foo.lower.assert_called_with()
            self.assertRaises(AttributeError, getattr, mock.foo, 'bar')
            mock.Bar()
            mock.Bar.assert_called_with()
            mock.Bar.a()
            mock.Bar.a.assert_called_with()
            self.assertRaises(TypeError, mock.Bar.a, 1)
            mock.Bar().a()
            mock.Bar().a.assert_called_with()
            self.assertRaises(TypeError, mock.Bar().a, 1)
            self.assertRaises(AttributeError, getattr, mock.Bar, 'b')
            self.assertRaises(AttributeError, getattr, mock.Bar(), 'b')

        def function(mock):
            if False:
                print('Hello World!')
            _test(mock)
            _test2(mock)
            _test2(mock(1))
            self.assertIs(mock, Foo)
            return mock
        test = patch(foo_name, autospec=True)(function)
        mock = test()
        self.assertIsNot(Foo, mock)
        test()
        module = sys.modules[__name__]
        test = patch.object(module, 'Foo', autospec=True)(function)
        mock = test()
        self.assertIsNot(Foo, mock)
        test()

    def test_autospec_function(self):
        if False:
            i = 10
            return i + 15

        @patch('%s.function' % __name__, autospec=True)
        def test(mock):
            if False:
                while True:
                    i = 10
            function.assert_not_called()
            self.assertRaises(AssertionError, function.assert_called)
            self.assertRaises(AssertionError, function.assert_called_once)
            function(1)
            self.assertRaises(AssertionError, function.assert_not_called)
            function.assert_called_with(1)
            function.assert_called()
            function.assert_called_once()
            function(2, 3)
            function.assert_called_with(2, 3)
            self.assertRaises(TypeError, function)
            self.assertRaises(AttributeError, getattr, function, 'foo')
        test()

    def test_autospec_keywords(self):
        if False:
            print('Hello World!')

        @patch('%s.function' % __name__, autospec=True, return_value=3)
        def test(mock_function):
            if False:
                return 10
            return function(1, 2)
        result = test()
        self.assertEqual(result, 3)

    def test_autospec_staticmethod(self):
        if False:
            return 10
        with patch('%s.Foo.static_method' % __name__, autospec=True) as method:
            Foo.static_method()
            method.assert_called_once_with()

    def test_autospec_classmethod(self):
        if False:
            return 10
        with patch('%s.Foo.class_method' % __name__, autospec=True) as method:
            Foo.class_method()
            method.assert_called_once_with()

    def test_autospec_with_new(self):
        if False:
            return 10
        patcher = patch('%s.function' % __name__, new=3, autospec=True)
        self.assertRaises(TypeError, patcher.start)
        module = sys.modules[__name__]
        patcher = patch.object(module, 'function', new=3, autospec=True)
        self.assertRaises(TypeError, patcher.start)

    def test_autospec_with_object(self):
        if False:
            return 10

        class Bar(Foo):
            extra = []
        patcher = patch(foo_name, autospec=Bar)
        mock = patcher.start()
        try:
            self.assertIsInstance(mock, Bar)
            self.assertIsInstance(mock.extra, list)
        finally:
            patcher.stop()

    def test_autospec_inherits(self):
        if False:
            print('Hello World!')
        FooClass = Foo
        patcher = patch(foo_name, autospec=True)
        mock = patcher.start()
        try:
            self.assertIsInstance(mock, FooClass)
            self.assertIsInstance(mock(3), FooClass)
        finally:
            patcher.stop()

    def test_autospec_name(self):
        if False:
            print('Hello World!')
        patcher = patch(foo_name, autospec=True)
        mock = patcher.start()
        try:
            self.assertIn(" name='Foo'", repr(mock))
            self.assertIn(" name='Foo.f'", repr(mock.f))
            self.assertIn(" name='Foo()'", repr(mock(None)))
            self.assertIn(" name='Foo().f'", repr(mock(None).f))
        finally:
            patcher.stop()

    def test_tracebacks(self):
        if False:
            print('Hello World!')

        @patch.object(Foo, 'f', object())
        def test():
            if False:
                print('Hello World!')
            raise AssertionError
        try:
            test()
        except:
            err = sys.exc_info()
        result = unittest.TextTestResult(None, None, 0)
        traceback = result._exc_info_to_string(err, self)
        self.assertIn('raise AssertionError', traceback)

    def test_new_callable_patch(self):
        if False:
            while True:
                i = 10
        patcher = patch(foo_name, new_callable=NonCallableMagicMock)
        m1 = patcher.start()
        patcher.stop()
        m2 = patcher.start()
        patcher.stop()
        self.assertIsNot(m1, m2)
        for mock in (m1, m2):
            self.assertNotCallable(m1)

    def test_new_callable_patch_object(self):
        if False:
            while True:
                i = 10
        patcher = patch.object(Foo, 'f', new_callable=NonCallableMagicMock)
        m1 = patcher.start()
        patcher.stop()
        m2 = patcher.start()
        patcher.stop()
        self.assertIsNot(m1, m2)
        for mock in (m1, m2):
            self.assertNotCallable(m1)

    def test_new_callable_keyword_arguments(self):
        if False:
            for i in range(10):
                print('nop')

        class Bar(object):
            kwargs = None

            def __init__(self, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                Bar.kwargs = kwargs
        patcher = patch(foo_name, new_callable=Bar, arg1=1, arg2=2)
        m = patcher.start()
        try:
            self.assertIs(type(m), Bar)
            self.assertEqual(Bar.kwargs, dict(arg1=1, arg2=2))
        finally:
            patcher.stop()

    def test_new_callable_spec(self):
        if False:
            return 10

        class Bar(object):
            kwargs = None

            def __init__(self, **kwargs):
                if False:
                    print('Hello World!')
                Bar.kwargs = kwargs
        patcher = patch(foo_name, new_callable=Bar, spec=Bar)
        patcher.start()
        try:
            self.assertEqual(Bar.kwargs, dict(spec=Bar))
        finally:
            patcher.stop()
        patcher = patch(foo_name, new_callable=Bar, spec_set=Bar)
        patcher.start()
        try:
            self.assertEqual(Bar.kwargs, dict(spec_set=Bar))
        finally:
            patcher.stop()

    def test_new_callable_create(self):
        if False:
            while True:
                i = 10
        non_existent_attr = '%s.weeeee' % foo_name
        p = patch(non_existent_attr, new_callable=NonCallableMock)
        self.assertRaises(AttributeError, p.start)
        p = patch(non_existent_attr, new_callable=NonCallableMock, create=True)
        m = p.start()
        try:
            self.assertNotCallable(m, magic=False)
        finally:
            p.stop()

    def test_new_callable_incompatible_with_new(self):
        if False:
            return 10
        self.assertRaises(ValueError, patch, foo_name, new=object(), new_callable=MagicMock)
        self.assertRaises(ValueError, patch.object, Foo, 'f', new=object(), new_callable=MagicMock)

    def test_new_callable_incompatible_with_autospec(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(ValueError, patch, foo_name, new_callable=MagicMock, autospec=True)
        self.assertRaises(ValueError, patch.object, Foo, 'f', new_callable=MagicMock, autospec=True)

    def test_new_callable_inherit_for_mocks(self):
        if False:
            return 10

        class MockSub(Mock):
            pass
        MockClasses = (NonCallableMock, NonCallableMagicMock, MagicMock, Mock, MockSub)
        for Klass in MockClasses:
            for arg in ('spec', 'spec_set'):
                kwargs = {arg: True}
                p = patch(foo_name, new_callable=Klass, **kwargs)
                m = p.start()
                try:
                    instance = m.return_value
                    self.assertRaises(AttributeError, getattr, instance, 'x')
                finally:
                    p.stop()

    def test_new_callable_inherit_non_mock(self):
        if False:
            i = 10
            return i + 15

        class NotAMock(object):

            def __init__(self, spec):
                if False:
                    print('Hello World!')
                self.spec = spec
        p = patch(foo_name, new_callable=NotAMock, spec=True)
        m = p.start()
        try:
            self.assertTrue(is_instance(m, NotAMock))
            self.assertRaises(AttributeError, getattr, m, 'return_value')
        finally:
            p.stop()
        self.assertEqual(m.spec, Foo)

    def test_new_callable_class_decorating(self):
        if False:
            for i in range(10):
                print('nop')
        test = self
        original = Foo

        class SomeTest(object):

            def _test(self, mock_foo):
                if False:
                    return 10
                test.assertIsNot(Foo, original)
                test.assertIs(Foo, mock_foo)
                test.assertIsInstance(Foo, SomeClass)

            def test_two(self, mock_foo):
                if False:
                    i = 10
                    return i + 15
                self._test(mock_foo)

            def test_one(self, mock_foo):
                if False:
                    print('Hello World!')
                self._test(mock_foo)
        SomeTest = patch(foo_name, new_callable=SomeClass)(SomeTest)
        SomeTest().test_one()
        SomeTest().test_two()
        self.assertIs(Foo, original)

    def test_patch_multiple(self):
        if False:
            i = 10
            return i + 15
        original_foo = Foo
        original_f = Foo.f
        original_g = Foo.g
        patcher1 = patch.multiple(foo_name, f=1, g=2)
        patcher2 = patch.multiple(Foo, f=1, g=2)
        for patcher in (patcher1, patcher2):
            patcher.start()
            try:
                self.assertIs(Foo, original_foo)
                self.assertEqual(Foo.f, 1)
                self.assertEqual(Foo.g, 2)
            finally:
                patcher.stop()
            self.assertIs(Foo, original_foo)
            self.assertEqual(Foo.f, original_f)
            self.assertEqual(Foo.g, original_g)

        @patch.multiple(foo_name, f=3, g=4)
        def test():
            if False:
                return 10
            self.assertIs(Foo, original_foo)
            self.assertEqual(Foo.f, 3)
            self.assertEqual(Foo.g, 4)
        test()

    def test_patch_multiple_no_kwargs(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, patch.multiple, foo_name)
        self.assertRaises(ValueError, patch.multiple, Foo)

    def test_patch_multiple_create_mocks(self):
        if False:
            print('Hello World!')
        original_foo = Foo
        original_f = Foo.f
        original_g = Foo.g

        @patch.multiple(foo_name, f=DEFAULT, g=3, foo=DEFAULT)
        def test(f, foo):
            if False:
                i = 10
                return i + 15
            self.assertIs(Foo, original_foo)
            self.assertIs(Foo.f, f)
            self.assertEqual(Foo.g, 3)
            self.assertIs(Foo.foo, foo)
            self.assertTrue(is_instance(f, MagicMock))
            self.assertTrue(is_instance(foo, MagicMock))
        test()
        self.assertEqual(Foo.f, original_f)
        self.assertEqual(Foo.g, original_g)

    def test_patch_multiple_create_mocks_different_order(self):
        if False:
            for i in range(10):
                print('nop')
        original_f = Foo.f
        original_g = Foo.g
        patcher = patch.object(Foo, 'f', 3)
        patcher.attribute_name = 'f'
        other = patch.object(Foo, 'g', DEFAULT)
        other.attribute_name = 'g'
        patcher.additional_patchers = [other]

        @patcher
        def test(g):
            if False:
                i = 10
                return i + 15
            self.assertIs(Foo.g, g)
            self.assertEqual(Foo.f, 3)
        test()
        self.assertEqual(Foo.f, original_f)
        self.assertEqual(Foo.g, original_g)

    def test_patch_multiple_stacked_decorators(self):
        if False:
            for i in range(10):
                print('nop')
        original_foo = Foo
        original_f = Foo.f
        original_g = Foo.g

        @patch.multiple(foo_name, f=DEFAULT)
        @patch.multiple(foo_name, foo=DEFAULT)
        @patch(foo_name + '.g')
        def test1(g, **kwargs):
            if False:
                i = 10
                return i + 15
            _test(g, **kwargs)

        @patch.multiple(foo_name, f=DEFAULT)
        @patch(foo_name + '.g')
        @patch.multiple(foo_name, foo=DEFAULT)
        def test2(g, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            _test(g, **kwargs)

        @patch(foo_name + '.g')
        @patch.multiple(foo_name, f=DEFAULT)
        @patch.multiple(foo_name, foo=DEFAULT)
        def test3(g, **kwargs):
            if False:
                print('Hello World!')
            _test(g, **kwargs)

        def _test(g, **kwargs):
            if False:
                print('Hello World!')
            f = kwargs.pop('f')
            foo = kwargs.pop('foo')
            self.assertFalse(kwargs)
            self.assertIs(Foo, original_foo)
            self.assertIs(Foo.f, f)
            self.assertIs(Foo.g, g)
            self.assertIs(Foo.foo, foo)
            self.assertTrue(is_instance(f, MagicMock))
            self.assertTrue(is_instance(g, MagicMock))
            self.assertTrue(is_instance(foo, MagicMock))
        test1()
        test2()
        test3()
        self.assertEqual(Foo.f, original_f)
        self.assertEqual(Foo.g, original_g)

    def test_patch_multiple_create_mocks_patcher(self):
        if False:
            for i in range(10):
                print('nop')
        original_foo = Foo
        original_f = Foo.f
        original_g = Foo.g
        patcher = patch.multiple(foo_name, f=DEFAULT, g=3, foo=DEFAULT)
        result = patcher.start()
        try:
            f = result['f']
            foo = result['foo']
            self.assertEqual(set(result), set(['f', 'foo']))
            self.assertIs(Foo, original_foo)
            self.assertIs(Foo.f, f)
            self.assertIs(Foo.foo, foo)
            self.assertTrue(is_instance(f, MagicMock))
            self.assertTrue(is_instance(foo, MagicMock))
        finally:
            patcher.stop()
        self.assertEqual(Foo.f, original_f)
        self.assertEqual(Foo.g, original_g)

    def test_patch_multiple_decorating_class(self):
        if False:
            for i in range(10):
                print('nop')
        test = self
        original_foo = Foo
        original_f = Foo.f
        original_g = Foo.g

        class SomeTest(object):

            def _test(self, f, foo):
                if False:
                    print('Hello World!')
                test.assertIs(Foo, original_foo)
                test.assertIs(Foo.f, f)
                test.assertEqual(Foo.g, 3)
                test.assertIs(Foo.foo, foo)
                test.assertTrue(is_instance(f, MagicMock))
                test.assertTrue(is_instance(foo, MagicMock))

            def test_two(self, f, foo):
                if False:
                    while True:
                        i = 10
                self._test(f, foo)

            def test_one(self, f, foo):
                if False:
                    for i in range(10):
                        print('nop')
                self._test(f, foo)
        SomeTest = patch.multiple(foo_name, f=DEFAULT, g=3, foo=DEFAULT)(SomeTest)
        thing = SomeTest()
        thing.test_one()
        thing.test_two()
        self.assertEqual(Foo.f, original_f)
        self.assertEqual(Foo.g, original_g)

    def test_patch_multiple_create(self):
        if False:
            return 10
        patcher = patch.multiple(Foo, blam='blam')
        self.assertRaises(AttributeError, patcher.start)
        patcher = patch.multiple(Foo, blam='blam', create=True)
        patcher.start()
        try:
            self.assertEqual(Foo.blam, 'blam')
        finally:
            patcher.stop()
        self.assertFalse(hasattr(Foo, 'blam'))

    def test_patch_multiple_spec_set(self):
        if False:
            i = 10
            return i + 15
        patcher = patch.multiple(Foo, foo=DEFAULT, spec_set=['a', 'b'])
        result = patcher.start()
        try:
            self.assertEqual(Foo.foo, result['foo'])
            Foo.foo.a(1)
            Foo.foo.b(2)
            Foo.foo.a.assert_called_with(1)
            Foo.foo.b.assert_called_with(2)
            self.assertRaises(AttributeError, setattr, Foo.foo, 'c', None)
        finally:
            patcher.stop()

    def test_patch_multiple_new_callable(self):
        if False:
            i = 10
            return i + 15

        class Thing(object):
            pass
        patcher = patch.multiple(Foo, f=DEFAULT, g=DEFAULT, new_callable=Thing)
        result = patcher.start()
        try:
            self.assertIs(Foo.f, result['f'])
            self.assertIs(Foo.g, result['g'])
            self.assertIsInstance(Foo.f, Thing)
            self.assertIsInstance(Foo.g, Thing)
            self.assertIsNot(Foo.f, Foo.g)
        finally:
            patcher.stop()

    def test_nested_patch_failure(self):
        if False:
            return 10
        original_f = Foo.f
        original_g = Foo.g

        @patch.object(Foo, 'g', 1)
        @patch.object(Foo, 'missing', 1)
        @patch.object(Foo, 'f', 1)
        def thing1():
            if False:
                i = 10
                return i + 15
            pass

        @patch.object(Foo, 'missing', 1)
        @patch.object(Foo, 'g', 1)
        @patch.object(Foo, 'f', 1)
        def thing2():
            if False:
                while True:
                    i = 10
            pass

        @patch.object(Foo, 'g', 1)
        @patch.object(Foo, 'f', 1)
        @patch.object(Foo, 'missing', 1)
        def thing3():
            if False:
                while True:
                    i = 10
            pass
        for func in (thing1, thing2, thing3):
            self.assertRaises(AttributeError, func)
            self.assertEqual(Foo.f, original_f)
            self.assertEqual(Foo.g, original_g)

    def test_new_callable_failure(self):
        if False:
            while True:
                i = 10
        original_f = Foo.f
        original_g = Foo.g
        original_foo = Foo.foo

        def crasher():
            if False:
                while True:
                    i = 10
            raise NameError('crasher')

        @patch.object(Foo, 'g', 1)
        @patch.object(Foo, 'foo', new_callable=crasher)
        @patch.object(Foo, 'f', 1)
        def thing1():
            if False:
                i = 10
                return i + 15
            pass

        @patch.object(Foo, 'foo', new_callable=crasher)
        @patch.object(Foo, 'g', 1)
        @patch.object(Foo, 'f', 1)
        def thing2():
            if False:
                print('Hello World!')
            pass

        @patch.object(Foo, 'g', 1)
        @patch.object(Foo, 'f', 1)
        @patch.object(Foo, 'foo', new_callable=crasher)
        def thing3():
            if False:
                return 10
            pass
        for func in (thing1, thing2, thing3):
            self.assertRaises(NameError, func)
            self.assertEqual(Foo.f, original_f)
            self.assertEqual(Foo.g, original_g)
            self.assertEqual(Foo.foo, original_foo)

    def test_patch_multiple_failure(self):
        if False:
            while True:
                i = 10
        original_f = Foo.f
        original_g = Foo.g
        patcher = patch.object(Foo, 'f', 1)
        patcher.attribute_name = 'f'
        good = patch.object(Foo, 'g', 1)
        good.attribute_name = 'g'
        bad = patch.object(Foo, 'missing', 1)
        bad.attribute_name = 'missing'
        for additionals in ([good, bad], [bad, good]):
            patcher.additional_patchers = additionals

            @patcher
            def func():
                if False:
                    for i in range(10):
                        print('nop')
                pass
            self.assertRaises(AttributeError, func)
            self.assertEqual(Foo.f, original_f)
            self.assertEqual(Foo.g, original_g)

    def test_patch_multiple_new_callable_failure(self):
        if False:
            for i in range(10):
                print('nop')
        original_f = Foo.f
        original_g = Foo.g
        original_foo = Foo.foo

        def crasher():
            if False:
                while True:
                    i = 10
            raise NameError('crasher')
        patcher = patch.object(Foo, 'f', 1)
        patcher.attribute_name = 'f'
        good = patch.object(Foo, 'g', 1)
        good.attribute_name = 'g'
        bad = patch.object(Foo, 'foo', new_callable=crasher)
        bad.attribute_name = 'foo'
        for additionals in ([good, bad], [bad, good]):
            patcher.additional_patchers = additionals

            @patcher
            def func():
                if False:
                    for i in range(10):
                        print('nop')
                pass
            self.assertRaises(NameError, func)
            self.assertEqual(Foo.f, original_f)
            self.assertEqual(Foo.g, original_g)
            self.assertEqual(Foo.foo, original_foo)

    def test_patch_multiple_string_subclasses(self):
        if False:
            while True:
                i = 10
        Foo = type('Foo', (str,), {'fish': 'tasty'})
        foo = Foo()

        @patch.multiple(foo, fish='nearly gone')
        def test():
            if False:
                i = 10
                return i + 15
            self.assertEqual(foo.fish, 'nearly gone')
        test()
        self.assertEqual(foo.fish, 'tasty')

    @patch('unittest.mock.patch.TEST_PREFIX', 'foo')
    def test_patch_test_prefix(self):
        if False:
            i = 10
            return i + 15

        class Foo(object):
            thing = 'original'

            def foo_one(self):
                if False:
                    print('Hello World!')
                return self.thing

            def foo_two(self):
                if False:
                    while True:
                        i = 10
                return self.thing

            def test_one(self):
                if False:
                    while True:
                        i = 10
                return self.thing

            def test_two(self):
                if False:
                    return 10
                return self.thing
        Foo = patch.object(Foo, 'thing', 'changed')(Foo)
        foo = Foo()
        self.assertEqual(foo.foo_one(), 'changed')
        self.assertEqual(foo.foo_two(), 'changed')
        self.assertEqual(foo.test_one(), 'original')
        self.assertEqual(foo.test_two(), 'original')

    @patch('unittest.mock.patch.TEST_PREFIX', 'bar')
    def test_patch_dict_test_prefix(self):
        if False:
            while True:
                i = 10

        class Foo(object):

            def bar_one(self):
                if False:
                    print('Hello World!')
                return dict(the_dict)

            def bar_two(self):
                if False:
                    return 10
                return dict(the_dict)

            def test_one(self):
                if False:
                    i = 10
                    return i + 15
                return dict(the_dict)

            def test_two(self):
                if False:
                    return 10
                return dict(the_dict)
        the_dict = {'key': 'original'}
        Foo = patch.dict(the_dict, key='changed')(Foo)
        foo = Foo()
        self.assertEqual(foo.bar_one(), {'key': 'changed'})
        self.assertEqual(foo.bar_two(), {'key': 'changed'})
        self.assertEqual(foo.test_one(), {'key': 'original'})
        self.assertEqual(foo.test_two(), {'key': 'original'})

    def test_patch_with_spec_mock_repr(self):
        if False:
            while True:
                i = 10
        for arg in ('spec', 'autospec', 'spec_set'):
            p = patch('%s.SomeClass' % __name__, **{arg: True})
            m = p.start()
            try:
                self.assertIn(" name='SomeClass'", repr(m))
                self.assertIn(" name='SomeClass.class_attribute'", repr(m.class_attribute))
                self.assertIn(" name='SomeClass()'", repr(m()))
                self.assertIn(" name='SomeClass().class_attribute'", repr(m().class_attribute))
            finally:
                p.stop()

    def test_patch_nested_autospec_repr(self):
        if False:
            print('Hello World!')
        with patch('unittest.test.testmock.support', autospec=True) as m:
            self.assertIn(" name='support.SomeClass.wibble()'", repr(m.SomeClass.wibble()))
            self.assertIn(" name='support.SomeClass().wibble()'", repr(m.SomeClass().wibble()))

    def test_mock_calls_with_patch(self):
        if False:
            for i in range(10):
                print('nop')
        for arg in ('spec', 'autospec', 'spec_set'):
            p = patch('%s.SomeClass' % __name__, **{arg: True})
            m = p.start()
            try:
                m.wibble()
                kalls = [call.wibble()]
                self.assertEqual(m.mock_calls, kalls)
                self.assertEqual(m.method_calls, kalls)
                self.assertEqual(m.wibble.mock_calls, [call()])
                result = m()
                kalls.append(call())
                self.assertEqual(m.mock_calls, kalls)
                result.wibble()
                kalls.append(call().wibble())
                self.assertEqual(m.mock_calls, kalls)
                self.assertEqual(result.mock_calls, [call.wibble()])
                self.assertEqual(result.wibble.mock_calls, [call()])
                self.assertEqual(result.method_calls, [call.wibble()])
            finally:
                p.stop()

    def test_patch_imports_lazily(self):
        if False:
            i = 10
            return i + 15
        p1 = patch('squizz.squozz')
        self.assertRaises(ImportError, p1.start)
        with uncache('squizz'):
            squizz = Mock()
            sys.modules['squizz'] = squizz
            squizz.squozz = 6
            p1 = patch('squizz.squozz')
            squizz.squozz = 3
            p1.start()
            p1.stop()
        self.assertEqual(squizz.squozz, 3)

    def test_patch_propagates_exc_on_exit(self):
        if False:
            for i in range(10):
                print('nop')

        class holder:
            exc_info = (None, None, None)

        class custom_patch(_patch):

            def __exit__(self, etype=None, val=None, tb=None):
                if False:
                    for i in range(10):
                        print('nop')
                _patch.__exit__(self, etype, val, tb)
                holder.exc_info = (etype, val, tb)
            stop = __exit__

        def with_custom_patch(target):
            if False:
                return 10
            (getter, attribute) = _get_target(target)
            return custom_patch(getter, attribute, DEFAULT, None, False, None, None, None, {})

        @with_custom_patch('squizz.squozz')
        def test(mock):
            if False:
                print('Hello World!')
            raise RuntimeError
        with uncache('squizz'):
            squizz = Mock()
            sys.modules['squizz'] = squizz
            self.assertRaises(RuntimeError, test)
        self.assertIs(holder.exc_info[0], RuntimeError)
        self.assertIsNotNone(holder.exc_info[1], 'exception value not propagated')
        self.assertIsNotNone(holder.exc_info[2], 'exception traceback not propagated')

    def test_create_and_specs(self):
        if False:
            i = 10
            return i + 15
        for kwarg in ('spec', 'spec_set', 'autospec'):
            p = patch('%s.doesnotexist' % __name__, create=True, **{kwarg: True})
            self.assertRaises(TypeError, p.start)
            self.assertRaises(NameError, lambda : doesnotexist)
            p = patch(MODNAME, create=True, **{kwarg: True})
            p.start()
            p.stop()

    def test_multiple_specs(self):
        if False:
            print('Hello World!')
        original = PTModule
        for kwarg in ('spec', 'spec_set'):
            p = patch(MODNAME, autospec=0, **{kwarg: 0})
            self.assertRaises(TypeError, p.start)
            self.assertIs(PTModule, original)
        for kwarg in ('spec', 'autospec'):
            p = patch(MODNAME, spec_set=0, **{kwarg: 0})
            self.assertRaises(TypeError, p.start)
            self.assertIs(PTModule, original)
        for kwarg in ('spec_set', 'autospec'):
            p = patch(MODNAME, spec=0, **{kwarg: 0})
            self.assertRaises(TypeError, p.start)
            self.assertIs(PTModule, original)

    def test_specs_false_instead_of_none(self):
        if False:
            return 10
        p = patch(MODNAME, spec=False, spec_set=False, autospec=False)
        mock = p.start()
        try:
            mock.does_not_exist
            mock.does_not_exist = 3
        finally:
            p.stop()

    def test_falsey_spec(self):
        if False:
            print('Hello World!')
        for kwarg in ('spec', 'autospec', 'spec_set'):
            p = patch(MODNAME, **{kwarg: 0})
            m = p.start()
            try:
                self.assertRaises(AttributeError, getattr, m, 'doesnotexit')
            finally:
                p.stop()

    def test_spec_set_true(self):
        if False:
            print('Hello World!')
        for kwarg in ('spec', 'autospec'):
            p = patch(MODNAME, spec_set=True, **{kwarg: True})
            m = p.start()
            try:
                self.assertRaises(AttributeError, setattr, m, 'doesnotexist', 'something')
                self.assertRaises(AttributeError, getattr, m, 'doesnotexist')
            finally:
                p.stop()

    def test_callable_spec_as_list(self):
        if False:
            while True:
                i = 10
        spec = ('__call__',)
        p = patch(MODNAME, spec=spec)
        m = p.start()
        try:
            self.assertTrue(callable(m))
        finally:
            p.stop()

    def test_not_callable_spec_as_list(self):
        if False:
            return 10
        spec = ('foo', 'bar')
        p = patch(MODNAME, spec=spec)
        m = p.start()
        try:
            self.assertFalse(callable(m))
        finally:
            p.stop()

    def test_patch_stopall(self):
        if False:
            i = 10
            return i + 15
        unlink = os.unlink
        chdir = os.chdir
        path = os.path
        patch('os.unlink', something).start()
        patch('os.chdir', something_else).start()

        @patch('os.path')
        def patched(mock_path):
            if False:
                for i in range(10):
                    print('nop')
            patch.stopall()
            self.assertIs(os.path, mock_path)
            self.assertIs(os.unlink, unlink)
            self.assertIs(os.chdir, chdir)
        patched()
        self.assertIs(os.path, path)

    def test_stopall_lifo(self):
        if False:
            i = 10
            return i + 15
        stopped = []

        class thing(object):
            one = two = three = None

        def get_patch(attribute):
            if False:
                print('Hello World!')

            class mypatch(_patch):

                def stop(self):
                    if False:
                        i = 10
                        return i + 15
                    stopped.append(attribute)
                    return super(mypatch, self).stop()
            return mypatch(lambda : thing, attribute, None, None, False, None, None, None, {})
        [get_patch(val).start() for val in ('one', 'two', 'three')]
        patch.stopall()
        self.assertEqual(stopped, ['three', 'two', 'one'])

    def test_patch_dict_stopall(self):
        if False:
            for i in range(10):
                print('nop')
        dic1 = {}
        dic2 = {1: 'a'}
        dic3 = {1: 'A', 2: 'B'}
        origdic1 = dic1.copy()
        origdic2 = dic2.copy()
        origdic3 = dic3.copy()
        patch.dict(dic1, {1: 'I', 2: 'II'}).start()
        patch.dict(dic2, {2: 'b'}).start()

        @patch.dict(dic3)
        def patched():
            if False:
                return 10
            del dic3[1]
        patched()
        self.assertNotEqual(dic1, origdic1)
        self.assertNotEqual(dic2, origdic2)
        self.assertEqual(dic3, origdic3)
        patch.stopall()
        self.assertEqual(dic1, origdic1)
        self.assertEqual(dic2, origdic2)
        self.assertEqual(dic3, origdic3)

    def test_patch_and_patch_dict_stopall(self):
        if False:
            return 10
        original_unlink = os.unlink
        original_chdir = os.chdir
        dic1 = {}
        dic2 = {1: 'A', 2: 'B'}
        origdic1 = dic1.copy()
        origdic2 = dic2.copy()
        patch('os.unlink', something).start()
        patch('os.chdir', something_else).start()
        patch.dict(dic1, {1: 'I', 2: 'II'}).start()
        patch.dict(dic2).start()
        del dic2[1]
        self.assertIsNot(os.unlink, original_unlink)
        self.assertIsNot(os.chdir, original_chdir)
        self.assertNotEqual(dic1, origdic1)
        self.assertNotEqual(dic2, origdic2)
        patch.stopall()
        self.assertIs(os.unlink, original_unlink)
        self.assertIs(os.chdir, original_chdir)
        self.assertEqual(dic1, origdic1)
        self.assertEqual(dic2, origdic2)

    def test_special_attrs(self):
        if False:
            print('Hello World!')

        def foo(x=0):
            if False:
                for i in range(10):
                    print('nop')
            'TEST'
            return x
        with patch.object(foo, '__defaults__', (1,)):
            self.assertEqual(foo(), 1)
        self.assertEqual(foo(), 0)
        orig_doc = foo.__doc__
        with patch.object(foo, '__doc__', 'FUN'):
            self.assertEqual(foo.__doc__, 'FUN')
        self.assertEqual(foo.__doc__, orig_doc)
        with patch.object(foo, '__module__', 'testpatch2'):
            self.assertEqual(foo.__module__, 'testpatch2')
        self.assertEqual(foo.__module__, 'unittest.test.testmock.testpatch')
        with patch.object(foo, '__annotations__', dict([('s', 1)])):
            self.assertEqual(foo.__annotations__, dict([('s', 1)]))
        self.assertEqual(foo.__annotations__, dict())

        def foo(*a, x=0):
            if False:
                return 10
            return x
        with patch.object(foo, '__kwdefaults__', dict([('x', 1)])):
            self.assertEqual(foo(), 1)
        self.assertEqual(foo(), 0)

    def test_patch_orderdict(self):
        if False:
            for i in range(10):
                print('nop')
        foo = OrderedDict()
        foo['a'] = object()
        foo['b'] = 'python'
        original = foo.copy()
        update_values = list(zip('cdefghijklmnopqrstuvwxyz', range(26)))
        patched_values = list(foo.items()) + update_values
        with patch.dict(foo, OrderedDict(update_values)):
            self.assertEqual(list(foo.items()), patched_values)
        self.assertEqual(foo, original)
        with patch.dict(foo, update_values):
            self.assertEqual(list(foo.items()), patched_values)
        self.assertEqual(foo, original)

    def test_dotted_but_module_not_loaded(self):
        if False:
            i = 10
            return i + 15
        import unittest.test.testmock.support
        with patch.dict('sys.modules'):
            del sys.modules['unittest.test.testmock.support']
            del sys.modules['unittest.test.testmock']
            del sys.modules['unittest.test']
            del sys.modules['unittest']

            @patch('unittest.test.testmock.support.X')
            def test(mock):
                if False:
                    print('Hello World!')
                pass
            test()

    def test_invalid_target(self):
        if False:
            while True:
                i = 10

        class Foo:
            pass
        for target in ['', 12, Foo()]:
            with self.subTest(target=target):
                with self.assertRaises(TypeError):
                    patch(target)

    def test_cant_set_kwargs_when_passing_a_mock(self):
        if False:
            for i in range(10):
                print('nop')

        @patch('unittest.test.testmock.support.X', new=object(), x=1)
        def test():
            if False:
                print('Hello World!')
            pass
        with self.assertRaises(TypeError):
            test()
if __name__ == '__main__':
    unittest.main()