import re
import sys
import types
import unittest
ZERO = 0

class Test(unittest.TestCase):
    if not hasattr(unittest.TestCase, 'assertRegex'):

        def assertRegex(self, value, regex):
            if False:
                while True:
                    i = 10
            self.assertTrue(re.search(regex, str(value)), "'%s' did not match '%s'" % (value, regex))
    if not hasattr(unittest.TestCase, 'assertCountEqual'):

        def assertCountEqual(self, first, second):
            if False:
                print('Hello World!')
            self.assertEqual(set(first), set(second))
            self.assertEqual(len(first), len(second))

    def test_init_subclass(self):
        if False:
            while True:
                i = 10

        class A:
            initialized = False

            def __init_subclass__(cls):
                if False:
                    i = 10
                    return i + 15
                super().__init_subclass__()
                cls.initialized = True

        class B(A):
            pass
        self.assertFalse(A.initialized)
        self.assertTrue(B.initialized)

    def test_init_subclass_dict(self):
        if False:
            return 10

        class A(dict):
            initialized = False

            def __init_subclass__(cls):
                if False:
                    i = 10
                    return i + 15
                super().__init_subclass__()
                cls.initialized = True

        class B(A):
            pass
        self.assertFalse(A.initialized)
        self.assertTrue(B.initialized)

    def test_init_subclass_kwargs(self):
        if False:
            return 10

        class A:

            def __init_subclass__(cls, **kwargs):
                if False:
                    print('Hello World!')
                cls.kwargs = kwargs

        class B(A, x=3):
            pass
        self.assertEqual(B.kwargs, dict(x=3))

    def test_init_subclass_error(self):
        if False:
            return 10

        class A:

            def __init_subclass__(cls):
                if False:
                    print('Hello World!')
                raise RuntimeError
        with self.assertRaises(RuntimeError):

            class B(A):
                pass

    def test_init_subclass_wrong(self):
        if False:
            i = 10
            return i + 15

        class A:

            def __init_subclass__(cls, whatever):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        with self.assertRaises(TypeError):

            class B(A):
                pass

    def test_init_subclass_skipped(self):
        if False:
            for i in range(10):
                print('nop')

        class BaseWithInit:

            def __init_subclass__(cls, **kwargs):
                if False:
                    return 10
                super().__init_subclass__(**kwargs)
                cls.initialized = cls

        class BaseWithoutInit(BaseWithInit):
            pass

        class A(BaseWithoutInit):
            pass
        self.assertIs(A.initialized, A)
        self.assertIs(BaseWithoutInit.initialized, BaseWithoutInit)

    def test_init_subclass_diamond(self):
        if False:
            print('Hello World!')

        class Base:

            def __init_subclass__(cls, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init_subclass__(**kwargs)
                cls.calls = []

        class Left(Base):
            pass

        class Middle:

            def __init_subclass__(cls, middle, **kwargs):
                if False:
                    print('Hello World!')
                super().__init_subclass__(**kwargs)
                cls.calls += [middle]

        class Right(Base):

            def __init_subclass__(cls, right='right', **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init_subclass__(**kwargs)
                cls.calls += [right]

        class A(Left, Middle, Right, middle='middle'):
            pass
        self.assertEqual(A.calls, ['right', 'middle'])
        self.assertEqual(Left.calls, [])
        self.assertEqual(Right.calls, [])

    def test_set_name(self):
        if False:
            for i in range(10):
                print('nop')

        class Descriptor:

            def __set_name__(self, owner, name):
                if False:
                    while True:
                        i = 10
                self.owner = owner
                self.name = name

        class A:
            d = Descriptor()
        self.assertEqual(A.d.name, 'd')
        self.assertIs(A.d.owner, A)

    def test_set_name_metaclass(self):
        if False:
            i = 10
            return i + 15

        class Meta(type):

            def __new__(cls, name, bases, ns):
                if False:
                    for i in range(10):
                        print('nop')
                ret = super().__new__(cls, name, bases, ns)
                self.assertEqual(ret.d.name, 'd')
                self.assertIs(ret.d.owner, ret)
                return 0

        class Descriptor:

            def __set_name__(self, owner, name):
                if False:
                    i = 10
                    return i + 15
                self.owner = owner
                self.name = name

        class A(metaclass=Meta):
            d = Descriptor()
        self.assertEqual(A, 0)

    def test_set_name_error(self):
        if False:
            return 10

        class Descriptor:

            def __set_name__(self, owner, name):
                if False:
                    while True:
                        i = 10
                1 / ZERO
        with self.assertRaises((RuntimeError, ZeroDivisionError)) as cm:

            class NotGoingToWork:
                attr = Descriptor()
        if sys.version_info >= (3, 12):
            notes = cm.exception.__notes__
            self.assertRegex(str(notes), '\\bNotGoingToWork\\b')
            self.assertRegex(str(notes), '\\battr\\b')
            self.assertRegex(str(notes), '\\bDescriptor\\b')
        else:
            exc = cm.exception
            self.assertRegex(str(exc), '\\bNotGoingToWork\\b')
            self.assertRegex(str(exc), '\\battr\\b')
            self.assertRegex(str(exc), '\\bDescriptor\\b')
            self.assertIsInstance(exc.__cause__, ZeroDivisionError)

    def test_set_name_wrong(self):
        if False:
            print('Hello World!')

        class Descriptor:

            def __set_name__(self):
                if False:
                    i = 10
                    return i + 15
                pass
        with self.assertRaises((RuntimeError, TypeError)) as cm:

            class NotGoingToWork:
                attr = Descriptor()
        if sys.version_info >= (3, 12):
            notes = cm.exception.__notes__
            self.assertRegex(str(notes), '\\bNotGoingToWork\\b')
            self.assertRegex(str(notes), '\\battr\\b')
            self.assertRegex(str(notes), '\\bDescriptor\\b')
        else:
            exc = cm.exception
            self.assertRegex(str(exc), '\\bNotGoingToWork\\b')
            self.assertRegex(str(exc), '\\battr\\b')
            self.assertRegex(str(exc), '\\bDescriptor\\b')
            self.assertIsInstance(exc.__cause__, TypeError)

    def test_set_name_lookup(self):
        if False:
            i = 10
            return i + 15
        resolved = []

        class NonDescriptor:

            def __getattr__(self, name):
                if False:
                    return 10
                resolved.append(name)

        class A:
            d = NonDescriptor()
        self.assertNotIn('__set_name__', resolved, '__set_name__ is looked up in instance dict')

    def test_set_name_init_subclass(self):
        if False:
            while True:
                i = 10

        class Descriptor:

            def __set_name__(self, owner, name):
                if False:
                    while True:
                        i = 10
                self.owner = owner
                self.name = name

        class Meta(type):

            def __new__(cls, name, bases, ns):
                if False:
                    print('Hello World!')
                self = super().__new__(cls, name, bases, ns)
                self.meta_owner = self.owner
                self.meta_name = self.name
                return self

        class A:

            def __init_subclass__(cls):
                if False:
                    return 10
                cls.owner = cls.d.owner
                cls.name = cls.d.name

        class B(A, metaclass=Meta):
            d = Descriptor()
        self.assertIs(B.owner, B)
        self.assertEqual(B.name, 'd')
        self.assertIs(B.meta_owner, B)
        self.assertEqual(B.name, 'd')

    def test_set_name_modifying_dict(self):
        if False:
            while True:
                i = 10
        notified = []

        class Descriptor:

            def __set_name__(self, owner, name):
                if False:
                    while True:
                        i = 10
                setattr(owner, name + 'x', None)
                notified.append(name)

        class A:
            a = Descriptor()
            b = Descriptor()
            c = Descriptor()
            d = Descriptor()
            e = Descriptor()
        self.assertCountEqual(notified, ['a', 'b', 'c', 'd', 'e'])

    def test_errors(self):
        if False:
            for i in range(10):
                print('nop')

        class MyMeta(type):
            pass
        with self.assertRaises(TypeError):

            class MyClass(metaclass=MyMeta, otherarg=1):
                pass
        with self.assertRaises(TypeError):
            types.new_class('MyClass', (object,), dict(metaclass=MyMeta, otherarg=1))
        types.prepare_class('MyClass', (object,), dict(metaclass=MyMeta, otherarg=1))

        class MyMeta(type):

            def __init__(self, name, bases, namespace, otherarg):
                if False:
                    while True:
                        i = 10
                super().__init__(name, bases, namespace)
        with self.assertRaises(TypeError):

            class MyClass(metaclass=MyMeta, otherarg=1):
                pass

        class MyMeta(type):

            def __new__(cls, name, bases, namespace, otherarg):
                if False:
                    print('Hello World!')
                return super().__new__(cls, name, bases, namespace)

            def __init__(self, name, bases, namespace, otherarg):
                if False:
                    print('Hello World!')
                super().__init__(name, bases, namespace)
                self.otherarg = otherarg

        class MyClass(metaclass=MyMeta, otherarg=1):
            pass
        self.assertEqual(MyClass.otherarg, 1)

    def test_errors_changed_pep487(self):
        if False:
            for i in range(10):
                print('nop')

        class MyMeta(type):

            def __new__(cls, name, bases, namespace):
                if False:
                    print('Hello World!')
                return super().__new__(cls, name=name, bases=bases, dict=namespace)
        with self.assertRaises(TypeError):

            class MyClass(metaclass=MyMeta):
                pass

        class MyMeta(type):

            def __new__(cls, name, bases, namespace, otherarg):
                if False:
                    return 10
                self = super().__new__(cls, name, bases, namespace)
                self.otherarg = otherarg
                return self

        class MyClass(metaclass=MyMeta, otherarg=1):
            pass
        self.assertEqual(MyClass.otherarg, 1)

    def test_type(self):
        if False:
            return 10
        t = type('NewClass', (object,), {})
        self.assertIsInstance(t, type)
        self.assertEqual(t.__name__, 'NewClass')
        with self.assertRaises(TypeError):
            type(name='NewClass', bases=(object,), dict={})
if __name__ == '__main__':
    unittest.main()