import abc
import sys
import unittest
from types import DynamicClassAttribute

class PropertyBase(Exception):
    pass

class PropertyGet(PropertyBase):
    pass

class PropertySet(PropertyBase):
    pass

class PropertyDel(PropertyBase):
    pass

class BaseClass(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._spam = 5

    @DynamicClassAttribute
    def spam(self):
        if False:
            return 10
        'BaseClass.getter'
        return self._spam

    @spam.setter
    def spam(self, value):
        if False:
            return 10
        self._spam = value

    @spam.deleter
    def spam(self):
        if False:
            i = 10
            return i + 15
        del self._spam

class SubClass(BaseClass):
    spam = BaseClass.__dict__['spam']

    @spam.getter
    def spam(self):
        if False:
            print('Hello World!')
        'SubClass.getter'
        raise PropertyGet(self._spam)

    @spam.setter
    def spam(self, value):
        if False:
            while True:
                i = 10
        raise PropertySet(self._spam)

    @spam.deleter
    def spam(self):
        if False:
            i = 10
            return i + 15
        raise PropertyDel(self._spam)

class PropertyDocBase(object):
    _spam = 1

    def _get_spam(self):
        if False:
            while True:
                i = 10
        return self._spam
    spam = DynamicClassAttribute(_get_spam, doc='spam spam spam')

class PropertyDocSub(PropertyDocBase):
    spam = PropertyDocBase.__dict__['spam']

    @spam.getter
    def spam(self):
        if False:
            print('Hello World!')
        'The decorator does not use this doc string'
        return self._spam

class PropertySubNewGetter(BaseClass):
    spam = BaseClass.__dict__['spam']

    @spam.getter
    def spam(self):
        if False:
            i = 10
            return i + 15
        'new docstring'
        return 5

class PropertyNewGetter(object):

    @DynamicClassAttribute
    def spam(self):
        if False:
            i = 10
            return i + 15
        'original docstring'
        return 1

    @spam.getter
    def spam(self):
        if False:
            while True:
                i = 10
        'new docstring'
        return 8

class ClassWithAbstractVirtualProperty(metaclass=abc.ABCMeta):

    @DynamicClassAttribute
    @abc.abstractmethod
    def color():
        if False:
            return 10
        pass

class ClassWithPropertyAbstractVirtual(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    @DynamicClassAttribute
    def color():
        if False:
            return 10
        pass

class PropertyTests(unittest.TestCase):

    def test_property_decorator_baseclass(self):
        if False:
            print('Hello World!')
        base = BaseClass()
        self.assertEqual(base.spam, 5)
        self.assertEqual(base._spam, 5)
        base.spam = 10
        self.assertEqual(base.spam, 10)
        self.assertEqual(base._spam, 10)
        delattr(base, 'spam')
        self.assertTrue(not hasattr(base, 'spam'))
        self.assertTrue(not hasattr(base, '_spam'))
        base.spam = 20
        self.assertEqual(base.spam, 20)
        self.assertEqual(base._spam, 20)

    def test_property_decorator_subclass(self):
        if False:
            return 10
        sub = SubClass()
        self.assertRaises(PropertyGet, getattr, sub, 'spam')
        self.assertRaises(PropertySet, setattr, sub, 'spam', None)
        self.assertRaises(PropertyDel, delattr, sub, 'spam')

    @unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def test_property_decorator_subclass_doc(self):
        if False:
            print('Hello World!')
        sub = SubClass()
        self.assertEqual(sub.__class__.__dict__['spam'].__doc__, 'SubClass.getter')

    @unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def test_property_decorator_baseclass_doc(self):
        if False:
            for i in range(10):
                print('nop')
        base = BaseClass()
        self.assertEqual(base.__class__.__dict__['spam'].__doc__, 'BaseClass.getter')

    def test_property_decorator_doc(self):
        if False:
            print('Hello World!')
        base = PropertyDocBase()
        sub = PropertyDocSub()
        self.assertEqual(base.__class__.__dict__['spam'].__doc__, 'spam spam spam')
        self.assertEqual(sub.__class__.__dict__['spam'].__doc__, 'spam spam spam')

    @unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def test_property_getter_doc_override(self):
        if False:
            return 10
        newgettersub = PropertySubNewGetter()
        self.assertEqual(newgettersub.spam, 5)
        self.assertEqual(newgettersub.__class__.__dict__['spam'].__doc__, 'new docstring')
        newgetter = PropertyNewGetter()
        self.assertEqual(newgetter.spam, 8)
        self.assertEqual(newgetter.__class__.__dict__['spam'].__doc__, 'new docstring')

    def test_property___isabstractmethod__descriptor(self):
        if False:
            while True:
                i = 10
        for val in (True, False, [], [1], '', '1'):

            class C(object):

                def foo(self):
                    if False:
                        while True:
                            i = 10
                    pass
                foo.__isabstractmethod__ = val
                foo = DynamicClassAttribute(foo)
            self.assertIs(C.__dict__['foo'].__isabstractmethod__, bool(val))

        class NotBool(object):

            def __bool__(self):
                if False:
                    i = 10
                    return i + 15
                raise ValueError()
            __len__ = __bool__
        with self.assertRaises(ValueError):

            class C(object):

                def foo(self):
                    if False:
                        print('Hello World!')
                    pass
                foo.__isabstractmethod__ = NotBool()
                foo = DynamicClassAttribute(foo)

    def test_abstract_virtual(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, ClassWithAbstractVirtualProperty)
        self.assertRaises(TypeError, ClassWithPropertyAbstractVirtual)

        class APV(ClassWithPropertyAbstractVirtual):
            pass
        self.assertRaises(TypeError, APV)

        class AVP(ClassWithAbstractVirtualProperty):
            pass
        self.assertRaises(TypeError, AVP)

        class Okay1(ClassWithAbstractVirtualProperty):

            @DynamicClassAttribute
            def color(self):
                if False:
                    print('Hello World!')
                return self._color

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self._color = 'cyan'
        with self.assertRaises(AttributeError):
            Okay1.color
        self.assertEqual(Okay1().color, 'cyan')

        class Okay2(ClassWithAbstractVirtualProperty):

            @DynamicClassAttribute
            def color(self):
                if False:
                    while True:
                        i = 10
                return self._color

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self._color = 'magenta'
        with self.assertRaises(AttributeError):
            Okay2.color
        self.assertEqual(Okay2().color, 'magenta')

class PropertySub(DynamicClassAttribute):
    """This is a subclass of DynamicClassAttribute"""

class PropertySubSlots(DynamicClassAttribute):
    """This is a subclass of DynamicClassAttribute that defines __slots__"""
    __slots__ = ()

class PropertySubclassTests(unittest.TestCase):

    @unittest.skipIf(hasattr(PropertySubSlots, '__doc__'), '__doc__ is already present, __slots__ will have no effect')
    def test_slots_docstring_copy_exception(self):
        if False:
            print('Hello World!')
        try:

            class Foo(object):

                @PropertySubSlots
                def spam(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    'Trying to copy this docstring will raise an exception'
                    return 1
                print('\n', spam.__doc__)
        except AttributeError:
            pass
        else:
            raise Exception('AttributeError not raised')

    @unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def test_docstring_copy(self):
        if False:
            return 10

        class Foo(object):

            @PropertySub
            def spam(self):
                if False:
                    for i in range(10):
                        print('nop')
                'spam wrapped in DynamicClassAttribute subclass'
                return 1
        self.assertEqual(Foo.__dict__['spam'].__doc__, 'spam wrapped in DynamicClassAttribute subclass')

    @unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def test_property_setter_copies_getter_docstring(self):
        if False:
            print('Hello World!')

        class Foo(object):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self._spam = 1

            @PropertySub
            def spam(self):
                if False:
                    return 10
                'spam wrapped in DynamicClassAttribute subclass'
                return self._spam

            @spam.setter
            def spam(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                'this docstring is ignored'
                self._spam = value
        foo = Foo()
        self.assertEqual(foo.spam, 1)
        foo.spam = 2
        self.assertEqual(foo.spam, 2)
        self.assertEqual(Foo.__dict__['spam'].__doc__, 'spam wrapped in DynamicClassAttribute subclass')

        class FooSub(Foo):
            spam = Foo.__dict__['spam']

            @spam.setter
            def spam(self, value):
                if False:
                    i = 10
                    return i + 15
                'another ignored docstring'
                self._spam = 'eggs'
        foosub = FooSub()
        self.assertEqual(foosub.spam, 1)
        foosub.spam = 7
        self.assertEqual(foosub.spam, 'eggs')
        self.assertEqual(FooSub.__dict__['spam'].__doc__, 'spam wrapped in DynamicClassAttribute subclass')

    @unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def test_property_new_getter_new_docstring(self):
        if False:
            return 10

        class Foo(object):

            @PropertySub
            def spam(self):
                if False:
                    while True:
                        i = 10
                'a docstring'
                return 1

            @spam.getter
            def spam(self):
                if False:
                    print('Hello World!')
                'a new docstring'
                return 2
        self.assertEqual(Foo.__dict__['spam'].__doc__, 'a new docstring')

        class FooBase(object):

            @PropertySub
            def spam(self):
                if False:
                    return 10
                'a docstring'
                return 1

        class Foo2(FooBase):
            spam = FooBase.__dict__['spam']

            @spam.getter
            def spam(self):
                if False:
                    i = 10
                    return i + 15
                'a new docstring'
                return 2
        self.assertEqual(Foo.__dict__['spam'].__doc__, 'a new docstring')
if __name__ == '__main__':
    unittest.main()