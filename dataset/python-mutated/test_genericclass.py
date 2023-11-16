import contextlib
import unittest
import sys

class TestMROEntry(unittest.TestCase):

    def test_mro_entry_signature(self):
        if False:
            return 10
        tested = []

        class B:
            ...

        class C:

            def __mro_entries__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                tested.extend([args, kwargs])
                return (C,)
        c = C()
        self.assertEqual(tested, [])

        class D(B, c):
            ...
        self.assertEqual(tested[0], ((B, c),))
        self.assertEqual(tested[1], {})

    def test_mro_entry(self):
        if False:
            print('Hello World!')
        tested = []

        class A:
            ...

        class B:
            ...

        class C:

            def __mro_entries__(self, bases):
                if False:
                    while True:
                        i = 10
                tested.append(bases)
                return (self.__class__,)
        c = C()
        self.assertEqual(tested, [])

        class D(A, c, B):
            ...
        self.assertEqual(tested[-1], (A, c, B))
        self.assertEqual(D.__bases__, (A, C, B))
        self.assertEqual(D.__orig_bases__, (A, c, B))
        self.assertEqual(D.__mro__, (D, A, C, B, object))
        d = D()

        class E(d):
            ...
        self.assertEqual(tested[-1], (d,))
        self.assertEqual(E.__bases__, (D,))

    def test_mro_entry_none(self):
        if False:
            return 10
        tested = []

        class A:
            ...

        class B:
            ...

        class C:

            def __mro_entries__(self, bases):
                if False:
                    print('Hello World!')
                tested.append(bases)
                return ()
        c = C()
        self.assertEqual(tested, [])

        class D(A, c, B):
            ...
        self.assertEqual(tested[-1], (A, c, B))
        self.assertEqual(D.__bases__, (A, B))
        self.assertEqual(D.__orig_bases__, (A, c, B))
        self.assertEqual(D.__mro__, (D, A, B, object))

        class E(c):
            ...
        self.assertEqual(tested[-1], (c,))
        self.assertEqual(E.__bases__, (object,))
        self.assertEqual(E.__orig_bases__, (c,))
        self.assertEqual(E.__mro__, (E, object))

    def test_mro_entry_with_builtins(self):
        if False:
            return 10
        tested = []

        class A:
            ...

        class C:

            def __mro_entries__(self, bases):
                if False:
                    return 10
                tested.append(bases)
                return (dict,)
        c = C()
        self.assertEqual(tested, [])

        class D(A, c):
            ...
        self.assertEqual(tested[-1], (A, c))
        self.assertEqual(D.__bases__, (A, dict))
        self.assertEqual(D.__orig_bases__, (A, c))
        self.assertEqual(D.__mro__, (D, A, dict, object))

    def test_mro_entry_with_builtins_2(self):
        if False:
            while True:
                i = 10
        tested = []

        class C:

            def __mro_entries__(self, bases):
                if False:
                    print('Hello World!')
                tested.append(bases)
                return (C,)
        c = C()
        self.assertEqual(tested, [])

        class D(c, dict):
            ...
        self.assertEqual(tested[-1], (c, dict))
        self.assertEqual(D.__bases__, (C, dict))
        self.assertEqual(D.__orig_bases__, (c, dict))
        self.assertEqual(D.__mro__, (D, C, dict, object))

    def test_mro_entry_errors(self):
        if False:
            i = 10
            return i + 15

        class C_too_many:

            def __mro_entries__(self, bases, something, other):
                if False:
                    print('Hello World!')
                return ()
        c = C_too_many()
        with self.assertRaises(TypeError):

            class D(c):
                ...

        class C_too_few:

            def __mro_entries__(self):
                if False:
                    return 10
                return ()
        d = C_too_few()
        with self.assertRaises(TypeError):

            class D(d):
                ...

    def test_mro_entry_errors_2(self):
        if False:
            for i in range(10):
                print('nop')

        class C_not_callable:
            __mro_entries__ = 'Surprise!'
        c = C_not_callable()
        with self.assertRaises(TypeError):

            class D(c):
                ...

        class C_not_tuple:

            def __mro_entries__(self):
                if False:
                    i = 10
                    return i + 15
                return object
        c = C_not_tuple()
        with self.assertRaises(TypeError):

            class D(c):
                ...

    def test_mro_entry_metaclass(self):
        if False:
            for i in range(10):
                print('nop')
        meta_args = []

        class Meta(type):

            def __new__(mcls, name, bases, ns):
                if False:
                    print('Hello World!')
                meta_args.extend([mcls, name, bases, ns])
                return super().__new__(mcls, name, bases, ns)

        class A:
            ...

        class C:

            def __mro_entries__(self, bases):
                if False:
                    return 10
                return (A,)
        c = C()

        class D(c, metaclass=Meta):
            x = 1
        self.assertEqual(meta_args[0], Meta)
        self.assertEqual(meta_args[1], 'D')
        self.assertEqual(meta_args[2], (A,))
        self.assertEqual(meta_args[3]['x'], 1)
        self.assertEqual(D.__bases__, (A,))
        self.assertEqual(D.__orig_bases__, (c,))
        self.assertEqual(D.__mro__, (D, A, object))
        self.assertEqual(D.__class__, Meta)

    def test_mro_entry_type_call(self):
        if False:
            return 10

        class C:

            def __mro_entries__(self, bases):
                if False:
                    i = 10
                    return i + 15
                return ()
        c = C()
        with self.assertRaisesRegex(TypeError, 'MRO entry resolution; use types.new_class()'):
            type('Bad', (c,), {})

class TestClassGetitem(unittest.TestCase):

    def test_no_class_getitem(self):
        if False:
            print('Hello World!')

        class C:
            ...
        if hasattr(sys, 'pypy_version_info') and sys.pypy_version_info < (7, 3, 8):
            err = AttributeError
        else:
            err = TypeError
        with self.assertRaises(err):
            C[int]

    def test_class_getitem(self):
        if False:
            return 10
        getitem_args = []

        class C:

            def __class_getitem__(*args, **kwargs):
                if False:
                    print('Hello World!')
                getitem_args.extend([args, kwargs])
                return None
        C[int, str]
        self.assertEqual(getitem_args[0], (C, (int, str)))
        self.assertEqual(getitem_args[1], {})

    def test_class_getitem_format(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __class_getitem__(cls, item):
                if False:
                    return 10
                return f'C[{item.__name__}]'
        self.assertEqual(C[int], 'C[int]')
        self.assertEqual(C[C], 'C[C]')

    def test_class_getitem_inheritance(self):
        if False:
            print('Hello World!')

        class C:

            def __class_getitem__(cls, item):
                if False:
                    while True:
                        i = 10
                return f'{cls.__name__}[{item.__name__}]'

        class D(C):
            ...
        self.assertEqual(D[int], 'D[int]')
        self.assertEqual(D[D], 'D[D]')

    def test_class_getitem_inheritance_2(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __class_getitem__(cls, item):
                if False:
                    for i in range(10):
                        print('nop')
                return 'Should not see this'

        class D(C):

            def __class_getitem__(cls, item):
                if False:
                    return 10
                return f'{cls.__name__}[{item.__name__}]'
        self.assertEqual(D[int], 'D[int]')
        self.assertEqual(D[D], 'D[D]')

    def test_class_getitem_classmethod(self):
        if False:
            for i in range(10):
                print('nop')

        class C:

            @classmethod
            def __class_getitem__(cls, item):
                if False:
                    for i in range(10):
                        print('nop')
                return f'{cls.__name__}[{item.__name__}]'

        class D(C):
            ...
        self.assertEqual(D[int], 'D[int]')
        self.assertEqual(D[D], 'D[D]')

    def test_class_getitem_patched(self):
        if False:
            i = 10
            return i + 15

        class C:

            def __init_subclass__(cls):
                if False:
                    i = 10
                    return i + 15

                def __class_getitem__(cls, item):
                    if False:
                        print('Hello World!')
                    return f'{cls.__name__}[{item.__name__}]'
                cls.__class_getitem__ = classmethod(__class_getitem__)

        class D(C):
            ...
        self.assertEqual(D[int], 'D[int]')
        self.assertEqual(D[D], 'D[D]')

    def test_class_getitem_with_builtins(self):
        if False:
            while True:
                i = 10

        class A(dict):
            called_with = None

            def __class_getitem__(cls, item):
                if False:
                    print('Hello World!')
                cls.called_with = item

        class B(A):
            pass
        self.assertIs(B.called_with, None)
        B[int]
        self.assertIs(B.called_with, int)

    def test_class_getitem_errors(self):
        if False:
            return 10

        class C_too_few:

            def __class_getitem__(cls):
                if False:
                    return 10
                return None
        with self.assertRaises(TypeError):
            C_too_few[int]

        class C_too_many:

            def __class_getitem__(cls, one, two):
                if False:
                    while True:
                        i = 10
                return None
        with self.assertRaises(TypeError):
            C_too_many[int]

    def test_class_getitem_errors_2(self):
        if False:
            while True:
                i = 10

        class C:

            def __class_getitem__(cls, item):
                if False:
                    for i in range(10):
                        print('nop')
                return None
        with self.assertRaises(TypeError):
            C()[int]

        class E:
            ...
        e = E()
        e.__class_getitem__ = lambda cls, item: 'This will not work'
        with self.assertRaises(TypeError):
            e[int]

        class C_not_callable:
            __class_getitem__ = 'Surprise!'
        with self.assertRaises(TypeError):
            C_not_callable[int]

    def test_class_getitem_metaclass(self):
        if False:
            for i in range(10):
                print('nop')

        class Meta(type):

            def __class_getitem__(cls, item):
                if False:
                    return 10
                return f'{cls.__name__}[{item.__name__}]'
        self.assertEqual(Meta[int], 'Meta[int]')

    def test_class_getitem_with_metaclass(self):
        if False:
            print('Hello World!')

        class Meta(type):
            pass

        class C(metaclass=Meta):

            def __class_getitem__(cls, item):
                if False:
                    i = 10
                    return i + 15
                return f'{cls.__name__}[{item.__name__}]'
        self.assertEqual(C[int], 'C[int]')

    def test_class_getitem_metaclass_first(self):
        if False:
            return 10

        class Meta(type):

            def __getitem__(cls, item):
                if False:
                    return 10
                return 'from metaclass'

        class C(metaclass=Meta):

            def __class_getitem__(cls, item):
                if False:
                    i = 10
                    return i + 15
                return 'from __class_getitem__'
        self.assertEqual(C[int], 'from metaclass')
if __name__ == '__main__':
    unittest.main()