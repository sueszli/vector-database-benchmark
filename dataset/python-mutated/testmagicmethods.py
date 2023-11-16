import math
import unittest
import os
from asyncio import iscoroutinefunction
from unittest.mock import AsyncMock, Mock, MagicMock, _magics

class TestMockingMagicMethods(unittest.TestCase):

    def test_deleting_magic_methods(self):
        if False:
            print('Hello World!')
        mock = Mock()
        self.assertFalse(hasattr(mock, '__getitem__'))
        mock.__getitem__ = Mock()
        self.assertTrue(hasattr(mock, '__getitem__'))
        del mock.__getitem__
        self.assertFalse(hasattr(mock, '__getitem__'))

    def test_magicmock_del(self):
        if False:
            i = 10
            return i + 15
        mock = MagicMock()
        del mock.__getitem__
        self.assertRaises(TypeError, lambda : mock['foo'])
        mock = MagicMock()
        mock['foo']
        del mock.__getitem__
        self.assertRaises(TypeError, lambda : mock['foo'])

    def test_magic_method_wrapping(self):
        if False:
            return 10
        mock = Mock()

        def f(self, name):
            if False:
                i = 10
                return i + 15
            return (self, 'fish')
        mock.__getitem__ = f
        self.assertIsNot(mock.__getitem__, f)
        self.assertEqual(mock['foo'], (mock, 'fish'))
        self.assertEqual(mock.__getitem__('foo'), (mock, 'fish'))
        mock.__getitem__ = mock
        self.assertIs(mock.__getitem__, mock)

    def test_magic_methods_isolated_between_mocks(self):
        if False:
            i = 10
            return i + 15
        mock1 = Mock()
        mock2 = Mock()
        mock1.__iter__ = Mock(return_value=iter([]))
        self.assertEqual(list(mock1), [])
        self.assertRaises(TypeError, lambda : list(mock2))

    def test_repr(self):
        if False:
            return 10
        mock = Mock()
        self.assertEqual(repr(mock), "<Mock id='%s'>" % id(mock))
        mock.__repr__ = lambda s: 'foo'
        self.assertEqual(repr(mock), 'foo')

    def test_str(self):
        if False:
            i = 10
            return i + 15
        mock = Mock()
        self.assertEqual(str(mock), object.__str__(mock))
        mock.__str__ = lambda s: 'foo'
        self.assertEqual(str(mock), 'foo')

    def test_dict_methods(self):
        if False:
            while True:
                i = 10
        mock = Mock()
        self.assertRaises(TypeError, lambda : mock['foo'])

        def _del():
            if False:
                return 10
            del mock['foo']

        def _set():
            if False:
                while True:
                    i = 10
            mock['foo'] = 3
        self.assertRaises(TypeError, _del)
        self.assertRaises(TypeError, _set)
        _dict = {}

        def getitem(s, name):
            if False:
                print('Hello World!')
            return _dict[name]

        def setitem(s, name, value):
            if False:
                while True:
                    i = 10
            _dict[name] = value

        def delitem(s, name):
            if False:
                for i in range(10):
                    print('nop')
            del _dict[name]
        mock.__setitem__ = setitem
        mock.__getitem__ = getitem
        mock.__delitem__ = delitem
        self.assertRaises(KeyError, lambda : mock['foo'])
        mock['foo'] = 'bar'
        self.assertEqual(_dict, {'foo': 'bar'})
        self.assertEqual(mock['foo'], 'bar')
        del mock['foo']
        self.assertEqual(_dict, {})

    def test_numeric(self):
        if False:
            for i in range(10):
                print('nop')
        original = mock = Mock()
        mock.value = 0
        self.assertRaises(TypeError, lambda : mock + 3)

        def add(self, other):
            if False:
                print('Hello World!')
            mock.value += other
            return self
        mock.__add__ = add
        self.assertEqual(mock + 3, mock)
        self.assertEqual(mock.value, 3)
        del mock.__add__

        def iadd(mock):
            if False:
                while True:
                    i = 10
            mock += 3
        self.assertRaises(TypeError, iadd, mock)
        mock.__iadd__ = add
        mock += 6
        self.assertEqual(mock, original)
        self.assertEqual(mock.value, 9)
        self.assertRaises(TypeError, lambda : 3 + mock)
        mock.__radd__ = add
        self.assertEqual(7 + mock, mock)
        self.assertEqual(mock.value, 16)

    def test_division(self):
        if False:
            return 10
        original = mock = Mock()
        mock.value = 32
        self.assertRaises(TypeError, lambda : mock / 2)

        def truediv(self, other):
            if False:
                return 10
            mock.value /= other
            return self
        mock.__truediv__ = truediv
        self.assertEqual(mock / 2, mock)
        self.assertEqual(mock.value, 16)
        del mock.__truediv__

        def itruediv(mock):
            if False:
                for i in range(10):
                    print('nop')
            mock /= 4
        self.assertRaises(TypeError, itruediv, mock)
        mock.__itruediv__ = truediv
        mock /= 8
        self.assertEqual(mock, original)
        self.assertEqual(mock.value, 2)
        self.assertRaises(TypeError, lambda : 8 / mock)
        mock.__rtruediv__ = truediv
        self.assertEqual(0.5 / mock, mock)
        self.assertEqual(mock.value, 4)

    def test_hash(self):
        if False:
            return 10
        mock = Mock()
        self.assertEqual(hash(mock), Mock.__hash__(mock))

        def _hash(s):
            if False:
                i = 10
                return i + 15
            return 3
        mock.__hash__ = _hash
        self.assertEqual(hash(mock), 3)

    def test_nonzero(self):
        if False:
            while True:
                i = 10
        m = Mock()
        self.assertTrue(bool(m))
        m.__bool__ = lambda s: False
        self.assertFalse(bool(m))

    def test_comparison(self):
        if False:
            i = 10
            return i + 15
        mock = Mock()

        def comp(s, o):
            if False:
                while True:
                    i = 10
            return True
        mock.__lt__ = mock.__gt__ = mock.__le__ = mock.__ge__ = comp
        self.assertTrue(mock < 3)
        self.assertTrue(mock > 3)
        self.assertTrue(mock <= 3)
        self.assertTrue(mock >= 3)
        self.assertRaises(TypeError, lambda : MagicMock() < object())
        self.assertRaises(TypeError, lambda : object() < MagicMock())
        self.assertRaises(TypeError, lambda : MagicMock() < MagicMock())
        self.assertRaises(TypeError, lambda : MagicMock() > object())
        self.assertRaises(TypeError, lambda : object() > MagicMock())
        self.assertRaises(TypeError, lambda : MagicMock() > MagicMock())
        self.assertRaises(TypeError, lambda : MagicMock() <= object())
        self.assertRaises(TypeError, lambda : object() <= MagicMock())
        self.assertRaises(TypeError, lambda : MagicMock() <= MagicMock())
        self.assertRaises(TypeError, lambda : MagicMock() >= object())
        self.assertRaises(TypeError, lambda : object() >= MagicMock())
        self.assertRaises(TypeError, lambda : MagicMock() >= MagicMock())

    def test_equality(self):
        if False:
            while True:
                i = 10
        for mock in (Mock(), MagicMock()):
            self.assertEqual(mock == mock, True)
            self.assertIsInstance(mock == mock, bool)
            self.assertEqual(mock != mock, False)
            self.assertIsInstance(mock != mock, bool)
            self.assertEqual(mock == object(), False)
            self.assertEqual(mock != object(), True)

            def eq(self, other):
                if False:
                    print('Hello World!')
                return other == 3
            mock.__eq__ = eq
            self.assertTrue(mock == 3)
            self.assertFalse(mock == 4)

            def ne(self, other):
                if False:
                    i = 10
                    return i + 15
                return other == 3
            mock.__ne__ = ne
            self.assertTrue(mock != 3)
            self.assertFalse(mock != 4)
        mock = MagicMock()
        mock.__eq__.return_value = True
        self.assertIsInstance(mock == 3, bool)
        self.assertEqual(mock == 3, True)
        mock.__ne__.return_value = False
        self.assertIsInstance(mock != 3, bool)
        self.assertEqual(mock != 3, False)

    def test_len_contains_iter(self):
        if False:
            print('Hello World!')
        mock = Mock()
        self.assertRaises(TypeError, len, mock)
        self.assertRaises(TypeError, iter, mock)
        self.assertRaises(TypeError, lambda : 'foo' in mock)
        mock.__len__ = lambda s: 6
        self.assertEqual(len(mock), 6)
        mock.__contains__ = lambda s, o: o == 3
        self.assertIn(3, mock)
        self.assertNotIn(6, mock)
        mock.__iter__ = lambda s: iter('foobarbaz')
        self.assertEqual(list(mock), list('foobarbaz'))

    def test_magicmock(self):
        if False:
            i = 10
            return i + 15
        mock = MagicMock()
        mock.__iter__.return_value = iter([1, 2, 3])
        self.assertEqual(list(mock), [1, 2, 3])
        getattr(mock, '__bool__').return_value = False
        self.assertFalse(hasattr(mock, '__nonzero__'))
        self.assertFalse(bool(mock))
        for entry in _magics:
            self.assertTrue(hasattr(mock, entry))
        self.assertFalse(hasattr(mock, '__imaginary__'))

    def test_magic_mock_equality(self):
        if False:
            print('Hello World!')
        mock = MagicMock()
        self.assertIsInstance(mock == object(), bool)
        self.assertIsInstance(mock != object(), bool)
        self.assertEqual(mock == object(), False)
        self.assertEqual(mock != object(), True)
        self.assertEqual(mock == mock, True)
        self.assertEqual(mock != mock, False)

    def test_asyncmock_defaults(self):
        if False:
            i = 10
            return i + 15
        mock = AsyncMock()
        self.assertEqual(int(mock), 1)
        self.assertEqual(complex(mock), 1j)
        self.assertEqual(float(mock), 1.0)
        self.assertNotIn(object(), mock)
        self.assertEqual(len(mock), 0)
        self.assertEqual(list(mock), [])
        self.assertEqual(hash(mock), object.__hash__(mock))
        self.assertEqual(str(mock), object.__str__(mock))
        self.assertTrue(bool(mock))
        self.assertEqual(round(mock), mock.__round__())
        self.assertEqual(math.trunc(mock), mock.__trunc__())
        self.assertEqual(math.floor(mock), mock.__floor__())
        self.assertEqual(math.ceil(mock), mock.__ceil__())
        self.assertTrue(iscoroutinefunction(mock.__aexit__))
        self.assertTrue(iscoroutinefunction(mock.__aenter__))
        self.assertIsInstance(mock.__aenter__, AsyncMock)
        self.assertIsInstance(mock.__aexit__, AsyncMock)
        self.assertEqual(oct(mock), '0o1')
        self.assertEqual(hex(mock), '0x1')

    def test_magicmock_defaults(self):
        if False:
            i = 10
            return i + 15
        mock = MagicMock()
        self.assertEqual(int(mock), 1)
        self.assertEqual(complex(mock), 1j)
        self.assertEqual(float(mock), 1.0)
        self.assertNotIn(object(), mock)
        self.assertEqual(len(mock), 0)
        self.assertEqual(list(mock), [])
        self.assertEqual(hash(mock), object.__hash__(mock))
        self.assertEqual(str(mock), object.__str__(mock))
        self.assertTrue(bool(mock))
        self.assertEqual(round(mock), mock.__round__())
        self.assertEqual(math.trunc(mock), mock.__trunc__())
        self.assertEqual(math.floor(mock), mock.__floor__())
        self.assertEqual(math.ceil(mock), mock.__ceil__())
        self.assertTrue(iscoroutinefunction(mock.__aexit__))
        self.assertTrue(iscoroutinefunction(mock.__aenter__))
        self.assertIsInstance(mock.__aenter__, AsyncMock)
        self.assertIsInstance(mock.__aexit__, AsyncMock)
        self.assertEqual(oct(mock), '0o1')
        self.assertEqual(hex(mock), '0x1')

    def test_magic_methods_fspath(self):
        if False:
            while True:
                i = 10
        mock = MagicMock()
        expected_path = mock.__fspath__()
        mock.reset_mock()
        self.assertEqual(os.fspath(mock), expected_path)
        mock.__fspath__.assert_called_once()

    def test_magic_methods_and_spec(self):
        if False:
            return 10

        class Iterable(object):

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                pass
        mock = Mock(spec=Iterable)
        self.assertRaises(AttributeError, lambda : mock.__iter__)
        mock.__iter__ = Mock(return_value=iter([]))
        self.assertEqual(list(mock), [])

        class NonIterable(object):
            pass
        mock = Mock(spec=NonIterable)
        self.assertRaises(AttributeError, lambda : mock.__iter__)

        def set_int():
            if False:
                i = 10
                return i + 15
            mock.__int__ = Mock(return_value=iter([]))
        self.assertRaises(AttributeError, set_int)
        mock = MagicMock(spec=Iterable)
        self.assertEqual(list(mock), [])
        self.assertRaises(AttributeError, set_int)

    def test_magic_methods_and_spec_set(self):
        if False:
            print('Hello World!')

        class Iterable(object):

            def __iter__(self):
                if False:
                    while True:
                        i = 10
                pass
        mock = Mock(spec_set=Iterable)
        self.assertRaises(AttributeError, lambda : mock.__iter__)
        mock.__iter__ = Mock(return_value=iter([]))
        self.assertEqual(list(mock), [])

        class NonIterable(object):
            pass
        mock = Mock(spec_set=NonIterable)
        self.assertRaises(AttributeError, lambda : mock.__iter__)

        def set_int():
            if False:
                return 10
            mock.__int__ = Mock(return_value=iter([]))
        self.assertRaises(AttributeError, set_int)
        mock = MagicMock(spec_set=Iterable)
        self.assertEqual(list(mock), [])
        self.assertRaises(AttributeError, set_int)

    def test_setting_unsupported_magic_method(self):
        if False:
            while True:
                i = 10
        mock = MagicMock()

        def set_setattr():
            if False:
                return 10
            mock.__setattr__ = lambda self, name: None
        self.assertRaisesRegex(AttributeError, "Attempting to set unsupported magic method '__setattr__'.", set_setattr)

    def test_attributes_and_return_value(self):
        if False:
            print('Hello World!')
        mock = MagicMock()
        attr = mock.foo

        def _get_type(obj):
            if False:
                while True:
                    i = 10
            return type(obj).__mro__[1]
        self.assertEqual(_get_type(attr), MagicMock)
        returned = mock()
        self.assertEqual(_get_type(returned), MagicMock)

    def test_magic_methods_are_magic_mocks(self):
        if False:
            while True:
                i = 10
        mock = MagicMock()
        self.assertIsInstance(mock.__getitem__, MagicMock)
        mock[1][2].__getitem__.return_value = 3
        self.assertEqual(mock[1][2][3], 3)

    def test_magic_method_reset_mock(self):
        if False:
            i = 10
            return i + 15
        mock = MagicMock()
        str(mock)
        self.assertTrue(mock.__str__.called)
        mock.reset_mock()
        self.assertFalse(mock.__str__.called)

    def test_dir(self):
        if False:
            while True:
                i = 10
        for mock in (Mock(), MagicMock()):

            def _dir(self):
                if False:
                    print('Hello World!')
                return ['foo']
            mock.__dir__ = _dir
            self.assertEqual(dir(mock), ['foo'])

    def test_bound_methods(self):
        if False:
            while True:
                i = 10
        m = Mock()
        m.__iter__ = [3].__iter__
        self.assertRaises(TypeError, iter, m)

    def test_magic_method_type(self):
        if False:
            i = 10
            return i + 15

        class Foo(MagicMock):
            pass
        foo = Foo()
        self.assertIsInstance(foo.__int__, Foo)

    def test_descriptor_from_class(self):
        if False:
            i = 10
            return i + 15
        m = MagicMock()
        type(m).__str__.return_value = 'foo'
        self.assertEqual(str(m), 'foo')

    def test_iterable_as_iter_return_value(self):
        if False:
            i = 10
            return i + 15
        m = MagicMock()
        m.__iter__.return_value = [1, 2, 3]
        self.assertEqual(list(m), [1, 2, 3])
        self.assertEqual(list(m), [1, 2, 3])
        m.__iter__.return_value = iter([4, 5, 6])
        self.assertEqual(list(m), [4, 5, 6])
        self.assertEqual(list(m), [])

    def test_matmul(self):
        if False:
            print('Hello World!')
        m = MagicMock()
        self.assertIsInstance(m @ 1, MagicMock)
        m.__matmul__.return_value = 42
        m.__rmatmul__.return_value = 666
        m.__imatmul__.return_value = 24
        self.assertEqual(m @ 1, 42)
        self.assertEqual(1 @ m, 666)
        m @= 24
        self.assertEqual(m, 24)

    def test_divmod_and_rdivmod(self):
        if False:
            return 10
        m = MagicMock()
        self.assertIsInstance(divmod(5, m), MagicMock)
        m.__divmod__.return_value = (2, 1)
        self.assertEqual(divmod(m, 2), (2, 1))
        m = MagicMock()
        foo = divmod(2, m)
        self.assertIsInstance(foo, MagicMock)
        foo_direct = m.__divmod__(2)
        self.assertIsInstance(foo_direct, MagicMock)
        bar = divmod(m, 2)
        self.assertIsInstance(bar, MagicMock)
        bar_direct = m.__rdivmod__(2)
        self.assertIsInstance(bar_direct, MagicMock)

    def test_magic_in_initialization(self):
        if False:
            print('Hello World!')
        m = MagicMock(**{'__str__.return_value': '12'})
        self.assertEqual(str(m), '12')

    def test_changing_magic_set_in_initialization(self):
        if False:
            return 10
        m = MagicMock(**{'__str__.return_value': '12'})
        m.__str__.return_value = '13'
        self.assertEqual(str(m), '13')
        m = MagicMock(**{'__str__.return_value': '12'})
        m.configure_mock(**{'__str__.return_value': '14'})
        self.assertEqual(str(m), '14')
if __name__ == '__main__':
    unittest.main()