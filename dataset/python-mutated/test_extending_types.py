"""
Test extending types via the numba.extending.* API.
"""
import operator
from numba import njit, literally
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError
from numba.core.extending import lower_builtin
from numba.core.extending import models, register_model
from numba.core.extending import make_attribute_wrapper
from numba.core.extending import type_callable
from numba.core.extending import overload
from numba.core.extending import typeof_impl
import unittest

def gen_mock_float():
    if False:
        i = 10
        return i + 15

    def mock_float(x):
        if False:
            while True:
                i = 10
        pass
    return mock_float

class TestExtTypDummy(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')

        class Dummy(object):

            def __init__(self, value):
                if False:
                    print('Hello World!')
                self.value = value

        class DummyType(types.Type):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(DummyType, self).__init__(name='Dummy')
        dummy_type = DummyType()

        @register_model(DummyType)
        class DummyModel(models.StructModel):

            def __init__(self, dmm, fe_type):
                if False:
                    i = 10
                    return i + 15
                members = [('value', types.intp)]
                models.StructModel.__init__(self, dmm, fe_type, members)
        make_attribute_wrapper(DummyType, 'value', 'value')

        @type_callable(Dummy)
        def type_dummy(context):
            if False:
                print('Hello World!')

            def typer(value):
                if False:
                    for i in range(10):
                        print('nop')
                return dummy_type
            return typer

        @lower_builtin(Dummy, types.intp)
        def impl_dummy(context, builder, sig, args):
            if False:
                i = 10
                return i + 15
            typ = sig.return_type
            [value] = args
            dummy = cgutils.create_struct_proxy(typ)(context, builder)
            dummy.value = value
            return dummy._getvalue()

        @typeof_impl.register(Dummy)
        def typeof_dummy(val, c):
            if False:
                i = 10
                return i + 15
            return DummyType()
        self.Dummy = Dummy
        self.DummyType = DummyType

    def _add_float_overload(self, mock_float_inst):
        if False:
            return 10

        @overload(mock_float_inst)
        def dummy_to_float(x):
            if False:
                while True:
                    i = 10
            if isinstance(x, self.DummyType):

                def codegen(x):
                    if False:
                        while True:
                            i = 10
                    return float(x.value)
                return codegen
            else:
                raise NumbaTypeError('cannot type float({})'.format(x))

    def test_overload_float(self):
        if False:
            return 10
        mock_float = gen_mock_float()
        self._add_float_overload(mock_float)
        Dummy = self.Dummy

        @njit
        def foo(x):
            if False:
                while True:
                    i = 10
            return mock_float(Dummy(x))
        self.assertEqual(foo(123), float(123))

    def test_overload_float_error_msg(self):
        if False:
            while True:
                i = 10
        mock_float = gen_mock_float()
        self._add_float_overload(mock_float)

        @njit
        def foo(x):
            if False:
                print('Hello World!')
            return mock_float(x)
        with self.assertRaises(TypingError) as raises:
            foo(1j)
        self.assertIn('cannot type float(complex128)', str(raises.exception))

    def test_unboxing(self):
        if False:
            while True:
                i = 10
        'A test for the unboxing logic on unknown type\n        '
        Dummy = self.Dummy

        @njit
        def foo(x):
            if False:
                while True:
                    i = 10
            bar(Dummy(x))

        @njit(no_cpython_wrapper=False)
        def bar(dummy_obj):
            if False:
                return 10
            pass
        foo(123)
        with self.assertRaises(TypeError) as raises:
            bar(Dummy(123))
        self.assertIn("can't unbox Dummy type", str(raises.exception))

    def test_boxing(self):
        if False:
            return 10
        'A test for the boxing logic on unknown type\n        '
        Dummy = self.Dummy

        @njit
        def foo(x):
            if False:
                print('Hello World!')
            return Dummy(x)
        with self.assertRaises(TypeError) as raises:
            foo(123)
        self.assertIn('cannot convert native Dummy to Python object', str(raises.exception))

    def test_issue5565_literal_getitem(self):
        if False:
            while True:
                i = 10
        (Dummy, DummyType) = (self.Dummy, self.DummyType)
        MAGIC_NUMBER = 12321

        @overload(operator.getitem)
        def dummy_getitem_ovld(self, idx):
            if False:
                while True:
                    i = 10
            if not isinstance(self, DummyType):
                return None
            if isinstance(idx, types.StringLiteral):

                def dummy_getitem_impl(self, idx):
                    if False:
                        for i in range(10):
                            print('nop')
                    return MAGIC_NUMBER
                return dummy_getitem_impl
            if isinstance(idx, types.UnicodeType):

                def dummy_getitem_impl(self, idx):
                    if False:
                        while True:
                            i = 10
                    return literally(idx)
                return dummy_getitem_impl
            return None

        @njit
        def test_impl(x, y):
            if False:
                return 10
            return Dummy(x)[y]
        var = 'abc'
        self.assertEqual(test_impl(1, var), MAGIC_NUMBER)