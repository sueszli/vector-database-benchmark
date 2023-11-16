import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import boolean, deferred_type, float32, float64, int16, int32, njit, optional, typeof
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy

class TestClass1(object):

    def __init__(self, x, y, z=1, *, a=5):
        if False:
            i = 10
            return i + 15
        self.x = x
        self.y = y
        self.z = z
        self.a = a

class TestClass2(object):

    def __init__(self, x, y, z=1, *args, a=5):
        if False:
            print('Hello World!')
        self.x = x
        self.y = y
        self.z = z
        self.args = args
        self.a = a

def _get_meminfo(box):
    if False:
        for i in range(10):
            print('nop')
    ptr = _box.box_get_meminfoptr(box)
    mi = MemInfo(ptr)
    mi.acquire()
    return mi

class TestJitClass(TestCase, MemoryLeakMixin):

    def _check_spec(self, spec=None, test_cls=None, all_expected=None):
        if False:
            while True:
                i = 10
        if test_cls is None:

            @jitclass(spec)
            class Test(object):

                def __init__(self):
                    if False:
                        return 10
                    pass
            test_cls = Test
        clsty = test_cls.class_type.instance_type
        names = list(clsty.struct.keys())
        values = list(clsty.struct.values())
        if all_expected is None:
            if isinstance(spec, OrderedDict):
                all_expected = spec.items()
            else:
                all_expected = spec
        assert all_expected is not None
        self.assertEqual(len(names), len(all_expected))
        for (got, expected) in zip(zip(names, values), all_expected):
            self.assertEqual(got[0], expected[0])
            self.assertEqual(got[1], expected[1])

    def test_ordereddict_spec(self):
        if False:
            for i in range(10):
                print('nop')
        spec = OrderedDict()
        spec['x'] = int32
        spec['y'] = float32
        self._check_spec(spec)

    def test_list_spec(self):
        if False:
            i = 10
            return i + 15
        spec = [('x', int32), ('y', float32)]
        self._check_spec(spec)

    def test_type_annotations(self):
        if False:
            return 10
        spec = [('x', int32)]

        @jitclass(spec)
        class Test1(object):
            x: int
            y: pt.List[float]

            def __init__(self):
                if False:
                    print('Hello World!')
                pass
        self._check_spec(spec, Test1, spec + [('y', types.ListType(float64))])

    def test_type_annotation_inheritance(self):
        if False:
            while True:
                i = 10

        class Foo:
            x: int

        @jitclass
        class Bar(Foo):
            y: float

            def __init__(self, value: float) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                self.x = int(value)
                self.y = value
        self._check_spec(test_cls=Bar, all_expected=[('x', typeof(0)), ('y', typeof(0.0))])

    def test_spec_errors(self):
        if False:
            print('Hello World!')
        spec1 = [('x', int), ('y', float32[:])]
        spec2 = [(1, int32), ('y', float32[:])]

        class Test(object):

            def __init__(self):
                if False:
                    print('Hello World!')
                pass
        with self.assertRaises(TypeError) as raises:
            jitclass(Test, spec1)
        self.assertIn('spec values should be Numba type instances', str(raises.exception))
        with self.assertRaises(TypeError) as raises:
            jitclass(Test, spec2)
        self.assertEqual(str(raises.exception), 'spec keys should be strings, got 1')

    def test_init_errors(self):
        if False:
            return 10

        @jitclass([])
        class Test:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                return 7
        with self.assertRaises(errors.TypingError) as raises:
            Test()
        self.assertIn('__init__() should return None, not', str(raises.exception))

    def _make_Float2AndArray(self):
        if False:
            while True:
                i = 10
        spec = OrderedDict()
        spec['x'] = float32
        spec['y'] = float32
        spec['arr'] = float32[:]

        @jitclass(spec)
        class Float2AndArray(object):

            def __init__(self, x, y, arr):
                if False:
                    while True:
                        i = 10
                self.x = x
                self.y = y
                self.arr = arr

            def add(self, val):
                if False:
                    i = 10
                    return i + 15
                self.x += val
                self.y += val
                return val
        return Float2AndArray

    def _make_Vector2(self):
        if False:
            while True:
                i = 10
        spec = OrderedDict()
        spec['x'] = int32
        spec['y'] = int32

        @jitclass(spec)
        class Vector2(object):

            def __init__(self, x, y):
                if False:
                    print('Hello World!')
                self.x = x
                self.y = y
        return Vector2

    def test_jit_class_1(self):
        if False:
            return 10
        Float2AndArray = self._make_Float2AndArray()
        Vector2 = self._make_Vector2()

        @njit
        def bar(obj):
            if False:
                for i in range(10):
                    print('nop')
            return obj.x + obj.y

        @njit
        def foo(a):
            if False:
                i = 10
                return i + 15
            obj = Float2AndArray(1, 2, a)
            obj.add(123)
            vec = Vector2(3, 4)
            return (bar(obj), bar(vec), obj.arr)
        inp = np.ones(10, dtype=np.float32)
        (a, b, c) = foo(inp)
        self.assertEqual(a, 123 + 1 + 123 + 2)
        self.assertEqual(b, 3 + 4)
        self.assertPreciseEqual(c, inp)

    def test_jitclass_usage_from_python(self):
        if False:
            return 10
        Float2AndArray = self._make_Float2AndArray()

        @njit
        def identity(obj):
            if False:
                for i in range(10):
                    print('nop')
            return obj

        @njit
        def retrieve_attributes(obj):
            if False:
                for i in range(10):
                    print('nop')
            return (obj.x, obj.y, obj.arr)
        arr = np.arange(10, dtype=np.float32)
        obj = Float2AndArray(1, 2, arr)
        obj_meminfo = _get_meminfo(obj)
        self.assertEqual(obj_meminfo.refcount, 2)
        self.assertEqual(obj_meminfo.data, _box.box_get_dataptr(obj))
        self.assertEqual(obj._numba_type_.class_type, Float2AndArray.class_type)
        other = identity(obj)
        other_meminfo = _get_meminfo(other)
        self.assertEqual(obj_meminfo.refcount, 4)
        self.assertEqual(other_meminfo.refcount, 4)
        self.assertEqual(other_meminfo.data, _box.box_get_dataptr(other))
        self.assertEqual(other_meminfo.data, obj_meminfo.data)
        del other, other_meminfo
        self.assertEqual(obj_meminfo.refcount, 2)
        (out_x, out_y, out_arr) = retrieve_attributes(obj)
        self.assertEqual(out_x, 1)
        self.assertEqual(out_y, 2)
        self.assertIs(out_arr, arr)
        self.assertEqual(obj.x, 1)
        self.assertEqual(obj.y, 2)
        self.assertIs(obj.arr, arr)
        self.assertEqual(obj.add(123), 123)
        self.assertEqual(obj.x, 1 + 123)
        self.assertEqual(obj.y, 2 + 123)
        obj.x = 333
        obj.y = 444
        obj.arr = newarr = np.arange(5, dtype=np.float32)
        self.assertEqual(obj.x, 333)
        self.assertEqual(obj.y, 444)
        self.assertIs(obj.arr, newarr)

    def test_jitclass_datalayout(self):
        if False:
            while True:
                i = 10
        spec = OrderedDict()
        spec['val'] = boolean

        @jitclass(spec)
        class Foo(object):

            def __init__(self, val):
                if False:
                    i = 10
                    return i + 15
                self.val = val
        self.assertTrue(Foo(True).val)
        self.assertFalse(Foo(False).val)

    def test_deferred_type(self):
        if False:
            print('Hello World!')
        node_type = deferred_type()
        spec = OrderedDict()
        spec['data'] = float32
        spec['next'] = optional(node_type)

        @njit
        def get_data(node):
            if False:
                i = 10
                return i + 15
            return node.data

        @jitclass(spec)
        class LinkedNode(object):

            def __init__(self, data, next):
                if False:
                    i = 10
                    return i + 15
                self.data = data
                self.next = next

            def get_next_data(self):
                if False:
                    for i in range(10):
                        print('nop')
                return get_data(self.next)

            def append_to_tail(self, other):
                if False:
                    i = 10
                    return i + 15
                cur = self
                while cur.next is not None:
                    cur = cur.next
                cur.next = other
        node_type.define(LinkedNode.class_type.instance_type)
        first = LinkedNode(123, None)
        self.assertEqual(first.data, 123)
        self.assertIsNone(first.next)
        second = LinkedNode(321, first)
        first_meminfo = _get_meminfo(first)
        second_meminfo = _get_meminfo(second)
        self.assertEqual(first_meminfo.refcount, 3)
        self.assertEqual(second.next.data, first.data)
        self.assertEqual(first_meminfo.refcount, 3)
        self.assertEqual(second_meminfo.refcount, 2)
        first_val = second.get_next_data()
        self.assertEqual(first_val, first.data)
        self.assertIsNone(first.next)
        second.append_to_tail(LinkedNode(567, None))
        self.assertIsNotNone(first.next)
        self.assertEqual(first.next.data, 567)
        self.assertIsNone(first.next.next)
        second.append_to_tail(LinkedNode(678, None))
        self.assertIsNotNone(first.next.next)
        self.assertEqual(first.next.next.data, 678)
        self.assertEqual(first_meminfo.refcount, 3)
        del second, second_meminfo
        self.assertEqual(first_meminfo.refcount, 2)

    def test_c_structure(self):
        if False:
            i = 10
            return i + 15
        spec = OrderedDict()
        spec['a'] = int32
        spec['b'] = int16
        spec['c'] = float64

        @jitclass(spec)
        class Struct(object):

            def __init__(self, a, b, c):
                if False:
                    return 10
                self.a = a
                self.b = b
                self.c = c
        st = Struct(43981, 239, 3.1415)

        class CStruct(ctypes.Structure):
            _fields_ = [('a', ctypes.c_int32), ('b', ctypes.c_int16), ('c', ctypes.c_double)]
        ptr = ctypes.c_void_p(_box.box_get_dataptr(st))
        cstruct = ctypes.cast(ptr, ctypes.POINTER(CStruct))[0]
        self.assertEqual(cstruct.a, st.a)
        self.assertEqual(cstruct.b, st.b)
        self.assertEqual(cstruct.c, st.c)

    def test_is(self):
        if False:
            print('Hello World!')
        Vector = self._make_Vector2()
        vec_a = Vector(1, 2)

        @njit
        def do_is(a, b):
            if False:
                while True:
                    i = 10
            return a is b
        with self.assertRaises(LoweringError) as raises:
            do_is(vec_a, vec_a)
        self.assertIn('no default `is` implementation', str(raises.exception))

    def test_isinstance(self):
        if False:
            for i in range(10):
                print('nop')
        Vector2 = self._make_Vector2()
        vec = Vector2(1, 2)
        self.assertIsInstance(vec, Vector2)

    def test_subclassing(self):
        if False:
            return 10
        Vector2 = self._make_Vector2()
        with self.assertRaises(TypeError) as raises:

            class SubV(Vector2):
                pass
        self.assertEqual(str(raises.exception), 'cannot subclass from a jitclass')

    def test_base_class(self):
        if False:
            print('Hello World!')

        class Base(object):

            def what(self):
                if False:
                    i = 10
                    return i + 15
                return self.attr

        @jitclass([('attr', int32)])
        class Test(Base):

            def __init__(self, attr):
                if False:
                    print('Hello World!')
                self.attr = attr
        obj = Test(123)
        self.assertEqual(obj.what(), 123)

    def test_globals(self):
        if False:
            i = 10
            return i + 15

        class Mine(object):
            constant = 123

            def __init__(self):
                if False:
                    while True:
                        i = 10
                pass
        with self.assertRaises(TypeError) as raises:
            jitclass(Mine)
        self.assertEqual(str(raises.exception), 'class members are not yet supported: constant')

    def test_user_getter_setter(self):
        if False:
            while True:
                i = 10

        @jitclass([('attr', int32)])
        class Foo(object):

            def __init__(self, attr):
                if False:
                    i = 10
                    return i + 15
                self.attr = attr

            @property
            def value(self):
                if False:
                    print('Hello World!')
                return self.attr + 1

            @value.setter
            def value(self, val):
                if False:
                    print('Hello World!')
                self.attr = val - 1
        foo = Foo(123)
        self.assertEqual(foo.attr, 123)
        self.assertEqual(foo.value, 123 + 1)
        foo.value = 789
        self.assertEqual(foo.attr, 789 - 1)
        self.assertEqual(foo.value, 789)

        @njit
        def bar(foo, val):
            if False:
                return 10
            a = foo.value
            foo.value = val
            b = foo.value
            c = foo.attr
            return (a, b, c)
        (a, b, c) = bar(foo, 567)
        self.assertEqual(a, 789)
        self.assertEqual(b, 567)
        self.assertEqual(c, 567 - 1)

    def test_user_deleter_error(self):
        if False:
            return 10

        class Foo(object):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                pass

            @property
            def value(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 1

            @value.deleter
            def value(self):
                if False:
                    while True:
                        i = 10
                pass
        with self.assertRaises(TypeError) as raises:
            jitclass(Foo)
        self.assertEqual(str(raises.exception), 'deleter is not supported: value')

    def test_name_shadowing_error(self):
        if False:
            return 10

        class Foo(object):

            def __init__(self):
                if False:
                    print('Hello World!')
                pass

            @property
            def my_property(self):
                if False:
                    return 10
                pass

            def my_method(self):
                if False:
                    print('Hello World!')
                pass
        with self.assertRaises(NameError) as raises:
            jitclass(Foo, [('my_property', int32)])
        self.assertEqual(str(raises.exception), 'name shadowing: my_property')
        with self.assertRaises(NameError) as raises:
            jitclass(Foo, [('my_method', int32)])
        self.assertEqual(str(raises.exception), 'name shadowing: my_method')

    def test_distinct_classes(self):
        if False:
            while True:
                i = 10

        @jitclass([('x', int32)])
        class Foo(object):

            def __init__(self, x):
                if False:
                    i = 10
                    return i + 15
                self.x = x + 2

            def run(self):
                if False:
                    print('Hello World!')
                return self.x + 1
        FirstFoo = Foo

        @jitclass([('x', int32)])
        class Foo(object):

            def __init__(self, x):
                if False:
                    return 10
                self.x = x - 2

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.x - 1
        SecondFoo = Foo
        foo = FirstFoo(5)
        self.assertEqual(foo.x, 7)
        self.assertEqual(foo.run(), 8)
        foo = SecondFoo(5)
        self.assertEqual(foo.x, 3)
        self.assertEqual(foo.run(), 2)

    def test_parameterized(self):
        if False:
            i = 10
            return i + 15

        class MyClass(object):

            def __init__(self, value):
                if False:
                    i = 10
                    return i + 15
                self.value = value

        def create_my_class(value):
            if False:
                print('Hello World!')
            cls = jitclass(MyClass, [('value', typeof(value))])
            return cls(value)
        a = create_my_class(123)
        self.assertEqual(a.value, 123)
        b = create_my_class(12.3)
        self.assertEqual(b.value, 12.3)
        c = create_my_class(np.array([123]))
        np.testing.assert_equal(c.value, [123])
        d = create_my_class(np.array([12.3]))
        np.testing.assert_equal(d.value, [12.3])

    def test_protected_attrs(self):
        if False:
            while True:
                i = 10
        spec = {'value': int32, '_value': float32, '__value': int32, '__value__': int32}

        @jitclass(spec)
        class MyClass(object):

            def __init__(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                self.value = value
                self._value = value / 2
                self.__value = value * 2
                self.__value__ = value - 1

            @property
            def private_value(self):
                if False:
                    print('Hello World!')
                return self.__value

            @property
            def _inner_value(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self._value

            @_inner_value.setter
            def _inner_value(self, v):
                if False:
                    return 10
                self._value = v

            @property
            def __private_value(self):
                if False:
                    return 10
                return self.__value

            @__private_value.setter
            def __private_value(self, v):
                if False:
                    i = 10
                    return i + 15
                self.__value = v

            def swap_private_value(self, new):
                if False:
                    print('Hello World!')
                old = self.__private_value
                self.__private_value = new
                return old

            def _protected_method(self, factor):
                if False:
                    for i in range(10):
                        print('nop')
                return self._value * factor

            def __private_method(self, factor):
                if False:
                    while True:
                        i = 10
                return self.__value * factor

            def check_private_method(self, factor):
                if False:
                    print('Hello World!')
                return self.__private_method(factor)
        value = 123
        inst = MyClass(value)
        self.assertEqual(inst.value, value)
        self.assertEqual(inst._value, value / 2)
        self.assertEqual(inst.private_value, value * 2)
        self.assertEqual(inst._inner_value, inst._value)
        freeze_inst_value = inst._value
        inst._inner_value -= 1
        self.assertEqual(inst._inner_value, freeze_inst_value - 1)
        self.assertEqual(inst.swap_private_value(321), value * 2)
        self.assertEqual(inst.swap_private_value(value * 2), 321)
        self.assertEqual(inst._protected_method(3), inst._value * 3)
        self.assertEqual(inst.check_private_method(3), inst.private_value * 3)
        self.assertEqual(inst.__value__, value - 1)
        inst.__value__ -= 100
        self.assertEqual(inst.__value__, value - 101)

        @njit
        def access_dunder(inst):
            if False:
                for i in range(10):
                    print('nop')
            return inst.__value
        with self.assertRaises(errors.TypingError) as raises:
            access_dunder(inst)
        self.assertIn('_TestJitClass__value', str(raises.exception))
        with self.assertRaises(AttributeError) as raises:
            access_dunder.py_func(inst)
        self.assertIn('_TestJitClass__value', str(raises.exception))

    @skip_if_typeguard
    def test_annotations(self):
        if False:
            while True:
                i = 10
        '\n        Methods with annotations should compile fine (issue #1911).\n        '
        from .annotation_usecases import AnnotatedClass
        spec = {'x': int32}
        cls = jitclass(AnnotatedClass, spec)
        obj = cls(5)
        self.assertEqual(obj.x, 5)
        self.assertEqual(obj.add(2), 7)

    def test_docstring(self):
        if False:
            print('Hello World!')

        @jitclass
        class Apple(object):
            """Class docstring"""

            def __init__(self):
                if False:
                    while True:
                        i = 10
                'init docstring'

            def foo(self):
                if False:
                    for i in range(10):
                        print('nop')
                'foo method docstring'

            @property
            def aval(self):
                if False:
                    i = 10
                    return i + 15
                'aval property docstring'
        self.assertEqual(Apple.__doc__, 'Class docstring')
        self.assertEqual(Apple.__init__.__doc__, 'init docstring')
        self.assertEqual(Apple.foo.__doc__, 'foo method docstring')
        self.assertEqual(Apple.aval.__doc__, 'aval property docstring')

    def test_kwargs(self):
        if False:
            while True:
                i = 10
        spec = [('a', int32), ('b', float64)]

        @jitclass(spec)
        class TestClass(object):

            def __init__(self, x, y, z):
                if False:
                    print('Hello World!')
                self.a = x * y
                self.b = z
        x = 2
        y = 2
        z = 1.1
        kwargs = {'y': y, 'z': z}
        tc = TestClass(x=2, **kwargs)
        self.assertEqual(tc.a, x * y)
        self.assertEqual(tc.b, z)

    def test_default_args(self):
        if False:
            i = 10
            return i + 15
        spec = [('x', int32), ('y', int32), ('z', int32)]

        @jitclass(spec)
        class TestClass(object):

            def __init__(self, x, y, z=1):
                if False:
                    print('Hello World!')
                self.x = x
                self.y = y
                self.z = z
        tc = TestClass(1, 2, 3)
        self.assertEqual(tc.x, 1)
        self.assertEqual(tc.y, 2)
        self.assertEqual(tc.z, 3)
        tc = TestClass(1, 2)
        self.assertEqual(tc.x, 1)
        self.assertEqual(tc.y, 2)
        self.assertEqual(tc.z, 1)
        tc = TestClass(y=2, z=5, x=1)
        self.assertEqual(tc.x, 1)
        self.assertEqual(tc.y, 2)
        self.assertEqual(tc.z, 5)

    def test_default_args_keyonly(self):
        if False:
            while True:
                i = 10
        spec = [('x', int32), ('y', int32), ('z', int32), ('a', int32)]
        TestClass = jitclass(TestClass1, spec)
        tc = TestClass(2, 3)
        self.assertEqual(tc.x, 2)
        self.assertEqual(tc.y, 3)
        self.assertEqual(tc.z, 1)
        self.assertEqual(tc.a, 5)
        tc = TestClass(y=4, x=2, a=42, z=100)
        self.assertEqual(tc.x, 2)
        self.assertEqual(tc.y, 4)
        self.assertEqual(tc.z, 100)
        self.assertEqual(tc.a, 42)
        tc = TestClass(y=4, x=2, a=42)
        self.assertEqual(tc.x, 2)
        self.assertEqual(tc.y, 4)
        self.assertEqual(tc.z, 1)
        self.assertEqual(tc.a, 42)
        tc = TestClass(y=4, x=2)
        self.assertEqual(tc.x, 2)
        self.assertEqual(tc.y, 4)
        self.assertEqual(tc.z, 1)
        self.assertEqual(tc.a, 5)

    def test_default_args_starargs_and_keyonly(self):
        if False:
            for i in range(10):
                print('nop')
        spec = [('x', int32), ('y', int32), ('z', int32), ('args', types.UniTuple(int32, 2)), ('a', int32)]
        with self.assertRaises(errors.UnsupportedError) as raises:
            jitclass(TestClass2, spec)
        msg = 'VAR_POSITIONAL argument type unsupported'
        self.assertIn(msg, str(raises.exception))

    def test_generator_method(self):
        if False:
            print('Hello World!')
        spec = []

        @jitclass(spec)
        class TestClass(object):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def gen(self, niter):
                if False:
                    return 10
                for i in range(niter):
                    yield np.arange(i)

        def expected_gen(niter):
            if False:
                while True:
                    i = 10
            for i in range(niter):
                yield np.arange(i)
        for niter in range(10):
            for (expect, got) in zip(expected_gen(niter), TestClass().gen(niter)):
                self.assertPreciseEqual(expect, got)

    def test_getitem(self):
        if False:
            print('Hello World!')
        spec = [('data', int32[:])]

        @jitclass(spec)
        class TestClass(object):

            def __init__(self):
                if False:
                    return 10
                self.data = np.zeros(10, dtype=np.int32)

            def __setitem__(self, key, data):
                if False:
                    while True:
                        i = 10
                self.data[key] = data

            def __getitem__(self, key):
                if False:
                    i = 10
                    return i + 15
                return self.data[key]

        @njit
        def create_and_set_indices():
            if False:
                return 10
            t = TestClass()
            t[1] = 1
            t[2] = 2
            t[3] = 3
            return t

        @njit
        def get_index(t, n):
            if False:
                for i in range(10):
                    print('nop')
            return t[n]
        t = create_and_set_indices()
        self.assertEqual(get_index(t, 1), 1)
        self.assertEqual(get_index(t, 2), 2)
        self.assertEqual(get_index(t, 3), 3)

    def test_getitem_unbox(self):
        if False:
            return 10
        spec = [('data', int32[:])]

        @jitclass(spec)
        class TestClass(object):

            def __init__(self):
                if False:
                    return 10
                self.data = np.zeros(10, dtype=np.int32)

            def __setitem__(self, key, data):
                if False:
                    return 10
                self.data[key] = data

            def __getitem__(self, key):
                if False:
                    for i in range(10):
                        print('nop')
                return self.data[key]
        t = TestClass()
        t[1] = 10

        @njit
        def set2return1(t):
            if False:
                while True:
                    i = 10
            t[2] = 20
            return t[1]
        t_1 = set2return1(t)
        self.assertEqual(t_1, 10)
        self.assertEqual(t[2], 20)

    def test_getitem_complex_key(self):
        if False:
            i = 10
            return i + 15
        spec = [('data', int32[:, :])]

        @jitclass(spec)
        class TestClass(object):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.data = np.zeros((10, 10), dtype=np.int32)

            def __setitem__(self, key, data):
                if False:
                    return 10
                self.data[int(key.real), int(key.imag)] = data

            def __getitem__(self, key):
                if False:
                    for i in range(10):
                        print('nop')
                return self.data[int(key.real), int(key.imag)]
        t = TestClass()
        t[complex(1, 1)] = 3

        @njit
        def get_key(t, real, imag):
            if False:
                i = 10
                return i + 15
            return t[complex(real, imag)]

        @njit
        def set_key(t, real, imag, data):
            if False:
                return 10
            t[complex(real, imag)] = data
        self.assertEqual(get_key(t, 1, 1), 3)
        set_key(t, 2, 2, 4)
        self.assertEqual(t[complex(2, 2)], 4)

    def test_getitem_tuple_key(self):
        if False:
            for i in range(10):
                print('nop')
        spec = [('data', int32[:, :])]

        @jitclass(spec)
        class TestClass(object):

            def __init__(self):
                if False:
                    return 10
                self.data = np.zeros((10, 10), dtype=np.int32)

            def __setitem__(self, key, data):
                if False:
                    while True:
                        i = 10
                self.data[key[0], key[1]] = data

            def __getitem__(self, key):
                if False:
                    for i in range(10):
                        print('nop')
                return self.data[key[0], key[1]]
        t = TestClass()
        t[1, 1] = 11

        @njit
        def get11(t):
            if False:
                return 10
            return t[1, 1]

        @njit
        def set22(t, data):
            if False:
                i = 10
                return i + 15
            t[2, 2] = data
        self.assertEqual(get11(t), 11)
        set22(t, 22)
        self.assertEqual(t[2, 2], 22)

    def test_getitem_slice_key(self):
        if False:
            for i in range(10):
                print('nop')
        spec = [('data', int32[:])]

        @jitclass(spec)
        class TestClass(object):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.data = np.zeros(10, dtype=np.int32)

            def __setitem__(self, slc, data):
                if False:
                    while True:
                        i = 10
                self.data[slc.start] = data
                self.data[slc.stop] = data + slc.step

            def __getitem__(self, slc):
                if False:
                    for i in range(10):
                        print('nop')
                return self.data[slc.start]
        t = TestClass()
        t[1:5:1] = 1
        self.assertEqual(t[1:1:1], 1)
        self.assertEqual(t[5:5:5], 2)

        @njit
        def get5(t):
            if False:
                return 10
            return t[5:6:1]
        self.assertEqual(get5(t), 2)

        @njit
        def set26(t, data):
            if False:
                print('Hello World!')
            t[2:6:1] = data
        set26(t, 2)
        self.assertEqual(t[2:2:1], 2)
        self.assertEqual(t[6:6:1], 3)

    def test_jitclass_longlabel_not_truncated(self):
        if False:
            return 10
        alphabet = [chr(ord('a') + x) for x in range(26)]
        spec = [(letter * 10, float64) for letter in alphabet]
        spec.extend([(letter.upper() * 10, float64) for letter in alphabet])

        @jitclass(spec)
        class TruncatedLabel(object):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.aaaaaaaaaa = 10.0

            def meth1(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.bbbbbbbbbb = random.gauss(self.aaaaaaaaaa, self.aaaaaaaaaa)

            def meth2(self):
                if False:
                    return 10
                self.meth1()
        TruncatedLabel().meth2()

    def test_pickling(self):
        if False:
            print('Hello World!')

        @jitclass
        class PickleTestSubject(object):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        inst = PickleTestSubject()
        ty = typeof(inst)
        self.assertIsInstance(ty, types.ClassInstanceType)
        pickled = pickle.dumps(ty)
        self.assertIs(pickle.loads(pickled), ty)

    def test_static_methods(self):
        if False:
            for i in range(10):
                print('nop')

        @jitclass([('x', int32)])
        class Test1:

            def __init__(self, x):
                if False:
                    print('Hello World!')
                self.x = x

            def increase(self, y):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = self.add(self.x, y)
                return self.x

            @staticmethod
            def add(a, b):
                if False:
                    return 10
                return a + b

            @staticmethod
            def sub(a, b):
                if False:
                    while True:
                        i = 10
                return a - b

        @jitclass([('x', int32)])
        class Test2:

            def __init__(self, x):
                if False:
                    i = 10
                    return i + 15
                self.x = x

            def increase(self, y):
                if False:
                    while True:
                        i = 10
                self.x = self.add(self.x, y)
                return self.x

            @staticmethod
            def add(a, b):
                if False:
                    return 10
                return a - b
        self.assertIsInstance(Test1.add, Dispatcher)
        self.assertIsInstance(Test1.sub, Dispatcher)
        self.assertIsInstance(Test2.add, Dispatcher)
        self.assertNotEqual(Test1.add, Test2.add)
        self.assertEqual(3, Test1.add(1, 2))
        self.assertEqual(-1, Test2.add(1, 2))
        self.assertEqual(4, Test1.sub(6, 2))
        t1 = Test1(0)
        t2 = Test2(0)
        self.assertEqual(1, t1.increase(1))
        self.assertEqual(-1, t2.increase(1))
        self.assertEqual(2, t1.add(1, 1))
        self.assertEqual(0, t1.sub(1, 1))
        self.assertEqual(0, t2.add(1, 1))
        self.assertEqual(2j, t1.add(1j, 1j))
        self.assertEqual(1j, t1.sub(2j, 1j))
        self.assertEqual('foobar', t1.add('foo', 'bar'))
        with self.assertRaises(AttributeError) as raises:
            Test2.sub(3, 1)
        self.assertIn("has no attribute 'sub'", str(raises.exception))
        with self.assertRaises(TypeError) as raises:
            Test1.add(3)
        self.assertIn('not enough arguments: expected 2, got 1', str(raises.exception))

        @jitclass([])
        class Test3:

            def __init__(self):
                if False:
                    print('Hello World!')
                pass

            @staticmethod
            def a_static_method(a, b):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def call_static(self):
                if False:
                    for i in range(10):
                        print('nop')
                return Test3.a_static_method(1, 2)
        invalid = Test3()
        with self.assertRaises(errors.TypingError) as raises:
            invalid.call_static()
        self.assertIn("Unknown attribute 'a_static_method'", str(raises.exception))

    def test_jitclass_decorator_usecases(self):
        if False:
            print('Hello World!')
        spec = OrderedDict(x=float64)

        @jitclass()
        class Test1:
            x: float

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = 0
        self.assertIsInstance(Test1, JitClassType)
        self.assertDictEqual(Test1.class_type.struct, spec)

        @jitclass(spec=spec)
        class Test2:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = 0
        self.assertIsInstance(Test2, JitClassType)
        self.assertDictEqual(Test2.class_type.struct, spec)

        @jitclass
        class Test3:
            x: float

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.x = 0
        self.assertIsInstance(Test3, JitClassType)
        self.assertDictEqual(Test3.class_type.struct, spec)

        @jitclass(spec)
        class Test4:

            def __init__(self):
                if False:
                    print('Hello World!')
                self.x = 0
        self.assertIsInstance(Test4, JitClassType)
        self.assertDictEqual(Test4.class_type.struct, spec)

    def test_jitclass_function_usecases(self):
        if False:
            i = 10
            return i + 15
        spec = OrderedDict(x=float64)

        class AnnotatedTest:
            x: float

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.x = 0
        JitTest1 = jitclass(AnnotatedTest)
        self.assertIsInstance(JitTest1, JitClassType)
        self.assertDictEqual(JitTest1.class_type.struct, spec)

        class UnannotatedTest:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = 0
        JitTest2 = jitclass(UnannotatedTest, spec)
        self.assertIsInstance(JitTest2, JitClassType)
        self.assertDictEqual(JitTest2.class_type.struct, spec)

    def test_jitclass_isinstance(self):
        if False:
            return 10
        spec = OrderedDict(value=int32)

        @jitclass(spec)
        class Foo(object):

            def __init__(self, value):
                if False:
                    for i in range(10):
                        print('nop')
                self.value = value

            def getValue(self):
                if False:
                    print('Hello World!')
                return self.value

            def getValueIncr(self):
                if False:
                    print('Hello World!')
                return self.value + 1

        @jitclass(spec)
        class Bar(object):

            def __init__(self, value):
                if False:
                    return 10
                self.value = value

            def getValue(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.value

        def test_jitclass_isinstance(obj):
            if False:
                return 10
            if isinstance(obj, (Foo, Bar)):
                x = obj.getValue()
                if isinstance(obj, Foo):
                    return (obj.getValueIncr() + x, 'Foo')
                else:
                    return (obj.getValue() + x, 'Bar')
            else:
                return 'no match'
        pyfunc = test_jitclass_isinstance
        cfunc = njit(test_jitclass_isinstance)
        self.assertIsInstance(Foo, JitClassType)
        self.assertEqual(pyfunc(Foo(3)), cfunc(Foo(3)))
        self.assertEqual(pyfunc(Bar(123)), cfunc(Bar(123)))
        self.assertEqual(pyfunc(0), cfunc(0))

    def test_jitclass_unsupported_dunder(self):
        if False:
            return 10
        with self.assertRaises(TypeError) as e:

            @jitclass
            class Foo(object):

                def __init__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    return

                def __enter__(self):
                    if False:
                        i = 10
                        return i + 15
                    return None
            Foo()
        self.assertIn("Method '__enter__' is not supported.", str(e.exception))

    def test_modulename(self):
        if False:
            print('Hello World!')

        @jitclass
        class TestModname(object):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.x = 12
        thisModule = __name__
        classModule = TestModname.__module__
        self.assertEqual(thisModule, classModule)

class TestJitClassOverloads(MemoryLeakMixin, TestCase):

    class PyList:

        def __init__(self):
            if False:
                return 10
            self.x = [0]

        def append(self, y):
            if False:
                for i in range(10):
                    print('nop')
            self.x.append(y)

        def clear(self):
            if False:
                i = 10
                return i + 15
            self.x.clear()

        def __abs__(self):
            if False:
                for i in range(10):
                    print('nop')
            return len(self.x) * 7

        def __bool__(self):
            if False:
                for i in range(10):
                    print('nop')
            return len(self.x) % 3 != 0

        def __complex__(self):
            if False:
                return 10
            c = complex(2)
            if self.x:
                c += self.x[0]
            return c

        def __contains__(self, y):
            if False:
                print('Hello World!')
            return y in self.x

        def __float__(self):
            if False:
                print('Hello World!')
            f = 3.1415
            if self.x:
                f += self.x[0]
            return f

        def __int__(self):
            if False:
                return 10
            i = 5
            if self.x:
                i += self.x[0]
            return i

        def __len__(self):
            if False:
                for i in range(10):
                    print('nop')
            return len(self.x) + 1

        def __str__(self):
            if False:
                return 10
            if len(self.x) == 0:
                return 'PyList empty'
            else:
                return 'PyList non-empty'

    @staticmethod
    def get_int_wrapper():
        if False:
            while True:
                i = 10

        @jitclass([('x', types.intp)])
        class IntWrapper:

            def __init__(self, value):
                if False:
                    return 10
                self.x = value

            def __eq__(self, other):
                if False:
                    i = 10
                    return i + 15
                return self.x == other.x

            def __hash__(self):
                if False:
                    print('Hello World!')
                return self.x

            def __lshift__(self, other):
                if False:
                    print('Hello World!')
                return IntWrapper(self.x << other.x)

            def __rshift__(self, other):
                if False:
                    i = 10
                    return i + 15
                return IntWrapper(self.x >> other.x)

            def __and__(self, other):
                if False:
                    i = 10
                    return i + 15
                return IntWrapper(self.x & other.x)

            def __or__(self, other):
                if False:
                    return 10
                return IntWrapper(self.x | other.x)

            def __xor__(self, other):
                if False:
                    while True:
                        i = 10
                return IntWrapper(self.x ^ other.x)
        return IntWrapper

    @staticmethod
    def get_float_wrapper():
        if False:
            while True:
                i = 10

        @jitclass([('x', types.float64)])
        class FloatWrapper:

            def __init__(self, value):
                if False:
                    print('Hello World!')
                self.x = value

            def __eq__(self, other):
                if False:
                    i = 10
                    return i + 15
                return self.x == other.x

            def __hash__(self):
                if False:
                    i = 10
                    return i + 15
                return self.x

            def __ge__(self, other):
                if False:
                    print('Hello World!')
                return self.x >= other.x

            def __gt__(self, other):
                if False:
                    return 10
                return self.x > other.x

            def __le__(self, other):
                if False:
                    return 10
                return self.x <= other.x

            def __lt__(self, other):
                if False:
                    return 10
                return self.x < other.x

            def __add__(self, other):
                if False:
                    i = 10
                    return i + 15
                return FloatWrapper(self.x + other.x)

            def __floordiv__(self, other):
                if False:
                    i = 10
                    return i + 15
                return FloatWrapper(self.x // other.x)

            def __mod__(self, other):
                if False:
                    while True:
                        i = 10
                return FloatWrapper(self.x % other.x)

            def __mul__(self, other):
                if False:
                    return 10
                return FloatWrapper(self.x * other.x)

            def __neg__(self, other):
                if False:
                    print('Hello World!')
                return FloatWrapper(-self.x)

            def __pos__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return FloatWrapper(+self.x)

            def __pow__(self, other):
                if False:
                    return 10
                return FloatWrapper(self.x ** other.x)

            def __sub__(self, other):
                if False:
                    print('Hello World!')
                return FloatWrapper(self.x - other.x)

            def __truediv__(self, other):
                if False:
                    print('Hello World!')
                return FloatWrapper(self.x / other.x)
        return FloatWrapper

    def assertSame(self, first, second, msg=None):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(type(first), type(second), msg=msg)
        self.assertEqual(first, second, msg=msg)

    def test_overloads(self):
        if False:
            print('Hello World!')
        JitList = jitclass({'x': types.List(types.intp)})(self.PyList)
        py_funcs = [lambda x: abs(x), lambda x: x.__abs__(), lambda x: bool(x), lambda x: x.__bool__(), lambda x: complex(x), lambda x: x.__complex__(), lambda x: 0 in x, lambda x: x.__contains__(0), lambda x: float(x), lambda x: x.__float__(), lambda x: int(x), lambda x: x.__int__(), lambda x: len(x), lambda x: x.__len__(), lambda x: str(x), lambda x: x.__str__(), lambda x: 1 if x else 0]
        jit_funcs = [njit(f) for f in py_funcs]
        py_list = self.PyList()
        jit_list = JitList()
        for (py_f, jit_f) in zip(py_funcs, jit_funcs):
            self.assertSame(py_f(py_list), py_f(jit_list))
            self.assertSame(py_f(py_list), jit_f(jit_list))
        py_list.append(2)
        jit_list.append(2)
        for (py_f, jit_f) in zip(py_funcs, jit_funcs):
            self.assertSame(py_f(py_list), py_f(jit_list))
            self.assertSame(py_f(py_list), jit_f(jit_list))
        py_list.append(-5)
        jit_list.append(-5)
        for (py_f, jit_f) in zip(py_funcs, jit_funcs):
            self.assertSame(py_f(py_list), py_f(jit_list))
            self.assertSame(py_f(py_list), jit_f(jit_list))
        py_list.clear()
        jit_list.clear()
        for (py_f, jit_f) in zip(py_funcs, jit_funcs):
            self.assertSame(py_f(py_list), py_f(jit_list))
            self.assertSame(py_f(py_list), jit_f(jit_list))

    def test_bool_fallback(self):
        if False:
            for i in range(10):
                print('nop')

        def py_b(x):
            if False:
                while True:
                    i = 10
            return bool(x)
        jit_b = njit(py_b)

        @jitclass([('x', types.List(types.intp))])
        class LenClass:

            def __init__(self, x):
                if False:
                    while True:
                        i = 10
                self.x = x

            def __len__(self):
                if False:
                    while True:
                        i = 10
                return len(self.x) % 4

            def append(self, y):
                if False:
                    for i in range(10):
                        print('nop')
                self.x.append(y)

            def pop(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.x.pop(0)
        obj = LenClass([1, 2, 3])
        self.assertTrue(py_b(obj))
        self.assertTrue(jit_b(obj))
        obj.append(4)
        self.assertFalse(py_b(obj))
        self.assertFalse(jit_b(obj))
        obj.pop()
        self.assertTrue(py_b(obj))
        self.assertTrue(jit_b(obj))

        @jitclass([('y', types.float64)])
        class NormalClass:

            def __init__(self, y):
                if False:
                    for i in range(10):
                        print('nop')
                self.y = y
        obj = NormalClass(0)
        self.assertTrue(py_b(obj))
        self.assertTrue(jit_b(obj))

    def test_numeric_fallback(self):
        if False:
            print('Hello World!')

        def py_c(x):
            if False:
                return 10
            return complex(x)

        def py_f(x):
            if False:
                return 10
            return float(x)

        def py_i(x):
            if False:
                i = 10
                return i + 15
            return int(x)
        jit_c = njit(py_c)
        jit_f = njit(py_f)
        jit_i = njit(py_i)

        @jitclass([])
        class FloatClass:

            def __init__(self):
                if False:
                    print('Hello World!')
                pass

            def __float__(self):
                if False:
                    while True:
                        i = 10
                return 3.1415
        obj = FloatClass()
        self.assertSame(py_c(obj), complex(3.1415))
        self.assertSame(jit_c(obj), complex(3.1415))
        self.assertSame(py_f(obj), 3.1415)
        self.assertSame(jit_f(obj), 3.1415)
        with self.assertRaises(TypeError) as e:
            py_i(obj)
        self.assertIn('int', str(e.exception))
        with self.assertRaises(TypingError) as e:
            jit_i(obj)
        self.assertIn('int', str(e.exception))

        @jitclass([])
        class IntClass:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def __int__(self):
                if False:
                    return 10
                return 7
        obj = IntClass()
        self.assertSame(py_i(obj), 7)
        self.assertSame(jit_i(obj), 7)
        with self.assertRaises(TypeError) as e:
            py_c(obj)
        self.assertIn('complex', str(e.exception))
        with self.assertRaises(TypingError) as e:
            jit_c(obj)
        self.assertIn('complex', str(e.exception))
        with self.assertRaises(TypeError) as e:
            py_f(obj)
        self.assertIn('float', str(e.exception))
        with self.assertRaises(TypingError) as e:
            jit_f(obj)
        self.assertIn('float', str(e.exception))

        @jitclass([])
        class IndexClass:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def __index__(self):
                if False:
                    while True:
                        i = 10
                return 1
        obj = IndexClass()
        self.assertSame(py_c(obj), complex(1))
        self.assertSame(jit_c(obj), complex(1))
        self.assertSame(py_f(obj), 1.0)
        self.assertSame(jit_f(obj), 1.0)
        self.assertSame(py_i(obj), 1)
        self.assertSame(jit_i(obj), 1)

        @jitclass([])
        class FloatIntIndexClass:

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def __float__(self):
                if False:
                    i = 10
                    return i + 15
                return 3.1415

            def __int__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 7

            def __index__(self):
                if False:
                    i = 10
                    return i + 15
                return 1
        obj = FloatIntIndexClass()
        self.assertSame(py_c(obj), complex(3.1415))
        self.assertSame(jit_c(obj), complex(3.1415))
        self.assertSame(py_f(obj), 3.1415)
        self.assertSame(jit_f(obj), 3.1415)
        self.assertSame(py_i(obj), 7)
        self.assertSame(jit_i(obj), 7)

    def test_arithmetic_logical(self):
        if False:
            while True:
                i = 10
        IntWrapper = self.get_int_wrapper()
        FloatWrapper = self.get_float_wrapper()
        float_py_funcs = [lambda x, y: x == y, lambda x, y: x != y, lambda x, y: x >= y, lambda x, y: x > y, lambda x, y: x <= y, lambda x, y: x < y, lambda x, y: x + y, lambda x, y: x // y, lambda x, y: x % y, lambda x, y: x * y, lambda x, y: x ** y, lambda x, y: x - y, lambda x, y: x / y]
        int_py_funcs = [lambda x, y: x == y, lambda x, y: x != y, lambda x, y: x << y, lambda x, y: x >> y, lambda x, y: x & y, lambda x, y: x | y, lambda x, y: x ^ y]
        test_values = [(0.0, 2.0), (1.234, 3.1415), (13.1, 1.01)]

        def unwrap(value):
            if False:
                while True:
                    i = 10
            return getattr(value, 'x', value)
        for (jit_f, (x, y)) in itertools.product(map(njit, float_py_funcs), test_values):
            py_f = jit_f.py_func
            expected = py_f(x, y)
            jit_x = FloatWrapper(x)
            jit_y = FloatWrapper(y)
            check = self.assertEqual if type(expected) is not float else self.assertAlmostEqual
            check(expected, jit_f(x, y))
            check(expected, unwrap(py_f(jit_x, jit_y)))
            check(expected, unwrap(jit_f(jit_x, jit_y)))
        for (jit_f, (x, y)) in itertools.product(map(njit, int_py_funcs), test_values):
            py_f = jit_f.py_func
            (x, y) = (int(x), int(y))
            expected = py_f(x, y)
            jit_x = IntWrapper(x)
            jit_y = IntWrapper(y)
            self.assertEqual(expected, jit_f(x, y))
            self.assertEqual(expected, unwrap(py_f(jit_x, jit_y)))
            self.assertEqual(expected, unwrap(jit_f(jit_x, jit_y)))

    def test_arithmetic_logical_inplace(self):
        if False:
            return 10
        JitIntWrapper = self.get_int_wrapper()
        JitFloatWrapper = self.get_float_wrapper()
        PyIntWrapper = JitIntWrapper.mro()[1]
        PyFloatWrapper = JitFloatWrapper.mro()[1]

        @jitclass([('x', types.intp)])
        class JitIntUpdateWrapper(PyIntWrapper):

            def __init__(self, value):
                if False:
                    return 10
                self.x = value

            def __ilshift__(self, other):
                if False:
                    return 10
                return JitIntUpdateWrapper(self.x << other.x)

            def __irshift__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return JitIntUpdateWrapper(self.x >> other.x)

            def __iand__(self, other):
                if False:
                    return 10
                return JitIntUpdateWrapper(self.x & other.x)

            def __ior__(self, other):
                if False:
                    while True:
                        i = 10
                return JitIntUpdateWrapper(self.x | other.x)

            def __ixor__(self, other):
                if False:
                    i = 10
                    return i + 15
                return JitIntUpdateWrapper(self.x ^ other.x)

        @jitclass({'x': types.float64})
        class JitFloatUpdateWrapper(PyFloatWrapper):

            def __init__(self, value):
                if False:
                    i = 10
                    return i + 15
                self.x = value

            def __iadd__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return JitFloatUpdateWrapper(self.x + 2.718 * other.x)

            def __ifloordiv__(self, other):
                if False:
                    return 10
                return JitFloatUpdateWrapper(self.x * 2.718 // other.x)

            def __imod__(self, other):
                if False:
                    while True:
                        i = 10
                return JitFloatUpdateWrapper(self.x % (other.x + 1))

            def __imul__(self, other):
                if False:
                    i = 10
                    return i + 15
                return JitFloatUpdateWrapper(self.x * other.x + 1)

            def __ipow__(self, other):
                if False:
                    while True:
                        i = 10
                return JitFloatUpdateWrapper(self.x ** other.x + 1)

            def __isub__(self, other):
                if False:
                    while True:
                        i = 10
                return JitFloatUpdateWrapper(self.x - 3.1415 * other.x)

            def __itruediv__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return JitFloatUpdateWrapper((self.x + 1) / other.x)
        PyIntUpdateWrapper = JitIntUpdateWrapper.mro()[1]
        PyFloatUpdateWrapper = JitFloatUpdateWrapper.mro()[1]

        def get_update_func(op):
            if False:
                print('Hello World!')
            template = f'\ndef f(x, y):\n    x {op}= y\n    return x\n'
            namespace = {}
            exec(template, namespace)
            return namespace['f']
        float_py_funcs = [get_update_func(op) for op in ['+', '//', '%', '*', '**', '-', '/']]
        int_py_funcs = [get_update_func(op) for op in ['<<', '>>', '&', '|', '^']]
        test_values = [(0.0, 2.0), (1.234, 3.1415), (13.1, 1.01)]
        for (jit_f, (py_cls, jit_cls), (x, y)) in itertools.product(map(njit, float_py_funcs), [(PyFloatWrapper, JitFloatWrapper), (PyFloatUpdateWrapper, JitFloatUpdateWrapper)], test_values):
            py_f = jit_f.py_func
            expected = py_f(py_cls(x), py_cls(y)).x
            self.assertAlmostEqual(expected, py_f(jit_cls(x), jit_cls(y)).x)
            self.assertAlmostEqual(expected, jit_f(jit_cls(x), jit_cls(y)).x)
        for (jit_f, (py_cls, jit_cls), (x, y)) in itertools.product(map(njit, int_py_funcs), [(PyIntWrapper, JitIntWrapper), (PyIntUpdateWrapper, JitIntUpdateWrapper)], test_values):
            (x, y) = (int(x), int(y))
            py_f = jit_f.py_func
            expected = py_f(py_cls(x), py_cls(y)).x
            self.assertEqual(expected, py_f(jit_cls(x), jit_cls(y)).x)
            self.assertEqual(expected, jit_f(jit_cls(x), jit_cls(y)).x)

    def test_hash_eq_ne(self):
        if False:
            return 10

        class HashEqTest:
            x: int

            def __init__(self, x):
                if False:
                    print('Hello World!')
                self.x = x

            def __hash__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.x % 10

            def __eq__(self, o):
                if False:
                    i = 10
                    return i + 15
                return (self.x - o.x) % 20 == 0

        class HashEqNeTest(HashEqTest):

            def __ne__(self, o):
                if False:
                    return 10
                return (self.x - o.x) % 20 > 1

        def py_hash(x):
            if False:
                while True:
                    i = 10
            return hash(x)

        def py_eq(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x == y

        def py_ne(x, y):
            if False:
                return 10
            return x != y

        def identity_decorator(f):
            if False:
                print('Hello World!')
            return f
        comparisons = [(0, 1), (2, 22), (7, 10), (3, 3)]
        for (base_cls, use_jit) in itertools.product([HashEqTest, HashEqNeTest], [False, True]):
            decorator = njit if use_jit else identity_decorator
            hash_func = decorator(py_hash)
            eq_func = decorator(py_eq)
            ne_func = decorator(py_ne)
            jit_cls = jitclass(base_cls)
            for v in [0, 2, 10, 24, -8]:
                self.assertEqual(hash_func(jit_cls(v)), v % 10)
            for (x, y) in comparisons:
                self.assertEqual(eq_func(jit_cls(x), jit_cls(y)), base_cls(x) == base_cls(y))
                self.assertEqual(ne_func(jit_cls(x), jit_cls(y)), base_cls(x) != base_cls(y))

    def test_bool_fallback_len(self):
        if False:
            while True:
                i = 10

        class NoBoolHasLen:

            def __init__(self, val):
                if False:
                    while True:
                        i = 10
                self.val = val

            def __len__(self):
                if False:
                    print('Hello World!')
                return self.val

            def get_bool(self):
                if False:
                    for i in range(10):
                        print('nop')
                return bool(self)
        py_class = NoBoolHasLen
        jitted_class = jitclass([('val', types.int64)])(py_class)
        py_class_0_bool = py_class(0).get_bool()
        py_class_2_bool = py_class(2).get_bool()
        jitted_class_0_bool = jitted_class(0).get_bool()
        jitted_class_2_bool = jitted_class(2).get_bool()
        self.assertEqual(py_class_0_bool, jitted_class_0_bool)
        self.assertEqual(py_class_2_bool, jitted_class_2_bool)
        self.assertEqual(type(py_class_0_bool), type(jitted_class_0_bool))
        self.assertEqual(type(py_class_2_bool), type(jitted_class_2_bool))

    def test_bool_fallback_default(self):
        if False:
            return 10

        class NoBoolNoLen:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                pass

            def get_bool(self):
                if False:
                    while True:
                        i = 10
                return bool(self)
        py_class = NoBoolNoLen
        jitted_class = jitclass([])(py_class)
        py_class_bool = py_class().get_bool()
        jitted_class_bool = jitted_class().get_bool()
        self.assertEqual(py_class_bool, jitted_class_bool)
        self.assertEqual(type(py_class_bool), type(jitted_class_bool))

    def test_operator_reflection(self):
        if False:
            return 10

        class OperatorsDefined:

            def __init__(self, x):
                if False:
                    i = 10
                    return i + 15
                self.x = x

            def __eq__(self, other):
                if False:
                    print('Hello World!')
                return self.x == other.x

            def __le__(self, other):
                if False:
                    i = 10
                    return i + 15
                return self.x <= other.x

            def __lt__(self, other):
                if False:
                    while True:
                        i = 10
                return self.x < other.x

            def __ge__(self, other):
                if False:
                    return 10
                return self.x >= other.x

            def __gt__(self, other):
                if False:
                    while True:
                        i = 10
                return self.x > other.x

        class NoOperatorsDefined:

            def __init__(self, x):
                if False:
                    return 10
                self.x = x
        spec = [('x', types.int32)]
        JitOperatorsDefined = jitclass(spec)(OperatorsDefined)
        JitNoOperatorsDefined = jitclass(spec)(NoOperatorsDefined)
        py_ops_defined = OperatorsDefined(2)
        py_ops_not_defined = NoOperatorsDefined(3)
        jit_ops_defined = JitOperatorsDefined(2)
        jit_ops_not_defined = JitNoOperatorsDefined(3)
        self.assertEqual(py_ops_not_defined == py_ops_defined, jit_ops_not_defined == jit_ops_defined)
        self.assertEqual(py_ops_not_defined <= py_ops_defined, jit_ops_not_defined <= jit_ops_defined)
        self.assertEqual(py_ops_not_defined < py_ops_defined, jit_ops_not_defined < jit_ops_defined)
        self.assertEqual(py_ops_not_defined >= py_ops_defined, jit_ops_not_defined >= jit_ops_defined)
        self.assertEqual(py_ops_not_defined > py_ops_defined, jit_ops_not_defined > jit_ops_defined)

    @skip_unless_scipy
    def test_matmul_operator(self):
        if False:
            while True:
                i = 10

        class ArrayAt:

            def __init__(self, array):
                if False:
                    print('Hello World!')
                self.arr = array

            def __matmul__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return self.arr @ other.arr

            def __rmatmul__(self, other):
                if False:
                    return 10
                return other.arr @ self.arr

            def __imatmul__(self, other):
                if False:
                    i = 10
                    return i + 15
                self.arr = self.arr @ other.arr
                return self

        class ArrayNoAt:

            def __init__(self, array):
                if False:
                    while True:
                        i = 10
                self.arr = array
        n = 3
        np.random.seed(1)
        vec = np.random.random(size=(n,))
        mat = np.random.random(size=(n, n))
        vector_noat = ArrayNoAt(vec)
        vector_at = ArrayAt(vec)
        jit_vector_noat = jitclass(ArrayNoAt, spec={'arr': float64[::1]})(vec)
        jit_vector_at = jitclass(ArrayAt, spec={'arr': float64[::1]})(vec)
        matrix_noat = ArrayNoAt(mat)
        matrix_at = ArrayAt(mat)
        jit_matrix_noat = jitclass(ArrayNoAt, spec={'arr': float64[:, ::1]})(mat)
        jit_matrix_at = jitclass(ArrayAt, spec={'arr': float64[:, ::1]})(mat)
        np.testing.assert_allclose(vector_at @ vector_noat, jit_vector_at @ jit_vector_noat)
        np.testing.assert_allclose(vector_at @ matrix_noat, jit_vector_at @ jit_matrix_noat)
        np.testing.assert_allclose(matrix_at @ vector_noat, jit_matrix_at @ jit_vector_noat)
        np.testing.assert_allclose(matrix_at @ matrix_noat, jit_matrix_at @ jit_matrix_noat)
        np.testing.assert_allclose(vector_noat @ vector_at, jit_vector_noat @ jit_vector_at)
        np.testing.assert_allclose(vector_noat @ matrix_at, jit_vector_noat @ jit_matrix_at)
        np.testing.assert_allclose(matrix_noat @ vector_at, jit_matrix_noat @ jit_vector_at)
        np.testing.assert_allclose(matrix_noat @ matrix_at, jit_matrix_noat @ jit_matrix_at)
        vector_at @= matrix_noat
        matrix_at @= matrix_noat
        jit_vector_at @= jit_matrix_noat
        jit_matrix_at @= jit_matrix_noat
        np.testing.assert_allclose(vector_at.arr, jit_vector_at.arr)
        np.testing.assert_allclose(matrix_at.arr, jit_matrix_at.arr)

    def test_arithmetic_logical_reflection(self):
        if False:
            return 10

        class OperatorsDefined:

            def __init__(self, x):
                if False:
                    print('Hello World!')
                self.x = x

            def __radd__(self, other):
                if False:
                    print('Hello World!')
                return other.x + self.x

            def __rsub__(self, other):
                if False:
                    i = 10
                    return i + 15
                return other.x - self.x

            def __rmul__(self, other):
                if False:
                    return 10
                return other.x * self.x

            def __rtruediv__(self, other):
                if False:
                    while True:
                        i = 10
                return other.x / self.x

            def __rfloordiv__(self, other):
                if False:
                    while True:
                        i = 10
                return other.x // self.x

            def __rmod__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return other.x % self.x

            def __rpow__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return other.x ** self.x

            def __rlshift__(self, other):
                if False:
                    while True:
                        i = 10
                return other.x << self.x

            def __rrshift__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return other.x >> self.x

            def __rand__(self, other):
                if False:
                    while True:
                        i = 10
                return other.x & self.x

            def __rxor__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return other.x ^ self.x

            def __ror__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return other.x | self.x

        class NoOperatorsDefined:

            def __init__(self, x):
                if False:
                    return 10
                self.x = x
        float_op = ['+', '-', '*', '**', '/', '//', '%']
        int_op = [*float_op, '<<', '>>', '&', '^', '|']
        for (test_type, test_op, test_value) in [(int32, int_op, (2, 4)), (float64, float_op, (2.0, 4.0)), (float64[::1], float_op, (np.array([1.0, 2.0, 4.0]), np.array([20.0, -24.0, 1.0])))]:
            spec = {'x': test_type}
            JitOperatorsDefined = jitclass(OperatorsDefined, spec)
            JitNoOperatorsDefined = jitclass(NoOperatorsDefined, spec)
            py_ops_defined = OperatorsDefined(test_value[0])
            py_ops_not_defined = NoOperatorsDefined(test_value[1])
            jit_ops_defined = JitOperatorsDefined(test_value[0])
            jit_ops_not_defined = JitNoOperatorsDefined(test_value[1])
            for op in test_op:
                if not 'array' in str(test_type):
                    self.assertEqual(eval(f'py_ops_not_defined {op} py_ops_defined'), eval(f'jit_ops_not_defined {op} jit_ops_defined'))
                else:
                    self.assertTupleEqual(tuple(eval(f'py_ops_not_defined {op} py_ops_defined')), tuple(eval(f'jit_ops_not_defined {op} jit_ops_defined')))

    def test_implicit_hash_compiles(self):
        if False:
            while True:
                i = 10

        class ImplicitHash:

            def __init__(self):
                if False:
                    return 10
                pass

            def __eq__(self, other):
                if False:
                    return 10
                return False
        jitted = jitclass([])(ImplicitHash)
        instance = jitted()
        self.assertFalse(instance == instance)
if __name__ == '__main__':
    unittest.main()