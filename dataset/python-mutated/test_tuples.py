import collections
import itertools
import numpy as np
from numba.core.compiler import compile_isolated
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
Rect = collections.namedtuple('Rect', ('width', 'height'))
Point = collections.namedtuple('Point', ('x', 'y', 'z'))
Point2 = collections.namedtuple('Point2', ('x', 'y', 'z'))
Empty = collections.namedtuple('Empty', ())

def tuple_return_usecase(a, b):
    if False:
        for i in range(10):
            print('nop')
    return (a, b)

def tuple_first(tup):
    if False:
        while True:
            i = 10
    (a, b) = tup
    return a

def tuple_second(tup):
    if False:
        while True:
            i = 10
    (a, b) = tup
    return b

def tuple_index(tup, idx):
    if False:
        for i in range(10):
            print('nop')
    return tup[idx]

def tuple_index_static(tup):
    if False:
        while True:
            i = 10
    return tup[-2]

def tuple_slice2(tup):
    if False:
        return 10
    return tup[1:-1]

def tuple_slice3(tup):
    if False:
        while True:
            i = 10
    return tup[1::2]

def len_usecase(tup):
    if False:
        return 10
    return len(tup)

def add_usecase(a, b):
    if False:
        print('Hello World!')
    return a + b

def eq_usecase(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a == b

def ne_usecase(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a != b

def gt_usecase(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a > b

def ge_usecase(a, b):
    if False:
        i = 10
        return i + 15
    return a >= b

def lt_usecase(a, b):
    if False:
        i = 10
        return i + 15
    return a < b

def le_usecase(a, b):
    if False:
        print('Hello World!')
    return a <= b

def in_usecase(a, b):
    if False:
        while True:
            i = 10
    return a in b

def bool_usecase(tup):
    if False:
        return 10
    return (bool(tup), 3 if tup else 2)

def getattr_usecase(tup):
    if False:
        for i in range(10):
            print('nop')
    return (tup.z, tup.y, tup.x)

def make_point(a, b, c):
    if False:
        while True:
            i = 10
    return Point(a, b, c)

def make_point_kws(a, b, c):
    if False:
        while True:
            i = 10
    return Point(z=c, y=b, x=a)

def make_point_nrt(n):
    if False:
        while True:
            i = 10
    r = Rect(list(range(n)), np.zeros(n + 1))
    p = Point(r, len(r.width), len(r.height))
    return p

def type_usecase(tup, *args):
    if False:
        while True:
            i = 10
    return type(tup)(*args)

def identity(tup):
    if False:
        i = 10
        return i + 15
    return tup

def index_method_usecase(tup, value):
    if False:
        while True:
            i = 10
    return tup.index(value)

def tuple_unpack_static_getitem_err():
    if False:
        for i in range(10):
            print('nop')
    (a, b, c, d) = ([], [], [], 0.0)
    a.append(1)
    b.append(1)
    return

class TestTupleLengthError(unittest.TestCase):

    def test_tuple_length_error(self):
        if False:
            i = 10
            return i + 15

        @njit
        def eattuple(tup):
            if False:
                return 10
            return len(tup)
        with self.assertRaises(errors.UnsupportedError) as raises:
            tup = tuple(range(1001))
            eattuple(tup)
        expected = "Tuple 'tup' length must be smaller than 1000"
        self.assertIn(expected, str(raises.exception))

class TestTupleTypeNotIterable(unittest.TestCase):
    """
    issue 4369
    raise an error if 'type' is not iterable
    """

    def test_namedtuple_types_exception(self):
        if False:
            return 10
        with self.assertRaises(errors.TypingError) as raises:
            types.NamedTuple(types.uint32, 'p')
        self.assertIn("Argument 'types' is not iterable", str(raises.exception))

    def test_tuple_types_exception(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(errors.TypingError) as raises:
            types.Tuple(types.uint32)
        self.assertIn("Argument 'types' is not iterable", str(raises.exception))

class TestTupleReturn(TestCase):

    def test_array_tuple(self):
        if False:
            i = 10
            return i + 15
        aryty = types.Array(types.float64, 1, 'C')
        cres = compile_isolated(tuple_return_usecase, (aryty, aryty))
        a = b = np.arange(5, dtype='float64')
        (ra, rb) = cres.entry_point(a, b)
        self.assertPreciseEqual(ra, a)
        self.assertPreciseEqual(rb, b)
        del a, b
        self.assertPreciseEqual(ra, rb)

    def test_scalar_tuple(self):
        if False:
            while True:
                i = 10
        scalarty = types.float32
        cres = compile_isolated(tuple_return_usecase, (scalarty, scalarty))
        a = b = 1
        (ra, rb) = cres.entry_point(a, b)
        self.assertEqual(ra, a)
        self.assertEqual(rb, b)

    def test_hetero_tuple(self):
        if False:
            return 10
        alltypes = []
        allvalues = []
        alltypes.append((types.int32, types.int64))
        allvalues.append((1, 2))
        alltypes.append((types.float32, types.float64))
        allvalues.append((1.125, 0.25))
        alltypes.append((types.int32, types.float64))
        allvalues.append((1231, 0.5))
        for ((ta, tb), (a, b)) in zip(alltypes, allvalues):
            cres = compile_isolated(tuple_return_usecase, (ta, tb))
            (ra, rb) = cres.entry_point(a, b)
            self.assertPreciseEqual((ra, rb), (a, b))

class TestTuplePassing(TestCase):

    def test_unituple(self):
        if False:
            print('Hello World!')
        tuple_type = types.UniTuple(types.int32, 2)
        cr_first = compile_isolated(tuple_first, (tuple_type,))
        cr_second = compile_isolated(tuple_second, (tuple_type,))
        self.assertPreciseEqual(cr_first.entry_point((4, 5)), 4)
        self.assertPreciseEqual(cr_second.entry_point((4, 5)), 5)

    def test_hetero_tuple(self):
        if False:
            i = 10
            return i + 15
        tuple_type = types.Tuple((types.int64, types.float32))
        cr_first = compile_isolated(tuple_first, (tuple_type,))
        cr_second = compile_isolated(tuple_second, (tuple_type,))
        self.assertPreciseEqual(cr_first.entry_point((2 ** 61, 1.5)), 2 ** 61)
        self.assertPreciseEqual(cr_second.entry_point((2 ** 61, 1.5)), 1.5)

    def test_size_mismatch(self):
        if False:
            while True:
                i = 10
        tuple_type = types.UniTuple(types.int32, 2)
        cr = compile_isolated(tuple_first, (tuple_type,))
        with self.assertRaises(ValueError) as raises:
            cr.entry_point((4, 5, 6))
        self.assertEqual(str(raises.exception), 'size mismatch for tuple, expected 2 element(s) but got 3')

class TestOperations(TestCase):

    def test_len(self):
        if False:
            while True:
                i = 10
        pyfunc = len_usecase
        cr = compile_isolated(pyfunc, [types.Tuple((types.int64, types.float32))])
        self.assertPreciseEqual(cr.entry_point((4, 5)), 2)
        cr = compile_isolated(pyfunc, [types.UniTuple(types.int64, 3)])
        self.assertPreciseEqual(cr.entry_point((4, 5, 6)), 3)

    def test_index_literal(self):
        if False:
            for i in range(10):
                print('nop')

        def pyfunc(tup, idx):
            if False:
                while True:
                    i = 10
            idx = literally(idx)
            return tup[idx]
        cfunc = njit(pyfunc)
        tup = (4, 3.1, 'sss')
        for i in range(len(tup)):
            self.assertPreciseEqual(cfunc(tup, i), tup[i])

    def test_index(self):
        if False:
            while True:
                i = 10
        pyfunc = tuple_index
        cr = compile_isolated(pyfunc, [types.UniTuple(types.int64, 3), types.int64])
        tup = (4, 3, 6)
        for i in range(len(tup)):
            self.assertPreciseEqual(cr.entry_point(tup, i), tup[i])
        for i in range(len(tup) + 1):
            self.assertPreciseEqual(cr.entry_point(tup, -i), tup[-i])
        with self.assertRaises(IndexError) as raises:
            cr.entry_point(tup, len(tup))
        self.assertEqual('tuple index out of range', str(raises.exception))
        with self.assertRaises(IndexError) as raises:
            cr.entry_point(tup, -(len(tup) + 1))
        self.assertEqual('tuple index out of range', str(raises.exception))
        cr = compile_isolated(pyfunc, [types.UniTuple(types.int64, 0), types.int64])
        with self.assertRaises(IndexError) as raises:
            cr.entry_point((), 0)
        self.assertEqual('tuple index out of range', str(raises.exception))
        cr = compile_isolated(pyfunc, [types.UniTuple(types.int64, 3), types.uintp])
        for i in range(len(tup)):
            self.assertPreciseEqual(cr.entry_point(tup, types.uintp(i)), tup[i])
        pyfunc = tuple_index_static
        for typ in (types.UniTuple(types.int64, 4), types.Tuple((types.int64, types.int32, types.int64, types.int32))):
            cr = compile_isolated(pyfunc, (typ,))
            tup = (4, 3, 42, 6)
            self.assertPreciseEqual(cr.entry_point(tup), pyfunc(tup))
        typ = types.UniTuple(types.int64, 1)
        with self.assertTypingError():
            cr = compile_isolated(pyfunc, (typ,))
        pyfunc = tuple_unpack_static_getitem_err
        with self.assertTypingError() as raises:
            cr = compile_isolated(pyfunc, ())
        msg = "Cannot infer the type of variable 'c', have imprecise type: list(undefined)<iv=None>."
        self.assertIn(msg, str(raises.exception))

    def test_in(self):
        if False:
            return 10
        pyfunc = in_usecase
        cr = compile_isolated(pyfunc, [types.int64, types.UniTuple(types.int64, 3)])
        tup = (4, 1, 5)
        for i in range(5):
            self.assertPreciseEqual(cr.entry_point(i, tup), pyfunc(i, tup))
        cr = compile_isolated(pyfunc, [types.int64, types.Tuple([])])
        self.assertPreciseEqual(cr.entry_point(1, ()), pyfunc(1, ()))

    def check_slice(self, pyfunc):
        if False:
            print('Hello World!')
        tup = (4, 5, 6, 7)
        cr = compile_isolated(pyfunc, [types.UniTuple(types.int64, 4)])
        self.assertPreciseEqual(cr.entry_point(tup), pyfunc(tup))
        cr = compile_isolated(pyfunc, [types.Tuple((types.int64, types.int32, types.int64, types.int32))])
        self.assertPreciseEqual(cr.entry_point(tup), pyfunc(tup))

    def test_slice2(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_slice(tuple_slice2)

    def test_slice3(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_slice(tuple_slice3)

    def test_bool(self):
        if False:
            while True:
                i = 10
        pyfunc = bool_usecase
        cr = compile_isolated(pyfunc, [types.Tuple((types.int64, types.int32))])
        args = ((4, 5),)
        self.assertPreciseEqual(cr.entry_point(*args), pyfunc(*args))
        cr = compile_isolated(pyfunc, [types.UniTuple(types.int64, 3)])
        args = ((4, 5, 6),)
        self.assertPreciseEqual(cr.entry_point(*args), pyfunc(*args))
        cr = compile_isolated(pyfunc, [types.Tuple(())])
        self.assertPreciseEqual(cr.entry_point(()), pyfunc(()))

    def test_add(self):
        if False:
            return 10
        pyfunc = add_usecase
        samples = [(types.Tuple(()), ()), (types.UniTuple(types.int32, 0), ()), (types.UniTuple(types.int32, 1), (42,)), (types.Tuple((types.int64, types.float32)), (3, 4.5))]
        for ((ta, a), (tb, b)) in itertools.product(samples, samples):
            cr = compile_isolated(pyfunc, (ta, tb))
            expected = pyfunc(a, b)
            got = cr.entry_point(a, b)
            self.assertPreciseEqual(got, expected, msg=(ta, tb))

    def _test_compare(self, pyfunc):
        if False:
            print('Hello World!')

        def eq(pyfunc, cfunc, args):
            if False:
                i = 10
                return i + 15
            self.assertIs(cfunc(*args), pyfunc(*args), 'mismatch for arguments %s' % (args,))
        argtypes = [types.Tuple((types.int64, types.float32)), types.UniTuple(types.int32, 2)]
        for (ta, tb) in itertools.product(argtypes, argtypes):
            cr = compile_isolated(pyfunc, (ta, tb))
            cfunc = cr.entry_point
            for args in [((4, 5), (4, 5)), ((4, 5), (4, 6)), ((4, 6), (4, 5)), ((4, 5), (5, 4))]:
                eq(pyfunc, cfunc, args)
        argtypes = [types.Tuple((types.int64, types.float32)), types.UniTuple(types.int32, 3)]
        cr = compile_isolated(pyfunc, tuple(argtypes))
        cfunc = cr.entry_point
        for args in [((4, 5), (4, 5, 6)), ((4, 5), (4, 4, 6)), ((4, 5), (4, 6, 7))]:
            eq(pyfunc, cfunc, args)

    def test_eq(self):
        if False:
            return 10
        self._test_compare(eq_usecase)

    def test_ne(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_compare(ne_usecase)

    def test_gt(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_compare(gt_usecase)

    def test_ge(self):
        if False:
            while True:
                i = 10
        self._test_compare(ge_usecase)

    def test_lt(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_compare(lt_usecase)

    def test_le(self):
        if False:
            return 10
        self._test_compare(le_usecase)

class TestNamedTuple(TestCase, MemoryLeakMixin):

    def test_unpack(self):
        if False:
            while True:
                i = 10

        def check(p):
            if False:
                i = 10
                return i + 15
            for pyfunc in (tuple_first, tuple_second):
                cfunc = jit(nopython=True)(pyfunc)
                self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check(Rect(4, 5))
        check(Rect(4, 5.5))

    def test_len(self):
        if False:
            print('Hello World!')

        def check(p):
            if False:
                for i in range(10):
                    print('nop')
            pyfunc = len_usecase
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check(Rect(4, 5))
        check(Point(4, 5, 6))
        check(Rect(4, 5.5))
        check(Point(4, 5.5, 6j))

    def test_index(self):
        if False:
            return 10
        pyfunc = tuple_index
        cfunc = jit(nopython=True)(pyfunc)
        p = Point(4, 5, 6)
        for i in range(len(p)):
            self.assertPreciseEqual(cfunc(p, i), pyfunc(p, i))
        for i in range(len(p)):
            self.assertPreciseEqual(cfunc(p, types.uintp(i)), pyfunc(p, i))

    def test_bool(self):
        if False:
            return 10

        def check(p):
            if False:
                while True:
                    i = 10
            pyfunc = bool_usecase
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check(Rect(4, 5))
        check(Rect(4, 5.5))
        check(Empty())

    def _test_compare(self, pyfunc):
        if False:
            return 10

        def eq(pyfunc, cfunc, args):
            if False:
                i = 10
                return i + 15
            self.assertIs(cfunc(*args), pyfunc(*args), 'mismatch for arguments %s' % (args,))
        cfunc = jit(nopython=True)(pyfunc)
        for (a, b) in [((4, 5), (4, 5)), ((4, 5), (4, 6)), ((4, 6), (4, 5)), ((4, 5), (5, 4))]:
            eq(pyfunc, cfunc, (Rect(*a), Rect(*b)))
        for (a, b) in [((4, 5), (4, 5, 6)), ((4, 5), (4, 4, 6)), ((4, 5), (4, 6, 7))]:
            eq(pyfunc, cfunc, (Rect(*a), Point(*b)))

    def test_eq(self):
        if False:
            i = 10
            return i + 15
        self._test_compare(eq_usecase)

    def test_ne(self):
        if False:
            print('Hello World!')
        self._test_compare(ne_usecase)

    def test_gt(self):
        if False:
            return 10
        self._test_compare(gt_usecase)

    def test_ge(self):
        if False:
            print('Hello World!')
        self._test_compare(ge_usecase)

    def test_lt(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_compare(lt_usecase)

    def test_le(self):
        if False:
            i = 10
            return i + 15
        self._test_compare(le_usecase)

    def test_getattr(self):
        if False:
            i = 10
            return i + 15
        pyfunc = getattr_usecase
        cfunc = jit(nopython=True)(pyfunc)
        for args in ((4, 5, 6), (4, 5.5, 6j)):
            p = Point(*args)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))

    def test_construct(self):
        if False:
            i = 10
            return i + 15

        def check(pyfunc):
            if False:
                for i in range(10):
                    print('nop')
            cfunc = jit(nopython=True)(pyfunc)
            for args in ((4, 5, 6), (4, 5.5, 6j)):
                expected = pyfunc(*args)
                got = cfunc(*args)
                self.assertIs(type(got), type(expected))
                self.assertPreciseEqual(got, expected)
        check(make_point)
        check(make_point_kws)

    def test_type(self):
        if False:
            i = 10
            return i + 15
        pyfunc = type_usecase
        cfunc = jit(nopython=True)(pyfunc)
        arg_tuples = [(4, 5, 6), (4, 5.5, 6j)]
        for (tup_args, args) in itertools.product(arg_tuples, arg_tuples):
            tup = Point(*tup_args)
            expected = pyfunc(tup, *args)
            got = cfunc(tup, *args)
            self.assertIs(type(got), type(expected))
            self.assertPreciseEqual(got, expected)

    def test_literal_unification(self):
        if False:
            for i in range(10):
                print('nop')

        @jit(nopython=True)
        def Data1(value):
            if False:
                i = 10
                return i + 15
            return Rect(value, -321)

        @jit(nopython=True)
        def call(i, j):
            if False:
                for i in range(10):
                    print('nop')
            if j == 0:
                result = Data1(i)
            else:
                result = Rect(i, j)
            return result
        r = call(123, 1321)
        self.assertEqual(r, Rect(width=123, height=1321))
        r = call(123, 0)
        self.assertEqual(r, Rect(width=123, height=-321))

    def test_string_literal_in_ctor(self):
        if False:
            while True:
                i = 10

        @jit(nopython=True)
        def foo():
            if False:
                i = 10
                return i + 15
            return Rect(10, 'somestring')
        r = foo()
        self.assertEqual(r, Rect(width=10, height='somestring'))

    def test_dispatcher_mistreat(self):
        if False:
            print('Hello World!')

        @jit(nopython=True)
        def foo(x):
            if False:
                while True:
                    i = 10
            return x
        in1 = (1, 2, 3)
        out1 = foo(in1)
        self.assertEqual(in1, out1)
        in2 = Point(1, 2, 3)
        out2 = foo(in2)
        self.assertEqual(in2, out2)
        self.assertEqual(len(foo.nopython_signatures), 2)
        self.assertEqual(foo.nopython_signatures[0].args[0], typeof(in1))
        self.assertEqual(foo.nopython_signatures[1].args[0], typeof(in2))
        in3 = Point2(1, 2, 3)
        out3 = foo(in3)
        self.assertEqual(in3, out3)
        self.assertEqual(len(foo.nopython_signatures), 3)
        self.assertEqual(foo.nopython_signatures[2].args[0], typeof(in3))

class TestTupleNRT(TestCase, MemoryLeakMixin):

    def test_tuple_add(self):
        if False:
            i = 10
            return i + 15

        def pyfunc(x):
            if False:
                for i in range(10):
                    print('nop')
            a = np.arange(3)
            return (a,) + (x,)
        cfunc = jit(nopython=True)(pyfunc)
        x = 123
        (expect_a, expect_x) = pyfunc(x)
        (got_a, got_x) = cfunc(x)
        np.testing.assert_equal(got_a, expect_a)
        self.assertEqual(got_x, expect_x)

class TestNamedTupleNRT(TestCase, MemoryLeakMixin):

    def test_return(self):
        if False:
            print('Hello World!')
        pyfunc = make_point_nrt
        cfunc = jit(nopython=True)(pyfunc)
        for arg in (3, 0):
            expected = pyfunc(arg)
            got = cfunc(arg)
            self.assertIs(type(got), type(expected))
            self.assertPreciseEqual(got, expected)

class TestConversions(TestCase):
    """
    Test implicit conversions between tuple types.
    """

    def check_conversion(self, fromty, toty, val):
        if False:
            i = 10
            return i + 15
        pyfunc = identity
        cr = compile_isolated(pyfunc, (fromty,), toty)
        cfunc = cr.entry_point
        res = cfunc(val)
        self.assertEqual(res, val)

    def test_conversions(self):
        if False:
            return 10
        check = self.check_conversion
        fromty = types.UniTuple(types.int32, 2)
        check(fromty, types.UniTuple(types.float32, 2), (4, 5))
        check(fromty, types.Tuple((types.float32, types.int16)), (4, 5))
        aty = types.UniTuple(types.int32, 0)
        bty = types.Tuple(())
        check(aty, bty, ())
        check(bty, aty, ())
        with self.assertRaises(errors.TypingError) as raises:
            check(fromty, types.Tuple((types.float32,)), (4, 5))
        msg = 'No conversion from UniTuple(int32 x 2) to UniTuple(float32 x 1)'
        self.assertIn(msg, str(raises.exception))

class TestMethods(TestCase):

    def test_index(self):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = index_method_usecase
        cfunc = jit(nopython=True)(pyfunc)
        self.assertEqual(cfunc((1, 2, 3), 2), 1)
        with self.assertRaises(ValueError) as raises:
            cfunc((1, 2, 3), 4)
        msg = 'tuple.index(x): x not in tuple'
        self.assertEqual(msg, str(raises.exception))

class TestTupleBuild(TestCase):

    def test_build_unpack(self):
        if False:
            print('Hello World!')

        def check(p):
            if False:
                return 10
            pyfunc = lambda a: (1, *a)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))
        check((4, 5.5))

    def test_build_unpack_assign_like(self):
        if False:
            for i in range(10):
                print('nop')

        def check(p):
            if False:
                return 10
            pyfunc = lambda a: (*a,)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))
        check((4, 5.5))

    def test_build_unpack_fail_on_list_assign_like(self):
        if False:
            print('Hello World!')

        def check(p):
            if False:
                return 10
            pyfunc = lambda a: (*a,)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        with self.assertRaises(errors.TypingError) as raises:
            check([4, 5])
        if utils.PYVERSION > (3, 8):
            msg1 = 'No implementation of function'
            self.assertIn(msg1, str(raises.exception))
            msg2 = 'tuple(reflected list('
            self.assertIn(msg2, str(raises.exception))
        else:
            msg = 'Only tuples are supported when unpacking a single item'
            self.assertIn(msg, str(raises.exception))

    def test_build_unpack_more(self):
        if False:
            return 10

        def check(p):
            if False:
                return 10
            pyfunc = lambda a: (1, *a, (1, 2), *a)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))
        check((4, 5.5))

    def test_build_unpack_call(self):
        if False:
            print('Hello World!')

        def check(p):
            if False:
                print('Hello World!')

            @jit
            def inner(*args):
                if False:
                    for i in range(10):
                        print('nop')
                return args
            pyfunc = lambda a: inner(1, *a)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))
        check((4, 5.5))

    def test_build_unpack_call_more(self):
        if False:
            return 10

        def check(p):
            if False:
                return 10

            @jit
            def inner(*args):
                if False:
                    i = 10
                    return i + 15
                return args
            pyfunc = lambda a: inner(1, *a, *(1, 2), *a)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))
        check((4, 5.5))

    def test_tuple_constructor(self):
        if False:
            while True:
                i = 10

        def check(pyfunc, arg):
            if False:
                while True:
                    i = 10
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(arg), pyfunc(arg))
        check(lambda _: tuple(), ())
        check(lambda a: tuple(a), (4, 5))
        check(lambda a: tuple(a), (4, 5.5))

    @unittest.skipIf(utils.PYVERSION < (3, 9), 'needs Python 3.9+')
    def test_unpack_with_predicate_fails(self):
        if False:
            for i in range(10):
                print('nop')

        @njit
        def foo():
            if False:
                while True:
                    i = 10
            a = (1,)
            b = (3, 2, 4)
            return (*(b if a[0] else (5, 6)),)
        with self.assertRaises(errors.UnsupportedError) as raises:
            foo()
        msg = 'op_LIST_EXTEND at the start of a block'
        self.assertIn(msg, str(raises.exception))

    def test_build_unpack_with_calls_in_unpack(self):
        if False:
            i = 10
            return i + 15

        def check(p):
            if False:
                print('Hello World!')

            def pyfunc(a):
                if False:
                    for i in range(10):
                        print('nop')
                z = [1, 2]
                return ((*a, z.append(3), z.extend(a), np.ones(3)), z)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((4, 5))

    def test_build_unpack_complicated(self):
        if False:
            return 10

        def check(p):
            if False:
                i = 10
                return i + 15

            def pyfunc(a):
                if False:
                    i = 10
                    return i + 15
                z = [1, 2]
                return ((*a, *(*a, a), *(a, (*(a, (1, 2), *(3,), *a), (a, 1, (2, 3), *a, 1), (1,))), *(z.append(4), z.extend(a))), z)
            cfunc = jit(nopython=True)(pyfunc)
            self.assertPreciseEqual(cfunc(p), pyfunc(p))
        check((10, 20))
if __name__ == '__main__':
    unittest.main()