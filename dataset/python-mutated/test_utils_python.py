import functools
import operator
import platform
from twisted.trial import unittest
from scrapy.utils.asyncgen import as_async_generator, collect_asyncgen
from scrapy.utils.defer import aiter_errback, deferred_f_from_coro_f
from scrapy.utils.python import MutableAsyncChain, MutableChain, binary_is_text, equal_attributes, get_func_args, memoizemethod_noargs, to_bytes, to_unicode, without_none_values
__doctests__ = ['scrapy.utils.python']

class MutableChainTest(unittest.TestCase):

    def test_mutablechain(self):
        if False:
            print('Hello World!')
        m = MutableChain(range(2), [2, 3], (4, 5))
        m.extend(range(6, 7))
        m.extend([7, 8])
        m.extend([9, 10], (11, 12))
        self.assertEqual(next(m), 0)
        self.assertEqual(m.__next__(), 1)
        self.assertEqual(list(m), list(range(2, 13)))

class MutableAsyncChainTest(unittest.TestCase):

    @staticmethod
    async def g1():
        for i in range(3):
            yield i

    @staticmethod
    async def g2():
        return
        yield

    @staticmethod
    async def g3():
        for i in range(7, 10):
            yield i

    @staticmethod
    async def g4():
        for i in range(3, 5):
            yield i
        1 / 0
        for i in range(5, 7):
            yield i

    @staticmethod
    async def collect_asyncgen_exc(asyncgen):
        results = []
        async for x in asyncgen:
            results.append(x)
        return results

    @deferred_f_from_coro_f
    async def test_mutableasyncchain(self):
        m = MutableAsyncChain(self.g1(), as_async_generator(range(3, 7)))
        m.extend(self.g2())
        m.extend(self.g3())
        self.assertEqual(await m.__anext__(), 0)
        results = await collect_asyncgen(m)
        self.assertEqual(results, list(range(1, 10)))

    @deferred_f_from_coro_f
    async def test_mutableasyncchain_exc(self):
        m = MutableAsyncChain(self.g1())
        m.extend(self.g4())
        m.extend(self.g3())
        results = await collect_asyncgen(aiter_errback(m, lambda _: None))
        self.assertEqual(results, list(range(5)))

class ToUnicodeTest(unittest.TestCase):

    def test_converting_an_utf8_encoded_string_to_unicode(self):
        if False:
            while True:
                i = 10
        self.assertEqual(to_unicode(b'lel\xc3\xb1e'), 'lelñe')

    def test_converting_a_latin_1_encoded_string_to_unicode(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(to_unicode(b'lel\xf1e', 'latin-1'), 'lelñe')

    def test_converting_a_unicode_to_unicode_should_return_the_same_object(self):
        if False:
            return 10
        self.assertEqual(to_unicode('ñeñeñe'), 'ñeñeñe')

    def test_converting_a_strange_object_should_raise_TypeError(self):
        if False:
            return 10
        self.assertRaises(TypeError, to_unicode, 423)

    def test_errors_argument(self):
        if False:
            while True:
                i = 10
        self.assertEqual(to_unicode(b'a\xedb', 'utf-8', errors='replace'), 'a�b')

class ToBytesTest(unittest.TestCase):

    def test_converting_a_unicode_object_to_an_utf_8_encoded_string(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(to_bytes('£ 49'), b'\xc2\xa3 49')

    def test_converting_a_unicode_object_to_a_latin_1_encoded_string(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(to_bytes('£ 49', 'latin-1'), b'\xa3 49')

    def test_converting_a_regular_bytes_to_bytes_should_return_the_same_object(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(to_bytes(b'lel\xf1e'), b'lel\xf1e')

    def test_converting_a_strange_object_should_raise_TypeError(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, to_bytes, unittest)

    def test_errors_argument(self):
        if False:
            while True:
                i = 10
        self.assertEqual(to_bytes('a�b', 'latin-1', errors='replace'), b'a?b')

class MemoizedMethodTest(unittest.TestCase):

    def test_memoizemethod_noargs(self):
        if False:
            for i in range(10):
                print('nop')

        class A:

            @memoizemethod_noargs
            def cached(self):
                if False:
                    while True:
                        i = 10
                return object()

            def noncached(self):
                if False:
                    for i in range(10):
                        print('nop')
                return object()
        a = A()
        one = a.cached()
        two = a.cached()
        three = a.noncached()
        assert one is two
        assert one is not three

class BinaryIsTextTest(unittest.TestCase):

    def test_binaryistext(self):
        if False:
            i = 10
            return i + 15
        assert binary_is_text(b'hello')

    def test_utf_16_strings_contain_null_bytes(self):
        if False:
            i = 10
            return i + 15
        assert binary_is_text('hello'.encode('utf-16'))

    def test_one_with_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        assert binary_is_text(b'<div>Price \xa3</div>')

    def test_real_binary_bytes(self):
        if False:
            for i in range(10):
                print('nop')
        assert not binary_is_text(b'\x02\xa3')

class UtilsPythonTestCase(unittest.TestCase):

    def test_equal_attributes(self):
        if False:
            return 10

        class Obj:
            pass
        a = Obj()
        b = Obj()
        self.assertFalse(equal_attributes(a, b, []))
        self.assertFalse(equal_attributes(a, b, ['x', 'y']))
        a.x = 1
        b.x = 1
        self.assertTrue(equal_attributes(a, b, ['x']))
        b.y = 2
        self.assertFalse(equal_attributes(a, b, ['x', 'y']))
        a.y = 2
        self.assertTrue(equal_attributes(a, b, ['x', 'y']))
        a.y = 1
        self.assertFalse(equal_attributes(a, b, ['x', 'y']))
        a.meta = {}
        b.meta = {}
        self.assertTrue(equal_attributes(a, b, ['meta']))
        a.meta['z'] = 1
        b.meta['z'] = 1
        get_z = operator.itemgetter('z')
        get_meta = operator.attrgetter('meta')

        def compare_z(obj):
            if False:
                return 10
            return get_z(get_meta(obj))
        self.assertTrue(equal_attributes(a, b, [compare_z, 'x']))
        a.meta['z'] = 2
        self.assertFalse(equal_attributes(a, b, [compare_z, 'x']))

    def test_get_func_args(self):
        if False:
            while True:
                i = 10

        def f1(a, b, c):
            if False:
                return 10
            pass

        def f2(a, b=None, c=None):
            if False:
                i = 10
                return i + 15
            pass

        def f3(a, b=None, *, c=None):
            if False:
                for i in range(10):
                    print('nop')
            pass

        class A:

            def __init__(self, a, b, c):
                if False:
                    return 10
                pass

            def method(self, a, b, c):
                if False:
                    while True:
                        i = 10
                pass

        class Callable:

            def __call__(self, a, b, c):
                if False:
                    return 10
                pass
        a = A(1, 2, 3)
        cal = Callable()
        partial_f1 = functools.partial(f1, None)
        partial_f2 = functools.partial(f1, b=None)
        partial_f3 = functools.partial(partial_f2, None)
        self.assertEqual(get_func_args(f1), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(f2), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(f3), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(A), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(a.method), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(partial_f1), ['b', 'c'])
        self.assertEqual(get_func_args(partial_f2), ['a', 'c'])
        self.assertEqual(get_func_args(partial_f3), ['c'])
        self.assertEqual(get_func_args(cal), ['a', 'b', 'c'])
        self.assertEqual(get_func_args(object), [])
        self.assertEqual(get_func_args(str.split, stripself=True), ['sep', 'maxsplit'])
        self.assertEqual(get_func_args(' '.join, stripself=True), ['iterable'])
        if platform.python_implementation() == 'CPython':
            self.assertEqual(get_func_args(operator.itemgetter(2)), [])
        elif platform.python_implementation() == 'PyPy':
            self.assertEqual(get_func_args(operator.itemgetter(2), stripself=True), ['obj'])

    def test_without_none_values(self):
        if False:
            print('Hello World!')
        self.assertEqual(without_none_values([1, None, 3, 4]), [1, 3, 4])
        self.assertEqual(without_none_values((1, None, 3, 4)), (1, 3, 4))
        self.assertEqual(without_none_values({'one': 1, 'none': None, 'three': 3, 'four': 4}), {'one': 1, 'three': 3, 'four': 4})