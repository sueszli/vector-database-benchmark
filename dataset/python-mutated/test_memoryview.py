"""Unit tests for the memoryview

   Some tests are in test_bytes. Many tests that require _testbuffer.ndarray
   are in test_buffer.
"""
import unittest
import test.support
import sys
import gc
import weakref
import array
import io
import copy
import pickle
from test.support import import_helper

class AbstractMemoryTests:
    source_bytes = b'abcdef'

    @property
    def _source(self):
        if False:
            return 10
        return self.source_bytes

    @property
    def _types(self):
        if False:
            for i in range(10):
                print('nop')
        return filter(None, [self.ro_type, self.rw_type])

    def check_getitem_with_type(self, tp):
        if False:
            return 10
        b = tp(self._source)
        oldrefcount = sys.getrefcount(b)
        m = self._view(b)
        self.assertEqual(m[0], ord(b'a'))
        self.assertIsInstance(m[0], int)
        self.assertEqual(m[5], ord(b'f'))
        self.assertEqual(m[-1], ord(b'f'))
        self.assertEqual(m[-6], ord(b'a'))
        self.assertRaises(IndexError, lambda : m[6])
        self.assertRaises(IndexError, lambda : m[-7])
        self.assertRaises(IndexError, lambda : m[sys.maxsize])
        self.assertRaises(IndexError, lambda : m[-sys.maxsize])
        self.assertRaises(TypeError, lambda : m[None])
        self.assertRaises(TypeError, lambda : m[0.0])
        self.assertRaises(TypeError, lambda : m['a'])
        m = None
        self.assertEqual(sys.getrefcount(b), oldrefcount)

    def test_getitem(self):
        if False:
            while True:
                i = 10
        for tp in self._types:
            self.check_getitem_with_type(tp)

    def test_iter(self):
        if False:
            print('Hello World!')
        for tp in self._types:
            b = tp(self._source)
            m = self._view(b)
            self.assertEqual(list(m), [m[i] for i in range(len(m))])

    def test_setitem_readonly(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.ro_type:
            self.skipTest('no read-only type to test')
        b = self.ro_type(self._source)
        oldrefcount = sys.getrefcount(b)
        m = self._view(b)

        def setitem(value):
            if False:
                for i in range(10):
                    print('nop')
            m[0] = value
        self.assertRaises(TypeError, setitem, b'a')
        self.assertRaises(TypeError, setitem, 65)
        self.assertRaises(TypeError, setitem, memoryview(b'a'))
        m = None
        self.assertEqual(sys.getrefcount(b), oldrefcount)

    def test_setitem_writable(self):
        if False:
            while True:
                i = 10
        if not self.rw_type:
            self.skipTest('no writable type to test')
        tp = self.rw_type
        b = self.rw_type(self._source)
        oldrefcount = sys.getrefcount(b)
        m = self._view(b)
        m[0] = ord(b'1')
        self._check_contents(tp, b, b'1bcdef')
        m[0:1] = tp(b'0')
        self._check_contents(tp, b, b'0bcdef')
        m[1:3] = tp(b'12')
        self._check_contents(tp, b, b'012def')
        m[1:1] = tp(b'')
        self._check_contents(tp, b, b'012def')
        m[:] = tp(b'abcdef')
        self._check_contents(tp, b, b'abcdef')
        m[0:3] = m[2:5]
        self._check_contents(tp, b, b'cdedef')
        m[:] = tp(b'abcdef')
        m[2:5] = m[0:3]
        self._check_contents(tp, b, b'ababcf')

        def setitem(key, value):
            if False:
                for i in range(10):
                    print('nop')
            m[key] = tp(value)
        self.assertRaises(IndexError, setitem, 6, b'a')
        self.assertRaises(IndexError, setitem, -7, b'a')
        self.assertRaises(IndexError, setitem, sys.maxsize, b'a')
        self.assertRaises(IndexError, setitem, -sys.maxsize, b'a')
        self.assertRaises(TypeError, setitem, 0.0, b'a')
        self.assertRaises(TypeError, setitem, (0,), b'a')
        self.assertRaises(TypeError, setitem, (slice(0, 1, 1), 0), b'a')
        self.assertRaises(TypeError, setitem, (0, slice(0, 1, 1)), b'a')
        self.assertRaises(TypeError, setitem, (0,), b'a')
        self.assertRaises(TypeError, setitem, 'a', b'a')
        slices = (slice(0, 1, 1), slice(0, 1, 2))
        self.assertRaises(NotImplementedError, setitem, slices, b'a')
        exc = ValueError if m.format == 'c' else TypeError
        self.assertRaises(exc, setitem, 0, b'')
        self.assertRaises(exc, setitem, 0, b'ab')
        self.assertRaises(ValueError, setitem, slice(1, 1), b'a')
        self.assertRaises(ValueError, setitem, slice(0, 2), b'a')
        m = None
        self.assertEqual(sys.getrefcount(b), oldrefcount)

    def test_delitem(self):
        if False:
            i = 10
            return i + 15
        for tp in self._types:
            b = tp(self._source)
            m = self._view(b)
            with self.assertRaises(TypeError):
                del m[1]
            with self.assertRaises(TypeError):
                del m[1:4]

    def test_tobytes(self):
        if False:
            print('Hello World!')
        for tp in self._types:
            m = self._view(tp(self._source))
            b = m.tobytes()
            expected = b''.join((self.getitem_type(bytes([c])) for c in b'abcdef'))
            self.assertEqual(b, expected)
            self.assertIsInstance(b, bytes)

    def test_tolist(self):
        if False:
            return 10
        for tp in self._types:
            m = self._view(tp(self._source))
            l = m.tolist()
            self.assertEqual(l, list(b'abcdef'))

    def test_compare(self):
        if False:
            return 10
        for tp in self._types:
            m = self._view(tp(self._source))
            for tp_comp in self._types:
                self.assertTrue(m == tp_comp(b'abcdef'))
                self.assertFalse(m != tp_comp(b'abcdef'))
                self.assertFalse(m == tp_comp(b'abcde'))
                self.assertTrue(m != tp_comp(b'abcde'))
                self.assertFalse(m == tp_comp(b'abcde1'))
                self.assertTrue(m != tp_comp(b'abcde1'))
            self.assertTrue(m == m)
            self.assertTrue(m == m[:])
            self.assertTrue(m[0:6] == m[:])
            self.assertFalse(m[0:5] == m)
            self.assertFalse(m == 'abcdef')
            self.assertTrue(m != 'abcdef')
            self.assertFalse('abcdef' == m)
            self.assertTrue('abcdef' != m)
            for c in (m, b'abcdef'):
                self.assertRaises(TypeError, lambda : m < c)
                self.assertRaises(TypeError, lambda : c <= m)
                self.assertRaises(TypeError, lambda : m >= c)
                self.assertRaises(TypeError, lambda : c > m)

    def check_attributes_with_type(self, tp):
        if False:
            i = 10
            return i + 15
        m = self._view(tp(self._source))
        self.assertEqual(m.format, self.format)
        self.assertEqual(m.itemsize, self.itemsize)
        self.assertEqual(m.ndim, 1)
        self.assertEqual(m.shape, (6,))
        self.assertEqual(len(m), 6)
        self.assertEqual(m.strides, (self.itemsize,))
        self.assertEqual(m.suboffsets, ())
        return m

    def test_attributes_readonly(self):
        if False:
            i = 10
            return i + 15
        if not self.ro_type:
            self.skipTest('no read-only type to test')
        m = self.check_attributes_with_type(self.ro_type)
        self.assertEqual(m.readonly, True)

    def test_attributes_writable(self):
        if False:
            return 10
        if not self.rw_type:
            self.skipTest('no writable type to test')
        m = self.check_attributes_with_type(self.rw_type)
        self.assertEqual(m.readonly, False)

    def test_getbuffer(self):
        if False:
            for i in range(10):
                print('nop')
        for tp in self._types:
            b = tp(self._source)
            oldrefcount = sys.getrefcount(b)
            m = self._view(b)
            oldviewrefcount = sys.getrefcount(m)
            s = str(m, 'utf-8')
            self._check_contents(tp, b, s.encode('utf-8'))
            self.assertEqual(sys.getrefcount(m), oldviewrefcount)
            m = None
            self.assertEqual(sys.getrefcount(b), oldrefcount)

    def test_gc(self):
        if False:
            print('Hello World!')
        for tp in self._types:
            if not isinstance(tp, type):
                continue

            class MyView:

                def __init__(self, base):
                    if False:
                        return 10
                    self.m = memoryview(base)

            class MySource(tp):
                pass

            class MyObject:
                pass
            b = MySource(tp(b'abc'))
            m = self._view(b)
            o = MyObject()
            b.m = m
            b.o = o
            wr = weakref.ref(o)
            b = m = o = None
            gc.collect()
            self.assertTrue(wr() is None, wr())
            m = MyView(tp(b'abc'))
            o = MyObject()
            m.x = m
            m.o = o
            wr = weakref.ref(o)
            m = o = None
            gc.collect()
            self.assertTrue(wr() is None, wr())

    def _check_released(self, m, tp):
        if False:
            while True:
                i = 10
        check = self.assertRaisesRegex(ValueError, 'released')
        with check:
            bytes(m)
        with check:
            m.tobytes()
        with check:
            m.tolist()
        with check:
            m[0]
        with check:
            m[0] = b'x'
        with check:
            len(m)
        with check:
            m.format
        with check:
            m.itemsize
        with check:
            m.ndim
        with check:
            m.readonly
        with check:
            m.shape
        with check:
            m.strides
        with check:
            with m:
                pass
        self.assertIn('released memory', str(m))
        self.assertIn('released memory', repr(m))
        self.assertEqual(m, m)
        self.assertNotEqual(m, memoryview(tp(self._source)))
        self.assertNotEqual(m, tp(self._source))

    def test_contextmanager(self):
        if False:
            while True:
                i = 10
        for tp in self._types:
            b = tp(self._source)
            m = self._view(b)
            with m as cm:
                self.assertIs(cm, m)
            self._check_released(m, tp)
            m = self._view(b)
            with m:
                m.release()

    def test_release(self):
        if False:
            print('Hello World!')
        for tp in self._types:
            b = tp(self._source)
            m = self._view(b)
            m.release()
            self._check_released(m, tp)
            m.release()
            self._check_released(m, tp)

    def test_writable_readonly(self):
        if False:
            while True:
                i = 10
        tp = self.ro_type
        if tp is None:
            self.skipTest('no read-only type to test')
        b = tp(self._source)
        m = self._view(b)
        i = io.BytesIO(b'ZZZZ')
        self.assertRaises(TypeError, i.readinto, m)

    def test_getbuf_fail(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, self._view, {})

    def test_hash(self):
        if False:
            for i in range(10):
                print('nop')
        tp = self.ro_type
        if tp is None:
            self.skipTest('no read-only type to test')
        b = tp(self._source)
        m = self._view(b)
        self.assertEqual(hash(m), hash(b'abcdef'))
        m.release()
        self.assertEqual(hash(m), hash(b'abcdef'))
        m = self._view(b)
        m.release()
        self.assertRaises(ValueError, hash, m)

    def test_hash_writable(self):
        if False:
            while True:
                i = 10
        tp = self.rw_type
        if tp is None:
            self.skipTest('no writable type to test')
        b = tp(self._source)
        m = self._view(b)
        self.assertRaises(ValueError, hash, m)

    def test_weakref(self):
        if False:
            while True:
                i = 10
        for tp in self._types:
            b = tp(self._source)
            m = self._view(b)
            L = []

            def callback(wr, b=b):
                if False:
                    return 10
                L.append(b)
            wr = weakref.ref(m, callback)
            self.assertIs(wr(), m)
            del m
            test.support.gc_collect()
            self.assertIs(wr(), None)
            self.assertIs(L[0], b)

    def test_reversed(self):
        if False:
            i = 10
            return i + 15
        for tp in self._types:
            b = tp(self._source)
            m = self._view(b)
            aslist = list(reversed(m.tolist()))
            self.assertEqual(list(reversed(m)), aslist)
            self.assertEqual(list(reversed(m)), list(m[::-1]))

    def test_toreadonly(self):
        if False:
            while True:
                i = 10
        for tp in self._types:
            b = tp(self._source)
            m = self._view(b)
            mm = m.toreadonly()
            self.assertTrue(mm.readonly)
            self.assertTrue(memoryview(mm).readonly)
            self.assertEqual(mm.tolist(), m.tolist())
            mm.release()
            m.tolist()

    def test_issue22668(self):
        if False:
            print('Hello World!')
        a = array.array('H', [256, 256, 256, 256])
        x = memoryview(a)
        m = x.cast('B')
        b = m.cast('H')
        c = b[0:2]
        d = memoryview(b)
        del b
        self.assertEqual(c[0], 256)
        self.assertEqual(d[0], 256)
        self.assertEqual(c.format, 'H')
        self.assertEqual(d.format, 'H')
        _ = m.cast('I')
        self.assertEqual(c[0], 256)
        self.assertEqual(d[0], 256)
        self.assertEqual(c.format, 'H')
        self.assertEqual(d.format, 'H')

class BaseBytesMemoryTests(AbstractMemoryTests):
    ro_type = bytes
    rw_type = bytearray
    getitem_type = bytes
    itemsize = 1
    format = 'B'

class BaseArrayMemoryTests(AbstractMemoryTests):
    ro_type = None
    rw_type = lambda self, b: array.array('i', list(b))
    getitem_type = lambda self, b: array.array('i', list(b)).tobytes()
    itemsize = array.array('i').itemsize
    format = 'i'

    @unittest.skip('XXX test should be adapted for non-byte buffers')
    def test_getbuffer(self):
        if False:
            print('Hello World!')
        pass

    @unittest.skip('XXX NotImplementedError: tolist() only supports byte views')
    def test_tolist(self):
        if False:
            return 10
        pass

class BaseMemoryviewTests:

    def _view(self, obj):
        if False:
            while True:
                i = 10
        return memoryview(obj)

    def _check_contents(self, tp, obj, contents):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(obj, tp(contents))

class BaseMemorySliceTests:
    source_bytes = b'XabcdefY'

    def _view(self, obj):
        if False:
            for i in range(10):
                print('nop')
        m = memoryview(obj)
        return m[1:7]

    def _check_contents(self, tp, obj, contents):
        if False:
            print('Hello World!')
        self.assertEqual(obj[1:7], tp(contents))

    def test_refs(self):
        if False:
            i = 10
            return i + 15
        for tp in self._types:
            m = memoryview(tp(self._source))
            oldrefcount = sys.getrefcount(m)
            m[1:2]
            self.assertEqual(sys.getrefcount(m), oldrefcount)

class BaseMemorySliceSliceTests:
    source_bytes = b'XabcdefY'

    def _view(self, obj):
        if False:
            print('Hello World!')
        m = memoryview(obj)
        return m[:7][1:]

    def _check_contents(self, tp, obj, contents):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(obj[1:7], tp(contents))

class BytesMemoryviewTest(unittest.TestCase, BaseMemoryviewTests, BaseBytesMemoryTests):

    def test_constructor(self):
        if False:
            print('Hello World!')
        for tp in self._types:
            ob = tp(self._source)
            self.assertTrue(memoryview(ob))
            self.assertTrue(memoryview(object=ob))
            self.assertRaises(TypeError, memoryview)
            self.assertRaises(TypeError, memoryview, ob, ob)
            self.assertRaises(TypeError, memoryview, argument=ob)
            self.assertRaises(TypeError, memoryview, ob, argument=True)

class ArrayMemoryviewTest(unittest.TestCase, BaseMemoryviewTests, BaseArrayMemoryTests):

    def test_array_assign(self):
        if False:
            i = 10
            return i + 15
        a = array.array('i', range(10))
        m = memoryview(a)
        new_a = array.array('i', range(9, -1, -1))
        m[:] = new_a
        self.assertEqual(a, new_a)

class BytesMemorySliceTest(unittest.TestCase, BaseMemorySliceTests, BaseBytesMemoryTests):
    pass

class ArrayMemorySliceTest(unittest.TestCase, BaseMemorySliceTests, BaseArrayMemoryTests):
    pass

class BytesMemorySliceSliceTest(unittest.TestCase, BaseMemorySliceSliceTests, BaseBytesMemoryTests):
    pass

class ArrayMemorySliceSliceTest(unittest.TestCase, BaseMemorySliceSliceTests, BaseArrayMemoryTests):
    pass

class OtherTest(unittest.TestCase):

    def test_ctypes_cast(self):
        if False:
            i = 10
            return i + 15
        ctypes = import_helper.import_module('ctypes')
        p6 = bytes(ctypes.c_double(0.6))
        d = ctypes.c_double()
        m = memoryview(d).cast('B')
        m[:2] = p6[:2]
        m[2:] = p6[2:]
        self.assertEqual(d.value, 0.6)
        for format in 'Bbc':
            with self.subTest(format):
                d = ctypes.c_double()
                m = memoryview(d).cast(format)
                m[:2] = memoryview(p6).cast(format)[:2]
                m[2:] = memoryview(p6).cast(format)[2:]
                self.assertEqual(d.value, 0.6)

    def test_memoryview_hex(self):
        if False:
            while True:
                i = 10
        x = b'0' * 200000
        m1 = memoryview(x)
        m2 = m1[::-1]
        self.assertEqual(m2.hex(), '30' * 200000)

    def test_copy(self):
        if False:
            while True:
                i = 10
        m = memoryview(b'abc')
        with self.assertRaises(TypeError):
            copy.copy(m)

    def test_pickle(self):
        if False:
            print('Hello World!')
        m = memoryview(b'abc')
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.assertRaises(TypeError):
                pickle.dumps(m, proto)
if __name__ == '__main__':
    unittest.main()