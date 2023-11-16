"""Unit tests for the PickleBuffer object.

Pickling tests themselves are in pickletester.py.
"""
import gc
from pickle import PickleBuffer
import weakref
import unittest
from test.support import import_helper

class B(bytes):
    pass

class PickleBufferTest(unittest.TestCase):

    def check_memoryview(self, pb, equiv):
        if False:
            i = 10
            return i + 15
        with memoryview(pb) as m:
            with memoryview(equiv) as expected:
                self.assertEqual(m.nbytes, expected.nbytes)
                self.assertEqual(m.readonly, expected.readonly)
                self.assertEqual(m.itemsize, expected.itemsize)
                self.assertEqual(m.shape, expected.shape)
                self.assertEqual(m.strides, expected.strides)
                self.assertEqual(m.c_contiguous, expected.c_contiguous)
                self.assertEqual(m.f_contiguous, expected.f_contiguous)
                self.assertEqual(m.format, expected.format)
                self.assertEqual(m.tobytes(), expected.tobytes())

    def test_constructor_failure(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            PickleBuffer()
        with self.assertRaises(TypeError):
            PickleBuffer('foo')
        m = memoryview(b'foo')
        m.release()
        with self.assertRaises(ValueError):
            PickleBuffer(m)

    def test_basics(self):
        if False:
            print('Hello World!')
        pb = PickleBuffer(b'foo')
        self.assertEqual(b'foo', bytes(pb))
        with memoryview(pb) as m:
            self.assertTrue(m.readonly)
        pb = PickleBuffer(bytearray(b'foo'))
        self.assertEqual(b'foo', bytes(pb))
        with memoryview(pb) as m:
            self.assertFalse(m.readonly)
            m[0] = 48
        self.assertEqual(b'0oo', bytes(pb))

    def test_release(self):
        if False:
            i = 10
            return i + 15
        pb = PickleBuffer(b'foo')
        pb.release()
        with self.assertRaises(ValueError) as raises:
            memoryview(pb)
        self.assertIn('operation forbidden on released PickleBuffer object', str(raises.exception))
        pb.release()

    def test_cycle(self):
        if False:
            print('Hello World!')
        b = B(b'foo')
        pb = PickleBuffer(b)
        b.cycle = pb
        wpb = weakref.ref(pb)
        del b, pb
        gc.collect()
        self.assertIsNone(wpb())

    def test_ndarray_2d(self):
        if False:
            while True:
                i = 10
        ndarray = import_helper.import_module('_testbuffer').ndarray
        arr = ndarray(list(range(12)), shape=(4, 3), format='<i')
        self.assertTrue(arr.c_contiguous)
        self.assertFalse(arr.f_contiguous)
        pb = PickleBuffer(arr)
        self.check_memoryview(pb, arr)
        arr = arr[::2]
        self.assertFalse(arr.c_contiguous)
        self.assertFalse(arr.f_contiguous)
        pb = PickleBuffer(arr)
        self.check_memoryview(pb, arr)
        arr = ndarray(list(range(12)), shape=(3, 4), strides=(4, 12), format='<i')
        self.assertTrue(arr.f_contiguous)
        self.assertFalse(arr.c_contiguous)
        pb = PickleBuffer(arr)
        self.check_memoryview(pb, arr)

    def check_raw(self, obj, equiv):
        if False:
            for i in range(10):
                print('nop')
        pb = PickleBuffer(obj)
        with pb.raw() as m:
            self.assertIsInstance(m, memoryview)
            self.check_memoryview(m, equiv)

    def test_raw(self):
        if False:
            return 10
        for obj in (b'foo', bytearray(b'foo')):
            with self.subTest(obj=obj):
                self.check_raw(obj, obj)

    def test_raw_ndarray(self):
        if False:
            i = 10
            return i + 15
        ndarray = import_helper.import_module('_testbuffer').ndarray
        arr = ndarray(list(range(3)), shape=(3,), format='<h')
        equiv = b'\x00\x00\x01\x00\x02\x00'
        self.check_raw(arr, equiv)
        arr = ndarray(list(range(6)), shape=(2, 3), format='<h')
        equiv = b'\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00'
        self.check_raw(arr, equiv)
        arr = ndarray(list(range(6)), shape=(2, 3), strides=(2, 4), format='<h')
        equiv = b'\x00\x00\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00'
        self.check_raw(arr, equiv)
        arr = ndarray(456, shape=(), format='<i')
        equiv = b'\xc8\x01\x00\x00'
        self.check_raw(arr, equiv)

    def check_raw_non_contiguous(self, obj):
        if False:
            print('Hello World!')
        pb = PickleBuffer(obj)
        with self.assertRaisesRegex(BufferError, 'non-contiguous'):
            pb.raw()

    def test_raw_non_contiguous(self):
        if False:
            print('Hello World!')
        ndarray = import_helper.import_module('_testbuffer').ndarray
        arr = ndarray(list(range(6)), shape=(6,), format='<i')[::2]
        self.check_raw_non_contiguous(arr)
        arr = ndarray(list(range(12)), shape=(4, 3), format='<i')[::2]
        self.check_raw_non_contiguous(arr)

    def test_raw_released(self):
        if False:
            while True:
                i = 10
        pb = PickleBuffer(b'foo')
        pb.release()
        with self.assertRaises(ValueError) as raises:
            pb.raw()
if __name__ == '__main__':
    unittest.main()