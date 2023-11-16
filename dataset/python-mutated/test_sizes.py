from ctypes import *
import unittest

class SizesTestCase(unittest.TestCase):

    def test_8(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(1, sizeof(c_int8))
        self.assertEqual(1, sizeof(c_uint8))

    def test_16(self):
        if False:
            return 10
        self.assertEqual(2, sizeof(c_int16))
        self.assertEqual(2, sizeof(c_uint16))

    def test_32(self):
        if False:
            return 10
        self.assertEqual(4, sizeof(c_int32))
        self.assertEqual(4, sizeof(c_uint32))

    def test_64(self):
        if False:
            print('Hello World!')
        self.assertEqual(8, sizeof(c_int64))
        self.assertEqual(8, sizeof(c_uint64))

    def test_size_t(self):
        if False:
            print('Hello World!')
        self.assertEqual(sizeof(c_void_p), sizeof(c_size_t))

    def test_ssize_t(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(sizeof(c_void_p), sizeof(c_ssize_t))
if __name__ == '__main__':
    unittest.main()