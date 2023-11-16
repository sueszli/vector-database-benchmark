import array
import operator
import unittest
import pywin32_testutil
import win32gui

class TestPyGetString(unittest.TestCase):

    def test_get_string(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ValueError, win32gui.PyGetString, 0)
        self.assertRaises(ValueError, win32gui.PyGetString, 1)
        self.assertRaises(ValueError, win32gui.PyGetString, 1, 1)

class TestPyGetMemory(unittest.TestCase):

    def test_ob(self):
        if False:
            print('Hello World!')
        test_data = b'\x00\x01\x02\x03\x04\x05\x06'
        c = array.array('b', test_data)
        (addr, buflen) = c.buffer_info()
        got = win32gui.PyGetMemory(addr, buflen)
        self.assertEqual(len(got), len(test_data))
        self.assertEqual(bytes(got), test_data)

    def test_memory_index(self):
        if False:
            while True:
                i = 10
        test_data = b'\x00\x01\x02\x03\x04\x05\x06'
        c = array.array('b', test_data)
        (addr, buflen) = c.buffer_info()
        got = win32gui.PyGetMemory(addr, buflen)
        self.assertEqual(got[0], 0)

    def test_memory_slice(self):
        if False:
            i = 10
            return i + 15
        test_data = b'\x00\x01\x02\x03\x04\x05\x06'
        c = array.array('b', test_data)
        (addr, buflen) = c.buffer_info()
        got = win32gui.PyGetMemory(addr, buflen)
        self.assertEqual(list(got[0:3]), [0, 1, 2])

    def test_real_view(self):
        if False:
            while True:
                i = 10
        test_data = b'\x00\x01\x02\x03\x04\x05\x06'
        c = array.array('b', test_data)
        (addr, buflen) = c.buffer_info()
        got = win32gui.PyGetMemory(addr, buflen)
        self.assertEqual(got[0], 0)
        c[0] = 1
        self.assertEqual(got[0], 1)

    def test_memory_not_writable(self):
        if False:
            return 10
        test_data = b'\x00\x01\x02\x03\x04\x05\x06'
        c = array.array('b', test_data)
        (addr, buflen) = c.buffer_info()
        got = win32gui.PyGetMemory(addr, buflen)
        self.assertRaises(TypeError, operator.setitem, got, 0, 1)
if __name__ == '__main__':
    unittest.main()