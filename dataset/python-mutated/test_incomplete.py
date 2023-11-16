import unittest
from ctypes import *

class MyTestCase(unittest.TestCase):

    def test_incomplete_example(self):
        if False:
            for i in range(10):
                print('nop')
        lpcell = POINTER('cell')

        class cell(Structure):
            _fields_ = [('name', c_char_p), ('next', lpcell)]
        SetPointerType(lpcell, cell)
        c1 = cell()
        c1.name = b'foo'
        c2 = cell()
        c2.name = b'bar'
        c1.next = pointer(c2)
        c2.next = pointer(c1)
        p = c1
        result = []
        for i in range(8):
            result.append(p.name)
            p = p.next[0]
        self.assertEqual(result, [b'foo', b'bar'] * 4)
        from ctypes import _pointer_type_cache
        del _pointer_type_cache[cell]
if __name__ == '__main__':
    unittest.main()