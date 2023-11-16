import unittest
import xdrlib

class XDRTest(unittest.TestCase):

    def test_xdr(self):
        if False:
            while True:
                i = 10
        p = xdrlib.Packer()
        s = b'hello world'
        a = [b'what', b'is', b'hapnin', b'doctor']
        p.pack_int(42)
        p.pack_int(-17)
        p.pack_uint(9)
        p.pack_bool(True)
        p.pack_bool(False)
        p.pack_uhyper(45)
        p.pack_float(1.9)
        p.pack_double(1.9)
        p.pack_string(s)
        p.pack_list(range(5), p.pack_uint)
        p.pack_array(a, p.pack_string)
        data = p.get_buffer()
        up = xdrlib.Unpacker(data)
        self.assertEqual(up.get_position(), 0)
        self.assertEqual(up.unpack_int(), 42)
        self.assertEqual(up.unpack_int(), -17)
        self.assertEqual(up.unpack_uint(), 9)
        self.assertTrue(up.unpack_bool() is True)
        pos = up.get_position()
        self.assertTrue(up.unpack_bool() is False)
        up.set_position(pos)
        self.assertTrue(up.unpack_bool() is False)
        self.assertEqual(up.unpack_uhyper(), 45)
        self.assertAlmostEqual(up.unpack_float(), 1.9)
        self.assertAlmostEqual(up.unpack_double(), 1.9)
        self.assertEqual(up.unpack_string(), s)
        self.assertEqual(up.unpack_list(up.unpack_uint), list(range(5)))
        self.assertEqual(up.unpack_array(up.unpack_string), a)
        up.done()
        self.assertRaises(EOFError, up.unpack_uint)

class ConversionErrorTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.packer = xdrlib.Packer()

    def assertRaisesConversion(self, *args):
        if False:
            print('Hello World!')
        self.assertRaises(xdrlib.ConversionError, *args)

    def test_pack_int(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaisesConversion(self.packer.pack_int, 'string')

    def test_pack_uint(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaisesConversion(self.packer.pack_uint, 'string')

    def test_float(self):
        if False:
            while True:
                i = 10
        self.assertRaisesConversion(self.packer.pack_float, 'string')

    def test_double(self):
        if False:
            return 10
        self.assertRaisesConversion(self.packer.pack_double, 'string')

    def test_uhyper(self):
        if False:
            return 10
        self.assertRaisesConversion(self.packer.pack_uhyper, 'string')
if __name__ == '__main__':
    unittest.main()