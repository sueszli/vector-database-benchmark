import unittest
from lbry.wallet.bcd_data_stream import BCDataStream

class TestBCDataStream(unittest.TestCase):

    def test_write_read(self):
        if False:
            for i in range(10):
                print('nop')
        s = BCDataStream()
        s.write_string(b'a' * 252)
        s.write_string(b'b' * 254)
        s.write_string(b'c' * (65535 + 1))
        s.write_boolean(True)
        s.write_boolean(False)
        s.reset()
        self.assertEqual(s.read_string(), b'a' * 252)
        self.assertEqual(s.read_string(), b'b' * 254)
        self.assertEqual(s.read_string(), b'c' * (65535 + 1))
        self.assertTrue(s.read_boolean())
        self.assertFalse(s.read_boolean())