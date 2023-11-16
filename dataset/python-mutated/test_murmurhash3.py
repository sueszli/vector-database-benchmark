import unittest
import pytest
import struct
pytestmark = pytest.mark.cosmosEmulator
from azure.cosmos._cosmos_murmurhash3 import murmurhash3_128
from azure.cosmos._cosmos_integers import UInt128

@pytest.mark.usefixtures('teardown')
class MurmurHash3Test(unittest.TestCase):
    """Python Murmurhash3 Tests and its compatibility with backend implementation..
        """
    string_low_value = 2792699143512860960
    string_high_value = 15069672278200047189
    test_seed = UInt128.create(0, 0)
    float_low_value = 16628891264555680919
    float_high_value = 12953474369317462

    def test_float_hash(self):
        if False:
            i = 10
            return i + 15
        ba = bytearray(struct.pack('d', 374.0))
        ret = murmurhash3_128(ba, self.test_seed)
        self.assertEqual(self.float_low_value, ret.get_low().value)
        self.assertEqual(self.float_high_value, ret.get_high().value)

    def test_string_hash(self):
        if False:
            while True:
                i = 10
        s = 'afdgdd'
        ba = bytearray()
        ba.extend(s.encode('utf-8'))
        ret = murmurhash3_128(ba, self.test_seed)
        self.assertEqual(self.string_low_value, ret.get_low().value)
        self.assertEqual(self.string_high_value, ret.get_high().value)
if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit as inst:
        if inst.args[0] is True:
            raise