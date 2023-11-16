import six
import unittest
from binascii import hexlify
from impacket.IP6_Address import IP6_Address

def hexl(b):
    if False:
        i = 10
        return i + 15
    return hexlify(b).decode('ascii')

class IP6AddressTests(unittest.TestCase):

    def test_bin(self):
        if False:
            i = 10
            return i + 15
        tests = (('A:B:C:D:E:F:1:2', '000a000b000c000d000e000f00010002', 'A:B:C:D:E:F:1:2'), ('A:B:0:D:E:F:0:2', '000a000b0000000d000e000f00000002', 'A:B::D:E:F:0:2'), ('A::BC:E:D', '000a000000000000000000bc000e000d', 'A::BC:E:D'), ('A::BCD:EFFF:D', '000a00000000000000000bcdefff000d', 'A::BCD:EFFF:D'), ('FE80:0000:0000:0000:020C:29FF:FE26:E251', 'fe80000000000000020c29fffe26e251', 'FE80::20C:29FF:FE26:E251'), ('::', '00000000000000000000000000000000', '::'), ('1::', '00010000000000000000000000000000', '1::'), ('::2', '00000000000000000000000000000002', '::2'))
        for (torig, thex, texp) in tests:
            ip = IP6_Address(torig)
            byt = ip.as_bytes()
            self.assertEqual(hexl(byt), thex)
            self.assertEqual(ip.as_string(), texp)

    def test_malformed(self):
        if False:
            print('Hello World!')
        with six.assertRaisesRegex(self, Exception, 'address size'):
            IP6_Address('ABCD:EFAB:1234:1234:1234:1234:1234:12345')
        with six.assertRaisesRegex(self, Exception, 'triple colon'):
            IP6_Address(':::')
        with six.assertRaisesRegex(self, Exception, 'triple colon'):
            IP6_Address('::::')
if __name__ == '__main__':
    unittest.main(verbosity=1)