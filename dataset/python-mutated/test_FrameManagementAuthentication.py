import unittest
from six import PY2
from impacket.dot11 import Dot11Types
from impacket.ImpactDecoder import RadioTapDecoder

class TestDot11ManagementAuthenticationFrames(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.rawframe = b'\x00\x00\x1c\x00\xef\x18\x00\x009Uo\x05<\x00\x00\x00\x10\x02\x85\t\xa0\x00\xb8\x9d`\x00\x00\x1b\xb0\x00:\x01\x00\x18\xf8lvBp\x1a\x04T\xe3\x86\x00\x18\xf8lvB0\xc8\x00\x00\x01\x00\x00\x00\xdd\t\x00\x10\x18\x02\x00\x10\x00\x00\x00\x8ad\xe9;'
        self.radiotap_decoder = RadioTapDecoder()
        radiotap = self.radiotap_decoder.decode(self.rawframe)
        if PY2:
            self.assertEqual(str(radiotap.__class__), 'impacket.dot11.RadioTap')
        else:
            self.assertEqual(str(radiotap.__class__), "<class 'impacket.dot11.RadioTap'>")
        self.dot11 = radiotap.child()
        if PY2:
            self.assertEqual(str(self.dot11.__class__), 'impacket.dot11.Dot11')
        else:
            self.assertEqual(str(self.dot11.__class__), "<class 'impacket.dot11.Dot11'>")
        type = self.dot11.get_type()
        self.assertEqual(type, Dot11Types.DOT11_TYPE_MANAGEMENT)
        subtype = self.dot11.get_subtype()
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_MANAGEMENT_AUTHENTICATION)
        typesubtype = self.dot11.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_MANAGEMENT_SUBTYPE_AUTHENTICATION)
        self.management_base = self.dot11.child()
        if PY2:
            self.assertEqual(str(self.management_base.__class__), 'impacket.dot11.Dot11ManagementFrame')
        else:
            self.assertEqual(str(self.management_base.__class__), "<class 'impacket.dot11.Dot11ManagementFrame'>")
        self.management_authentication = self.management_base.child()
        if PY2:
            self.assertEqual(str(self.management_authentication.__class__), 'impacket.dot11.Dot11ManagementAuthentication')
        else:
            self.assertEqual(str(self.management_authentication.__class__), "<class 'impacket.dot11.Dot11ManagementAuthentication'>")

    def test_01(self):
        if False:
            print('Hello World!')
        'Test Header and Tail Size field'
        self.assertEqual(self.management_base.get_header_size(), 22)
        self.assertEqual(self.management_base.get_tail_size(), 0)
        self.assertEqual(self.management_authentication.get_header_size(), 17)
        self.assertEqual(self.management_authentication.get_tail_size(), 0)

    def test_02(self):
        if False:
            return 10
        'Test Duration field'
        self.assertEqual(self.management_base.get_duration(), 314)
        self.management_base.set_duration(4660)
        self.assertEqual(self.management_base.get_duration(), 4660)

    def test_03(self):
        if False:
            while True:
                i = 10
        'Test Destination Address field'
        addr = self.management_base.get_destination_address()
        self.assertEqual(addr.tolist(), [0, 24, 248, 108, 118, 66])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_destination_address(addr)
        self.assertEqual(self.management_base.get_destination_address().tolist(), [18, 24, 248, 108, 118, 52])

    def test_04(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Source Address field'
        addr = self.management_base.get_source_address()
        self.assertEqual(addr.tolist(), [112, 26, 4, 84, 227, 134])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_source_address(addr)
        self.assertEqual(self.management_base.get_source_address().tolist(), [18, 26, 4, 84, 227, 52])

    def test_05(self):
        if False:
            while True:
                i = 10
        'Test BSSID Address field'
        addr = self.management_base.get_bssid()
        self.assertEqual(addr.tolist(), [0, 24, 248, 108, 118, 66])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_bssid(addr)
        self.assertEqual(self.management_base.get_bssid().tolist(), [18, 24, 248, 108, 118, 52])

    def test_06(self):
        if False:
            while True:
                i = 10
        'Test Sequence control field'
        self.assertEqual(self.management_base.get_sequence_control(), 51248)
        self.management_base.set_sequence_control(4660)
        self.assertEqual(self.management_base.get_sequence_control(), 4660)

    def test_07(self):
        if False:
            return 10
        'Test Fragment number field'
        self.assertEqual(self.management_base.get_fragment_number(), 0)
        self.management_base.set_fragment_number(241)
        self.assertEqual(self.management_base.get_fragment_number(), 1)

    def test_08(self):
        if False:
            print('Hello World!')
        'Test Sequence number field'
        self.assertEqual(self.management_base.get_sequence_number(), 3203)
        self.management_base.set_sequence_number(62004)
        self.assertEqual(self.management_base.get_sequence_number(), 564)

    def test_09(self):
        if False:
            i = 10
            return i + 15
        'Test Management Frame Data field'
        frame_body = b'\x00\x00\x01\x00\x00\x00\xdd\t\x00\x10\x18\x02\x00\x10\x00\x00\x00'
        self.assertEqual(self.management_base.get_frame_body(), frame_body)

    def test_10(self):
        if False:
            while True:
                i = 10
        'Test Management Frame Authentication Algorithm field'
        self.assertEqual(self.management_authentication.get_authentication_algorithm(), 0)
        self.management_authentication.set_authentication_algorithm(34661)
        self.assertEqual(self.management_authentication.get_authentication_algorithm(), 34661)

    def test_11(self):
        if False:
            i = 10
            return i + 15
        'Test Management Frame Authentication Sequence field'
        self.assertEqual(self.management_authentication.get_authentication_sequence(), 1)
        self.management_authentication.set_authentication_sequence(34661)
        self.assertEqual(self.management_authentication.get_authentication_sequence(), 34661)

    def test_12(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Management Frame Authentication Status field'
        self.assertEqual(self.management_authentication.get_authentication_status(), 0)
        self.management_authentication.set_authentication_status(34661)
        self.assertEqual(self.management_authentication.get_authentication_status(), 34661)

    def test_13(self):
        if False:
            while True:
                i = 10
        'Test Management Vendor Specific getter/setter methods'
        self.assertEqual(self.management_authentication.get_vendor_specific(), [(b'\x00\x10\x18', b'\x02\x00\x10\x00\x00\x00')])
        self.management_authentication.add_vendor_specific(b'\x00\x00@', b'\x04\x04\x04\x04\x04\x04')
        self.assertEqual(self.management_authentication.get_vendor_specific(), [(b'\x00\x10\x18', b'\x02\x00\x10\x00\x00\x00'), (b'\x00\x00@', b'\x04\x04\x04\x04\x04\x04')])
        self.assertEqual(self.management_authentication.get_header_size(), 28)
if __name__ == '__main__':
    unittest.main(verbosity=1)