import unittest
from six import PY2
from impacket.dot11 import Dot11Types
from impacket.ImpactDecoder import RadioTapDecoder

class TestDot11ManagementAssociationResponseFrames(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.rawframe = b'\x00\x00\x1c\x00\xef\x18\x00\x00\xc2L\xfa\x00<\x00\x00\x00\x10\x02\x85\t\xa0\x00\xb4\x9e_\x00\x00\x16\x10\x00:\x01p\x1a\x04T\xe3\x86\x00\x18\xf8lvB\x00\x18\xf8lvB\xf0\x02\x11\x04\x00\x00\x04\xc0\x01\x08\x82\x84\x8b\x96$0Hl2\x04\x0c\x12\x18`\xdd\t\x00\x10\x18\x02\x02\xf0\x00\x00\x00f%\xdf7'
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
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_MANAGEMENT_ASSOCIATION_RESPONSE)
        typesubtype = self.dot11.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_MANAGEMENT_SUBTYPE_ASSOCIATION_RESPONSE)
        self.management_base = self.dot11.child()
        if PY2:
            self.assertEqual(str(self.management_base.__class__), 'impacket.dot11.Dot11ManagementFrame')
        else:
            self.assertEqual(str(self.management_base.__class__), "<class 'impacket.dot11.Dot11ManagementFrame'>")
        self.management_association_response = self.management_base.child()
        if PY2:
            self.assertEqual(str(self.management_association_response.__class__), 'impacket.dot11.Dot11ManagementAssociationResponse')
        else:
            self.assertEqual(str(self.management_association_response.__class__), "<class 'impacket.dot11.Dot11ManagementAssociationResponse'>")

    def test_01(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Header and Tail Size field'
        self.assertEqual(self.management_base.get_header_size(), 22)
        self.assertEqual(self.management_base.get_tail_size(), 0)
        self.assertEqual(self.management_association_response.get_header_size(), 33)
        self.assertEqual(self.management_association_response.get_tail_size(), 0)

    def test_02(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Duration field'
        self.assertEqual(self.management_base.get_duration(), 314)
        self.management_base.set_duration(4660)
        self.assertEqual(self.management_base.get_duration(), 4660)

    def test_03(self):
        if False:
            return 10
        'Test Destination Address field'
        addr = self.management_base.get_destination_address()
        self.assertEqual(addr.tolist(), [112, 26, 4, 84, 227, 134])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_destination_address(addr)
        self.assertEqual(self.management_base.get_destination_address().tolist(), [18, 26, 4, 84, 227, 52])

    def test_04(self):
        if False:
            while True:
                i = 10
        'Test Source Address field'
        addr = self.management_base.get_source_address()
        self.assertEqual(addr.tolist(), [0, 24, 248, 108, 118, 66])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_source_address(addr)
        self.assertEqual(self.management_base.get_source_address().tolist(), [18, 24, 248, 108, 118, 52])

    def test_05(self):
        if False:
            print('Hello World!')
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
        self.assertEqual(self.management_base.get_sequence_control(), 752)
        self.management_base.set_sequence_control(4660)
        self.assertEqual(self.management_base.get_sequence_control(), 4660)

    def test_07(self):
        if False:
            print('Hello World!')
        'Test Fragment number field'
        self.assertEqual(self.management_base.get_fragment_number(), 0)
        self.management_base.set_fragment_number(241)
        self.assertEqual(self.management_base.get_fragment_number(), 1)

    def test_08(self):
        if False:
            while True:
                i = 10
        'Test Sequence number field'
        self.assertEqual(self.management_base.get_sequence_number(), 47)
        self.management_base.set_sequence_number(62004)
        self.assertEqual(self.management_base.get_sequence_number(), 564)

    def test_09(self):
        if False:
            return 10
        'Test Management Frame Data field'
        frame_body = b'\x11\x04\x00\x00\x04\xc0\x01\x08\x82\x84\x8b\x96$0Hl2\x04\x0c\x12\x18`\xdd\t\x00\x10\x18\x02\x02\xf0\x00\x00\x00'
        self.assertEqual(self.management_base.get_frame_body(), frame_body)

    def test_10(self):
        if False:
            while True:
                i = 10
        'Test Management Association Response Capabilities field'
        self.assertEqual(self.management_association_response.get_capabilities(), 1041)
        self.management_association_response.set_capabilities(17185)
        self.assertEqual(self.management_association_response.get_capabilities(), 17185)

    def test_11(self):
        if False:
            i = 10
            return i + 15
        'Test Management Association Response Status Code field'
        self.assertEqual(self.management_association_response.get_status_code(), 0)
        self.management_association_response.set_status_code(17185)
        self.assertEqual(self.management_association_response.get_status_code(), 17185)

    def test_12(self):
        if False:
            i = 10
            return i + 15
        'Test Management Association Response Association ID field'
        self.assertEqual(self.management_association_response.get_association_id(), 49156)
        self.management_association_response.set_association_id(17185)
        self.assertEqual(self.management_association_response.get_association_id(), 17185)

    def test_13(self):
        if False:
            i = 10
            return i + 15
        'Test Management Association Response Supported_rates getter/setter methods'
        self.assertEqual(self.management_association_response.get_supported_rates(), (130, 132, 139, 150, 36, 48, 72, 108))
        self.assertEqual(self.management_association_response.get_supported_rates(human_readable=True), (1.0, 2.0, 5.5, 11.0, 18.0, 24.0, 36.0, 54.0))
        self.management_association_response.set_supported_rates((18, 152, 36, 176, 72, 96))
        self.assertEqual(self.management_association_response.get_supported_rates(), (18, 152, 36, 176, 72, 96))
        self.assertEqual(self.management_association_response.get_supported_rates(human_readable=True), (9.0, 12.0, 18.0, 24.0, 36.0, 48.0))
        self.assertEqual(self.management_association_response.get_header_size(), 33 - 2)

    def test_14(self):
        if False:
            i = 10
            return i + 15
        'Test Management Vendor Specific getter/setter methods'
        self.assertEqual(self.management_association_response.get_vendor_specific(), [(b'\x00\x10\x18', b'\x02\x02\xf0\x00\x00\x00')])
        self.management_association_response.add_vendor_specific(b'\x00\x00@', b'\x04\x04\x04\x04\x04\x04')
        self.assertEqual(self.management_association_response.get_vendor_specific(), [(b'\x00\x10\x18', b'\x02\x02\xf0\x00\x00\x00'), (b'\x00\x00@', b'\x04\x04\x04\x04\x04\x04')])
        self.assertEqual(self.management_association_response.get_header_size(), 33 + 11)
if __name__ == '__main__':
    unittest.main(verbosity=1)