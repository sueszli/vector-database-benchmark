import unittest
from six import PY2
from impacket.dot11 import Dot11Types
from impacket.ImpactDecoder import RadioTapDecoder

class TestDot11ManagementReassociationRequestFrames(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.rawframe = b'\x00\x00\x1c\x00\xef\x18\x00\x00\x9aK\x87\xae;\x00\x00\x00\x10\x02\x85\t\xa0\x00\xb5\x9d`\x00\x00\x18 \x00:\x01\x00\x18\xf8lvBp\x1a\x04T\xe3\x86\x00\x18\xf8lvB\x00\x081\x04\n\x00\x00\x18\xf8lvB\x00\x05ddwrt\x01\x08\x82\x84\x8b\x96$0Hl!\x02\n\x11$\x02\x01\x0e0\x14\x01\x00\x00\x0f\xac\x04\x01\x00\x00\x0f\xac\x04\x01\x00\x00\x0f\xac\x02\x08\x002\x04\x0c\x12\x18`\xdd\t\x00\x10\x18\x02\x00\x10\x00\x00\x00p\x97\x1cA'
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
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_MANAGEMENT_REASSOCIATION_REQUEST)
        typesubtype = self.dot11.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_MANAGEMENT_SUBTYPE_REASSOCIATION_REQUEST)
        self.management_base = self.dot11.child()
        if PY2:
            self.assertEqual(str(self.management_base.__class__), 'impacket.dot11.Dot11ManagementFrame')
        else:
            self.assertEqual(str(self.management_base.__class__), "<class 'impacket.dot11.Dot11ManagementFrame'>")
        self.management_reassociation_request = self.management_base.child()
        if PY2:
            self.assertEqual(str(self.management_reassociation_request.__class__), 'impacket.dot11.Dot11ManagementReassociationRequest')
        else:
            self.assertEqual(str(self.management_reassociation_request.__class__), "<class 'impacket.dot11.Dot11ManagementReassociationRequest'>")

    def test_01(self):
        if False:
            print('Hello World!')
        'Test Header and Tail Size field'
        self.assertEqual(self.management_base.get_header_size(), 22)
        self.assertEqual(self.management_base.get_tail_size(), 0)
        self.assertEqual(self.management_reassociation_request.get_header_size(), 74)
        self.assertEqual(self.management_reassociation_request.get_tail_size(), 0)

    def test_02(self):
        if False:
            while True:
                i = 10
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
            while True:
                i = 10
        'Test Source Address field'
        addr = self.management_base.get_source_address()
        self.assertEqual(addr.tolist(), [112, 26, 4, 84, 227, 134])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_source_address(addr)
        self.assertEqual(self.management_base.get_source_address().tolist(), [18, 26, 4, 84, 227, 52])

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
            print('Hello World!')
        'Test Sequence control field'
        self.assertEqual(self.management_base.get_sequence_control(), 2048)
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
            while True:
                i = 10
        'Test Sequence number field'
        self.assertEqual(self.management_base.get_sequence_number(), 128)
        self.management_base.set_sequence_number(62004)
        self.assertEqual(self.management_base.get_sequence_number(), 564)

    def test_09(self):
        if False:
            while True:
                i = 10
        'Test Management Frame Data field'
        frame_body = b'1\x04\n\x00\x00\x18\xf8lvB\x00\x05ddwrt\x01\x08\x82\x84\x8b\x96$0Hl!\x02\n\x11$\x02\x01\x0e0\x14\x01\x00\x00\x0f\xac\x04\x01\x00\x00\x0f\xac\x04\x01\x00\x00\x0f\xac\x02\x08\x002\x04\x0c\x12\x18`\xdd\t\x00\x10\x18\x02\x00\x10\x00\x00\x00'
        self.assertEqual(self.management_base.get_frame_body(), frame_body)

    def test_10(self):
        if False:
            i = 10
            return i + 15
        'Test Management Reassociation Request Capabilities field'
        self.assertEqual(self.management_reassociation_request.get_capabilities(), 1073)
        self.management_reassociation_request.set_capabilities(17185)
        self.assertEqual(self.management_reassociation_request.get_capabilities(), 17185)

    def test_11(self):
        if False:
            return 10
        'Test Management Reassociation Request Listen Interval field'
        self.assertEqual(self.management_reassociation_request.get_listen_interval(), 10)
        self.management_reassociation_request.set_listen_interval(17185)
        self.assertEqual(self.management_reassociation_request.get_listen_interval(), 17185)

    def test_12(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Management Reassociation Request Current AP field'
        addr = self.management_reassociation_request.get_current_ap()
        self.assertEqual(addr.tolist(), [0, 24, 248, 108, 118, 66])
        addr[0] = 18
        addr[5] = 52
        self.management_reassociation_request.set_current_ap(addr)
        self.assertEqual(self.management_reassociation_request.get_current_ap().tolist(), [18, 24, 248, 108, 118, 52])

    def test_13(self):
        if False:
            while True:
                i = 10
        'Test Management Reassociation Request Ssid getter/setter methods'
        act_ssid = b'ddwrt'
        new_ssid = b'holala'
        self.assertEqual(self.management_reassociation_request.get_ssid(), act_ssid)
        self.management_reassociation_request.set_ssid(new_ssid)
        self.assertEqual(self.management_reassociation_request.get_ssid(), new_ssid)
        self.assertEqual(self.management_reassociation_request.get_header_size(), 74 + 1)

    def test_14(self):
        if False:
            return 10
        'Test Management Ressociation Request Supported_rates getter/setter methods'
        self.assertEqual(self.management_reassociation_request.get_supported_rates(), (130, 132, 139, 150, 36, 48, 72, 108))
        self.assertEqual(self.management_reassociation_request.get_supported_rates(human_readable=True), (1.0, 2.0, 5.5, 11.0, 18.0, 24.0, 36.0, 54.0))
        self.management_reassociation_request.set_supported_rates((18, 152, 36, 176, 72, 96))
        self.assertEqual(self.management_reassociation_request.get_supported_rates(), (18, 152, 36, 176, 72, 96))
        self.assertEqual(self.management_reassociation_request.get_supported_rates(human_readable=True), (9.0, 12.0, 18.0, 24.0, 36.0, 48.0))
        self.assertEqual(self.management_reassociation_request.get_header_size(), 74 - 2)

    def test_15(self):
        if False:
            i = 10
            return i + 15
        'Test Management Association Request RSN getter/setter methods'
        self.assertEqual(self.management_reassociation_request.get_rsn(), b'\x01\x00\x00\x0f\xac\x04\x01\x00\x00\x0f\xac\x04\x01\x00\x00\x0f\xac\x02\x08\x00')
        self.management_reassociation_request.set_rsn(b'\xff\x00\x00\x0f\xac\x04\x01\x00\x00\x0f\xac\x04\x01\x00\x00\x0f\xac\x02\x08\xff')
        self.assertEqual(self.management_reassociation_request.get_rsn(), b'\xff\x00\x00\x0f\xac\x04\x01\x00\x00\x0f\xac\x04\x01\x00\x00\x0f\xac\x02\x08\xff')
        self.assertEqual(self.management_reassociation_request.get_header_size(), 74)

    def test_16(self):
        if False:
            print('Hello World!')
        'Test Management Vendor Specific getter/setter methods'
        self.assertEqual(self.management_reassociation_request.get_vendor_specific(), [(b'\x00\x10\x18', b'\x02\x00\x10\x00\x00\x00')])
        self.management_reassociation_request.add_vendor_specific(b'\x00\x00@', b'\x04\x04\x04\x04\x04\x04')
        self.assertEqual(self.management_reassociation_request.get_vendor_specific(), [(b'\x00\x10\x18', b'\x02\x00\x10\x00\x00\x00'), (b'\x00\x00@', b'\x04\x04\x04\x04\x04\x04')])
        self.assertEqual(self.management_reassociation_request.get_header_size(), 74 + 11)
if __name__ == '__main__':
    unittest.main(verbosity=1)