import unittest
from six import PY2
from impacket.dot11 import Dot11Types
from impacket.ImpactDecoder import RadioTapDecoder

class TestDot11ManagementProbeRequestFrames(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.rawProbeRequestframe = b'\x00\x00\x18\x00.H\x00\x00\x00\x02\x85\t\xa0\x00\xda\x01\x00\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\xff\xff\xff\xff\xff\xff\x00#M\x13\xf9\x1b\xff\xff\xff\xff\xff\xff\x90E\x00\x05dlink\x01\x08\x02\x04\x0b\x16\x0c\x12\x18$2\x040H`l'
        self.radiotap_decoder = RadioTapDecoder()
        radiotap = self.radiotap_decoder.decode(self.rawProbeRequestframe)
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
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_MANAGEMENT_PROBE_REQUEST)
        typesubtype = self.dot11.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_MANAGEMENT_SUBTYPE_PROBE_REQUEST)
        self.management_base = self.dot11.child()
        if PY2:
            self.assertEqual(str(self.management_base.__class__), 'impacket.dot11.Dot11ManagementFrame')
        else:
            self.assertEqual(str(self.management_base.__class__), "<class 'impacket.dot11.Dot11ManagementFrame'>")
        self.management_probe_request = self.management_base.child()
        if PY2:
            self.assertEqual(str(self.management_probe_request.__class__), 'impacket.dot11.Dot11ManagementProbeRequest')
        else:
            self.assertEqual(str(self.management_probe_request.__class__), "<class 'impacket.dot11.Dot11ManagementProbeRequest'>")

    def test_01(self):
        if False:
            return 10
        'Test Header and Tail Size field'
        self.assertEqual(self.management_base.get_header_size(), 22)
        self.assertEqual(self.management_base.get_tail_size(), 0)
        self.assertEqual(self.management_probe_request.get_header_size(), 23)
        self.assertEqual(self.management_probe_request.get_tail_size(), 0)

    def test_02(self):
        if False:
            while True:
                i = 10
        'Test Duration field'
        self.assertEqual(self.management_base.get_duration(), 0)
        self.management_base.set_duration(4660)
        self.assertEqual(self.management_base.get_duration(), 4660)

    def test_03(self):
        if False:
            return 10
        'Test Destination Address field'
        addr = self.management_base.get_destination_address()
        self.assertEqual(addr.tolist(), [255, 255, 255, 255, 255, 255])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_destination_address(addr)
        self.assertEqual(self.management_base.get_destination_address().tolist(), [18, 255, 255, 255, 255, 52])

    def test_04(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Source Address field'
        addr = self.management_base.get_source_address()
        self.assertEqual(addr.tolist(), [0, 35, 77, 19, 249, 27])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_source_address(addr)
        self.assertEqual(self.management_base.get_source_address().tolist(), [18, 35, 77, 19, 249, 52])

    def test_05(self):
        if False:
            i = 10
            return i + 15
        'Test BSSID Address field'
        addr = self.management_base.get_bssid()
        self.assertEqual(addr.tolist(), [255, 255, 255, 255, 255, 255])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_bssid(addr)
        self.assertEqual(self.management_base.get_bssid().tolist(), [18, 255, 255, 255, 255, 52])

    def test_06(self):
        if False:
            i = 10
            return i + 15
        'Test Sequence control field'
        self.assertEqual(self.management_base.get_sequence_control(), 17808)
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
            return 10
        'Test Sequence number field'
        self.assertEqual(self.management_base.get_sequence_number(), 1113)
        self.management_base.set_sequence_number(62004)
        self.assertEqual(self.management_base.get_sequence_number(), 564)

    def test_09(self):
        if False:
            return 10
        'Test Management Frame Data field'
        frame_body = b'\x00\x05dlink\x01\x08\x02\x04\x0b\x16\x0c\x12\x18$2\x040H`l'
        self.assertEqual(self.management_base.get_frame_body(), frame_body)

    def test_10(self):
        if False:
            while True:
                i = 10
        'Test Management ssid getter/setter methods'
        act_ssid = b'dlink'
        new_ssid = b'holala'
        self.assertEqual(self.management_probe_request.get_ssid(), act_ssid)
        self.management_probe_request.set_ssid(new_ssid)
        self.assertEqual(self.management_probe_request.get_ssid(), new_ssid)
        self.assertEqual(self.management_probe_request.get_header_size(), 23 + len(new_ssid) - len(act_ssid))

    def test_11(self):
        if False:
            return 10
        'Test Management supported_rates getter/setter methods'
        self.assertEqual(self.management_probe_request.get_supported_rates(), (2, 4, 11, 22, 12, 18, 24, 36))
        self.assertEqual(self.management_probe_request.get_supported_rates(human_readable=True), (1.0, 2.0, 5.5, 11.0, 6.0, 9.0, 12.0, 18.0))
        self.management_probe_request.set_supported_rates((4, 11, 22, 12, 18, 24))
        self.assertEqual(self.management_probe_request.get_supported_rates(), (4, 11, 22, 12, 18, 24))
        self.assertEqual(self.management_probe_request.get_supported_rates(human_readable=True), (2.0, 5.5, 11.0, 6.0, 9.0, 12.0))
        self.assertEqual(self.management_probe_request.get_header_size(), 23 - 2)
if __name__ == '__main__':
    unittest.main(verbosity=1)