import unittest
from six import PY2
from impacket.dot11 import Dot11Types
from impacket.ImpactDecoder import RadioTapDecoder

class TestDot11ManagementProbeResponseFrames(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.rawProbeResponseframe = b'\x00\x00\x18\x00.H\x00\x00\x00\x02\x85\t\xa0\x00\xb0\x01\x00\x00\x00\x00\x00\x00\x00\x00P\x00:\x01\x00!\xfe9?w\x00\x1b\x112f#\x00\x1b\x112f# s\x7f\xa0"\xf8?\x01\x00\x00d\x00\x11\x04\x00\x07freebsd\x01\x08\x82\x84\x8b\x96$0Hl\x03\x01\x06*\x01\x04/\x01\x042\x04\x0c\x12\x18`\xddu\x00P\xf2\x04\x10J\x00\x01\x10\x10D\x00\x01\x02\x10A\x00\x01\x00\x10;\x00\x01\x03\x10G\x00\x10\x11N\xf7F\xa9\xc6\xfb\x1dp\x1b\x00\x1b\x112f#\x10!\x00\x06D-Link\x10#\x00\x07DIR-320\x10$\x00\x07DIR-320\x10B\x00\x0800000000\x10T\x00\x08\x00\x06\x00P\xf2\x04\x00\x01\x10\x11\x00\x07DIR-320\x10\x08\x00\x02\x00\x8e\xdd\x05\x00P\xf2\x05\x00\xdd\t\x00\x10\x18\x02\x01\xf0\x00\x00\x00\xdd\x18\x00P\xf2\x01\x01\x00\x00P\xf2\x02\x01\x00\x00P\xf2\x02\x01\x00\x00P\xf2\x02\x00\x00'
        self.radiotap_decoder = RadioTapDecoder()
        radiotap = self.radiotap_decoder.decode(self.rawProbeResponseframe)
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
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_MANAGEMENT_PROBE_RESPONSE)
        typesubtype = self.dot11.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_MANAGEMENT_SUBTYPE_PROBE_RESPONSE)
        self.management_base = self.dot11.child()
        if PY2:
            self.assertEqual(str(self.management_base.__class__), 'impacket.dot11.Dot11ManagementFrame')
        else:
            self.assertEqual(str(self.management_base.__class__), "<class 'impacket.dot11.Dot11ManagementFrame'>")
        self.management_probe_response = self.management_base.child()
        if PY2:
            self.assertEqual(str(self.management_probe_response.__class__), 'impacket.dot11.Dot11ManagementProbeResponse')
        else:
            self.assertEqual(str(self.management_probe_response.__class__), "<class 'impacket.dot11.Dot11ManagementProbeResponse'>")

    def test_01(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Header and Tail Size field'
        self.assertEqual(self.management_base.get_header_size(), 22)
        self.assertEqual(self.management_base.get_tail_size(), 0)
        self.assertEqual(self.management_probe_response.get_header_size(), 209)
        self.assertEqual(self.management_probe_response.get_tail_size(), 0)

    def test_02(self):
        if False:
            i = 10
            return i + 15
        'Test Duration field'
        self.assertEqual(self.management_base.get_duration(), 314)
        self.management_base.set_duration(4660)
        self.assertEqual(self.management_base.get_duration(), 4660)

    def test_03(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Destination Address field'
        addr = self.management_base.get_destination_address()
        self.assertEqual(addr.tolist(), [0, 33, 254, 57, 63, 119])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_destination_address(addr)
        self.assertEqual(self.management_base.get_destination_address().tolist(), [18, 33, 254, 57, 63, 52])

    def test_04(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Source Address field'
        addr = self.management_base.get_source_address()
        self.assertEqual(addr.tolist(), [0, 27, 17, 50, 102, 35])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_source_address(addr)
        self.assertEqual(self.management_base.get_source_address().tolist(), [18, 27, 17, 50, 102, 52])

    def test_05(self):
        if False:
            print('Hello World!')
        'Test BSSID Address field'
        addr = self.management_base.get_bssid()
        self.assertEqual(addr.tolist(), [0, 27, 17, 50, 102, 35])
        addr[0] = 18
        addr[5] = 52
        self.management_base.set_bssid(addr)
        self.assertEqual(self.management_base.get_bssid().tolist(), [18, 27, 17, 50, 102, 52])

    def test_06(self):
        if False:
            print('Hello World!')
        'Test Sequence control field'
        self.assertEqual(self.management_base.get_sequence_control(), 29472)
        self.management_base.set_sequence_control(4660)
        self.assertEqual(self.management_base.get_sequence_control(), 4660)

    def test_07(self):
        if False:
            while True:
                i = 10
        'Test Fragment number field'
        self.assertEqual(self.management_base.get_fragment_number(), 0)
        self.management_base.set_fragment_number(241)
        self.assertEqual(self.management_base.get_fragment_number(), 1)

    def test_08(self):
        if False:
            return 10
        'Test Sequence number field'
        self.assertEqual(self.management_base.get_sequence_number(), 1842)
        self.management_base.set_sequence_number(62004)
        self.assertEqual(self.management_base.get_sequence_number(), 564)

    def test_09(self):
        if False:
            print('Hello World!')
        'Test Management Frame Data field'
        frame_body = b'\x7f\xa0"\xf8?\x01\x00\x00d\x00\x11\x04\x00\x07freebsd\x01\x08\x82\x84\x8b\x96$0Hl\x03\x01\x06*\x01\x04/\x01\x042\x04\x0c\x12\x18`\xddu\x00P\xf2\x04\x10J\x00\x01\x10\x10D\x00\x01\x02\x10A\x00\x01\x00\x10;\x00\x01\x03\x10G\x00\x10\x11N\xf7F\xa9\xc6\xfb\x1dp\x1b\x00\x1b\x112f#\x10!\x00\x06D-Link\x10#\x00\x07DIR-320\x10$\x00\x07DIR-320\x10B\x00\x0800000000\x10T\x00\x08\x00\x06\x00P\xf2\x04\x00\x01\x10\x11\x00\x07DIR-320\x10\x08\x00\x02\x00\x8e\xdd\x05\x00P\xf2\x05\x00\xdd\t\x00\x10\x18\x02\x01\xf0\x00\x00\x00\xdd\x18\x00P\xf2\x01\x01\x00\x00P\xf2\x02\x01\x00\x00P\xf2\x02\x01\x00\x00P\xf2\x02\x00\x00'
        self.assertEqual(self.management_base.get_frame_body(), frame_body)

    def test_10(self):
        if False:
            while True:
                i = 10
        'Test Management Beacon Timestamp field'
        self.assertEqual(self.management_probe_response.get_timestamp(), 1374257586303)
        self.management_probe_response.set_timestamp(9756277976800118119)
        self.assertEqual(self.management_probe_response.get_timestamp(), 9756277976800118119)

    def test_11(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Management Beacon Interval field'
        self.assertEqual(self.management_probe_response.get_beacon_interval(), 100)
        self.management_probe_response.set_beacon_interval(17185)
        self.assertEqual(self.management_probe_response.get_beacon_interval(), 17185)

    def test_12(self):
        if False:
            while True:
                i = 10
        'Test Management Beacon Capabilities field'
        self.assertEqual(self.management_probe_response.get_capabilities(), 1041)
        self.management_probe_response.set_capabilities(17185)
        self.assertEqual(self.management_probe_response.get_capabilities(), 17185)

    def test_13(self):
        if False:
            print('Hello World!')
        'Test Management ssid getter/setter methods'
        act_ssid = b'freebsd'
        new_ssid = b'holala'
        self.assertEqual(self.management_probe_response.get_ssid(), act_ssid)
        self.management_probe_response.set_ssid(new_ssid)
        self.assertEqual(self.management_probe_response.get_ssid(), new_ssid)
        self.assertEqual(self.management_probe_response.get_header_size(), 209 - 1)

    def test_14(self):
        if False:
            while True:
                i = 10
        'Test Management supported_rates getter/setter methods'
        self.assertEqual(self.management_probe_response.get_supported_rates(), (130, 132, 139, 150, 36, 48, 72, 108))
        self.assertEqual(self.management_probe_response.get_supported_rates(human_readable=True), (1.0, 2.0, 5.5, 11.0, 18.0, 24.0, 36.0, 54.0))
        self.management_probe_response.set_supported_rates((132, 139, 150, 36, 48, 72))
        self.assertEqual(self.management_probe_response.get_supported_rates(), (132, 139, 150, 36, 48, 72))
        self.assertEqual(self.management_probe_response.get_supported_rates(human_readable=True), (2.0, 5.5, 11.0, 18.0, 24.0, 36.0))
        self.assertEqual(self.management_probe_response.get_header_size(), 209 - 2)

    def test_15(self):
        if False:
            return 10
        'Test Management DS Parameter Set getter/setter methods'
        self.assertEqual(self.management_probe_response.get_ds_parameter_set(), 6)
        self.management_probe_response.set_ds_parameter_set(40)
        self.assertEqual(self.management_probe_response.get_ds_parameter_set(), 40)
        self.assertEqual(self.management_probe_response.get_header_size(), 209)

    def test_16(self):
        if False:
            i = 10
            return i + 15
        'Test Management Vendor Specific getter/setter methods'
        self.assertEqual(self.management_probe_response.get_vendor_specific(), [(b'\x00P\xf2', b'\x04\x10J\x00\x01\x10\x10D\x00\x01\x02\x10A\x00\x01\x00\x10;\x00\x01\x03\x10G\x00\x10\x11N\xf7F\xa9\xc6\xfb\x1dp\x1b\x00\x1b\x112f#\x10!\x00\x06D-Link\x10#\x00\x07DIR-320\x10$\x00\x07DIR-320\x10B\x00\x0800000000\x10T\x00\x08\x00\x06\x00P\xf2\x04\x00\x01\x10\x11\x00\x07DIR-320\x10\x08\x00\x02\x00\x8e'), (b'\x00P\xf2', b'\x05\x00'), (b'\x00\x10\x18', b'\x02\x01\xf0\x00\x00\x00'), (b'\x00P\xf2', b'\x01\x01\x00\x00P\xf2\x02\x01\x00\x00P\xf2\x02\x01\x00\x00P\xf2\x02\x00\x00')])
        self.management_probe_response.add_vendor_specific(b'\x00\x00@', b'\x04\x04\x04\x04\x04\x04')
        self.assertEqual(self.management_probe_response.get_vendor_specific(), [(b'\x00P\xf2', b'\x04\x10J\x00\x01\x10\x10D\x00\x01\x02\x10A\x00\x01\x00\x10;\x00\x01\x03\x10G\x00\x10\x11N\xf7F\xa9\xc6\xfb\x1dp\x1b\x00\x1b\x112f#\x10!\x00\x06D-Link\x10#\x00\x07DIR-320\x10$\x00\x07DIR-320\x10B\x00\x0800000000\x10T\x00\x08\x00\x06\x00P\xf2\x04\x00\x01\x10\x11\x00\x07DIR-320\x10\x08\x00\x02\x00\x8e'), (b'\x00P\xf2', b'\x05\x00'), (b'\x00\x10\x18', b'\x02\x01\xf0\x00\x00\x00'), (b'\x00P\xf2', b'\x01\x01\x00\x00P\xf2\x02\x01\x00\x00P\xf2\x02\x01\x00\x00P\xf2\x02\x00\x00'), (b'\x00\x00@', b'\x04\x04\x04\x04\x04\x04')])
        self.assertEqual(self.management_probe_response.get_header_size(), 209 + 6 + 3 + 2)
if __name__ == '__main__':
    unittest.main(verbosity=1)