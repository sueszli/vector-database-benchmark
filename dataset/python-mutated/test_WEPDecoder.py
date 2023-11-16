import unittest
from six import PY2
from binascii import unhexlify
from impacket.dot11 import Dot11, Dot11Types, Dot11DataFrame, Dot11WEP, Dot11WEPData
from impacket.ImpactPacket import IP, ICMP
from impacket.Dot11KeyManager import KeyManager
from impacket.ImpactDecoder import Dot11Decoder

class TestDot11WEPData(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.dot11frame = b'\x08A,\x00\x00!)h3]\x00\x18\xde|7\x9f\x00!)h3[\xf0\xd6\x0c1e\x00\x8d#\x81\xe9%\x1c\xb5\xaa\x83\xd2\xc7\x16\xban\xe1\x8e}:,q\xc0\x0fj\xb8/\xbcT\xc4\xb0\x14\xab\x03\x11^\xde\xcc\xab+\x18\xeb\xeb%\x0fu\xebk\xf5\x7f\xd6\\\xb9\xe1\xb2nP\xbaK\xb4\x8b\x9f4q\xda\x9e\xcf\x12\xcb\x8f6\x1b\x02S'
        d = Dot11(self.dot11frame, FCS_at_end=False)
        self.assertEqual(d.get_type(), Dot11Types.DOT11_TYPE_DATA)
        self.assertEqual(d.get_subtype(), Dot11Types.DOT11_SUBTYPE_DATA)
        self.assertEqual(d.get_type_n_subtype(), Dot11Types.DOT11_TYPE_DATA_SUBTYPE_DATA)
        data = Dot11DataFrame(d.get_body_as_string())
        d.contains(data)
        self.wep_header = Dot11WEP(data.body_string)
        data.contains(self.wep_header)
        self.wep_data = Dot11WEPData(self.wep_header.body_string)
        self.wep_header.contains(self.wep_data)
        self.km = KeyManager()
        self.km.add_key([0, 33, 41, 104, 51, 93], unhexlify(b'999cbb701ca2ef030e302dcc35'))

    def test_01(self):
        if False:
            while True:
                i = 10
        'Test WEPHeader is_WEP method'
        self.assertEqual(self.wep_header.is_WEP(), True)

    def test_02(self):
        if False:
            return 10
        'Test Packet Hierarchy'
        dot11_decoder = Dot11Decoder()
        dot11_decoder.FCS_at_end(False)
        dot11_decoder.set_key_manager(self.km)
        in0 = dot11_decoder.decode(self.dot11frame)
        if PY2:
            self.assertEqual(str(in0.__class__), 'impacket.dot11.Dot11')
        else:
            self.assertEqual(str(in0.__class__), "<class 'impacket.dot11.Dot11'>")
        in1 = in0.child()
        if PY2:
            self.assertEqual(str(in1.__class__), 'impacket.dot11.Dot11DataFrame')
        else:
            self.assertEqual(str(in1.__class__), "<class 'impacket.dot11.Dot11DataFrame'>")
        in2 = in1.child()
        if PY2:
            self.assertEqual(str(in2.__class__), 'impacket.dot11.Dot11WEP')
        else:
            self.assertEqual(str(in2.__class__), "<class 'impacket.dot11.Dot11WEP'>")
        in3 = in2.child()
        if PY2:
            self.assertEqual(str(in3.__class__), 'impacket.dot11.Dot11WEPData')
        else:
            self.assertEqual(str(in3.__class__), "<class 'impacket.dot11.Dot11WEPData'>")
        in4 = in3.child()
        if PY2:
            self.assertEqual(str(in4.__class__), 'impacket.dot11.LLC')
        else:
            self.assertEqual(str(in4.__class__), "<class 'impacket.dot11.LLC'>")
        in5 = in4.child()
        if PY2:
            self.assertEqual(str(in5.__class__), 'impacket.dot11.SNAP')
        else:
            self.assertEqual(str(in5.__class__), "<class 'impacket.dot11.SNAP'>")
        in6 = in5.child()
        in7 = in6.child()
        in8 = in7.child()
        self.assertEqual(in8.get_packet(), b'abcdefghijklmnopqrstuvwabcdefghi')

    def test_03(self):
        if False:
            return 10
        'Test WEPHeader IV getter and setter methods'
        self.assertEqual(self.wep_header.get_iv(), 799077)
        self.wep_header.set_iv(1967361)
        self.assertEqual(self.wep_header.get_iv(), 1967361)

    def test_04(self):
        if False:
            print('Hello World!')
        'Test WEPHeader keyID getter and setter methods'
        self.assertEqual(self.wep_header.get_keyid(), 0)
        self.wep_header.set_iv(3)
        self.assertEqual(self.wep_header.get_iv(), 3)

    def test_05(self):
        if False:
            print('Hello World!')
        'Test WEPData ICV getter and setter methods'
        dot11_decoder = Dot11Decoder()
        dot11_decoder.FCS_at_end(False)
        dot11_decoder.set_key_manager(self.km)
        dot11_decoder.decode(self.dot11frame)
        wepdata = dot11_decoder.get_protocol(Dot11WEPData)
        self.assertEqual(wepdata.get_icv(), 2717464965)
        self.assertEqual(wepdata.get_computed_icv(), 2717464965)
        self.assertEqual(wepdata.get_icv(), wepdata.get_computed_icv())
        wepdata.set_icv(287454020)
        self.assertEqual(wepdata.get_icv(), 287454020)

    def test_06(self):
        if False:
            i = 10
            return i + 15
        'Test WEPData body decryption'
        dot11_decoder = Dot11Decoder()
        dot11_decoder.FCS_at_end(False)
        dot11_decoder.set_key_manager(self.km)
        dot11_decoder.decode(self.dot11frame)
        dot11_decoder.get_protocol(Dot11WEP)
        wepdata = dot11_decoder.get_protocol(Dot11WEPData)
        decrypted = b'\xaa\xaa\x03\x00\x00\x00\x08\x00E\x00\x00<\xa6\x07\x00\x00\x80\x01\xeeZ\xc0\xa8\x01f@\xe9\xa3g\x08\x00\xc5V\x04\x00\x84\x05abcdefghijklmnopqrstuvwabcdefghi\xa1\xf99\x85'
        self.assertEqual(wepdata.get_packet(), decrypted)
        self.assertEqual(wepdata.check_icv(), True)
        ip = dot11_decoder.get_protocol(IP)
        self.assertEqual(ip.get_ip_src(), '192.168.1.102')
        self.assertEqual(ip.get_ip_dst(), '64.233.163.103')
        icmp = dot11_decoder.get_protocol(ICMP)
        self.assertEqual(icmp.get_icmp_type(), icmp.ICMP_ECHO)
        self.assertEqual(icmp.get_icmp_id(), 1024)
if __name__ == '__main__':
    unittest.main(verbosity=1)