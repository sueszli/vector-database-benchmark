import unittest
from impacket.dot11 import Dot11, Dot11Types, Dot11ControlFrameCFEndCFACK

class TestDot11FrameControlCFEndCFACK(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.frame_orig = b'\xf4t\xde\xed\xe5V\x85\xf8\xd2;\x96\xae\x0f\xb0\xd9\x8a\x03\x028\x00'
        d = Dot11(self.frame_orig)
        type = d.get_type()
        self.assertEqual(type, Dot11Types.DOT11_TYPE_CONTROL)
        subtype = d.get_subtype()
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_CONTROL_CF_END_CF_ACK)
        typesubtype = d.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_CONTROL_SUBTYPE_CF_END_CF_ACK)
        self.cfendcfack = Dot11ControlFrameCFEndCFACK(d.get_body_as_string())
        d.contains(self.cfendcfack)

    def test_01_HeaderTailSize(self):
        if False:
            return 10
        'Test Header and Tail Size field'
        self.assertEqual(self.cfendcfack.get_header_size(), 14)
        self.assertEqual(self.cfendcfack.get_tail_size(), 0)

    def test_02_Duration(self):
        if False:
            return 10
        'Test Duration field'
        self.assertEqual(self.cfendcfack.get_duration(), 60894)
        self.cfendcfack.set_duration(4660)
        self.assertEqual(self.cfendcfack.get_duration(), 4660)

    def test_03_RA(self):
        if False:
            print('Hello World!')
        'Test RA field'
        ra = self.cfendcfack.get_ra()
        self.assertEqual(ra.tolist(), [229, 86, 133, 248, 210, 59])
        ra[0] = 18
        ra[5] = 52
        self.cfendcfack.set_ra(ra)
        self.assertEqual(self.cfendcfack.get_ra().tolist(), [18, 86, 133, 248, 210, 52])

    def test_04_BSSID(self):
        if False:
            while True:
                i = 10
        'Test BSS ID field'
        bssid = self.cfendcfack.get_bssid()
        self.assertEqual(bssid.tolist(), [150, 174, 15, 176, 217, 138])
        bssid[0] = 18
        bssid[5] = 52
        self.cfendcfack.set_bssid(bssid)
        self.assertEqual(self.cfendcfack.get_bssid().tolist(), [18, 174, 15, 176, 217, 52])
if __name__ == '__main__':
    unittest.main(verbosity=1)