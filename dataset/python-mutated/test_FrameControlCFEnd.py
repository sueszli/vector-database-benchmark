import unittest
from impacket.dot11 import Dot11, Dot11Types, Dot11ControlFrameCFEnd

class TestDot11FrameControlCFEnd(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.frame_orig = b'\xe4\x00\x00\x00\xff\xff\xff\xff\xff\xff\x00\x19\xe0\x98\x04\xd4\xad\x9c<\xc0'
        d = Dot11(self.frame_orig)
        type = d.get_type()
        self.assertEqual(type, Dot11Types.DOT11_TYPE_CONTROL)
        subtype = d.get_subtype()
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_CONTROL_CF_END)
        typesubtype = d.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_CONTROL_SUBTYPE_CF_END)
        self.cfend = Dot11ControlFrameCFEnd(d.get_body_as_string())
        d.contains(self.cfend)

    def test_01_HeaderTailSize(self):
        if False:
            print('Hello World!')
        'Test Header and Tail Size field'
        self.assertEqual(self.cfend.get_header_size(), 14)
        self.assertEqual(self.cfend.get_tail_size(), 0)

    def test_02_Duration(self):
        if False:
            print('Hello World!')
        'Test Duration field'
        self.assertEqual(self.cfend.get_duration(), 0)
        self.cfend.set_duration(4660)
        self.assertEqual(self.cfend.get_duration(), 4660)

    def test_03_RA(self):
        if False:
            while True:
                i = 10
        'Test RA field'
        ra = self.cfend.get_ra()
        self.assertEqual(ra.tolist(), [255, 255, 255, 255, 255, 255])
        ra[0] = 18
        ra[5] = 52
        self.cfend.set_ra(ra)
        self.assertEqual(self.cfend.get_ra().tolist(), [18, 255, 255, 255, 255, 52])

    def test_04_BSSID(self):
        if False:
            while True:
                i = 10
        'Test BSS ID field'
        bssid = self.cfend.get_bssid()
        self.assertEqual(bssid.tolist(), [0, 25, 224, 152, 4, 212])
        bssid[0] = 18
        bssid[5] = 52
        self.cfend.set_bssid(bssid)
        self.assertEqual(self.cfend.get_bssid().tolist(), [18, 25, 224, 152, 4, 52])
if __name__ == '__main__':
    unittest.main(verbosity=1)