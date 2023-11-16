import unittest
from impacket.dot11 import Dot11, Dot11Types, Dot11ControlFramePSPoll

class TestDot11FrameControlPSPoll(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.frame_orig = b'\xa6s\xf1\xafH\x06\xee#+\xc9\xfe\xbe\xe5\x05L\n\x04\xa0\x00\x0f'
        d = Dot11(self.frame_orig)
        type = d.get_type()
        self.assertEqual(type, Dot11Types.DOT11_TYPE_CONTROL)
        subtype = d.get_subtype()
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_CONTROL_POWERSAVE_POLL)
        typesubtype = d.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_CONTROL_SUBTYPE_POWERSAVE_POLL)
        self.pspoll = Dot11ControlFramePSPoll(d.get_body_as_string())
        d.contains(self.pspoll)

    def test_01_HeaderTailSize(self):
        if False:
            print('Hello World!')
        'Test Header and Tail Size field'
        self.assertEqual(self.pspoll.get_header_size(), 14)
        self.assertEqual(self.pspoll.get_tail_size(), 0)

    def test_02_AID(self):
        if False:
            while True:
                i = 10
        'Test AID field'
        self.assertEqual(self.pspoll.get_aid(), 45041)
        self.pspoll.set_aid(4660)
        self.assertEqual(self.pspoll.get_aid(), 4660)

    def test_03_BSSID(self):
        if False:
            for i in range(10):
                print('nop')
        'Test BSS ID field'
        bssid = self.pspoll.get_bssid()
        self.assertEqual(bssid.tolist(), [72, 6, 238, 35, 43, 201])
        bssid[0] = 18
        bssid[5] = 52
        self.pspoll.set_bssid(bssid)
        self.assertEqual(self.pspoll.get_bssid().tolist(), [18, 6, 238, 35, 43, 52])

    def test_04_TA(self):
        if False:
            i = 10
            return i + 15
        'Test TA field'
        ta = self.pspoll.get_ta()
        self.assertEqual(ta.tolist(), [254, 190, 229, 5, 76, 10])
        ta[0] = 18
        ta[5] = 52
        self.pspoll.set_ta(ta)
        self.assertEqual(self.pspoll.get_ta().tolist(), [18, 190, 229, 5, 76, 52])
if __name__ == '__main__':
    unittest.main(verbosity=1)