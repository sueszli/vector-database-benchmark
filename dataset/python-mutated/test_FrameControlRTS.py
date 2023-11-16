import unittest
from impacket.dot11 import Dot11, Dot11Types, Dot11ControlFrameRTS

class TestDot11FrameControlRTS(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.frame_orig = b'\xb4\x00\x81\x01\x00\x08T\xac/\x85\x00#M\t\x86\xfe\x99uCs'
        d = Dot11(self.frame_orig)
        type = d.get_type()
        self.assertEqual(type, Dot11Types.DOT11_TYPE_CONTROL)
        subtype = d.get_subtype()
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_CONTROL_REQUEST_TO_SEND)
        typesubtype = d.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_CONTROL_SUBTYPE_REQUEST_TO_SEND)
        self.rts = Dot11ControlFrameRTS(d.get_body_as_string())
        d.contains(self.rts)

    def test_01_HeaderTailSize(self):
        if False:
            while True:
                i = 10
        'Test Header and Tail Size field'
        self.assertEqual(self.rts.get_header_size(), 14)
        self.assertEqual(self.rts.get_tail_size(), 0)

    def test_02_Duration(self):
        if False:
            return 10
        'Test Duration field'
        self.assertEqual(self.rts.get_duration(), 385)
        self.rts.set_duration(4660)
        self.assertEqual(self.rts.get_duration(), 4660)

    def test_03_RA(self):
        if False:
            return 10
        'Test RA field'
        ra = self.rts.get_ra()
        self.assertEqual(ra.tolist(), [0, 8, 84, 172, 47, 133])
        ra[0] = 18
        ra[5] = 52
        self.rts.set_ra(ra)
        self.assertEqual(self.rts.get_ra().tolist(), [18, 8, 84, 172, 47, 52])

    def test_04_TA(self):
        if False:
            return 10
        'Test TA field'
        ta = self.rts.get_ta()
        self.assertEqual(ta.tolist(), [0, 35, 77, 9, 134, 254])
        ta[0] = 18
        ta[5] = 52
        self.rts.set_ta(ta)
        self.assertEqual(self.rts.get_ta().tolist(), [18, 35, 77, 9, 134, 52])
if __name__ == '__main__':
    unittest.main(verbosity=1)