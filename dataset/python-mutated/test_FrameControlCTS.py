import unittest
from impacket.dot11 import Dot11, Dot11Types, Dot11ControlFrameCTS

class TestDot11FrameControlCTS(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.frame_orig = b'\xc4\x00;\x12\x00\x19\xe0\x98\x04\xd4+\x8ae\x17'
        d = Dot11(self.frame_orig)
        type = d.get_type()
        self.assertEqual(type, Dot11Types.DOT11_TYPE_CONTROL)
        subtype = d.get_subtype()
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_CONTROL_CLEAR_TO_SEND)
        typesubtype = d.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_CONTROL_SUBTYPE_CLEAR_TO_SEND)
        self.cts = Dot11ControlFrameCTS(d.get_body_as_string())
        d.contains(self.cts)

    def test_01_HeaderTailSize(self):
        if False:
            i = 10
            return i + 15
        'Test Header and Tail Size field'
        self.assertEqual(self.cts.get_header_size(), 8)
        self.assertEqual(self.cts.get_tail_size(), 0)

    def test_02_Duration(self):
        if False:
            i = 10
            return i + 15
        'Test Duration field'
        self.assertEqual(self.cts.get_duration(), 4667)
        self.cts.set_duration(4660)
        self.assertEqual(self.cts.get_duration(), 4660)

    def test_03_RA(self):
        if False:
            for i in range(10):
                print('nop')
        'Test RA field'
        ra = self.cts.get_ra()
        self.assertEqual(ra.tolist(), [0, 25, 224, 152, 4, 212])
        ra[0] = 18
        ra[5] = 52
        self.cts.set_ra(ra)
        self.assertEqual(self.cts.get_ra().tolist(), [18, 25, 224, 152, 4, 52])
if __name__ == '__main__':
    unittest.main(verbosity=1)