import unittest
from impacket.dot11 import Dot11, Dot11Types, Dot11ControlFrameACK

class TestDot11FrameControlACK(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.frame_orig = b'\xd4\x00\x00\x00\x00\x08T\xac/\x85\xb7\x7f\xc3\x9e'
        d = Dot11(self.frame_orig)
        type = d.get_type()
        self.assertEqual(type, Dot11Types.DOT11_TYPE_CONTROL)
        subtype = d.get_subtype()
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_CONTROL_ACKNOWLEDGMENT)
        typesubtype = d.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_CONTROL_SUBTYPE_ACKNOWLEDGMENT)
        self.ack = Dot11ControlFrameACK(d.get_body_as_string())
        d.contains(self.ack)

    def test_01_HeaderTailSize(self):
        if False:
            print('Hello World!')
        'Test Header and Tail Size field'
        self.assertEqual(self.ack.get_header_size(), 8)
        self.assertEqual(self.ack.get_tail_size(), 0)

    def test_02_Duration(self):
        if False:
            while True:
                i = 10
        'Test Duration field'
        self.assertEqual(self.ack.get_duration(), 0)
        self.ack.set_duration(4660)
        self.assertEqual(self.ack.get_duration(), 4660)

    def test_03_RA(self):
        if False:
            while True:
                i = 10
        'Test RA field'
        ra = self.ack.get_ra()
        self.assertEqual(ra.tolist(), [0, 8, 84, 172, 47, 133])
        ra[0] = 18
        ra[5] = 52
        self.ack.set_ra(ra)
        self.assertEqual(self.ack.get_ra().tolist(), [18, 8, 84, 172, 47, 52])
if __name__ == '__main__':
    unittest.main(verbosity=1)