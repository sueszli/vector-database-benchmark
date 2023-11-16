import unittest
from six import PY2
import impacket.dot11
import impacket.ImpactPacket
from impacket.ImpactDecoder import RadioTapDecoder

class TestRadioTapDecoder(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.RadioTapData = b'\x00\x00 \x00g\x08\x04\x000\x03\x1a%\x00\x00\x00\x00"\x0c\xd9\xa0\x02\x00\x00\x00@\x01\x00\x00<\x14$\x11\x08\x02\x00\x00\xff\xff\xff\xff\xff\xff\x06\x03\x7f\x07\xa0\x16\x00\x19\xe3\xd3SR\x90\x7f\xaa\xaa\x03\x00\x00\x00\x08\x06\x00\x01\x08\x00\x06\x04\x00\x01\x00\x19\xe3\xd3SR\xa9\xfe\xf7\x00\x00\x00\x00\x00\x00\x00C\x08\x0e6'
        self.radiotap_decoder = RadioTapDecoder()
        self.in0 = self.radiotap_decoder.decode(self.RadioTapData)
        self.in1 = self.in0.child()
        self.in2 = self.in1.child()
        self.in3 = self.in2.child()
        self.in4 = self.in3.child()
        self.in5 = self.in4.child()
        self.in6 = self.in5.child()

    def test_00(self):
        if False:
            i = 10
            return i + 15
        'Test RadioTap decoder'
        if PY2:
            self.assertEqual(str(self.in0.__class__), 'impacket.dot11.RadioTap')
        else:
            self.assertEqual(str(self.in0.__class__), "<class 'impacket.dot11.RadioTap'>")

    def test_01(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Dot11 decoder'
        if PY2:
            self.assertEqual(str(self.in1.__class__), 'impacket.dot11.Dot11')
        else:
            self.assertEqual(str(self.in1.__class__), "<class 'impacket.dot11.Dot11'>")

    def test_02(self):
        if False:
            i = 10
            return i + 15
        'Test Dot11DataFrame decoder'
        if PY2:
            self.assertEqual(str(self.in2.__class__), 'impacket.dot11.Dot11DataFrame')
        else:
            self.assertEqual(str(self.in2.__class__), "<class 'impacket.dot11.Dot11DataFrame'>")

    def test_03(self):
        if False:
            return 10
        'Test LLC decoder'
        if PY2:
            self.assertEqual(str(self.in3.__class__), 'impacket.dot11.LLC')
        else:
            self.assertEqual(str(self.in3.__class__), "<class 'impacket.dot11.LLC'>")

    def test_04(self):
        if False:
            for i in range(10):
                print('nop')
        'Test SNAP decoder'
        if PY2:
            self.assertEqual(str(self.in4.__class__), 'impacket.dot11.SNAP')
        else:
            self.assertEqual(str(self.in4.__class__), "<class 'impacket.dot11.SNAP'>")

    def test_06(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Protocol Finder'
        p = self.radiotap_decoder.get_protocol(impacket.dot11.RadioTap)
        if PY2:
            self.assertEqual(str(p.__class__), 'impacket.dot11.RadioTap')
        else:
            self.assertEqual(str(p.__class__), "<class 'impacket.dot11.RadioTap'>")
        p = self.radiotap_decoder.get_protocol(impacket.dot11.Dot11)
        if PY2:
            self.assertEqual(str(p.__class__), 'impacket.dot11.Dot11')
        else:
            self.assertEqual(str(p.__class__), "<class 'impacket.dot11.Dot11'>")
        p = self.radiotap_decoder.get_protocol(impacket.dot11.Dot11DataFrame)
        if PY2:
            self.assertEqual(str(p.__class__), 'impacket.dot11.Dot11DataFrame')
        else:
            self.assertEqual(str(p.__class__), "<class 'impacket.dot11.Dot11DataFrame'>")
        p = self.radiotap_decoder.get_protocol(impacket.dot11.LLC)
        if PY2:
            self.assertEqual(str(p.__class__), 'impacket.dot11.LLC')
        else:
            self.assertEqual(str(p.__class__), "<class 'impacket.dot11.LLC'>")
        p = self.radiotap_decoder.get_protocol(impacket.dot11.SNAP)
        if PY2:
            self.assertEqual(str(p.__class__), 'impacket.dot11.SNAP')
        else:
            self.assertEqual(str(p.__class__), "<class 'impacket.dot11.SNAP'>")
        p = self.radiotap_decoder.get_protocol(impacket.dot11.Dot11WPA)
        self.assertEqual(p, None)
if __name__ == '__main__':
    unittest.main(verbosity=1)