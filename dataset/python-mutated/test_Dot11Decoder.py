import unittest
from six import PY2
from impacket.ImpactDecoder import Dot11Decoder

class TestDot11Decoder(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.WEPKey = None
        self.WEPData = b'\x08A:\x01\x00\x17?DO\x96\x00\x13\xceg\x0es\x00\x17?DO\x96\xb0\x04\xeb\xcd\x8b\x00n\xdf\x9369Z9fk\x96\xd1z\xe1\xae\xb6\x11"\xfd\xf0\xd4\rj\xb8\xb1\xe6.\x1f%}d\x1a\x07\xd5\x86\xd2\x194\xb5\xf7\x8ab3Yn\x89\x01sP\x12\xbb\xde\x17\xdd\xb5\xd45'
        dot11_decoder = Dot11Decoder()
        self.in0 = dot11_decoder.decode(self.WEPData)
        self.in1 = self.in0.child()
        self.in2 = self.in1.child()
        self.in3 = self.in2.child()
        if self.WEPKey:
            self.in4 = self.in3.child()
            self.in5 = self.in4.child()

    def test_01_Dot11Decoder(self):
        if False:
            while True:
                i = 10
        'Test Dot11 decoder'
        if PY2:
            self.assertEqual(str(self.in0.__class__), 'impacket.dot11.Dot11')
        else:
            self.assertEqual(str(self.in0.__class__), "<class 'impacket.dot11.Dot11'>")

    def test_02_Dot11DataFrameDecoder(self):
        if False:
            while True:
                i = 10
        'Test Dot11DataFrame decoder'
        if PY2:
            self.assertEqual(str(self.in1.__class__), 'impacket.dot11.Dot11DataFrame')
        else:
            self.assertEqual(str(self.in1.__class__), "<class 'impacket.dot11.Dot11DataFrame'>")

    def test_03_Dot11WEP(self):
        if False:
            print('Hello World!')
        'Test Dot11WEP decoder'
        if PY2:
            self.assertEqual(str(self.in2.__class__), 'impacket.dot11.Dot11WEP')
        else:
            self.assertEqual(str(self.in2.__class__), "<class 'impacket.dot11.Dot11WEP'>")

    def test_04_Dot11WEPData(self):
        if False:
            i = 10
            return i + 15
        'Test Dot11WEPData decoder'
        if not self.WEPKey:
            return
        self.assertEqual(str(self.in3.__class__), 'impacket.dot11.Dot11WEPData')
        wepdata = b'n\xdf\x9369Z9fk\x96\xd1z\xe1\xae\xb6\x11"\xfd\xf0\xd4\rj\xb8\xb1\xe6.\x1f%}d\x1a\x07\xd5\x86\xd2\x194\xb5\xf7\x8ab3Yn\x89\x01sP\x12\xbb\xde\x17'
        self.assertEqual(self.in3.get_packet(), wepdata)

    def test_05_LLC(self):
        if False:
            i = 10
            return i + 15
        'Test LLC decoder'
        if self.WEPKey:
            self.assertEqual(str(self.in4.__class__), 'impacket.dot11.LLC')

    def test_06_Data(self):
        if False:
            i = 10
            return i + 15
        'Test LLC Data decoder'
        if self.WEPKey:
            dataclass = self.in4.__class__
        else:
            dataclass = self.in3.__class__
        self.assertGreater(str(dataclass).find('ImpactPacket.Data'), 0)
if __name__ == '__main__':
    unittest.main(verbosity=1)