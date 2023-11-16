import unittest
from impacket.dot11 import Dot11, Dot11Types, Dot11DataFrame, Dot11WPA2, Dot11WPA2Data

class TestDot11WPA2Data(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.frame_orig = b'\x08I$\x00\x00!)h3]\x00\x15\xaf\xe4\xf1\x0f\x00!)h3[\xe01\x1b\x13\x00 \x00\x00\x00\x00\x84}j0\x8c`~;"\xdc\x16\xc1K(\xd3&v\x9d.Y\x961>\x01oa\xa2Y\xc8\xdc\xd3\xc4\xad|\xcc2\xa8\x9f\xf6\x03\x02\xe1\xac\x1d\x1e\x02\x8a\xcd[\x94 -\xfcn7@.F\x17\x19\x0c\xc04\x07\xae\xe7w\xaf\xf9\x9fAS'
        d = Dot11(self.frame_orig)
        self.assertEqual(d.get_type(), Dot11Types.DOT11_TYPE_DATA)
        self.assertEqual(d.get_subtype(), Dot11Types.DOT11_SUBTYPE_DATA)
        self.assertEqual(d.get_type_n_subtype(), Dot11Types.DOT11_TYPE_DATA_SUBTYPE_DATA)
        data = Dot11DataFrame(d.get_body_as_string())
        d.contains(data)
        self.wpa2_header = Dot11WPA2(data.body_string)
        data.contains(self.wpa2_header)
        self.wpa2_data = Dot11WPA2Data(self.wpa2_header.body_string)
        self.wpa2_header.contains(self.wpa2_data)

    def test_01_is_WPA2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test WPA2Header is_WPA2 method'
        self.assertEqual(self.wpa2_header.is_WPA2(), True)

    def test_03_extIV(self):
        if False:
            for i in range(10):
                print('nop')
        'Test WPA2Header extIV getter and setter methods'
        self.assertEqual(self.wpa2_header.get_extIV(), 1)
        self.wpa2_header.set_extIV(0)
        self.assertEqual(self.wpa2_header.get_extIV(), 0)

    def test_04_keyid(self):
        if False:
            print('Hello World!')
        'Test WPA2Header keyID getter and setter methods'
        self.assertEqual(self.wpa2_header.get_keyid(), 0)
        self.wpa2_header.set_keyid(3)
        self.assertEqual(self.wpa2_header.get_keyid(), 3)

    def test_06_PNs(self):
        if False:
            print('Hello World!')
        'Test WPA2Data PN0 to PN5 getter and setter methods'
        self.assertEqual(self.wpa2_header.get_PN0(), 27)
        self.wpa2_header.set_PN0(171)
        self.assertEqual(self.wpa2_header.get_PN0(), 171)
        self.assertEqual(self.wpa2_header.get_PN1(), 19)
        self.wpa2_header.set_PN1(171)
        self.assertEqual(self.wpa2_header.get_PN1(), 171)
        self.assertEqual(self.wpa2_header.get_PN2(), 0)
        self.wpa2_header.set_PN2(171)
        self.assertEqual(self.wpa2_header.get_PN2(), 171)
        self.assertEqual(self.wpa2_header.get_PN3(), 0)
        self.wpa2_header.set_PN3(171)
        self.assertEqual(self.wpa2_header.get_PN3(), 171)
        self.assertEqual(self.wpa2_header.get_PN4(), 0)
        self.wpa2_header.set_PN4(171)
        self.assertEqual(self.wpa2_header.get_PN4(), 171)
        self.assertEqual(self.wpa2_header.get_PN5(), 0)
        self.wpa2_header.set_PN5(171)
        self.assertEqual(self.wpa2_header.get_PN5(), 171)

    def test_07_data(self):
        if False:
            return 10
        'Test WPA2Data body'
        data = b'\x84}j0\x8c`~;"\xdc\x16\xc1K(\xd3&v\x9d.Y\x961>\x01oa\xa2Y\xc8\xdc\xd3\xc4\xad|\xcc2\xa8\x9f\xf6\x03\x02\xe1\xac\x1d\x1e\x02\x8a\xcd[\x94 -\xfcn7@.F\x17\x19'
        self.assertEqual(self.wpa2_data.body_string, data)

    def test_08_mic(self):
        if False:
            print('Hello World!')
        'Test WPA2Data MIC field'
        mic = b'\x0c\xc04\x07\xae\xe7w\xaf'
        self.assertEqual(self.wpa2_data.get_MIC(), mic)
        mic = b'\x01\x02\x03\x04\xff\xfe\xfd\xfc'
        self.wpa2_data.set_MIC(mic)
        self.assertEqual(self.wpa2_data.get_MIC(), mic)
if __name__ == '__main__':
    unittest.main(verbosity=1)