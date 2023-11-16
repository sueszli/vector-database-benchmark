import unittest
from impacket.dot11 import Dot11, Dot11Types, Dot11DataFrame, Dot11WPA, Dot11WPAData

class TestDot11WPAData(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.frame_orig = b'\x08B\x00\x00\xff\xff\xff\xff\xff\xff\x00!)h3]\x00\x1b\xfc\x1e\xca@\xa0\x16\x02"Z`\x00\x00\x00\x00\xa2\x0ew6\xea\x90v\x0fz\x9fnlx\xb9\xe0>\xb4\x9d\t\xca\xde\xef\x95X(\x97\x17FSCA+*\xc6\xbe\xe4Y`\xf0\x17\x1d \x8c\xca<&\r]k\x10\x81\xbc\xc6\xba\x90\xa5w\x0e\x83\xd0\xd0\xb9\xdd\xbf\x80\xbfe\x17\xee\xc0:R24u\xac\x0c\xc2\xbb%(\x8fj\xe6\x96zSJw\xcc+\xe5\x9a\x9as\xc2\x08LB\x15\xe9&\xa0\xcep\x0eP\x9b-\xa2n\xcb\x92T\xc0m\xbc\x13\xfeM\xd8k\x8cv\x98\x9aqMQ\xb1\xf5O\xe2C\x1b\xfao\\\x98j:dOP\xc4\t}\x10?\xa2d\xd9\xadnD\xe3\x84=+w\x11\xd8\x04\x9d\x9d\xd425\xe8=\xeb\xd5\x9a\xde\xf3\xb5Ag\x94\xf9\xb1\xe0z\xea3\xb2\x00\xefj.l;\xea#I#\xc2\xca$S\xea\xc0~\x8c\xcfs\xcb-\x0c\x8e\xdb{\x9e\nf\x81\x90'
        d = Dot11(self.frame_orig)
        self.assertEqual(d.get_type(), Dot11Types.DOT11_TYPE_DATA)
        self.assertEqual(d.get_subtype(), Dot11Types.DOT11_SUBTYPE_DATA)
        self.assertEqual(d.get_type_n_subtype(), Dot11Types.DOT11_TYPE_DATA_SUBTYPE_DATA)
        data = Dot11DataFrame(d.get_body_as_string())
        d.contains(data)
        self.wpa_header = Dot11WPA(data.body_string)
        data.contains(self.wpa_header)
        self.wpa_data = Dot11WPAData(self.wpa_header.body_string)
        self.wpa_header.contains(self.wpa_data)

    def test_01_is_WPA(self):
        if False:
            i = 10
            return i + 15
        'Test WPAHeader is_WPA method'
        self.assertEqual(self.wpa_header.is_WPA(), True)

    def test_03_extIV(self):
        if False:
            for i in range(10):
                print('nop')
        'Test WPAHeader extIV getter and setter methods'
        self.assertEqual(self.wpa_header.get_extIV(), 1)
        self.wpa_header.set_extIV(0)
        self.assertEqual(self.wpa_header.get_extIV(), 0)

    def test_04_keyid(self):
        if False:
            for i in range(10):
                print('nop')
        'Test WPAHeader keyID getter and setter methods'
        self.assertEqual(self.wpa_header.get_keyid(), 1)
        self.wpa_header.set_keyid(3)
        self.assertEqual(self.wpa_header.get_keyid(), 3)

    def test_06_WEPSeed(self):
        if False:
            for i in range(10):
                print('nop')
        'Test WPAData WEPSeed getter and setter methods'
        self.assertEqual(self.wpa_header.get_WEPSeed(), 34)
        self.wpa_header.set_WEPSeed(171)
        self.assertEqual(self.wpa_header.get_WEPSeed(), 171)

    def test_07_TSCs(self):
        if False:
            for i in range(10):
                print('nop')
        'Test WPAData TSC0 to TSC5 getter and setter methods'
        self.assertEqual(self.wpa_header.get_TSC0(), 90)
        self.wpa_header.set_TSC0(171)
        self.assertEqual(self.wpa_header.get_TSC0(), 171)
        self.assertEqual(self.wpa_header.get_TSC1(), 2)
        self.wpa_header.set_TSC1(171)
        self.assertEqual(self.wpa_header.get_TSC1(), 171)
        self.assertEqual(self.wpa_header.get_TSC2(), 0)
        self.wpa_header.set_TSC2(171)
        self.assertEqual(self.wpa_header.get_TSC2(), 171)
        self.assertEqual(self.wpa_header.get_TSC3(), 0)
        self.wpa_header.set_TSC3(171)
        self.assertEqual(self.wpa_header.get_TSC3(), 171)
        self.assertEqual(self.wpa_header.get_TSC4(), 0)
        self.wpa_header.set_TSC4(171)
        self.assertEqual(self.wpa_header.get_TSC4(), 171)
        self.assertEqual(self.wpa_header.get_TSC5(), 0)
        self.wpa_header.set_TSC5(171)
        self.assertEqual(self.wpa_header.get_TSC5(), 171)

    def test_08_data(self):
        if False:
            print('Hello World!')
        'Test WPAData body'
        data = b'\xa2\x0ew6\xea\x90v\x0fz\x9fnlx\xb9\xe0>\xb4\x9d\t\xca\xde\xef\x95X(\x97\x17FSCA+*\xc6\xbe\xe4Y`\xf0\x17\x1d \x8c\xca<&\r]k\x10\x81\xbc\xc6\xba\x90\xa5w\x0e\x83\xd0\xd0\xb9\xdd\xbf\x80\xbfe\x17\xee\xc0:R24u\xac\x0c\xc2\xbb%(\x8fj\xe6\x96zSJw\xcc+\xe5\x9a\x9as\xc2\x08LB\x15\xe9&\xa0\xcep\x0eP\x9b-\xa2n\xcb\x92T\xc0m\xbc\x13\xfeM\xd8k\x8cv\x98\x9aqMQ\xb1\xf5O\xe2C\x1b\xfao\\\x98j:dOP\xc4\t}\x10?\xa2d\xd9\xadnD\xe3\x84=+w\x11\xd8\x04\x9d\x9d\xd425\xe8=\xeb\xd5\x9a\xde\xf3\xb5Ag\x94\xf9\xb1\xe0z\xea3\xb2\x00\xefj.l;\xea#I#\xc2\xca$S\xea'
        self.assertEqual(self.wpa_data.body_string, data)

    def test_09_mic(self):
        if False:
            for i in range(10):
                print('nop')
        'Test WPAData MIC field'
        mic = b'\xc0~\x8c\xcfs\xcb-\x0c'
        self.assertEqual(self.wpa_data.get_MIC(), mic)
        mic = b'\x01\x02\x03\x04\xff\xfe\xfd\xfc'
        self.wpa_data.set_MIC(mic)
        self.assertEqual(self.wpa_data.get_MIC(), mic)
        self.assertEqual(self.wpa_data.get_icv(), 2396748702)

    def test_10_get_icv(self):
        if False:
            for i in range(10):
                print('nop')
        'Test WPAData ICV field'
        self.assertEqual(self.wpa_data.get_icv(), 2396748702)
if __name__ == '__main__':
    unittest.main(verbosity=1)