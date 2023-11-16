import unittest
from impacket.dot11 import Dot11, Dot11Types

class TestDot11Common(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        a = b'\xd4\x00\x00\x00\x00\x08T\xac/\x85\xb7\x7f\xc3\x9e'
        self.dot11fc = Dot11(a)

    def test_01_HeaderSize(self):
        if False:
            while True:
                i = 10
        'Test Header Size field'
        self.assertEqual(self.dot11fc.get_header_size(), 2)

    def test_01_TailSize(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Tail Size field'
        self.assertEqual(self.dot11fc.get_tail_size(), 4)

    def test_02_Version(self):
        if False:
            return 10
        'Test Version field'
        self.assertEqual(self.dot11fc.get_version(), 0)
        self.dot11fc.set_version(3)
        self.assertEqual(self.dot11fc.get_version(), 3)

    def test_03_Type(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Type field'
        self.assertEqual(self.dot11fc.get_type(), 1)
        self.dot11fc.set_type(3)
        self.assertEqual(self.dot11fc.get_type(), 3)

    def test_04_SubType(self):
        if False:
            print('Hello World!')
        'Test Subtype field'
        self.assertEqual(self.dot11fc.get_subtype(), 13)
        self.dot11fc.set_subtype(5)
        self.assertEqual(self.dot11fc.get_subtype(), 5)

    def test_05_ToDS(self):
        if False:
            for i in range(10):
                print('nop')
        'Test toDS field'
        self.assertEqual(self.dot11fc.get_toDS(), 0)
        self.dot11fc.set_toDS(1)
        self.assertEqual(self.dot11fc.get_toDS(), 1)

    def test_06_FromDS(self):
        if False:
            i = 10
            return i + 15
        'Test fromDS field'
        self.assertEqual(self.dot11fc.get_fromDS(), 0)
        self.dot11fc.set_fromDS(1)
        self.assertEqual(self.dot11fc.get_fromDS(), 1)

    def test_07_MoreFrag(self):
        if False:
            print('Hello World!')
        'Test More Frag field'
        self.assertEqual(self.dot11fc.get_moreFrag(), 0)
        self.dot11fc.set_moreFrag(1)
        self.assertEqual(self.dot11fc.get_moreFrag(), 1)

    def test_08_Retry(self):
        if False:
            print('Hello World!')
        'Test Retry field'
        self.assertEqual(self.dot11fc.get_retry(), 0)
        self.dot11fc.set_retry(1)
        self.assertEqual(self.dot11fc.get_retry(), 1)

    def test_09_PowerManagement(self):
        if False:
            while True:
                i = 10
        'Test Power Management field'
        self.assertEqual(self.dot11fc.get_powerManagement(), 0)
        self.dot11fc.set_powerManagement(1)
        self.assertEqual(self.dot11fc.get_powerManagement(), 1)

    def test_10_MoreData(self):
        if False:
            for i in range(10):
                print('nop')
        'Test More Data field'
        self.assertEqual(self.dot11fc.get_moreData(), 0)
        self.dot11fc.set_moreData(1)
        self.assertEqual(self.dot11fc.get_moreData(), 1)

    def test_12_Order(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Order field'
        self.assertEqual(self.dot11fc.get_order(), 0)
        self.dot11fc.set_order(1)
        self.assertEqual(self.dot11fc.get_order(), 1)

    def test_13_latest(self):
        if False:
            print('Hello World!')
        'Test complete frame hexs'
        self.dot11fc.set_type_n_subtype(Dot11Types.DOT11_TYPE_CONTROL_SUBTYPE_POWERSAVE_POLL)
        self.dot11fc.set_order(1)
        self.dot11fc.set_moreData(1)
        self.dot11fc.set_retry(1)
        self.dot11fc.set_fromDS(1)
        frame = self.dot11fc.get_packet()
        self.assertEqual(frame, b'\xa4\xaa\x00\x00\x00\x08T\xac/\x85\xb7\x7f\xc3\x9e')
if __name__ == '__main__':
    unittest.main(verbosity=1)