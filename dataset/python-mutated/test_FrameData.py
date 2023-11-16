import unittest
from impacket.dot11 import Dot11, Dot11Types, Dot11DataFrame

class TestDot11DataFrames(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.frame_orig = b'\x08\x010\x00\x00\x08T\xac/\x85\x00#M\t\x86\xfe\x00\x08T\xac/\x85@D\xaa\xaa\x03\x00\x00\x00\x08\x00E\x00\x00(r7@\x00\x80\x06l"\xc0\xa8\x01\x02\xc3z\x97Q\xd7\xa0\x00P\xa5\xa5\xb1\xe0\x12\x1c\xa9\xe1P\x10NuYt\x00\x00\xed\x13"\x91'
        d = Dot11(self.frame_orig)
        type = d.get_type()
        self.assertEqual(type, Dot11Types.DOT11_TYPE_DATA)
        subtype = d.get_subtype()
        self.assertEqual(subtype, Dot11Types.DOT11_SUBTYPE_DATA)
        typesubtype = d.get_type_n_subtype()
        self.assertEqual(typesubtype, Dot11Types.DOT11_TYPE_DATA_SUBTYPE_DATA)
        self.data = Dot11DataFrame(d.get_body_as_string())
        d.contains(self.data)

    def test_01_HeaderSize(self):
        if False:
            return 10
        'Test Header and Tail Size field'
        self.assertEqual(self.data.get_header_size(), 22)
        self.assertEqual(self.data.get_tail_size(), 0)

    def test_02_Duration(self):
        if False:
            return 10
        'Test Duration field'
        self.assertEqual(self.data.get_duration(), 48)
        self.data.set_duration(4660)
        self.assertEqual(self.data.get_duration(), 4660)

    def test_03_Address_1(self):
        if False:
            i = 10
            return i + 15
        'Test Address 1 field'
        addr = self.data.get_address1()
        self.assertEqual(addr.tolist(), [0, 8, 84, 172, 47, 133])
        addr[0] = 18
        addr[5] = 52
        self.data.set_address1(addr)
        self.assertEqual(self.data.get_address1().tolist(), [18, 8, 84, 172, 47, 52])

    def test_04_Address_2(self):
        if False:
            return 10
        'Test Address 2 field'
        addr = self.data.get_address2()
        self.assertEqual(addr.tolist(), [0, 35, 77, 9, 134, 254])
        addr[0] = 18
        addr[5] = 52
        self.data.set_address2(addr)
        self.assertEqual(self.data.get_address2().tolist(), [18, 35, 77, 9, 134, 52])

    def test_05_Address_3(self):
        if False:
            print('Hello World!')
        'Test Address 3 field'
        addr = self.data.get_address3()
        self.assertEqual(addr.tolist(), [0, 8, 84, 172, 47, 133])
        addr[0] = 18
        addr[5] = 52
        self.data.set_address3(addr)
        self.assertEqual(self.data.get_address3().tolist(), [18, 8, 84, 172, 47, 52])

    def test_06_sequence_control(self):
        if False:
            print('Hello World!')
        'Test Sequence control field'
        self.assertEqual(self.data.get_sequence_control(), 17472)
        self.data.set_sequence_control(4660)
        self.assertEqual(self.data.get_sequence_control(), 4660)

    def test_07_fragment_number(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Fragment number field'
        self.assertEqual(self.data.get_fragment_number(), 0)
        self.data.set_fragment_number(241)
        self.assertEqual(self.data.get_fragment_number(), 1)

    def test_08_sequence_number(self):
        if False:
            return 10
        'Test Sequence number field'
        self.assertEqual(self.data.get_sequence_number(), 1092)
        self.data.set_sequence_number(62004)
        self.assertEqual(self.data.get_sequence_number(), 564)

    def test_09_frame_data(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Frame Data field'
        frame_body = b'\xaa\xaa\x03\x00\x00\x00\x08\x00E\x00\x00(r7@\x00\x80\x06l"\xc0\xa8\x01\x02\xc3z\x97Q\xd7\xa0\x00P\xa5\xa5\xb1\xe0\x12\x1c\xa9\xe1P\x10NuYt\x00\x00'
        self.assertEqual(self.data.get_frame_body(), frame_body)
if __name__ == '__main__':
    unittest.main(verbosity=1)