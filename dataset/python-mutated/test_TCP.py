import unittest
from impacket.ImpactPacket import TCP

class TestTCP(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.frame = b'\xec\xef\x00P\xa8\xbd\xeaL\x00\x00\x00\x00\xa0\x02\x16\xd0d\xcb\x00\x00\x02\x04\x05\xb4\x04\x02\x08\n\x00\xdc\xd6\x12\x00\x00\x00\x00\x01\x03\x03\x06'
        self.tcp = TCP(self.frame)

    def test_01(self):
        if False:
            print('Hello World!')
        'Test TCP get_packet'
        self.assertEqual(self.tcp.get_packet(), self.frame)

    def test_02(self):
        if False:
            print('Hello World!')
        'Test TCP getters'
        self.assertEqual(self.tcp.get_th_sport(), 60655)
        self.assertEqual(self.tcp.get_th_dport(), 80)
        self.assertEqual(self.tcp.get_th_off() * 4, 40)
        self.assertEqual(self.tcp.get_th_flags(), 2)
        self.assertEqual(self.tcp.get_th_win(), 5840)
        self.assertEqual(self.tcp.get_th_sum(), 25803)
        self.assertEqual(self.tcp.get_SYN(), 1)
        self.assertEqual(self.tcp.get_RST(), 0)

    def test_03(self):
        if False:
            i = 10
            return i + 15
        'Test TCP port setters'
        self.tcp.set_th_sport(54321)
        self.assertEqual(self.tcp.get_th_sport(), 54321)
        self.tcp.set_th_dport(81)
        self.assertEqual(self.tcp.get_th_dport(), 81)

    def test_04(self):
        if False:
            i = 10
            return i + 15
        'Test TCP offset setters'
        flags = int('10101010', 2)
        self.tcp.set_th_flags(flags)
        self.assertEqual(self.tcp.get_th_flags(), flags)
        self.tcp.set_th_off(4)
        self.assertEqual(self.tcp.get_th_off(), 4)
        self.assertEqual(self.tcp.get_th_flags(), flags)

    def test_05(self):
        if False:
            for i in range(10):
                print('nop')
        'Test TCP win setters'
        self.tcp.set_th_win(12345)
        self.assertEqual(self.tcp.get_th_win(), 12345)

    def test_06(self):
        if False:
            for i in range(10):
                print('nop')
        'Test TCP checksum setters'
        self.tcp.set_th_sum(65278)
        self.assertEqual(self.tcp.get_th_sum(), 65278)

    def test_07(self):
        if False:
            for i in range(10):
                print('nop')
        'Test TCP flags setters'
        self.tcp.set_th_flags(3)
        self.assertEqual(self.tcp.get_th_flags(), 3)
        self.tcp.set_ACK()
        self.assertEqual(self.tcp.get_ACK(), 1)
        self.assertEqual(self.tcp.get_SYN(), 1)
        self.assertEqual(self.tcp.get_FIN(), 1)
        self.assertEqual(self.tcp.get_RST(), 0)
        self.assertEqual(self.tcp.get_th_flags(), 19)

    def test_08(self):
        if False:
            return 10
        'Test TCP reset_flags'
        self.tcp.set_th_flags(19)
        self.assertEqual(self.tcp.get_th_flags(), 19)
        self.assertEqual(self.tcp.get_ACK(), 1)
        self.assertEqual(self.tcp.get_SYN(), 1)
        self.assertEqual(self.tcp.get_FIN(), 1)
        self.assertEqual(self.tcp.get_RST(), 0)
        self.tcp.reset_flags(2)
        self.assertEqual(self.tcp.get_th_flags(), 17)
        flags = int('10011', 2)
        self.tcp.set_th_flags(flags)
        self.assertEqual(self.tcp.get_th_flags(), 19)
        self.tcp.reset_flags(int('000010', 2))
        self.assertEqual(self.tcp.get_th_flags(), 17)
        flags = int('10011', 2)
        self.tcp.set_th_flags(flags)
        self.assertEqual(self.tcp.get_th_flags(), 19)
        self.tcp.reset_flags(int('010001', 2))
        self.assertEqual(self.tcp.get_th_flags(), 2)

    def test_09(self):
        if False:
            print('Hello World!')
        'Test TCP set_flags'
        flags = int('10101010', 2)
        self.tcp.set_flags(flags)
        self.assertEqual(self.tcp.get_FIN(), 0)
        self.assertEqual(self.tcp.get_SYN(), 1)
        self.assertEqual(self.tcp.get_RST(), 0)
        self.assertEqual(self.tcp.get_PSH(), 1)
        self.assertEqual(self.tcp.get_ACK(), 0)
        self.assertEqual(self.tcp.get_URG(), 1)
        self.assertEqual(self.tcp.get_ECE(), 0)
        self.assertEqual(self.tcp.get_CWR(), 1)
        self.assertEqual(self.tcp.get_th_flags(), 170)
if __name__ == '__main__':
    unittest.main(verbosity=1)