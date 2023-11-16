import unittest
from impacket.dot11 import ProtocolPacket

class PacketTest(ProtocolPacket):

    def __init__(self, aBuffer=None):
        if False:
            for i in range(10):
                print('nop')
        header_size = 7
        tail_size = 5
        ProtocolPacket.__init__(self, header_size, tail_size)
        if aBuffer:
            self.load_packet(aBuffer)

class TestDot11HierarchicalUpdate(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.rawpacket1 = b'Header1Body1Tail1'
        self.rawpacket2 = b'Header2' + self.rawpacket1 + b'Tail2'
        self.rawpacket3 = b'Header3' + self.rawpacket2 + b'Tail3'
        self.packet1 = PacketTest(self.rawpacket1)
        self.packet2 = PacketTest(self.rawpacket2)
        self.packet2.contains(self.packet1)
        self.packet3 = PacketTest(self.rawpacket3)
        self.packet3.contains(self.packet2)

    def test_01_StartupPacketsStringTest(self):
        if False:
            i = 10
            return i + 15
        'ProtocolPacket - get_packet initial string test'
        self.assertEqual(self.packet1.get_packet(), b'Header1Body1Tail1')
        self.assertEqual(self.packet2.get_packet(), b'Header2Header1Body1Tail1Tail2')
        self.assertEqual(self.packet3.get_packet(), b'Header3Header2Header1Body1Tail1Tail2Tail3')

    def test_02_StartupPacketsSizeTest(self):
        if False:
            print('Hello World!')
        'ProtocolPacket - Initial size getters test'
        self.assertEqual(self.packet1.get_size(), 7 + 5 + 5)
        self.assertEqual(self.packet1.get_header_size(), 7)
        self.assertEqual(self.packet1.get_body_size(), 5)
        self.assertEqual(self.packet1.get_tail_size(), 5)
        self.assertEqual(self.packet2.get_size(), 7 + (7 + 5 + 5) + 5)
        self.assertEqual(self.packet2.get_header_size(), 7)
        self.assertEqual(self.packet2.get_body_size(), 7 + 5 + 5)
        self.assertEqual(self.packet2.get_tail_size(), 5)
        self.assertEqual(self.packet3.get_size(), 7 + (7 + (7 + 5 + 5) + 5) + 5)
        self.assertEqual(self.packet3.get_header_size(), 7)
        self.assertEqual(self.packet3.get_body_size(), 7 + 7 + 5 + 5 + 5)
        self.assertEqual(self.packet3.get_tail_size(), 5)

    def test_03_ChildModificationTest(self):
        if False:
            i = 10
            return i + 15
        'ProtocolPacket - get_packet hierarchical update test'
        self.packet1.load_body(b'**NewBody**')
        self.assertEqual(self.packet1.get_packet(), b'Header1**NewBody**Tail1')
        self.assertEqual(self.packet2.get_packet(), b'Header2Header1**NewBody**Tail1Tail2')
        self.assertEqual(self.packet3.get_packet(), b'Header3Header2Header1**NewBody**Tail1Tail2Tail3')

    def test_04_ChildModificationTest(self):
        if False:
            print('Hello World!')
        'ProtocolPacket - size getters hierarchical update test'
        self.packet1.load_body(b'**NewBody**')
        self.assertEqual(self.packet1.get_size(), 7 + 11 + 5)
        self.assertEqual(self.packet1.get_header_size(), 7)
        self.assertEqual(self.packet1.get_body_size(), 11)
        self.assertEqual(self.packet1.get_tail_size(), 5)
        self.assertEqual(self.packet2.get_size(), 7 + (7 + 11 + 5) + 5)
        self.assertEqual(self.packet2.get_header_size(), 7)
        self.assertEqual(self.packet2.get_body_size(), 7 + 11 + 5)
        self.assertEqual(self.packet2.get_tail_size(), 5)
        self.assertEqual(self.packet3.get_size(), 7 + (7 + (7 + 11 + 5) + 5) + 5)
        self.assertEqual(self.packet3.get_header_size(), 7)
        self.assertEqual(self.packet3.get_body_size(), 7 + (7 + 11 + 5) + 5)
        self.assertEqual(self.packet3.get_tail_size(), 5)

    def test_05_ChildModificationTest(self):
        if False:
            while True:
                i = 10
        'ProtocolPacket - body packet hierarchical update test'
        self.packet1.load_body(b'**NewBody**')
        self.assertEqual(self.packet1.body.get_buffer_as_string(), b'**NewBody**')
        self.assertEqual(self.packet2.body.get_buffer_as_string(), b'Header1**NewBody**Tail1')
        self.assertEqual(self.packet3.body.get_buffer_as_string(), b'Header2Header1**NewBody**Tail1Tail2')

    def test_06_ChildModificationTest(self):
        if False:
            while True:
                i = 10
        'ProtocolPacket - get_body_as_string packet hierarchical update test'
        self.packet1.load_body(b'**NewBody**')
        self.assertEqual(self.packet1.get_body_as_string(), b'**NewBody**')
        self.assertEqual(self.packet2.get_body_as_string(), b'Header1**NewBody**Tail1')
        self.assertEqual(self.packet3.get_body_as_string(), b'Header2Header1**NewBody**Tail1Tail2')

    def test_07_ChildModificationTest(self):
        if False:
            i = 10
            return i + 15
        'ProtocolPacket - load_body child hierarchy update test'
        self.assertEqual(self.packet1.parent(), self.packet2)
        self.assertEqual(self.packet2.parent(), self.packet3)
        self.assertEqual(self.packet3.child(), self.packet2)
        self.assertEqual(self.packet2.child(), self.packet1)
        self.packet2.load_body(b'Header1**NewBody**Tail1')
        self.assertEqual(self.packet1.parent(), None)
        self.assertEqual(self.packet2.parent(), self.packet3)
        self.assertEqual(self.packet3.child(), self.packet2)
        self.assertEqual(self.packet2.child(), None)
        self.assertEqual(self.packet1.body.get_buffer_as_string(), b'Body1')
        self.assertEqual(self.packet2.body.get_buffer_as_string(), b'Header1**NewBody**Tail1')
        self.assertEqual(self.packet3.body.get_buffer_as_string(), b'Header2Header1**NewBody**Tail1Tail2')
if __name__ == '__main__':
    unittest.main(verbosity=1)