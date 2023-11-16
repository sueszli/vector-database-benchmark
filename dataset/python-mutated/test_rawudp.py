from __future__ import annotations
from twisted.internet import protocol
from twisted.pair import rawudp
from twisted.trial import unittest

class MyProtocol(protocol.DatagramProtocol):

    def __init__(self, expecting: list[tuple[bytes, bytes, int]]) -> None:
        if False:
            while True:
                i = 10
        self.expecting = list(expecting)

    def datagramReceived(self, data: bytes, peer: tuple[bytes, int]) -> None:
        if False:
            i = 10
            return i + 15
        (host, port) = peer
        assert self.expecting, 'Got a packet when not expecting anymore.'
        (expectData, expectHost, expectPort) = self.expecting.pop(0)
        assert expectData == data, 'Expected data {!r}, got {!r}'.format(expectData, data)
        assert expectHost == host, 'Expected host {!r}, got {!r}'.format(expectHost, host)
        assert expectPort == port, 'Expected port %d=0x%04x, got %d=0x%04x' % (expectPort, expectPort, port, port)

class RawUDPTests(unittest.TestCase):

    def testPacketParsing(self) -> None:
        if False:
            return 10
        proto = rawudp.RawUDPProtocol()
        p1 = MyProtocol([(b'foobar', b'testHost', 17314)])
        proto.addProto(61455, p1)
        proto.datagramReceived(b'C\xa2\xf0\x0f\x00\x06\xde\xadfoobar', partial=0, dest=b'dummy', source=b'testHost', protocol=b'dummy', version=b'dummy', ihl=b'dummy', tos=b'dummy', tot_len=b'dummy', fragment_id=b'dummy', fragment_offset=b'dummy', dont_fragment=b'dummy', more_fragments=b'dummy', ttl=b'dummy')
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting

    def testMultiplePackets(self) -> None:
        if False:
            print('Hello World!')
        proto = rawudp.RawUDPProtocol()
        p1 = MyProtocol([(b'foobar', b'testHost', 17314), (b'quux', b'otherHost', 13310)])
        proto.addProto(61455, p1)
        proto.datagramReceived(b'C\xa2\xf0\x0f\x00\x06\xde\xadfoobar', partial=0, dest=b'dummy', source=b'testHost', protocol=b'dummy', version=b'dummy', ihl=b'dummy', tos=b'dummy', tot_len=b'dummy', fragment_id=b'dummy', fragment_offset=b'dummy', dont_fragment=b'dummy', more_fragments=b'dummy', ttl=b'dummy')
        proto.datagramReceived(b'3\xfe\xf0\x0f\x00\x05\xde\xadquux', partial=0, dest=b'dummy', source=b'otherHost', protocol=b'dummy', version=b'dummy', ihl=b'dummy', tos=b'dummy', tot_len=b'dummy', fragment_id=b'dummy', fragment_offset=b'dummy', dont_fragment=b'dummy', more_fragments=b'dummy', ttl=b'dummy')
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting

    def testMultipleSameProtos(self) -> None:
        if False:
            while True:
                i = 10
        proto = rawudp.RawUDPProtocol()
        p1 = MyProtocol([(b'foobar', b'testHost', 17314)])
        p2 = MyProtocol([(b'foobar', b'testHost', 17314)])
        proto.addProto(61455, p1)
        proto.addProto(61455, p2)
        proto.datagramReceived(b'C\xa2\xf0\x0f\x00\x06\xde\xadfoobar', partial=0, dest=b'dummy', source=b'testHost', protocol=b'dummy', version=b'dummy', ihl=b'dummy', tos=b'dummy', tot_len=b'dummy', fragment_id=b'dummy', fragment_offset=b'dummy', dont_fragment=b'dummy', more_fragments=b'dummy', ttl=b'dummy')
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting
        assert not p2.expecting, 'Should not expect any more packets, but still want %r' % p2.expecting

    def testWrongProtoNotSeen(self) -> None:
        if False:
            print('Hello World!')
        proto = rawudp.RawUDPProtocol()
        p1 = MyProtocol([])
        proto.addProto(1, p1)
        proto.datagramReceived(b'C\xa2\xf0\x0f\x00\x06\xde\xadfoobar', partial=0, dest=b'dummy', source=b'testHost', protocol=b'dummy', version=b'dummy', ihl=b'dummy', tos=b'dummy', tot_len=b'dummy', fragment_id=b'dummy', fragment_offset=b'dummy', dont_fragment=b'dummy', more_fragments=b'dummy', ttl=b'dummy')

    def testDemuxing(self) -> None:
        if False:
            i = 10
            return i + 15
        proto = rawudp.RawUDPProtocol()
        p1 = MyProtocol([(b'foobar', b'testHost', 17314), (b'quux', b'otherHost', 13310)])
        proto.addProto(61455, p1)
        p2 = MyProtocol([(b'quux', b'otherHost', 41985), (b'foobar', b'testHost', 41730)])
        proto.addProto(45136, p2)
        proto.datagramReceived(b'\xa4\x01\xb0P\x00\x05\xde\xadquux', partial=0, dest=b'dummy', source=b'otherHost', protocol=b'dummy', version=b'dummy', ihl=b'dummy', tos=b'dummy', tot_len=b'dummy', fragment_id=b'dummy', fragment_offset=b'dummy', dont_fragment=b'dummy', more_fragments=b'dummy', ttl=b'dummy')
        proto.datagramReceived(b'C\xa2\xf0\x0f\x00\x06\xde\xadfoobar', partial=0, dest=b'dummy', source=b'testHost', protocol=b'dummy', version=b'dummy', ihl=b'dummy', tos=b'dummy', tot_len=b'dummy', fragment_id=b'dummy', fragment_offset=b'dummy', dont_fragment=b'dummy', more_fragments=b'dummy', ttl=b'dummy')
        proto.datagramReceived(b'3\xfe\xf0\x0f\x00\x05\xde\xadquux', partial=0, dest=b'dummy', source=b'otherHost', protocol=b'dummy', version=b'dummy', ihl=b'dummy', tos=b'dummy', tot_len=b'dummy', fragment_id=b'dummy', fragment_offset=b'dummy', dont_fragment=b'dummy', more_fragments=b'dummy', ttl=b'dummy')
        proto.datagramReceived(b'\xa3\x02\xb0P\x00\x06\xde\xadfoobar', partial=0, dest=b'dummy', source=b'testHost', protocol=b'dummy', version=b'dummy', ihl=b'dummy', tos=b'dummy', tot_len=b'dummy', fragment_id=b'dummy', fragment_offset=b'dummy', dont_fragment=b'dummy', more_fragments=b'dummy', ttl=b'dummy')
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting
        assert not p2.expecting, 'Should not expect any more packets, but still want %r' % p2.expecting

    def testAddingBadProtos_WrongLevel(self) -> None:
        if False:
            i = 10
            return i + 15
        'Adding a wrong level protocol raises an exception.'
        e = rawudp.RawUDPProtocol()
        try:
            e.addProto(42, 'silliness')
        except TypeError as e:
            if e.args == ('Added protocol must be an instance of DatagramProtocol',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooSmall(self) -> None:
        if False:
            return 10
        'Adding a protocol with a negative number raises an exception.'
        e = rawudp.RawUDPProtocol()
        try:
            e.addProto(-1, protocol.DatagramProtocol())
        except TypeError as e:
            if e.args == ('Added protocol must be positive or zero',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooBig(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adding a protocol with a number >=2**16 raises an exception.'
        e = rawudp.RawUDPProtocol()
        try:
            e.addProto(2 ** 16, protocol.DatagramProtocol())
        except TypeError as e:
            if e.args == ('Added protocol must fit in 16 bits',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooBig2(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adding a protocol with a number >=2**16 raises an exception.'
        e = rawudp.RawUDPProtocol()
        try:
            e.addProto(2 ** 16 + 1, protocol.DatagramProtocol())
        except TypeError as e:
            if e.args == ('Added protocol must fit in 16 bits',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')