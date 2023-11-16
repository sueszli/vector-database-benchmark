from __future__ import annotations
from zope import interface
from twisted.pair import ip, raw
from twisted.python import components
from twisted.trial import unittest

@interface.implementer(raw.IRawDatagramProtocol)
class MyProtocol:

    def __init__(self, expecting: list[tuple[bytes, dict[str, str | int]]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.expecting = list(expecting)

    def datagramReceived(self, data: bytes, partial: int, source: str, dest: str, protocol: int, version: int, ihl: int, tos: int, tot_len: int, fragment_id: int, fragment_offset: int, dont_fragment: int, more_fragments: int, ttl: int) -> None:
        if False:
            i = 10
            return i + 15
        assert self.expecting, 'Got a packet when not expecting anymore.'
        (expectData, expectKw) = self.expecting.pop(0)
        expectKwKeys = list(sorted(expectKw.keys()))
        localVariables = locals()
        for k in expectKwKeys:
            assert expectKw[k] == localVariables[k], f'Expected {k}={expectKw[k]!r}, got {localVariables[k]!r}'
        assert expectData == data, f'Expected {expectData!r}, got {data!r}'

    def addProto(self, num: object, proto: object) -> None:
        if False:
            return 10
        pass

class IPTests(unittest.TestCase):

    def testPacketParsing(self) -> None:
        if False:
            print('Hello World!')
        proto = ip.IPProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': '1.2.3.4', 'source': '5.6.7.8', 'protocol': 15, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192})])
        proto.addProto(15, p1)
        proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\x0f' + b'FE' + b'\x05\x06\x07\x08' + b'\x01\x02\x03\x04' + b'foobar', partial=0, dest='dummy', source='dummy', protocol='dummy')
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting

    def testMultiplePackets(self) -> None:
        if False:
            return 10
        proto = ip.IPProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': '1.2.3.4', 'source': '5.6.7.8', 'protocol': 15, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192}), (b'quux', {'partial': 1, 'dest': '5.4.3.2', 'source': '6.7.8.9', 'protocol': 15, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192})])
        proto.addProto(15, p1)
        proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\x0f' + b'FE' + b'\x05\x06\x07\x08' + b'\x01\x02\x03\x04' + b'foobar', partial=0, dest='dummy', source='dummy', protocol='dummy')
        proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\x0f' + b'FE' + b'\x06\x07\x08\t' + b'\x05\x04\x03\x02' + b'quux', partial=1, dest='dummy', source='dummy', protocol='dummy')
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting

    def testMultipleSameProtos(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        proto = ip.IPProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': '1.2.3.4', 'source': '5.6.7.8', 'protocol': 15, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192})])
        p2 = MyProtocol([(b'foobar', {'partial': 0, 'dest': '1.2.3.4', 'source': '5.6.7.8', 'protocol': 15, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192})])
        proto.addProto(15, p1)
        proto.addProto(15, p2)
        proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\x0f' + b'FE' + b'\x05\x06\x07\x08' + b'\x01\x02\x03\x04' + b'foobar', partial=0, dest='dummy', source='dummy', protocol='dummy')
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting
        assert not p2.expecting, 'Should not expect any more packets, but still want %r' % p2.expecting

    def testWrongProtoNotSeen(self) -> None:
        if False:
            return 10
        proto = ip.IPProtocol()
        p1 = MyProtocol([])
        proto.addProto(1, p1)
        proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\x0f' + b'FE' + b'\x05\x06\x07\x08' + b'\x01\x02\x03\x04' + b'foobar', partial=0, dest='dummy', source='dummy', protocol='dummy')

    def testDemuxing(self) -> None:
        if False:
            return 10
        proto = ip.IPProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': '1.2.3.4', 'source': '5.6.7.8', 'protocol': 15, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192}), (b'quux', {'partial': 1, 'dest': '5.4.3.2', 'source': '6.7.8.9', 'protocol': 15, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192})])
        proto.addProto(15, p1)
        p2 = MyProtocol([(b'quux', {'partial': 1, 'dest': '5.4.3.2', 'source': '6.7.8.9', 'protocol': 10, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192}), (b'foobar', {'partial': 0, 'dest': '1.2.3.4', 'source': '5.6.7.8', 'protocol': 10, 'version': 4, 'ihl': 20, 'tos': 7, 'tot_len': 20 + 6, 'fragment_id': 57005, 'fragment_offset': 7919, 'dont_fragment': 0, 'more_fragments': 1, 'ttl': 192})])
        proto.addProto(10, p2)
        proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\n' + b'FE' + b'\x06\x07\x08\t' + b'\x05\x04\x03\x02' + b'quux', partial=1, dest='dummy', source='dummy', protocol='dummy')
        proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\x0f' + b'FE' + b'\x05\x06\x07\x08' + b'\x01\x02\x03\x04' + b'foobar', partial=0, dest='dummy', source='dummy', protocol='dummy')
        proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\x0f' + b'FE' + b'\x06\x07\x08\t' + b'\x05\x04\x03\x02' + b'quux', partial=1, dest='dummy', source='dummy', protocol='dummy')
        proto.datagramReceived(b'T' + b'\x07' + b'\x00\x1a' + b'\xde\xad' + b'\xbe\xef' + b'\xc0' + b'\n' + b'FE' + b'\x05\x06\x07\x08' + b'\x01\x02\x03\x04' + b'foobar', partial=0, dest='dummy', source='dummy', protocol='dummy')
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting
        assert not p2.expecting, 'Should not expect any more packets, but still want %r' % p2.expecting

    def testAddingBadProtos_WrongLevel(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adding a wrong level protocol raises an exception.'
        e = ip.IPProtocol()
        try:
            e.addProto(42, 'silliness')
        except components.CannotAdapt:
            pass
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooSmall(self) -> None:
        if False:
            print('Hello World!')
        'Adding a protocol with a negative number raises an exception.'
        e = ip.IPProtocol()
        try:
            e.addProto(-1, MyProtocol([]))
        except TypeError as e:
            if e.args == ('Added protocol must be positive or zero',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooBig(self) -> None:
        if False:
            print('Hello World!')
        'Adding a protocol with a number >=2**32 raises an exception.'
        e = ip.IPProtocol()
        try:
            e.addProto(2 ** 32, MyProtocol([]))
        except TypeError as e:
            if e.args == ('Added protocol must fit in 32 bits',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooBig2(self) -> None:
        if False:
            print('Hello World!')
        'Adding a protocol with a number >=2**32 raises an exception.'
        e = ip.IPProtocol()
        try:
            e.addProto(2 ** 32 + 1, MyProtocol([]))
        except TypeError as e:
            if e.args == ('Added protocol must fit in 32 bits',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')