from zope.interface import implementer
from twisted.pair import ethernet, raw
from twisted.python import components
from twisted.trial import unittest

@implementer(raw.IRawPacketProtocol)
class MyProtocol:

    def __init__(self, expecting):
        if False:
            i = 10
            return i + 15
        self.expecting = list(expecting)

    def addProto(self, num, proto):
        if False:
            print('Hello World!')
        '\n        Not implemented\n        '

    def datagramReceived(self, data, partial, dest, source, protocol):
        if False:
            return 10
        assert self.expecting, 'Got a packet when not expecting anymore.'
        expect = self.expecting.pop(0)
        localVariables = locals()
        params = {'partial': partial, 'dest': dest, 'source': source, 'protocol': protocol}
        assert expect == (data, params), 'Expected {!r}, got {!r}'.format(expect, (data, params))

class EthernetTests(unittest.TestCase):

    def testPacketParsing(self):
        if False:
            print('Hello World!')
        proto = ethernet.EthernetProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'987654', 'protocol': 2048})])
        proto.addProto(2048, p1)
        proto.datagramReceived(b'123456987654\x08\x00foobar', partial=0)
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting

    def testMultiplePackets(self):
        if False:
            for i in range(10):
                print('nop')
        proto = ethernet.EthernetProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'987654', 'protocol': 2048}), (b'quux', {'partial': 1, 'dest': b'012345', 'source': b'abcdef', 'protocol': 2048})])
        proto.addProto(2048, p1)
        proto.datagramReceived(b'123456987654\x08\x00foobar', partial=0)
        proto.datagramReceived(b'012345abcdef\x08\x00quux', partial=1)
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting

    def testMultipleSameProtos(self):
        if False:
            print('Hello World!')
        proto = ethernet.EthernetProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'987654', 'protocol': 2048})])
        p2 = MyProtocol([(b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'987654', 'protocol': 2048})])
        proto.addProto(2048, p1)
        proto.addProto(2048, p2)
        proto.datagramReceived(b'123456987654\x08\x00foobar', partial=0)
        assert not p1.expecting, 'Should not expect any more packets, but still want {!r}'.format(p1.expecting)
        assert not p2.expecting, 'Should not expect any more packets, but still want {!r}'.format(p2.expecting)

    def testWrongProtoNotSeen(self):
        if False:
            while True:
                i = 10
        proto = ethernet.EthernetProtocol()
        p1 = MyProtocol([])
        proto.addProto(2049, p1)
        proto.datagramReceived(b'123456987654\x08\x00foobar', partial=0)
        proto.datagramReceived(b'012345abcdef\x08\x00quux', partial=1)

    def testDemuxing(self):
        if False:
            return 10
        proto = ethernet.EthernetProtocol()
        p1 = MyProtocol([(b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'987654', 'protocol': 2048}), (b'quux', {'partial': 1, 'dest': b'012345', 'source': b'abcdef', 'protocol': 2048})])
        proto.addProto(2048, p1)
        p2 = MyProtocol([(b'quux', {'partial': 1, 'dest': b'012345', 'source': b'abcdef', 'protocol': 2054}), (b'foobar', {'partial': 0, 'dest': b'123456', 'source': b'987654', 'protocol': 2054})])
        proto.addProto(2054, p2)
        proto.datagramReceived(b'123456987654\x08\x00foobar', partial=0)
        proto.datagramReceived(b'012345abcdef\x08\x06quux', partial=1)
        proto.datagramReceived(b'123456987654\x08\x06foobar', partial=0)
        proto.datagramReceived(b'012345abcdef\x08\x00quux', partial=1)
        assert not p1.expecting, 'Should not expect any more packets, but still want %r' % p1.expecting
        assert not p2.expecting, 'Should not expect any more packets, but still want %r' % p2.expecting

    def testAddingBadProtos_WrongLevel(self):
        if False:
            print('Hello World!')
        'Adding a wrong level protocol raises an exception.'
        e = ethernet.EthernetProtocol()
        try:
            e.addProto(42, 'silliness')
        except components.CannotAdapt:
            pass
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooSmall(self):
        if False:
            print('Hello World!')
        'Adding a protocol with a negative number raises an exception.'
        e = ethernet.EthernetProtocol()
        try:
            e.addProto(-1, MyProtocol([]))
        except TypeError as e:
            if e.args == ('Added protocol must be positive or zero',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooBig(self):
        if False:
            while True:
                i = 10
        'Adding a protocol with a number >=2**16 raises an exception.'
        e = ethernet.EthernetProtocol()
        try:
            e.addProto(2 ** 16, MyProtocol([]))
        except TypeError as e:
            if e.args == ('Added protocol must fit in 16 bits',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')

    def testAddingBadProtos_TooBig2(self):
        if False:
            i = 10
            return i + 15
        'Adding a protocol with a number >=2**16 raises an exception.'
        e = ethernet.EthernetProtocol()
        try:
            e.addProto(2 ** 16 + 1, MyProtocol([]))
        except TypeError as e:
            if e.args == ('Added protocol must fit in 16 bits',):
                pass
            else:
                raise
        else:
            raise AssertionError('addProto must raise an exception for bad protocols')