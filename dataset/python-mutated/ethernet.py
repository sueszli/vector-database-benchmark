"""Support for working directly with ethernet frames"""
import struct
from zope.interface import Interface, implementer
from twisted.internet import protocol
from twisted.pair import raw

class IEthernetProtocol(Interface):
    """An interface for protocols that handle Ethernet frames"""

    def addProto(num, proto):
        if False:
            for i in range(10):
                print('nop')
        'Add an IRawPacketProtocol protocol'

    def datagramReceived(data, partial):
        if False:
            for i in range(10):
                print('nop')
        'An Ethernet frame has been received'

class EthernetHeader:

    def __init__(self, data):
        if False:
            i = 10
            return i + 15
        (self.dest, self.source, self.proto) = struct.unpack('!6s6sH', data[:6 + 6 + 2])

@implementer(IEthernetProtocol)
class EthernetProtocol(protocol.AbstractDatagramProtocol):

    def __init__(self):
        if False:
            return 10
        self.etherProtos = {}

    def addProto(self, num, proto):
        if False:
            print('Hello World!')
        proto = raw.IRawPacketProtocol(proto)
        if num < 0:
            raise TypeError('Added protocol must be positive or zero')
        if num >= 2 ** 16:
            raise TypeError('Added protocol must fit in 16 bits')
        if num not in self.etherProtos:
            self.etherProtos[num] = []
        self.etherProtos[num].append(proto)

    def datagramReceived(self, data, partial=0):
        if False:
            return 10
        header = EthernetHeader(data[:14])
        for proto in self.etherProtos.get(header.proto, ()):
            proto.datagramReceived(data=data[14:], partial=partial, dest=header.dest, source=header.source, protocol=header.proto)