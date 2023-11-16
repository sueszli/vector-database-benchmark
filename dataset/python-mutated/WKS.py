import socket
import struct
import dns.immutable
import dns.ipv4
import dns.rdata
try:
    _proto_tcp = socket.getprotobyname('tcp')
    _proto_udp = socket.getprotobyname('udp')
except OSError:
    _proto_tcp = 6
    _proto_udp = 17

@dns.immutable.immutable
class WKS(dns.rdata.Rdata):
    """WKS record"""
    __slots__ = ['address', 'protocol', 'bitmap']

    def __init__(self, rdclass, rdtype, address, protocol, bitmap):
        if False:
            i = 10
            return i + 15
        super().__init__(rdclass, rdtype)
        self.address = self._as_ipv4_address(address)
        self.protocol = self._as_uint8(protocol)
        self.bitmap = self._as_bytes(bitmap)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        bits = []
        for (i, byte) in enumerate(self.bitmap):
            for j in range(0, 8):
                if byte & 128 >> j:
                    bits.append(str(i * 8 + j))
        text = ' '.join(bits)
        return '%s %d %s' % (self.address, self.protocol, text)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            while True:
                i = 10
        address = tok.get_string()
        protocol = tok.get_string()
        if protocol.isdigit():
            protocol = int(protocol)
        else:
            protocol = socket.getprotobyname(protocol)
        bitmap = bytearray()
        for token in tok.get_remaining():
            value = token.unescape().value
            if value.isdigit():
                serv = int(value)
            else:
                if protocol != _proto_udp and protocol != _proto_tcp:
                    raise NotImplementedError('protocol must be TCP or UDP')
                if protocol == _proto_udp:
                    protocol_text = 'udp'
                else:
                    protocol_text = 'tcp'
                serv = socket.getservbyname(value, protocol_text)
            i = serv // 8
            l = len(bitmap)
            if l < i + 1:
                for _ in range(l, i + 1):
                    bitmap.append(0)
            bitmap[i] = bitmap[i] | 128 >> serv % 8
        bitmap = dns.rdata._truncate_bitmap(bitmap)
        return cls(rdclass, rdtype, address, protocol, bitmap)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            i = 10
            return i + 15
        file.write(dns.ipv4.inet_aton(self.address))
        protocol = struct.pack('!B', self.protocol)
        file.write(protocol)
        file.write(self.bitmap)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            for i in range(10):
                print('nop')
        address = parser.get_bytes(4)
        protocol = parser.get_uint8()
        bitmap = parser.get_remaining()
        return cls(rdclass, rdtype, address, protocol, bitmap)