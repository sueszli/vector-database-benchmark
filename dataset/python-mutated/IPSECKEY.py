import base64
import struct
import dns.exception
import dns.immutable
import dns.rdtypes.util

class Gateway(dns.rdtypes.util.Gateway):
    name = 'IPSECKEY gateway'

@dns.immutable.immutable
class IPSECKEY(dns.rdata.Rdata):
    """IPSECKEY record"""
    __slots__ = ['precedence', 'gateway_type', 'algorithm', 'gateway', 'key']

    def __init__(self, rdclass, rdtype, precedence, gateway_type, algorithm, gateway, key):
        if False:
            while True:
                i = 10
        super().__init__(rdclass, rdtype)
        gateway = Gateway(gateway_type, gateway)
        self.precedence = self._as_uint8(precedence)
        self.gateway_type = gateway.type
        self.algorithm = self._as_uint8(algorithm)
        self.gateway = gateway.gateway
        self.key = self._as_bytes(key)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        gateway = Gateway(self.gateway_type, self.gateway).to_text(origin, relativize)
        return '%d %d %d %s %s' % (self.precedence, self.gateway_type, self.algorithm, gateway, dns.rdata._base64ify(self.key, **kw))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            return 10
        precedence = tok.get_uint8()
        gateway_type = tok.get_uint8()
        algorithm = tok.get_uint8()
        gateway = Gateway.from_text(gateway_type, tok, origin, relativize, relativize_to)
        b64 = tok.concatenate_remaining_identifiers().encode()
        key = base64.b64decode(b64)
        return cls(rdclass, rdtype, precedence, gateway_type, algorithm, gateway.gateway, key)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            i = 10
            return i + 15
        header = struct.pack('!BBB', self.precedence, self.gateway_type, self.algorithm)
        file.write(header)
        Gateway(self.gateway_type, self.gateway).to_wire(file, compress, origin, canonicalize)
        file.write(self.key)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            while True:
                i = 10
        header = parser.get_struct('!BBB')
        gateway_type = header[1]
        gateway = Gateway.from_wire_parser(gateway_type, parser, origin)
        key = parser.get_remaining()
        return cls(rdclass, rdtype, header[0], gateway_type, header[2], gateway.gateway, key)