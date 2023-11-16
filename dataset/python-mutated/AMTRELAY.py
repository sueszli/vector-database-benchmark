import struct
import dns.exception
import dns.immutable
import dns.rdtypes.util

class Relay(dns.rdtypes.util.Gateway):
    name = 'AMTRELAY relay'

    @property
    def relay(self):
        if False:
            print('Hello World!')
        return self.gateway

@dns.immutable.immutable
class AMTRELAY(dns.rdata.Rdata):
    """AMTRELAY record"""
    __slots__ = ['precedence', 'discovery_optional', 'relay_type', 'relay']

    def __init__(self, rdclass, rdtype, precedence, discovery_optional, relay_type, relay):
        if False:
            i = 10
            return i + 15
        super().__init__(rdclass, rdtype)
        relay = Relay(relay_type, relay)
        self.precedence = self._as_uint8(precedence)
        self.discovery_optional = self._as_bool(discovery_optional)
        self.relay_type = relay.type
        self.relay = relay.relay

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            return 10
        relay = Relay(self.relay_type, self.relay).to_text(origin, relativize)
        return '%d %d %d %s' % (self.precedence, self.discovery_optional, self.relay_type, relay)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            i = 10
            return i + 15
        precedence = tok.get_uint8()
        discovery_optional = tok.get_uint8()
        if discovery_optional > 1:
            raise dns.exception.SyntaxError('expecting 0 or 1')
        discovery_optional = bool(discovery_optional)
        relay_type = tok.get_uint8()
        if relay_type > 127:
            raise dns.exception.SyntaxError('expecting an integer <= 127')
        relay = Relay.from_text(relay_type, tok, origin, relativize, relativize_to)
        return cls(rdclass, rdtype, precedence, discovery_optional, relay_type, relay.relay)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        relay_type = self.relay_type | self.discovery_optional << 7
        header = struct.pack('!BB', self.precedence, relay_type)
        file.write(header)
        Relay(self.relay_type, self.relay).to_wire(file, compress, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            return 10
        (precedence, relay_type) = parser.get_struct('!BB')
        discovery_optional = bool(relay_type >> 7)
        relay_type &= 127
        relay = Relay.from_wire_parser(relay_type, parser, origin)
        return cls(rdclass, rdtype, precedence, discovery_optional, relay_type, relay.relay)