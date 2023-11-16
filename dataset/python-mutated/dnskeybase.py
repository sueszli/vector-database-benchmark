import base64
import enum
import struct
import dns.dnssectypes
import dns.exception
import dns.immutable
import dns.rdata
__all__ = ['SEP', 'REVOKE', 'ZONE']

class Flag(enum.IntFlag):
    SEP = 1
    REVOKE = 128
    ZONE = 256

@dns.immutable.immutable
class DNSKEYBase(dns.rdata.Rdata):
    """Base class for rdata that is like a DNSKEY record"""
    __slots__ = ['flags', 'protocol', 'algorithm', 'key']

    def __init__(self, rdclass, rdtype, flags, protocol, algorithm, key):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        self.flags = Flag(self._as_uint16(flags))
        self.protocol = self._as_uint8(protocol)
        self.algorithm = dns.dnssectypes.Algorithm.make(algorithm)
        self.key = self._as_bytes(key)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            while True:
                i = 10
        return '%d %d %d %s' % (self.flags, self.protocol, self.algorithm, dns.rdata._base64ify(self.key, **kw))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            for i in range(10):
                print('nop')
        flags = tok.get_uint16()
        protocol = tok.get_uint8()
        algorithm = tok.get_string()
        b64 = tok.concatenate_remaining_identifiers().encode()
        key = base64.b64decode(b64)
        return cls(rdclass, rdtype, flags, protocol, algorithm, key)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            return 10
        header = struct.pack('!HBB', self.flags, self.protocol, self.algorithm)
        file.write(header)
        file.write(self.key)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            i = 10
            return i + 15
        header = parser.get_struct('!HBB')
        key = parser.get_remaining()
        return cls(rdclass, rdtype, header[0], header[1], header[2], key)
SEP = Flag.SEP
REVOKE = Flag.REVOKE
ZONE = Flag.ZONE