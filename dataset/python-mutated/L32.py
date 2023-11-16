import struct
import dns.immutable
import dns.rdata

@dns.immutable.immutable
class L32(dns.rdata.Rdata):
    """L32 record"""
    __slots__ = ['preference', 'locator32']

    def __init__(self, rdclass, rdtype, preference, locator32):
        if False:
            return 10
        super().__init__(rdclass, rdtype)
        self.preference = self._as_uint16(preference)
        self.locator32 = self._as_ipv4_address(locator32)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            while True:
                i = 10
        return f'{self.preference} {self.locator32}'

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            while True:
                i = 10
        preference = tok.get_uint16()
        nodeid = tok.get_identifier()
        return cls(rdclass, rdtype, preference, nodeid)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        file.write(struct.pack('!H', self.preference))
        file.write(dns.ipv4.inet_aton(self.locator32))

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            print('Hello World!')
        preference = parser.get_uint16()
        locator32 = parser.get_remaining()
        return cls(rdclass, rdtype, preference, locator32)