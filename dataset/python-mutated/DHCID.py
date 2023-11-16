import base64
import dns.exception
import dns.immutable
import dns.rdata

@dns.immutable.immutable
class DHCID(dns.rdata.Rdata):
    """DHCID record"""
    __slots__ = ['data']

    def __init__(self, rdclass, rdtype, data):
        if False:
            while True:
                i = 10
        super().__init__(rdclass, rdtype)
        self.data = self._as_bytes(data)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        return dns.rdata._base64ify(self.data, **kw)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            for i in range(10):
                print('nop')
        b64 = tok.concatenate_remaining_identifiers().encode()
        data = base64.b64decode(b64)
        return cls(rdclass, rdtype, data)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            while True:
                i = 10
        file.write(self.data)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            return 10
        data = parser.get_remaining()
        return cls(rdclass, rdtype, data)