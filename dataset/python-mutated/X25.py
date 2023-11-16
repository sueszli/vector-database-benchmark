import struct
import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer

@dns.immutable.immutable
class X25(dns.rdata.Rdata):
    """X25 record"""
    __slots__ = ['address']

    def __init__(self, rdclass, rdtype, address):
        if False:
            return 10
        super().__init__(rdclass, rdtype)
        self.address = self._as_bytes(address, True, 255)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            for i in range(10):
                print('nop')
        return '"%s"' % dns.rdata._escapify(self.address)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            return 10
        address = tok.get_string()
        return cls(rdclass, rdtype, address)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        l = len(self.address)
        assert l < 256
        file.write(struct.pack('!B', l))
        file.write(self.address)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            while True:
                i = 10
        address = parser.get_counted_bytes()
        return cls(rdclass, rdtype, address)