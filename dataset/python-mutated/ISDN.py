import struct
import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer

@dns.immutable.immutable
class ISDN(dns.rdata.Rdata):
    """ISDN record"""
    __slots__ = ['address', 'subaddress']

    def __init__(self, rdclass, rdtype, address, subaddress):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        self.address = self._as_bytes(address, True, 255)
        self.subaddress = self._as_bytes(subaddress, True, 255)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            return 10
        if self.subaddress:
            return '"{}" "{}"'.format(dns.rdata._escapify(self.address), dns.rdata._escapify(self.subaddress))
        else:
            return '"%s"' % dns.rdata._escapify(self.address)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            return 10
        address = tok.get_string()
        tokens = tok.get_remaining(max_tokens=1)
        if len(tokens) >= 1:
            subaddress = tokens[0].unescape().value
        else:
            subaddress = ''
        return cls(rdclass, rdtype, address, subaddress)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            i = 10
            return i + 15
        l = len(self.address)
        assert l < 256
        file.write(struct.pack('!B', l))
        file.write(self.address)
        l = len(self.subaddress)
        if l > 0:
            assert l < 256
            file.write(struct.pack('!B', l))
            file.write(self.subaddress)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            for i in range(10):
                print('nop')
        address = parser.get_counted_bytes()
        if parser.remaining() > 0:
            subaddress = parser.get_counted_bytes()
        else:
            subaddress = b''
        return cls(rdclass, rdtype, address, subaddress)