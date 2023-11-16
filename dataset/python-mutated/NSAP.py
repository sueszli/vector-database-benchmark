import binascii
import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer

@dns.immutable.immutable
class NSAP(dns.rdata.Rdata):
    """NSAP record."""
    __slots__ = ['address']

    def __init__(self, rdclass, rdtype, address):
        if False:
            i = 10
            return i + 15
        super().__init__(rdclass, rdtype)
        self.address = self._as_bytes(address)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        return '0x%s' % binascii.hexlify(self.address).decode()

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            for i in range(10):
                print('nop')
        address = tok.get_string()
        if address[0:2] != '0x':
            raise dns.exception.SyntaxError('string does not start with 0x')
        address = address[2:].replace('.', '')
        if len(address) % 2 != 0:
            raise dns.exception.SyntaxError('hexstring has odd length')
        address = binascii.unhexlify(address.encode())
        return cls(rdclass, rdtype, address)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        file.write(self.address)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            i = 10
            return i + 15
        address = parser.get_remaining()
        return cls(rdclass, rdtype, address)