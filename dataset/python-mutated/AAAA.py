import dns.exception
import dns.immutable
import dns.ipv6
import dns.rdata
import dns.tokenizer

@dns.immutable.immutable
class AAAA(dns.rdata.Rdata):
    """AAAA record."""
    __slots__ = ['address']

    def __init__(self, rdclass, rdtype, address):
        if False:
            print('Hello World!')
        super().__init__(rdclass, rdtype)
        self.address = self._as_ipv6_address(address)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            return 10
        return self.address

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            for i in range(10):
                print('nop')
        address = tok.get_identifier()
        return cls(rdclass, rdtype, address)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            i = 10
            return i + 15
        file.write(dns.ipv6.inet_aton(self.address))

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            print('Hello World!')
        address = parser.get_remaining()
        return cls(rdclass, rdtype, address)