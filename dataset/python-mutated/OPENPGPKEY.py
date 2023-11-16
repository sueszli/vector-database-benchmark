import base64
import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer

@dns.immutable.immutable
class OPENPGPKEY(dns.rdata.Rdata):
    """OPENPGPKEY record"""

    def __init__(self, rdclass, rdtype, key):
        if False:
            print('Hello World!')
        super().__init__(rdclass, rdtype)
        self.key = self._as_bytes(key)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        return dns.rdata._base64ify(self.key, chunksize=None, **kw)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            return 10
        b64 = tok.concatenate_remaining_identifiers().encode()
        key = base64.b64decode(b64)
        return cls(rdclass, rdtype, key)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        file.write(self.key)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            print('Hello World!')
        key = parser.get_remaining()
        return cls(rdclass, rdtype, key)