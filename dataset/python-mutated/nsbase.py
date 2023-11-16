"""NS-like base classes."""
import dns.exception
import dns.immutable
import dns.name
import dns.rdata

@dns.immutable.immutable
class NSBase(dns.rdata.Rdata):
    """Base class for rdata that is like an NS record."""
    __slots__ = ['target']

    def __init__(self, rdclass, rdtype, target):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        self.target = self._as_name(target)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            for i in range(10):
                print('nop')
        target = self.target.choose_relativity(origin, relativize)
        return str(target)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            for i in range(10):
                print('nop')
        target = tok.get_name(origin, relativize, relativize_to)
        return cls(rdclass, rdtype, target)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        self.target.to_wire(file, compress, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            return 10
        target = parser.get_name(origin)
        return cls(rdclass, rdtype, target)

@dns.immutable.immutable
class UncompressedNS(NSBase):
    """Base class for rdata that is like an NS record, but whose name
    is not compressed when convert to DNS wire format, and whose
    digestable form is not downcased."""

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            print('Hello World!')
        self.target.to_wire(file, None, origin, False)