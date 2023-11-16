import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdatatype
import dns.rdtypes.util

@dns.immutable.immutable
class Bitmap(dns.rdtypes.util.Bitmap):
    type_name = 'NSEC'

@dns.immutable.immutable
class NSEC(dns.rdata.Rdata):
    """NSEC record"""
    __slots__ = ['next', 'windows']

    def __init__(self, rdclass, rdtype, next, windows):
        if False:
            i = 10
            return i + 15
        super().__init__(rdclass, rdtype)
        self.next = self._as_name(next)
        if not isinstance(windows, Bitmap):
            windows = Bitmap(windows)
        self.windows = tuple(windows.windows)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        next = self.next.choose_relativity(origin, relativize)
        text = Bitmap(self.windows).to_text()
        return '{}{}'.format(next, text)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            for i in range(10):
                print('nop')
        next = tok.get_name(origin, relativize, relativize_to)
        windows = Bitmap.from_text(tok)
        return cls(rdclass, rdtype, next, windows)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            return 10
        self.next.to_wire(file, None, origin, False)
        Bitmap(self.windows).to_wire(file)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            return 10
        next = parser.get_name(origin)
        bitmap = Bitmap.from_wire_parser(parser)
        return cls(rdclass, rdtype, next, bitmap)