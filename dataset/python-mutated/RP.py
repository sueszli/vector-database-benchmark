import dns.exception
import dns.immutable
import dns.name
import dns.rdata

@dns.immutable.immutable
class RP(dns.rdata.Rdata):
    """RP record"""
    __slots__ = ['mbox', 'txt']

    def __init__(self, rdclass, rdtype, mbox, txt):
        if False:
            while True:
                i = 10
        super().__init__(rdclass, rdtype)
        self.mbox = self._as_name(mbox)
        self.txt = self._as_name(txt)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            i = 10
            return i + 15
        mbox = self.mbox.choose_relativity(origin, relativize)
        txt = self.txt.choose_relativity(origin, relativize)
        return '{} {}'.format(str(mbox), str(txt))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            return 10
        mbox = tok.get_name(origin, relativize, relativize_to)
        txt = tok.get_name(origin, relativize, relativize_to)
        return cls(rdclass, rdtype, mbox, txt)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            i = 10
            return i + 15
        self.mbox.to_wire(file, None, origin, canonicalize)
        self.txt.to_wire(file, None, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            print('Hello World!')
        mbox = parser.get_name(origin)
        txt = parser.get_name(origin)
        return cls(rdclass, rdtype, mbox, txt)