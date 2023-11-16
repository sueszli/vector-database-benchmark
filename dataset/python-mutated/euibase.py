import binascii
import dns.immutable
import dns.rdata

@dns.immutable.immutable
class EUIBase(dns.rdata.Rdata):
    """EUIxx record"""
    __slots__ = ['eui']

    def __init__(self, rdclass, rdtype, eui):
        if False:
            while True:
                i = 10
        super().__init__(rdclass, rdtype)
        self.eui = self._as_bytes(eui)
        if len(self.eui) != self.byte_len:
            raise dns.exception.FormError('EUI%s rdata has to have %s bytes' % (self.byte_len * 8, self.byte_len))

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            return 10
        return dns.rdata._hexify(self.eui, chunksize=2, separator=b'-', **kw)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            print('Hello World!')
        text = tok.get_string()
        if len(text) != cls.text_len:
            raise dns.exception.SyntaxError('Input text must have %s characters' % cls.text_len)
        for i in range(2, cls.byte_len * 3 - 1, 3):
            if text[i] != '-':
                raise dns.exception.SyntaxError('Dash expected at position %s' % i)
        text = text.replace('-', '')
        try:
            data = binascii.unhexlify(text.encode())
        except (ValueError, TypeError) as ex:
            raise dns.exception.SyntaxError('Hex decoding error: %s' % str(ex))
        return cls(rdclass, rdtype, data)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            print('Hello World!')
        file.write(self.eui)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            while True:
                i = 10
        eui = parser.get_bytes(cls.byte_len)
        return cls(rdclass, rdtype, eui)