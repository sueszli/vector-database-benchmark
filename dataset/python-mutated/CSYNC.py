import struct
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdatatype
import dns.rdtypes.util

@dns.immutable.immutable
class Bitmap(dns.rdtypes.util.Bitmap):
    type_name = 'CSYNC'

@dns.immutable.immutable
class CSYNC(dns.rdata.Rdata):
    """CSYNC record"""
    __slots__ = ['serial', 'flags', 'windows']

    def __init__(self, rdclass, rdtype, serial, flags, windows):
        if False:
            while True:
                i = 10
        super().__init__(rdclass, rdtype)
        self.serial = self._as_uint32(serial)
        self.flags = self._as_uint16(flags)
        if not isinstance(windows, Bitmap):
            windows = Bitmap(windows)
        self.windows = tuple(windows.windows)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        text = Bitmap(self.windows).to_text()
        return '%d %d%s' % (self.serial, self.flags, text)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            i = 10
            return i + 15
        serial = tok.get_uint32()
        flags = tok.get_uint16()
        bitmap = Bitmap.from_text(tok)
        return cls(rdclass, rdtype, serial, flags, bitmap)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            print('Hello World!')
        file.write(struct.pack('!IH', self.serial, self.flags))
        Bitmap(self.windows).to_wire(file)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            i = 10
            return i + 15
        (serial, flags) = parser.get_struct('!IH')
        bitmap = Bitmap.from_wire_parser(parser)
        return cls(rdclass, rdtype, serial, flags, bitmap)