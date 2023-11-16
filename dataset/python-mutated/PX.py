import struct
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdtypes.util

@dns.immutable.immutable
class PX(dns.rdata.Rdata):
    """PX record."""
    __slots__ = ['preference', 'map822', 'mapx400']

    def __init__(self, rdclass, rdtype, preference, map822, mapx400):
        if False:
            i = 10
            return i + 15
        super().__init__(rdclass, rdtype)
        self.preference = self._as_uint16(preference)
        self.map822 = self._as_name(map822)
        self.mapx400 = self._as_name(mapx400)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            i = 10
            return i + 15
        map822 = self.map822.choose_relativity(origin, relativize)
        mapx400 = self.mapx400.choose_relativity(origin, relativize)
        return '%d %s %s' % (self.preference, map822, mapx400)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            print('Hello World!')
        preference = tok.get_uint16()
        map822 = tok.get_name(origin, relativize, relativize_to)
        mapx400 = tok.get_name(origin, relativize, relativize_to)
        return cls(rdclass, rdtype, preference, map822, mapx400)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            while True:
                i = 10
        pref = struct.pack('!H', self.preference)
        file.write(pref)
        self.map822.to_wire(file, None, origin, canonicalize)
        self.mapx400.to_wire(file, None, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            while True:
                i = 10
        preference = parser.get_uint16()
        map822 = parser.get_name(origin)
        mapx400 = parser.get_name(origin)
        return cls(rdclass, rdtype, preference, map822, mapx400)

    def _processing_priority(self):
        if False:
            return 10
        return self.preference

    @classmethod
    def _processing_order(cls, iterable):
        if False:
            while True:
                i = 10
        return dns.rdtypes.util.priority_processing_order(iterable)