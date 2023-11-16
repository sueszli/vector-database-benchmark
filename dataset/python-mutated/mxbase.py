"""MX-like base classes."""
import struct
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdtypes.util

@dns.immutable.immutable
class MXBase(dns.rdata.Rdata):
    """Base class for rdata that is like an MX record."""
    __slots__ = ['preference', 'exchange']

    def __init__(self, rdclass, rdtype, preference, exchange):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        self.preference = self._as_uint16(preference)
        self.exchange = self._as_name(exchange)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            for i in range(10):
                print('nop')
        exchange = self.exchange.choose_relativity(origin, relativize)
        return '%d %s' % (self.preference, exchange)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            return 10
        preference = tok.get_uint16()
        exchange = tok.get_name(origin, relativize, relativize_to)
        return cls(rdclass, rdtype, preference, exchange)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            while True:
                i = 10
        pref = struct.pack('!H', self.preference)
        file.write(pref)
        self.exchange.to_wire(file, compress, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            i = 10
            return i + 15
        preference = parser.get_uint16()
        exchange = parser.get_name(origin)
        return cls(rdclass, rdtype, preference, exchange)

    def _processing_priority(self):
        if False:
            print('Hello World!')
        return self.preference

    @classmethod
    def _processing_order(cls, iterable):
        if False:
            print('Hello World!')
        return dns.rdtypes.util.priority_processing_order(iterable)

@dns.immutable.immutable
class UncompressedMX(MXBase):
    """Base class for rdata that is like an MX record, but whose name
    is not compressed when converted to DNS wire format, and whose
    digestable form is not downcased."""

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        super()._to_wire(file, None, origin, False)

@dns.immutable.immutable
class UncompressedDowncasingMX(MXBase):
    """Base class for rdata that is like an MX record, but whose name
    is not compressed when convert to DNS wire format."""

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        super()._to_wire(file, None, origin, canonicalize)