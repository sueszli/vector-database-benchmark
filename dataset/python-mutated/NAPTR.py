import struct
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdtypes.util

def _write_string(file, s):
    if False:
        print('Hello World!')
    l = len(s)
    assert l < 256
    file.write(struct.pack('!B', l))
    file.write(s)

@dns.immutable.immutable
class NAPTR(dns.rdata.Rdata):
    """NAPTR record"""
    __slots__ = ['order', 'preference', 'flags', 'service', 'regexp', 'replacement']

    def __init__(self, rdclass, rdtype, order, preference, flags, service, regexp, replacement):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        self.flags = self._as_bytes(flags, True, 255)
        self.service = self._as_bytes(service, True, 255)
        self.regexp = self._as_bytes(regexp, True, 255)
        self.order = self._as_uint16(order)
        self.preference = self._as_uint16(preference)
        self.replacement = self._as_name(replacement)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            for i in range(10):
                print('nop')
        replacement = self.replacement.choose_relativity(origin, relativize)
        return '%d %d "%s" "%s" "%s" %s' % (self.order, self.preference, dns.rdata._escapify(self.flags), dns.rdata._escapify(self.service), dns.rdata._escapify(self.regexp), replacement)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            return 10
        order = tok.get_uint16()
        preference = tok.get_uint16()
        flags = tok.get_string()
        service = tok.get_string()
        regexp = tok.get_string()
        replacement = tok.get_name(origin, relativize, relativize_to)
        return cls(rdclass, rdtype, order, preference, flags, service, regexp, replacement)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        two_ints = struct.pack('!HH', self.order, self.preference)
        file.write(two_ints)
        _write_string(file, self.flags)
        _write_string(file, self.service)
        _write_string(file, self.regexp)
        self.replacement.to_wire(file, compress, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            while True:
                i = 10
        (order, preference) = parser.get_struct('!HH')
        strings = []
        for _ in range(3):
            s = parser.get_counted_bytes()
            strings.append(s)
        replacement = parser.get_name(origin)
        return cls(rdclass, rdtype, order, preference, strings[0], strings[1], strings[2], replacement)

    def _processing_priority(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.order, self.preference)

    @classmethod
    def _processing_order(cls, iterable):
        if False:
            print('Hello World!')
        return dns.rdtypes.util.priority_processing_order(iterable)