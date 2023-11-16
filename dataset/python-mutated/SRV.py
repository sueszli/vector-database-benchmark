import struct
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdtypes.util

@dns.immutable.immutable
class SRV(dns.rdata.Rdata):
    """SRV record"""
    __slots__ = ['priority', 'weight', 'port', 'target']

    def __init__(self, rdclass, rdtype, priority, weight, port, target):
        if False:
            return 10
        super().__init__(rdclass, rdtype)
        self.priority = self._as_uint16(priority)
        self.weight = self._as_uint16(weight)
        self.port = self._as_uint16(port)
        self.target = self._as_name(target)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            while True:
                i = 10
        target = self.target.choose_relativity(origin, relativize)
        return '%d %d %d %s' % (self.priority, self.weight, self.port, target)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            i = 10
            return i + 15
        priority = tok.get_uint16()
        weight = tok.get_uint16()
        port = tok.get_uint16()
        target = tok.get_name(origin, relativize, relativize_to)
        return cls(rdclass, rdtype, priority, weight, port, target)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        three_ints = struct.pack('!HHH', self.priority, self.weight, self.port)
        file.write(three_ints)
        self.target.to_wire(file, compress, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            i = 10
            return i + 15
        (priority, weight, port) = parser.get_struct('!HHH')
        target = parser.get_name(origin)
        return cls(rdclass, rdtype, priority, weight, port, target)

    def _processing_priority(self):
        if False:
            while True:
                i = 10
        return self.priority

    def _processing_weight(self):
        if False:
            while True:
                i = 10
        return self.weight

    @classmethod
    def _processing_order(cls, iterable):
        if False:
            print('Hello World!')
        return dns.rdtypes.util.weighted_processing_order(iterable)