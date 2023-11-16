import struct
import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer

@dns.immutable.immutable
class HINFO(dns.rdata.Rdata):
    """HINFO record"""
    __slots__ = ['cpu', 'os']

    def __init__(self, rdclass, rdtype, cpu, os):
        if False:
            while True:
                i = 10
        super().__init__(rdclass, rdtype)
        self.cpu = self._as_bytes(cpu, True, 255)
        self.os = self._as_bytes(os, True, 255)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            while True:
                i = 10
        return '"{}" "{}"'.format(dns.rdata._escapify(self.cpu), dns.rdata._escapify(self.os))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            while True:
                i = 10
        cpu = tok.get_string(max_length=255)
        os = tok.get_string(max_length=255)
        return cls(rdclass, rdtype, cpu, os)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        l = len(self.cpu)
        assert l < 256
        file.write(struct.pack('!B', l))
        file.write(self.cpu)
        l = len(self.os)
        assert l < 256
        file.write(struct.pack('!B', l))
        file.write(self.os)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            for i in range(10):
                print('nop')
        cpu = parser.get_counted_bytes()
        os = parser.get_counted_bytes()
        return cls(rdclass, rdtype, cpu, os)