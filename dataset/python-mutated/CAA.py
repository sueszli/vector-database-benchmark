import struct
import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer

@dns.immutable.immutable
class CAA(dns.rdata.Rdata):
    """CAA (Certification Authority Authorization) record"""
    __slots__ = ['flags', 'tag', 'value']

    def __init__(self, rdclass, rdtype, flags, tag, value):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        self.flags = self._as_uint8(flags)
        self.tag = self._as_bytes(tag, True, 255)
        if not tag.isalnum():
            raise ValueError('tag is not alphanumeric')
        self.value = self._as_bytes(value)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            return 10
        return '%u %s "%s"' % (self.flags, dns.rdata._escapify(self.tag), dns.rdata._escapify(self.value))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            print('Hello World!')
        flags = tok.get_uint8()
        tag = tok.get_string().encode()
        value = tok.get_string().encode()
        return cls(rdclass, rdtype, flags, tag, value)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            return 10
        file.write(struct.pack('!B', self.flags))
        l = len(self.tag)
        assert l < 256
        file.write(struct.pack('!B', l))
        file.write(self.tag)
        file.write(self.value)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            return 10
        flags = parser.get_uint8()
        tag = parser.get_counted_bytes()
        value = parser.get_remaining()
        return cls(rdclass, rdtype, flags, tag, value)