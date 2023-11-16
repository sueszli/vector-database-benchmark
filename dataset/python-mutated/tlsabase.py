import binascii
import struct
import dns.immutable
import dns.rdata
import dns.rdatatype

@dns.immutable.immutable
class TLSABase(dns.rdata.Rdata):
    """Base class for TLSA and SMIMEA records"""
    __slots__ = ['usage', 'selector', 'mtype', 'cert']

    def __init__(self, rdclass, rdtype, usage, selector, mtype, cert):
        if False:
            return 10
        super().__init__(rdclass, rdtype)
        self.usage = self._as_uint8(usage)
        self.selector = self._as_uint8(selector)
        self.mtype = self._as_uint8(mtype)
        self.cert = self._as_bytes(cert)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        kw = kw.copy()
        chunksize = kw.pop('chunksize', 128)
        return '%d %d %d %s' % (self.usage, self.selector, self.mtype, dns.rdata._hexify(self.cert, chunksize=chunksize, **kw))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            for i in range(10):
                print('nop')
        usage = tok.get_uint8()
        selector = tok.get_uint8()
        mtype = tok.get_uint8()
        cert = tok.concatenate_remaining_identifiers().encode()
        cert = binascii.unhexlify(cert)
        return cls(rdclass, rdtype, usage, selector, mtype, cert)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        header = struct.pack('!BBB', self.usage, self.selector, self.mtype)
        file.write(header)
        file.write(self.cert)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            for i in range(10):
                print('nop')
        header = parser.get_struct('BBB')
        cert = parser.get_remaining()
        return cls(rdclass, rdtype, header[0], header[1], header[2], cert)