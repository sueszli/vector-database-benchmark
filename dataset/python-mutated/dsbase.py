import binascii
import struct
import dns.dnssectypes
import dns.immutable
import dns.rdata
import dns.rdatatype

@dns.immutable.immutable
class DSBase(dns.rdata.Rdata):
    """Base class for rdata that is like a DS record"""
    __slots__ = ['key_tag', 'algorithm', 'digest_type', 'digest']
    _digest_length_by_type = {1: 20, 2: 32, 3: 32, 4: 48}

    def __init__(self, rdclass, rdtype, key_tag, algorithm, digest_type, digest):
        if False:
            while True:
                i = 10
        super().__init__(rdclass, rdtype)
        self.key_tag = self._as_uint16(key_tag)
        self.algorithm = dns.dnssectypes.Algorithm.make(algorithm)
        self.digest_type = dns.dnssectypes.DSDigest.make(self._as_uint8(digest_type))
        self.digest = self._as_bytes(digest)
        try:
            if len(self.digest) != self._digest_length_by_type[self.digest_type]:
                raise ValueError('digest length inconsistent with digest type')
        except KeyError:
            if self.digest_type == 0:
                raise ValueError('digest type 0 is reserved')

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            i = 10
            return i + 15
        kw = kw.copy()
        chunksize = kw.pop('chunksize', 128)
        return '%d %d %d %s' % (self.key_tag, self.algorithm, self.digest_type, dns.rdata._hexify(self.digest, chunksize=chunksize, **kw))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            i = 10
            return i + 15
        key_tag = tok.get_uint16()
        algorithm = tok.get_string()
        digest_type = tok.get_uint8()
        digest = tok.concatenate_remaining_identifiers().encode()
        digest = binascii.unhexlify(digest)
        return cls(rdclass, rdtype, key_tag, algorithm, digest_type, digest)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            print('Hello World!')
        header = struct.pack('!HBB', self.key_tag, self.algorithm, self.digest_type)
        file.write(header)
        file.write(self.digest)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            return 10
        header = parser.get_struct('!HBB')
        digest = parser.get_remaining()
        return cls(rdclass, rdtype, header[0], header[1], header[2], digest)