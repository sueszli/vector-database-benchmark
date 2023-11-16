import binascii
import struct
import dns.immutable
import dns.rdata
import dns.rdatatype
import dns.zonetypes

@dns.immutable.immutable
class ZONEMD(dns.rdata.Rdata):
    """ZONEMD record"""
    __slots__ = ['serial', 'scheme', 'hash_algorithm', 'digest']

    def __init__(self, rdclass, rdtype, serial, scheme, hash_algorithm, digest):
        if False:
            while True:
                i = 10
        super().__init__(rdclass, rdtype)
        self.serial = self._as_uint32(serial)
        self.scheme = dns.zonetypes.DigestScheme.make(scheme)
        self.hash_algorithm = dns.zonetypes.DigestHashAlgorithm.make(hash_algorithm)
        self.digest = self._as_bytes(digest)
        if self.scheme == 0:
            raise ValueError('scheme 0 is reserved')
        if self.hash_algorithm == 0:
            raise ValueError('hash_algorithm 0 is reserved')
        hasher = dns.zonetypes._digest_hashers.get(self.hash_algorithm)
        if hasher and hasher().digest_size != len(self.digest):
            raise ValueError('digest length inconsistent with hash algorithm')

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            for i in range(10):
                print('nop')
        kw = kw.copy()
        chunksize = kw.pop('chunksize', 128)
        return '%d %d %d %s' % (self.serial, self.scheme, self.hash_algorithm, dns.rdata._hexify(self.digest, chunksize=chunksize, **kw))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            print('Hello World!')
        serial = tok.get_uint32()
        scheme = tok.get_uint8()
        hash_algorithm = tok.get_uint8()
        digest = tok.concatenate_remaining_identifiers().encode()
        digest = binascii.unhexlify(digest)
        return cls(rdclass, rdtype, serial, scheme, hash_algorithm, digest)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            while True:
                i = 10
        header = struct.pack('!IBB', self.serial, self.scheme, self.hash_algorithm)
        file.write(header)
        file.write(self.digest)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            for i in range(10):
                print('nop')
        header = parser.get_struct('!IBB')
        digest = parser.get_remaining()
        return cls(rdclass, rdtype, header[0], header[1], header[2], digest)