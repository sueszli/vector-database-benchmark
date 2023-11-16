import binascii
import struct
import dns.immutable
import dns.rdata
import dns.rdatatype

@dns.immutable.immutable
class SSHFP(dns.rdata.Rdata):
    """SSHFP record"""
    __slots__ = ['algorithm', 'fp_type', 'fingerprint']

    def __init__(self, rdclass, rdtype, algorithm, fp_type, fingerprint):
        if False:
            while True:
                i = 10
        super().__init__(rdclass, rdtype)
        self.algorithm = self._as_uint8(algorithm)
        self.fp_type = self._as_uint8(fp_type)
        self.fingerprint = self._as_bytes(fingerprint, True)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        kw = kw.copy()
        chunksize = kw.pop('chunksize', 128)
        return '%d %d %s' % (self.algorithm, self.fp_type, dns.rdata._hexify(self.fingerprint, chunksize=chunksize, **kw))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            while True:
                i = 10
        algorithm = tok.get_uint8()
        fp_type = tok.get_uint8()
        fingerprint = tok.concatenate_remaining_identifiers().encode()
        fingerprint = binascii.unhexlify(fingerprint)
        return cls(rdclass, rdtype, algorithm, fp_type, fingerprint)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        header = struct.pack('!BB', self.algorithm, self.fp_type)
        file.write(header)
        file.write(self.fingerprint)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            print('Hello World!')
        header = parser.get_struct('BB')
        fingerprint = parser.get_remaining()
        return cls(rdclass, rdtype, header[0], header[1], fingerprint)