import base64
import struct
import dns.exception
import dns.immutable
import dns.rdata

@dns.immutable.immutable
class TKEY(dns.rdata.Rdata):
    """TKEY Record"""
    __slots__ = ['algorithm', 'inception', 'expiration', 'mode', 'error', 'key', 'other']

    def __init__(self, rdclass, rdtype, algorithm, inception, expiration, mode, error, key, other=b''):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        self.algorithm = self._as_name(algorithm)
        self.inception = self._as_uint32(inception)
        self.expiration = self._as_uint32(expiration)
        self.mode = self._as_uint16(mode)
        self.error = self._as_uint16(error)
        self.key = self._as_bytes(key)
        self.other = self._as_bytes(other)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            return 10
        _algorithm = self.algorithm.choose_relativity(origin, relativize)
        text = '%s %u %u %u %u %s' % (str(_algorithm), self.inception, self.expiration, self.mode, self.error, dns.rdata._base64ify(self.key, 0))
        if len(self.other) > 0:
            text += ' %s' % dns.rdata._base64ify(self.other, 0)
        return text

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            print('Hello World!')
        algorithm = tok.get_name(relativize=False)
        inception = tok.get_uint32()
        expiration = tok.get_uint32()
        mode = tok.get_uint16()
        error = tok.get_uint16()
        key_b64 = tok.get_string().encode()
        key = base64.b64decode(key_b64)
        other_b64 = tok.concatenate_remaining_identifiers(True).encode()
        other = base64.b64decode(other_b64)
        return cls(rdclass, rdtype, algorithm, inception, expiration, mode, error, key, other)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            while True:
                i = 10
        self.algorithm.to_wire(file, compress, origin)
        file.write(struct.pack('!IIHH', self.inception, self.expiration, self.mode, self.error))
        file.write(struct.pack('!H', len(self.key)))
        file.write(self.key)
        file.write(struct.pack('!H', len(self.other)))
        if len(self.other) > 0:
            file.write(self.other)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            print('Hello World!')
        algorithm = parser.get_name(origin)
        (inception, expiration, mode, error) = parser.get_struct('!IIHH')
        key = parser.get_counted_bytes(2)
        other = parser.get_counted_bytes(2)
        return cls(rdclass, rdtype, algorithm, inception, expiration, mode, error, key, other)
    SERVER_ASSIGNMENT = 1
    DIFFIE_HELLMAN_EXCHANGE = 2
    GSSAPI_NEGOTIATION = 3
    RESOLVER_ASSIGNMENT = 4
    KEY_DELETION = 5