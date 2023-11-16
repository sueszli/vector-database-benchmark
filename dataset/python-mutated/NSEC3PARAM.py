import binascii
import struct
import dns.exception
import dns.immutable
import dns.rdata

@dns.immutable.immutable
class NSEC3PARAM(dns.rdata.Rdata):
    """NSEC3PARAM record"""
    __slots__ = ['algorithm', 'flags', 'iterations', 'salt']

    def __init__(self, rdclass, rdtype, algorithm, flags, iterations, salt):
        if False:
            print('Hello World!')
        super().__init__(rdclass, rdtype)
        self.algorithm = self._as_uint8(algorithm)
        self.flags = self._as_uint8(flags)
        self.iterations = self._as_uint16(iterations)
        self.salt = self._as_bytes(salt, True, 255)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            i = 10
            return i + 15
        if self.salt == b'':
            salt = '-'
        else:
            salt = binascii.hexlify(self.salt).decode()
        return '%u %u %u %s' % (self.algorithm, self.flags, self.iterations, salt)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            while True:
                i = 10
        algorithm = tok.get_uint8()
        flags = tok.get_uint8()
        iterations = tok.get_uint16()
        salt = tok.get_string()
        if salt == '-':
            salt = ''
        else:
            salt = binascii.unhexlify(salt.encode())
        return cls(rdclass, rdtype, algorithm, flags, iterations, salt)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            return 10
        l = len(self.salt)
        file.write(struct.pack('!BBHB', self.algorithm, self.flags, self.iterations, l))
        file.write(self.salt)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            print('Hello World!')
        (algorithm, flags, iterations) = parser.get_struct('!BBH')
        salt = parser.get_counted_bytes()
        return cls(rdclass, rdtype, algorithm, flags, iterations, salt)