import base64
import struct
import dns.dnssectypes
import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer
_ctype_by_value = {1: 'PKIX', 2: 'SPKI', 3: 'PGP', 4: 'IPKIX', 5: 'ISPKI', 6: 'IPGP', 7: 'ACPKIX', 8: 'IACPKIX', 253: 'URI', 254: 'OID'}
_ctype_by_name = {'PKIX': 1, 'SPKI': 2, 'PGP': 3, 'IPKIX': 4, 'ISPKI': 5, 'IPGP': 6, 'ACPKIX': 7, 'IACPKIX': 8, 'URI': 253, 'OID': 254}

def _ctype_from_text(what):
    if False:
        i = 10
        return i + 15
    v = _ctype_by_name.get(what)
    if v is not None:
        return v
    return int(what)

def _ctype_to_text(what):
    if False:
        return 10
    v = _ctype_by_value.get(what)
    if v is not None:
        return v
    return str(what)

@dns.immutable.immutable
class CERT(dns.rdata.Rdata):
    """CERT record"""
    __slots__ = ['certificate_type', 'key_tag', 'algorithm', 'certificate']

    def __init__(self, rdclass, rdtype, certificate_type, key_tag, algorithm, certificate):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        self.certificate_type = self._as_uint16(certificate_type)
        self.key_tag = self._as_uint16(key_tag)
        self.algorithm = self._as_uint8(algorithm)
        self.certificate = self._as_bytes(certificate)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        certificate_type = _ctype_to_text(self.certificate_type)
        return '%s %d %s %s' % (certificate_type, self.key_tag, dns.dnssectypes.Algorithm.to_text(self.algorithm), dns.rdata._base64ify(self.certificate, **kw))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            while True:
                i = 10
        certificate_type = _ctype_from_text(tok.get_string())
        key_tag = tok.get_uint16()
        algorithm = dns.dnssectypes.Algorithm.from_text(tok.get_string())
        b64 = tok.concatenate_remaining_identifiers().encode()
        certificate = base64.b64decode(b64)
        return cls(rdclass, rdtype, certificate_type, key_tag, algorithm, certificate)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            while True:
                i = 10
        prefix = struct.pack('!HHB', self.certificate_type, self.key_tag, self.algorithm)
        file.write(prefix)
        file.write(self.certificate)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            i = 10
            return i + 15
        (certificate_type, key_tag, algorithm) = parser.get_struct('!HHB')
        certificate = parser.get_remaining()
        return cls(rdclass, rdtype, certificate_type, key_tag, algorithm, certificate)