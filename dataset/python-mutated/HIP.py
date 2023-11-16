import base64
import binascii
import struct
import dns.exception
import dns.immutable
import dns.rdata
import dns.rdatatype

@dns.immutable.immutable
class HIP(dns.rdata.Rdata):
    """HIP record"""
    __slots__ = ['hit', 'algorithm', 'key', 'servers']

    def __init__(self, rdclass, rdtype, hit, algorithm, key, servers):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        self.hit = self._as_bytes(hit, True, 255)
        self.algorithm = self._as_uint8(algorithm)
        self.key = self._as_bytes(key, True)
        self.servers = self._as_tuple(servers, self._as_name)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        hit = binascii.hexlify(self.hit).decode()
        key = base64.b64encode(self.key).replace(b'\n', b'').decode()
        text = ''
        servers = []
        for server in self.servers:
            servers.append(server.choose_relativity(origin, relativize))
        if len(servers) > 0:
            text += ' ' + ' '.join((x.to_unicode() for x in servers))
        return '%u %s %s%s' % (self.algorithm, hit, key, text)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            for i in range(10):
                print('nop')
        algorithm = tok.get_uint8()
        hit = binascii.unhexlify(tok.get_string().encode())
        key = base64.b64decode(tok.get_string().encode())
        servers = []
        for token in tok.get_remaining():
            server = tok.as_name(token, origin, relativize, relativize_to)
            servers.append(server)
        return cls(rdclass, rdtype, hit, algorithm, key, servers)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        lh = len(self.hit)
        lk = len(self.key)
        file.write(struct.pack('!BBH', lh, self.algorithm, lk))
        file.write(self.hit)
        file.write(self.key)
        for server in self.servers:
            server.to_wire(file, None, origin, False)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            print('Hello World!')
        (lh, algorithm, lk) = parser.get_struct('!BBH')
        hit = parser.get_bytes(lh)
        key = parser.get_bytes(lk)
        servers = []
        while parser.remaining() > 0:
            server = parser.get_name(origin)
            servers.append(server)
        return cls(rdclass, rdtype, hit, algorithm, key, servers)