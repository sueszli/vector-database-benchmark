import struct
import dns.exception
import dns.immutable
import dns.name
import dns.rdata

@dns.immutable.immutable
class SOA(dns.rdata.Rdata):
    """SOA record"""
    __slots__ = ['mname', 'rname', 'serial', 'refresh', 'retry', 'expire', 'minimum']

    def __init__(self, rdclass, rdtype, mname, rname, serial, refresh, retry, expire, minimum):
        if False:
            print('Hello World!')
        super().__init__(rdclass, rdtype)
        self.mname = self._as_name(mname)
        self.rname = self._as_name(rname)
        self.serial = self._as_uint32(serial)
        self.refresh = self._as_ttl(refresh)
        self.retry = self._as_ttl(retry)
        self.expire = self._as_ttl(expire)
        self.minimum = self._as_ttl(minimum)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            i = 10
            return i + 15
        mname = self.mname.choose_relativity(origin, relativize)
        rname = self.rname.choose_relativity(origin, relativize)
        return '%s %s %d %d %d %d %d' % (mname, rname, self.serial, self.refresh, self.retry, self.expire, self.minimum)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            for i in range(10):
                print('nop')
        mname = tok.get_name(origin, relativize, relativize_to)
        rname = tok.get_name(origin, relativize, relativize_to)
        serial = tok.get_uint32()
        refresh = tok.get_ttl()
        retry = tok.get_ttl()
        expire = tok.get_ttl()
        minimum = tok.get_ttl()
        return cls(rdclass, rdtype, mname, rname, serial, refresh, retry, expire, minimum)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        self.mname.to_wire(file, compress, origin, canonicalize)
        self.rname.to_wire(file, compress, origin, canonicalize)
        five_ints = struct.pack('!IIIII', self.serial, self.refresh, self.retry, self.expire, self.minimum)
        file.write(five_ints)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            print('Hello World!')
        mname = parser.get_name(origin)
        rname = parser.get_name(origin)
        return cls(rdclass, rdtype, mname, rname, *parser.get_struct('!IIIII'))