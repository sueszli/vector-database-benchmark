import binascii
import codecs
import struct
import dns.exception
import dns.immutable
import dns.ipv4
import dns.ipv6
import dns.rdata
import dns.tokenizer

@dns.immutable.immutable
class APLItem:
    """An APL list item."""
    __slots__ = ['family', 'negation', 'address', 'prefix']

    def __init__(self, family, negation, address, prefix):
        if False:
            print('Hello World!')
        self.family = dns.rdata.Rdata._as_uint16(family)
        self.negation = dns.rdata.Rdata._as_bool(negation)
        if self.family == 1:
            self.address = dns.rdata.Rdata._as_ipv4_address(address)
            self.prefix = dns.rdata.Rdata._as_int(prefix, 0, 32)
        elif self.family == 2:
            self.address = dns.rdata.Rdata._as_ipv6_address(address)
            self.prefix = dns.rdata.Rdata._as_int(prefix, 0, 128)
        else:
            self.address = dns.rdata.Rdata._as_bytes(address, max_length=127)
            self.prefix = dns.rdata.Rdata._as_uint8(prefix)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if self.negation:
            return '!%d:%s/%s' % (self.family, self.address, self.prefix)
        else:
            return '%d:%s/%s' % (self.family, self.address, self.prefix)

    def to_wire(self, file):
        if False:
            while True:
                i = 10
        if self.family == 1:
            address = dns.ipv4.inet_aton(self.address)
        elif self.family == 2:
            address = dns.ipv6.inet_aton(self.address)
        else:
            address = binascii.unhexlify(self.address)
        last = 0
        for i in range(len(address) - 1, -1, -1):
            if address[i] != 0:
                last = i + 1
                break
        address = address[0:last]
        l = len(address)
        assert l < 128
        if self.negation:
            l |= 128
        header = struct.pack('!HBB', self.family, self.prefix, l)
        file.write(header)
        file.write(address)

@dns.immutable.immutable
class APL(dns.rdata.Rdata):
    """APL record."""
    __slots__ = ['items']

    def __init__(self, rdclass, rdtype, items):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        for item in items:
            if not isinstance(item, APLItem):
                raise ValueError('item not an APLItem')
        self.items = tuple(items)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            print('Hello World!')
        return ' '.join(map(str, self.items))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            while True:
                i = 10
        items = []
        for token in tok.get_remaining():
            item = token.unescape().value
            if item[0] == '!':
                negation = True
                item = item[1:]
            else:
                negation = False
            (family, rest) = item.split(':', 1)
            family = int(family)
            (address, prefix) = rest.split('/', 1)
            prefix = int(prefix)
            item = APLItem(family, negation, address, prefix)
            items.append(item)
        return cls(rdclass, rdtype, items)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            print('Hello World!')
        for item in self.items:
            item.to_wire(file)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            for i in range(10):
                print('nop')
        items = []
        while parser.remaining() > 0:
            header = parser.get_struct('!HBB')
            afdlen = header[2]
            if afdlen > 127:
                negation = True
                afdlen -= 128
            else:
                negation = False
            address = parser.get_bytes(afdlen)
            l = len(address)
            if header[0] == 1:
                if l < 4:
                    address += b'\x00' * (4 - l)
            elif header[0] == 2:
                if l < 16:
                    address += b'\x00' * (16 - l)
            else:
                address = codecs.encode(address, 'hex_codec')
            item = APLItem(header[0], negation, address, header[1])
            items.append(item)
        return cls(rdclass, rdtype, items)