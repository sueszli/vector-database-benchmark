"""EDNS Options"""
import math
import socket
import struct
from typing import Any, Dict, Optional, Union
import dns.enum
import dns.inet
import dns.rdata
import dns.wire

class OptionType(dns.enum.IntEnum):
    NSID = 3
    DAU = 5
    DHU = 6
    N3U = 7
    ECS = 8
    EXPIRE = 9
    COOKIE = 10
    KEEPALIVE = 11
    PADDING = 12
    CHAIN = 13
    EDE = 15

    @classmethod
    def _maximum(cls):
        if False:
            print('Hello World!')
        return 65535

class Option:
    """Base class for all EDNS option types."""

    def __init__(self, otype: Union[OptionType, str]):
        if False:
            return 10
        'Initialize an option.\n\n        *otype*, a ``dns.edns.OptionType``, is the option type.\n        '
        self.otype = OptionType.make(otype)

    def to_wire(self, file: Optional[Any]=None) -> Optional[bytes]:
        if False:
            for i in range(10):
                print('nop')
        'Convert an option to wire format.\n\n        Returns a ``bytes`` or ``None``.\n\n        '
        raise NotImplementedError

    @classmethod
    def from_wire_parser(cls, otype: OptionType, parser: 'dns.wire.Parser') -> 'Option':
        if False:
            print('Hello World!')
        'Build an EDNS option object from wire format.\n\n        *otype*, a ``dns.edns.OptionType``, is the option type.\n\n        *parser*, a ``dns.wire.Parser``, the parser, which should be\n        restructed to the option length.\n\n        Returns a ``dns.edns.Option``.\n        '
        raise NotImplementedError

    def _cmp(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Compare an EDNS option with another option of the same type.\n\n        Returns < 0 if < *other*, 0 if == *other*, and > 0 if > *other*.\n        '
        wire = self.to_wire()
        owire = other.to_wire()
        if wire == owire:
            return 0
        if wire > owire:
            return 1
        return -1

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Option):
            return False
        if self.otype != other.otype:
            return False
        return self._cmp(other) == 0

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Option):
            return True
        if self.otype != other.otype:
            return True
        return self._cmp(other) != 0

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Option) or self.otype != other.otype:
            return NotImplemented
        return self._cmp(other) < 0

    def __le__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Option) or self.otype != other.otype:
            return NotImplemented
        return self._cmp(other) <= 0

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Option) or self.otype != other.otype:
            return NotImplemented
        return self._cmp(other) >= 0

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Option) or self.otype != other.otype:
            return NotImplemented
        return self._cmp(other) > 0

    def __str__(self):
        if False:
            return 10
        return self.to_text()

class GenericOption(Option):
    """Generic Option Class

    This class is used for EDNS option types for which we have no better
    implementation.
    """

    def __init__(self, otype: Union[OptionType, str], data: Union[bytes, str]):
        if False:
            while True:
                i = 10
        super().__init__(otype)
        self.data = dns.rdata.Rdata._as_bytes(data, True)

    def to_wire(self, file: Optional[Any]=None) -> Optional[bytes]:
        if False:
            i = 10
            return i + 15
        if file:
            file.write(self.data)
            return None
        else:
            return self.data

    def to_text(self) -> str:
        if False:
            print('Hello World!')
        return 'Generic %d' % self.otype

    @classmethod
    def from_wire_parser(cls, otype: Union[OptionType, str], parser: 'dns.wire.Parser') -> Option:
        if False:
            print('Hello World!')
        return cls(otype, parser.get_remaining())

class ECSOption(Option):
    """EDNS Client Subnet (ECS, RFC7871)"""

    def __init__(self, address: str, srclen: Optional[int]=None, scopelen: int=0):
        if False:
            while True:
                i = 10
        '*address*, a ``str``, is the client address information.\n\n        *srclen*, an ``int``, the source prefix length, which is the\n        leftmost number of bits of the address to be used for the\n        lookup.  The default is 24 for IPv4 and 56 for IPv6.\n\n        *scopelen*, an ``int``, the scope prefix length.  This value\n        must be 0 in queries, and should be set in responses.\n        '
        super().__init__(OptionType.ECS)
        af = dns.inet.af_for_address(address)
        if af == socket.AF_INET6:
            self.family = 2
            if srclen is None:
                srclen = 56
            address = dns.rdata.Rdata._as_ipv6_address(address)
            srclen = dns.rdata.Rdata._as_int(srclen, 0, 128)
            scopelen = dns.rdata.Rdata._as_int(scopelen, 0, 128)
        elif af == socket.AF_INET:
            self.family = 1
            if srclen is None:
                srclen = 24
            address = dns.rdata.Rdata._as_ipv4_address(address)
            srclen = dns.rdata.Rdata._as_int(srclen, 0, 32)
            scopelen = dns.rdata.Rdata._as_int(scopelen, 0, 32)
        else:
            raise ValueError('Bad address family')
        assert srclen is not None
        self.address = address
        self.srclen = srclen
        self.scopelen = scopelen
        addrdata = dns.inet.inet_pton(af, address)
        nbytes = int(math.ceil(srclen / 8.0))
        self.addrdata = addrdata[:nbytes]
        nbits = srclen % 8
        if nbits != 0:
            last = struct.pack('B', ord(self.addrdata[-1:]) & 255 << 8 - nbits)
            self.addrdata = self.addrdata[:-1] + last

    def to_text(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'ECS {}/{} scope/{}'.format(self.address, self.srclen, self.scopelen)

    @staticmethod
    def from_text(text: str) -> Option:
        if False:
            print('Hello World!')
        "Convert a string into a `dns.edns.ECSOption`\n\n        *text*, a `str`, the text form of the option.\n\n        Returns a `dns.edns.ECSOption`.\n\n        Examples:\n\n        >>> import dns.edns\n        >>>\n        >>> # basic example\n        >>> dns.edns.ECSOption.from_text('1.2.3.4/24')\n        >>>\n        >>> # also understands scope\n        >>> dns.edns.ECSOption.from_text('1.2.3.4/24/32')\n        >>>\n        >>> # IPv6\n        >>> dns.edns.ECSOption.from_text('2001:4b98::1/64/64')\n        >>>\n        >>> # it understands results from `dns.edns.ECSOption.to_text()`\n        >>> dns.edns.ECSOption.from_text('ECS 1.2.3.4/24/32')\n        "
        optional_prefix = 'ECS'
        tokens = text.split()
        ecs_text = None
        if len(tokens) == 1:
            ecs_text = tokens[0]
        elif len(tokens) == 2:
            if tokens[0] != optional_prefix:
                raise ValueError('could not parse ECS from "{}"'.format(text))
            ecs_text = tokens[1]
        else:
            raise ValueError('could not parse ECS from "{}"'.format(text))
        n_slashes = ecs_text.count('/')
        if n_slashes == 1:
            (address, tsrclen) = ecs_text.split('/')
            tscope = '0'
        elif n_slashes == 2:
            (address, tsrclen, tscope) = ecs_text.split('/')
        else:
            raise ValueError('could not parse ECS from "{}"'.format(text))
        try:
            scope = int(tscope)
        except ValueError:
            raise ValueError('invalid scope ' + '"{}": scope must be an integer'.format(tscope))
        try:
            srclen = int(tsrclen)
        except ValueError:
            raise ValueError('invalid srclen ' + '"{}": srclen must be an integer'.format(tsrclen))
        return ECSOption(address, srclen, scope)

    def to_wire(self, file: Optional[Any]=None) -> Optional[bytes]:
        if False:
            i = 10
            return i + 15
        value = struct.pack('!HBB', self.family, self.srclen, self.scopelen) + self.addrdata
        if file:
            file.write(value)
            return None
        else:
            return value

    @classmethod
    def from_wire_parser(cls, otype: Union[OptionType, str], parser: 'dns.wire.Parser') -> Option:
        if False:
            for i in range(10):
                print('nop')
        (family, src, scope) = parser.get_struct('!HBB')
        addrlen = int(math.ceil(src / 8.0))
        prefix = parser.get_bytes(addrlen)
        if family == 1:
            pad = 4 - addrlen
            addr = dns.ipv4.inet_ntoa(prefix + b'\x00' * pad)
        elif family == 2:
            pad = 16 - addrlen
            addr = dns.ipv6.inet_ntoa(prefix + b'\x00' * pad)
        else:
            raise ValueError('unsupported family')
        return cls(addr, src, scope)

class EDECode(dns.enum.IntEnum):
    OTHER = 0
    UNSUPPORTED_DNSKEY_ALGORITHM = 1
    UNSUPPORTED_DS_DIGEST_TYPE = 2
    STALE_ANSWER = 3
    FORGED_ANSWER = 4
    DNSSEC_INDETERMINATE = 5
    DNSSEC_BOGUS = 6
    SIGNATURE_EXPIRED = 7
    SIGNATURE_NOT_YET_VALID = 8
    DNSKEY_MISSING = 9
    RRSIGS_MISSING = 10
    NO_ZONE_KEY_BIT_SET = 11
    NSEC_MISSING = 12
    CACHED_ERROR = 13
    NOT_READY = 14
    BLOCKED = 15
    CENSORED = 16
    FILTERED = 17
    PROHIBITED = 18
    STALE_NXDOMAIN_ANSWER = 19
    NOT_AUTHORITATIVE = 20
    NOT_SUPPORTED = 21
    NO_REACHABLE_AUTHORITY = 22
    NETWORK_ERROR = 23
    INVALID_DATA = 24

    @classmethod
    def _maximum(cls):
        if False:
            for i in range(10):
                print('nop')
        return 65535

class EDEOption(Option):
    """Extended DNS Error (EDE, RFC8914)"""

    def __init__(self, code: Union[EDECode, str], text: Optional[str]=None):
        if False:
            while True:
                i = 10
        '*code*, a ``dns.edns.EDECode`` or ``str``, the info code of the\n        extended error.\n\n        *text*, a ``str`` or ``None``, specifying additional information about\n        the error.\n        '
        super().__init__(OptionType.EDE)
        self.code = EDECode.make(code)
        if text is not None and (not isinstance(text, str)):
            raise ValueError('text must be string or None')
        self.text = text

    def to_text(self) -> str:
        if False:
            print('Hello World!')
        output = f'EDE {self.code}'
        if self.text is not None:
            output += f': {self.text}'
        return output

    def to_wire(self, file: Optional[Any]=None) -> Optional[bytes]:
        if False:
            i = 10
            return i + 15
        value = struct.pack('!H', self.code)
        if self.text is not None:
            value += self.text.encode('utf8')
        if file:
            file.write(value)
            return None
        else:
            return value

    @classmethod
    def from_wire_parser(cls, otype: Union[OptionType, str], parser: 'dns.wire.Parser') -> Option:
        if False:
            while True:
                i = 10
        code = EDECode.make(parser.get_uint16())
        text = parser.get_remaining()
        if text:
            if text[-1] == 0:
                text = text[:-1]
            btext = text.decode('utf8')
        else:
            btext = None
        return cls(code, btext)
_type_to_class: Dict[OptionType, Any] = {OptionType.ECS: ECSOption, OptionType.EDE: EDEOption}

def get_option_class(otype: OptionType) -> Any:
    if False:
        i = 10
        return i + 15
    'Return the class for the specified option type.\n\n    The GenericOption class is used if a more specific class is not\n    known.\n    '
    cls = _type_to_class.get(otype)
    if cls is None:
        cls = GenericOption
    return cls

def option_from_wire_parser(otype: Union[OptionType, str], parser: 'dns.wire.Parser') -> Option:
    if False:
        return 10
    'Build an EDNS option object from wire format.\n\n    *otype*, an ``int``, is the option type.\n\n    *parser*, a ``dns.wire.Parser``, the parser, which should be\n    restricted to the option length.\n\n    Returns an instance of a subclass of ``dns.edns.Option``.\n    '
    otype = OptionType.make(otype)
    cls = get_option_class(otype)
    return cls.from_wire_parser(otype, parser)

def option_from_wire(otype: Union[OptionType, str], wire: bytes, current: int, olen: int) -> Option:
    if False:
        print('Hello World!')
    'Build an EDNS option object from wire format.\n\n    *otype*, an ``int``, is the option type.\n\n    *wire*, a ``bytes``, is the wire-format message.\n\n    *current*, an ``int``, is the offset in *wire* of the beginning\n    of the rdata.\n\n    *olen*, an ``int``, is the length of the wire-format option data\n\n    Returns an instance of a subclass of ``dns.edns.Option``.\n    '
    parser = dns.wire.Parser(wire, current)
    with parser.restrict_to(olen):
        return option_from_wire_parser(otype, parser)

def register_type(implementation: Any, otype: OptionType) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Register the implementation of an option type.\n\n    *implementation*, a ``class``, is a subclass of ``dns.edns.Option``.\n\n    *otype*, an ``int``, is the option type.\n    '
    _type_to_class[otype] = implementation
NSID = OptionType.NSID
DAU = OptionType.DAU
DHU = OptionType.DHU
N3U = OptionType.N3U
ECS = OptionType.ECS
EXPIRE = OptionType.EXPIRE
COOKIE = OptionType.COOKIE
KEEPALIVE = OptionType.KEEPALIVE
PADDING = OptionType.PADDING
CHAIN = OptionType.CHAIN
EDE = OptionType.EDE