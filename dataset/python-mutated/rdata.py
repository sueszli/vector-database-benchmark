"""DNS rdata."""
import base64
import binascii
import inspect
import io
import itertools
import random
from importlib import import_module
from typing import Any, Dict, Optional, Tuple, Union
import dns.exception
import dns.immutable
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.tokenizer
import dns.ttl
import dns.wire
_chunksize = 32
_allow_relative_comparisons = True

class NoRelativeRdataOrdering(dns.exception.DNSException):
    """An attempt was made to do an ordered comparison of one or more
    rdata with relative names.  The only reliable way of sorting rdata
    is to use non-relativized rdata.

    """

def _wordbreak(data, chunksize=_chunksize, separator=b' '):
    if False:
        for i in range(10):
            print('nop')
    'Break a binary string into chunks of chunksize characters separated by\n    a space.\n    '
    if not chunksize:
        return data.decode()
    return separator.join([data[i:i + chunksize] for i in range(0, len(data), chunksize)]).decode()

def _hexify(data, chunksize=_chunksize, separator=b' ', **kw):
    if False:
        i = 10
        return i + 15
    'Convert a binary string into its hex encoding, broken up into chunks\n    of chunksize characters separated by a separator.\n    '
    return _wordbreak(binascii.hexlify(data), chunksize, separator)

def _base64ify(data, chunksize=_chunksize, separator=b' ', **kw):
    if False:
        print('Hello World!')
    'Convert a binary string into its base64 encoding, broken up into chunks\n    of chunksize characters separated by a separator.\n    '
    return _wordbreak(base64.b64encode(data), chunksize, separator)
__escaped = b'"\\'

def _escapify(qstring):
    if False:
        i = 10
        return i + 15
    'Escape the characters in a quoted string which need it.'
    if isinstance(qstring, str):
        qstring = qstring.encode()
    if not isinstance(qstring, bytearray):
        qstring = bytearray(qstring)
    text = ''
    for c in qstring:
        if c in __escaped:
            text += '\\' + chr(c)
        elif c >= 32 and c < 127:
            text += chr(c)
        else:
            text += '\\%03d' % c
    return text

def _truncate_bitmap(what):
    if False:
        for i in range(10):
            print('nop')
    "Determine the index of greatest byte that isn't all zeros, and\n    return the bitmap that contains all the bytes less than that index.\n    "
    for i in range(len(what) - 1, -1, -1):
        if what[i] != 0:
            return what[0:i + 1]
    return what[0:1]
_constify = dns.immutable.constify

@dns.immutable.immutable
class Rdata:
    """Base class for all DNS rdata types."""
    __slots__ = ['rdclass', 'rdtype', 'rdcomment']

    def __init__(self, rdclass, rdtype):
        if False:
            for i in range(10):
                print('nop')
        'Initialize an rdata.\n\n        *rdclass*, an ``int`` is the rdataclass of the Rdata.\n\n        *rdtype*, an ``int`` is the rdatatype of the Rdata.\n        '
        self.rdclass = self._as_rdataclass(rdclass)
        self.rdtype = self._as_rdatatype(rdtype)
        self.rdcomment = None

    def _get_all_slots(self):
        if False:
            i = 10
            return i + 15
        return itertools.chain.from_iterable((getattr(cls, '__slots__', []) for cls in self.__class__.__mro__))

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        state = {}
        for slot in self._get_all_slots():
            state[slot] = getattr(self, slot)
        return state

    def __setstate__(self, state):
        if False:
            while True:
                i = 10
        for (slot, val) in state.items():
            object.__setattr__(self, slot, val)
        if not hasattr(self, 'rdcomment'):
            object.__setattr__(self, 'rdcomment', None)

    def covers(self) -> dns.rdatatype.RdataType:
        if False:
            while True:
                i = 10
        'Return the type a Rdata covers.\n\n        DNS SIG/RRSIG rdatas apply to a specific type; this type is\n        returned by the covers() function.  If the rdata type is not\n        SIG or RRSIG, dns.rdatatype.NONE is returned.  This is useful when\n        creating rdatasets, allowing the rdataset to contain only RRSIGs\n        of a particular type, e.g. RRSIG(NS).\n\n        Returns a ``dns.rdatatype.RdataType``.\n        '
        return dns.rdatatype.NONE

    def extended_rdatatype(self) -> int:
        if False:
            while True:
                i = 10
        'Return a 32-bit type value, the least significant 16 bits of\n        which are the ordinary DNS type, and the upper 16 bits of which are\n        the "covered" type, if any.\n\n        Returns an ``int``.\n        '
        return self.covers() << 16 | self.rdtype

    def to_text(self, origin: Optional[dns.name.Name]=None, relativize: bool=True, **kw: Dict[str, Any]) -> str:
        if False:
            while True:
                i = 10
        'Convert an rdata to text format.\n\n        Returns a ``str``.\n        '
        raise NotImplementedError

    def _to_wire(self, file: Optional[Any], compress: Optional[dns.name.CompressType]=None, origin: Optional[dns.name.Name]=None, canonicalize: bool=False) -> bytes:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def to_wire(self, file: Optional[Any]=None, compress: Optional[dns.name.CompressType]=None, origin: Optional[dns.name.Name]=None, canonicalize: bool=False) -> bytes:
        if False:
            i = 10
            return i + 15
        'Convert an rdata to wire format.\n\n        Returns a ``bytes`` or ``None``.\n        '
        if file:
            return self._to_wire(file, compress, origin, canonicalize)
        else:
            f = io.BytesIO()
            self._to_wire(f, compress, origin, canonicalize)
            return f.getvalue()

    def to_generic(self, origin: Optional[dns.name.Name]=None) -> 'dns.rdata.GenericRdata':
        if False:
            while True:
                i = 10
        'Creates a dns.rdata.GenericRdata equivalent of this rdata.\n\n        Returns a ``dns.rdata.GenericRdata``.\n        '
        return dns.rdata.GenericRdata(self.rdclass, self.rdtype, self.to_wire(origin=origin))

    def to_digestable(self, origin: Optional[dns.name.Name]=None) -> bytes:
        if False:
            print('Hello World!')
        'Convert rdata to a format suitable for digesting in hashes.  This\n        is also the DNSSEC canonical form.\n\n        Returns a ``bytes``.\n        '
        return self.to_wire(origin=origin, canonicalize=True)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        covers = self.covers()
        if covers == dns.rdatatype.NONE:
            ctext = ''
        else:
            ctext = '(' + dns.rdatatype.to_text(covers) + ')'
        return '<DNS ' + dns.rdataclass.to_text(self.rdclass) + ' ' + dns.rdatatype.to_text(self.rdtype) + ctext + ' rdata: ' + str(self) + '>'

    def __str__(self):
        if False:
            return 10
        return self.to_text()

    def _cmp(self, other):
        if False:
            i = 10
            return i + 15
        'Compare an rdata with another rdata of the same rdtype and\n        rdclass.\n\n        For rdata with only absolute names:\n            Return < 0 if self < other in the DNSSEC ordering, 0 if self\n            == other, and > 0 if self > other.\n        For rdata with at least one relative names:\n            The rdata sorts before any rdata with only absolute names.\n            When compared with another relative rdata, all names are\n            made absolute as if they were relative to the root, as the\n            proper origin is not available.  While this creates a stable\n            ordering, it is NOT guaranteed to be the DNSSEC ordering.\n            In the future, all ordering comparisons for rdata with\n            relative names will be disallowed.\n        '
        try:
            our = self.to_digestable()
            our_relative = False
        except dns.name.NeedAbsoluteNameOrOrigin:
            if _allow_relative_comparisons:
                our = self.to_digestable(dns.name.root)
            our_relative = True
        try:
            their = other.to_digestable()
            their_relative = False
        except dns.name.NeedAbsoluteNameOrOrigin:
            if _allow_relative_comparisons:
                their = other.to_digestable(dns.name.root)
            their_relative = True
        if _allow_relative_comparisons:
            if our_relative != their_relative:
                if our_relative:
                    return -1
                else:
                    return 1
        elif our_relative or their_relative:
            raise NoRelativeRdataOrdering
        if our == their:
            return 0
        elif our > their:
            return 1
        else:
            return -1

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Rdata):
            return False
        if self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return False
        our_relative = False
        their_relative = False
        try:
            our = self.to_digestable()
        except dns.name.NeedAbsoluteNameOrOrigin:
            our = self.to_digestable(dns.name.root)
            our_relative = True
        try:
            their = other.to_digestable()
        except dns.name.NeedAbsoluteNameOrOrigin:
            their = other.to_digestable(dns.name.root)
            their_relative = True
        if our_relative != their_relative:
            return False
        return our == their

    def __ne__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, Rdata):
            return True
        if self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return True
        return not self.__eq__(other)

    def __lt__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, Rdata) or self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return NotImplemented
        return self._cmp(other) < 0

    def __le__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, Rdata) or self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return NotImplemented
        return self._cmp(other) <= 0

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Rdata) or self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return NotImplemented
        return self._cmp(other) >= 0

    def __gt__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, Rdata) or self.rdclass != other.rdclass or self.rdtype != other.rdtype:
            return NotImplemented
        return self._cmp(other) > 0

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.to_digestable(dns.name.root))

    @classmethod
    def from_text(cls, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, tok: dns.tokenizer.Tokenizer, origin: Optional[dns.name.Name]=None, relativize: bool=True, relativize_to: Optional[dns.name.Name]=None) -> 'Rdata':
        if False:
            return 10
        raise NotImplementedError

    @classmethod
    def from_wire_parser(cls, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, parser: dns.wire.Parser, origin: Optional[dns.name.Name]=None) -> 'Rdata':
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def replace(self, **kwargs: Any) -> 'Rdata':
        if False:
            while True:
                i = 10
        '\n        Create a new Rdata instance based on the instance replace was\n        invoked on. It is possible to pass different parameters to\n        override the corresponding properties of the base Rdata.\n\n        Any field specific to the Rdata type can be replaced, but the\n        *rdtype* and *rdclass* fields cannot.\n\n        Returns an instance of the same Rdata subclass as *self*.\n        '
        parameters = inspect.signature(self.__init__).parameters
        for key in kwargs:
            if key == 'rdcomment':
                continue
            if key not in parameters:
                raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, key))
            if key in ('rdclass', 'rdtype'):
                raise AttributeError("Cannot overwrite '{}' attribute '{}'".format(self.__class__.__name__, key))
        args = (kwargs.get(key, getattr(self, key)) for key in parameters)
        rd = self.__class__(*args)
        rdcomment = kwargs.get('rdcomment', self.rdcomment)
        if rdcomment is not None:
            object.__setattr__(rd, 'rdcomment', rdcomment)
        return rd

    @classmethod
    def _as_rdataclass(cls, value):
        if False:
            while True:
                i = 10
        return dns.rdataclass.RdataClass.make(value)

    @classmethod
    def _as_rdatatype(cls, value):
        if False:
            while True:
                i = 10
        return dns.rdatatype.RdataType.make(value)

    @classmethod
    def _as_bytes(cls, value: Any, encode: bool=False, max_length: Optional[int]=None, empty_ok: bool=True) -> bytes:
        if False:
            print('Hello World!')
        if encode and isinstance(value, str):
            bvalue = value.encode()
        elif isinstance(value, bytearray):
            bvalue = bytes(value)
        elif isinstance(value, bytes):
            bvalue = value
        else:
            raise ValueError('not bytes')
        if max_length is not None and len(bvalue) > max_length:
            raise ValueError('too long')
        if not empty_ok and len(bvalue) == 0:
            raise ValueError('empty bytes not allowed')
        return bvalue

    @classmethod
    def _as_name(cls, value):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, str):
            return dns.name.from_text(value)
        elif not isinstance(value, dns.name.Name):
            raise ValueError('not a name')
        return value

    @classmethod
    def _as_uint8(cls, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, int):
            raise ValueError('not an integer')
        if value < 0 or value > 255:
            raise ValueError('not a uint8')
        return value

    @classmethod
    def _as_uint16(cls, value):
        if False:
            return 10
        if not isinstance(value, int):
            raise ValueError('not an integer')
        if value < 0 or value > 65535:
            raise ValueError('not a uint16')
        return value

    @classmethod
    def _as_uint32(cls, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, int):
            raise ValueError('not an integer')
        if value < 0 or value > 4294967295:
            raise ValueError('not a uint32')
        return value

    @classmethod
    def _as_uint48(cls, value):
        if False:
            return 10
        if not isinstance(value, int):
            raise ValueError('not an integer')
        if value < 0 or value > 281474976710655:
            raise ValueError('not a uint48')
        return value

    @classmethod
    def _as_int(cls, value, low=None, high=None):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, int):
            raise ValueError('not an integer')
        if low is not None and value < low:
            raise ValueError('value too small')
        if high is not None and value > high:
            raise ValueError('value too large')
        return value

    @classmethod
    def _as_ipv4_address(cls, value):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, str):
            dns.ipv4.inet_aton(value)
            return value
        elif isinstance(value, bytes):
            return dns.ipv4.inet_ntoa(value)
        else:
            raise ValueError('not an IPv4 address')

    @classmethod
    def _as_ipv6_address(cls, value):
        if False:
            print('Hello World!')
        if isinstance(value, str):
            dns.ipv6.inet_aton(value)
            return value
        elif isinstance(value, bytes):
            return dns.ipv6.inet_ntoa(value)
        else:
            raise ValueError('not an IPv6 address')

    @classmethod
    def _as_bool(cls, value):
        if False:
            print('Hello World!')
        if isinstance(value, bool):
            return value
        else:
            raise ValueError('not a boolean')

    @classmethod
    def _as_ttl(cls, value):
        if False:
            return 10
        if isinstance(value, int):
            return cls._as_int(value, 0, dns.ttl.MAX_TTL)
        elif isinstance(value, str):
            return dns.ttl.from_text(value)
        else:
            raise ValueError('not a TTL')

    @classmethod
    def _as_tuple(cls, value, as_value):
        if False:
            return 10
        try:
            return (as_value(value),)
        except Exception:
            return tuple((as_value(v) for v in value))

    @classmethod
    def _processing_order(cls, iterable):
        if False:
            return 10
        items = list(iterable)
        random.shuffle(items)
        return items

@dns.immutable.immutable
class GenericRdata(Rdata):
    """Generic Rdata Class

    This class is used for rdata types for which we have no better
    implementation.  It implements the DNS "unknown RRs" scheme.
    """
    __slots__ = ['data']

    def __init__(self, rdclass, rdtype, data):
        if False:
            i = 10
            return i + 15
        super().__init__(rdclass, rdtype)
        self.data = data

    def to_text(self, origin: Optional[dns.name.Name]=None, relativize: bool=True, **kw: Dict[str, Any]) -> str:
        if False:
            i = 10
            return i + 15
        return '\\# %d ' % len(self.data) + _hexify(self.data, **kw)

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            while True:
                i = 10
        token = tok.get()
        if not token.is_identifier() or token.value != '\\#':
            raise dns.exception.SyntaxError('generic rdata does not start with \\#')
        length = tok.get_int()
        hex = tok.concatenate_remaining_identifiers(True).encode()
        data = binascii.unhexlify(hex)
        if len(data) != length:
            raise dns.exception.SyntaxError('generic rdata hex data has wrong length')
        return cls(rdclass, rdtype, data)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            for i in range(10):
                print('nop')
        file.write(self.data)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            i = 10
            return i + 15
        return cls(rdclass, rdtype, parser.get_remaining())
_rdata_classes: Dict[Tuple[dns.rdataclass.RdataClass, dns.rdatatype.RdataType], Any] = {}
_module_prefix = 'dns.rdtypes'

def get_rdata_class(rdclass, rdtype):
    if False:
        i = 10
        return i + 15
    cls = _rdata_classes.get((rdclass, rdtype))
    if not cls:
        cls = _rdata_classes.get((dns.rdatatype.ANY, rdtype))
        if not cls:
            rdclass_text = dns.rdataclass.to_text(rdclass)
            rdtype_text = dns.rdatatype.to_text(rdtype)
            rdtype_text = rdtype_text.replace('-', '_')
            try:
                mod = import_module('.'.join([_module_prefix, rdclass_text, rdtype_text]))
                cls = getattr(mod, rdtype_text)
                _rdata_classes[rdclass, rdtype] = cls
            except ImportError:
                try:
                    mod = import_module('.'.join([_module_prefix, 'ANY', rdtype_text]))
                    cls = getattr(mod, rdtype_text)
                    _rdata_classes[dns.rdataclass.ANY, rdtype] = cls
                    _rdata_classes[rdclass, rdtype] = cls
                except ImportError:
                    pass
    if not cls:
        cls = GenericRdata
        _rdata_classes[rdclass, rdtype] = cls
    return cls

def from_text(rdclass: Union[dns.rdataclass.RdataClass, str], rdtype: Union[dns.rdatatype.RdataType, str], tok: Union[dns.tokenizer.Tokenizer, str], origin: Optional[dns.name.Name]=None, relativize: bool=True, relativize_to: Optional[dns.name.Name]=None, idna_codec: Optional[dns.name.IDNACodec]=None) -> Rdata:
    if False:
        for i in range(10):
            print('nop')
    'Build an rdata object from text format.\n\n    This function attempts to dynamically load a class which\n    implements the specified rdata class and type.  If there is no\n    class-and-type-specific implementation, the GenericRdata class\n    is used.\n\n    Once a class is chosen, its from_text() class method is called\n    with the parameters to this function.\n\n    If *tok* is a ``str``, then a tokenizer is created and the string\n    is used as its input.\n\n    *rdclass*, a ``dns.rdataclass.RdataClass`` or ``str``, the rdataclass.\n\n    *rdtype*, a ``dns.rdatatype.RdataType`` or ``str``, the rdatatype.\n\n    *tok*, a ``dns.tokenizer.Tokenizer`` or a ``str``.\n\n    *origin*, a ``dns.name.Name`` (or ``None``), the\n    origin to use for relative names.\n\n    *relativize*, a ``bool``.  If true, name will be relativized.\n\n    *relativize_to*, a ``dns.name.Name`` (or ``None``), the origin to use\n    when relativizing names.  If not set, the *origin* value will be used.\n\n    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA\n    encoder/decoder to use if a tokenizer needs to be created.  If\n    ``None``, the default IDNA 2003 encoder/decoder is used.  If a\n    tokenizer is not created, then the codec associated with the tokenizer\n    is the one that is used.\n\n    Returns an instance of the chosen Rdata subclass.\n\n    '
    if isinstance(tok, str):
        tok = dns.tokenizer.Tokenizer(tok, idna_codec=idna_codec)
    rdclass = dns.rdataclass.RdataClass.make(rdclass)
    rdtype = dns.rdatatype.RdataType.make(rdtype)
    cls = get_rdata_class(rdclass, rdtype)
    with dns.exception.ExceptionWrapper(dns.exception.SyntaxError):
        rdata = None
        if cls != GenericRdata:
            token = tok.get()
            tok.unget(token)
            if token.is_identifier() and token.value == '\\#':
                grdata = GenericRdata.from_text(rdclass, rdtype, tok, origin, relativize, relativize_to)
                rdata = from_wire(rdclass, rdtype, grdata.data, 0, len(grdata.data), origin)
                rwire = rdata.to_wire()
                if rwire != grdata.data:
                    raise dns.exception.SyntaxError('compressed data in generic syntax form of known rdatatype')
        if rdata is None:
            rdata = cls.from_text(rdclass, rdtype, tok, origin, relativize, relativize_to)
        token = tok.get_eol_as_token()
        if token.comment is not None:
            object.__setattr__(rdata, 'rdcomment', token.comment)
        return rdata

def from_wire_parser(rdclass: Union[dns.rdataclass.RdataClass, str], rdtype: Union[dns.rdatatype.RdataType, str], parser: dns.wire.Parser, origin: Optional[dns.name.Name]=None) -> Rdata:
    if False:
        print('Hello World!')
    'Build an rdata object from wire format\n\n    This function attempts to dynamically load a class which\n    implements the specified rdata class and type.  If there is no\n    class-and-type-specific implementation, the GenericRdata class\n    is used.\n\n    Once a class is chosen, its from_wire() class method is called\n    with the parameters to this function.\n\n    *rdclass*, a ``dns.rdataclass.RdataClass`` or ``str``, the rdataclass.\n\n    *rdtype*, a ``dns.rdatatype.RdataType`` or ``str``, the rdatatype.\n\n    *parser*, a ``dns.wire.Parser``, the parser, which should be\n    restricted to the rdata length.\n\n    *origin*, a ``dns.name.Name`` (or ``None``).  If not ``None``,\n    then names will be relativized to this origin.\n\n    Returns an instance of the chosen Rdata subclass.\n    '
    rdclass = dns.rdataclass.RdataClass.make(rdclass)
    rdtype = dns.rdatatype.RdataType.make(rdtype)
    cls = get_rdata_class(rdclass, rdtype)
    with dns.exception.ExceptionWrapper(dns.exception.FormError):
        return cls.from_wire_parser(rdclass, rdtype, parser, origin)

def from_wire(rdclass: Union[dns.rdataclass.RdataClass, str], rdtype: Union[dns.rdatatype.RdataType, str], wire: bytes, current: int, rdlen: int, origin: Optional[dns.name.Name]=None) -> Rdata:
    if False:
        i = 10
        return i + 15
    'Build an rdata object from wire format\n\n    This function attempts to dynamically load a class which\n    implements the specified rdata class and type.  If there is no\n    class-and-type-specific implementation, the GenericRdata class\n    is used.\n\n    Once a class is chosen, its from_wire() class method is called\n    with the parameters to this function.\n\n    *rdclass*, an ``int``, the rdataclass.\n\n    *rdtype*, an ``int``, the rdatatype.\n\n    *wire*, a ``bytes``, the wire-format message.\n\n    *current*, an ``int``, the offset in wire of the beginning of\n    the rdata.\n\n    *rdlen*, an ``int``, the length of the wire-format rdata\n\n    *origin*, a ``dns.name.Name`` (or ``None``).  If not ``None``,\n    then names will be relativized to this origin.\n\n    Returns an instance of the chosen Rdata subclass.\n    '
    parser = dns.wire.Parser(wire, current)
    with parser.restrict_to(rdlen):
        return from_wire_parser(rdclass, rdtype, parser, origin)

class RdatatypeExists(dns.exception.DNSException):
    """DNS rdatatype already exists."""
    supp_kwargs = {'rdclass', 'rdtype'}
    fmt = 'The rdata type with class {rdclass:d} and rdtype {rdtype:d} ' + 'already exists.'

def register_type(implementation: Any, rdtype: int, rdtype_text: str, is_singleton: bool=False, rdclass: dns.rdataclass.RdataClass=dns.rdataclass.IN) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Dynamically register a module to handle an rdatatype.\n\n    *implementation*, a module implementing the type in the usual dnspython\n    way.\n\n    *rdtype*, an ``int``, the rdatatype to register.\n\n    *rdtype_text*, a ``str``, the textual form of the rdatatype.\n\n    *is_singleton*, a ``bool``, indicating if the type is a singleton (i.e.\n    RRsets of the type can have only one member.)\n\n    *rdclass*, the rdataclass of the type, or ``dns.rdataclass.ANY`` if\n    it applies to all classes.\n    '
    rdtype = dns.rdatatype.RdataType.make(rdtype)
    existing_cls = get_rdata_class(rdclass, rdtype)
    if existing_cls != GenericRdata or dns.rdatatype.is_metatype(rdtype):
        raise RdatatypeExists(rdclass=rdclass, rdtype=rdtype)
    _rdata_classes[rdclass, rdtype] = getattr(implementation, rdtype_text.replace('-', '_'))
    dns.rdatatype.register_type(rdtype, rdtype_text, is_singleton)