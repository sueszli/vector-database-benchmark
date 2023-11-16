"""TXT-like base class."""
import struct
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer

@dns.immutable.immutable
class TXTBase(dns.rdata.Rdata):
    """Base class for rdata that is like a TXT record (see RFC 1035)."""
    __slots__ = ['strings']

    def __init__(self, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, strings: Iterable[Union[bytes, str]]):
        if False:
            print('Hello World!')
        'Initialize a TXT-like rdata.\n\n        *rdclass*, an ``int`` is the rdataclass of the Rdata.\n\n        *rdtype*, an ``int`` is the rdatatype of the Rdata.\n\n        *strings*, a tuple of ``bytes``\n        '
        super().__init__(rdclass, rdtype)
        self.strings: Tuple[bytes] = self._as_tuple(strings, lambda x: self._as_bytes(x, True, 255))

    def to_text(self, origin: Optional[dns.name.Name]=None, relativize: bool=True, **kw: Dict[str, Any]) -> str:
        if False:
            return 10
        txt = ''
        prefix = ''
        for s in self.strings:
            txt += '{}"{}"'.format(prefix, dns.rdata._escapify(s))
            prefix = ' '
        return txt

    @classmethod
    def from_text(cls, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, tok: dns.tokenizer.Tokenizer, origin: Optional[dns.name.Name]=None, relativize: bool=True, relativize_to: Optional[dns.name.Name]=None) -> dns.rdata.Rdata:
        if False:
            i = 10
            return i + 15
        strings = []
        for token in tok.get_remaining():
            token = token.unescape_to_bytes()
            if not (token.is_quoted_string() or token.is_identifier()):
                raise dns.exception.SyntaxError('expected a string')
            if len(token.value) > 255:
                raise dns.exception.SyntaxError('string too long')
            strings.append(token.value)
        if len(strings) == 0:
            raise dns.exception.UnexpectedEnd
        return cls(rdclass, rdtype, strings)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            return 10
        for s in self.strings:
            l = len(s)
            assert l < 256
            file.write(struct.pack('!B', l))
            file.write(s)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            i = 10
            return i + 15
        strings = []
        while parser.remaining() > 0:
            s = parser.get_counted_bytes()
            strings.append(s)
        return cls(rdclass, rdtype, strings)