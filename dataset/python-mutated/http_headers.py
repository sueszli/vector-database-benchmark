"""
An API for storing HTTP header names and values.
"""
from collections.abc import Sequence as _Sequence
from typing import AnyStr, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union, overload
from twisted.python.compat import cmp, comparable
_T = TypeVar('_T')

def _dashCapitalize(name: bytes) -> bytes:
    if False:
        i = 10
        return i + 15
    "\n    Return a byte string which is capitalized using '-' as a word separator.\n\n    @param name: The name of the header to capitalize.\n\n    @return: The given header capitalized using '-' as a word separator.\n    "
    return b'-'.join([word.capitalize() for word in name.split(b'-')])

def _sanitizeLinearWhitespace(headerComponent: bytes) -> bytes:
    if False:
        print('Hello World!')
    '\n    Replace linear whitespace (C{\\n}, C{\\r\\n}, C{\\r}) in a header key\n    or value with a single space.\n\n    @param headerComponent: The header key or value to sanitize.\n\n    @return: The sanitized header key or value.\n    '
    return b' '.join(headerComponent.splitlines())

@comparable
class Headers:
    """
    Stores HTTP headers in a key and multiple value format.

    When passed L{str}, header names (e.g. 'Content-Type')
    are encoded using ISO-8859-1 and header values (e.g.
    'text/html;charset=utf-8') are encoded using UTF-8. Some methods that return
    values will return them in the same type as the name given.

    If the header keys or values cannot be encoded or decoded using the rules
    above, using just L{bytes} arguments to the methods of this class will
    ensure no decoding or encoding is done, and L{Headers} will treat the keys
    and values as opaque byte strings.

    @cvar _caseMappings: A L{dict} that maps lowercase header names
        to their canonicalized representation.

    @ivar _rawHeaders: A L{dict} mapping header names as L{bytes} to L{list}s of
        header values as L{bytes}.
    """
    _caseMappings = {b'content-md5': b'Content-MD5', b'dnt': b'DNT', b'etag': b'ETag', b'p3p': b'P3P', b'te': b'TE', b'www-authenticate': b'WWW-Authenticate', b'x-xss-protection': b'X-XSS-Protection'}

    def __init__(self, rawHeaders: Optional[Mapping[AnyStr, Sequence[AnyStr]]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._rawHeaders: Dict[bytes, List[bytes]] = {}
        if rawHeaders is not None:
            for (name, values) in rawHeaders.items():
                self.setRawHeaders(name, values)

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Return a string fully describing the headers set on this object.\n        '
        return '{}({!r})'.format(self.__class__.__name__, self._rawHeaders)

    def __cmp__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Define L{Headers} instances as being equal to each other if they have\n        the same raw headers.\n        '
        if isinstance(other, Headers):
            return cmp(sorted(self._rawHeaders.items()), sorted(other._rawHeaders.items()))
        return NotImplemented

    def _encodeName(self, name: Union[str, bytes]) -> bytes:
        if False:
            print('Hello World!')
        "\n        Encode the name of a header (eg 'Content-Type') to an ISO-8859-1 encoded\n        bytestring if required.\n\n        @param name: A HTTP header name\n\n        @return: C{name}, encoded if required, lowercased\n        "
        if isinstance(name, str):
            return name.lower().encode('iso-8859-1')
        return name.lower()

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a copy of itself with the same headers set.\n\n        @return: A new L{Headers}\n        '
        return self.__class__(self._rawHeaders)

    def hasHeader(self, name: AnyStr) -> bool:
        if False:
            return 10
        '\n        Check for the existence of a given header.\n\n        @param name: The name of the HTTP header to check for.\n\n        @return: C{True} if the header exists, otherwise C{False}.\n        '
        return self._encodeName(name) in self._rawHeaders

    def removeHeader(self, name: AnyStr) -> None:
        if False:
            while True:
                i = 10
        '\n        Remove the named header from this header object.\n\n        @param name: The name of the HTTP header to remove.\n\n        @return: L{None}\n        '
        self._rawHeaders.pop(self._encodeName(name), None)

    @overload
    def setRawHeaders(self, name: Union[str, bytes], values: Sequence[bytes]) -> None:
        if False:
            while True:
                i = 10
        ...

    @overload
    def setRawHeaders(self, name: Union[str, bytes], values: Sequence[str]) -> None:
        if False:
            print('Hello World!')
        ...

    @overload
    def setRawHeaders(self, name: Union[str, bytes], values: Sequence[Union[str, bytes]]) -> None:
        if False:
            print('Hello World!')
        ...

    def setRawHeaders(self, name: Union[str, bytes], values: object) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Sets the raw representation of the given header.\n\n        @param name: The name of the HTTP header to set the values for.\n\n        @param values: A list of strings each one being a header value of\n            the given name.\n\n        @raise TypeError: Raised if C{values} is not a sequence of L{bytes}\n            or L{str}, or if C{name} is not L{bytes} or L{str}.\n\n        @return: L{None}\n        '
        if not isinstance(values, _Sequence):
            raise TypeError('Header entry %r should be sequence but found instance of %r instead' % (name, type(values)))
        if not isinstance(name, (bytes, str)):
            raise TypeError(f'Header name is an instance of {type(name)!r}, not bytes or str')
        for (count, value) in enumerate(values):
            if not isinstance(value, (bytes, str)):
                raise TypeError('Header value at position %s is an instance of %r, not bytes or str' % (count, type(value)))
        _name = _sanitizeLinearWhitespace(self._encodeName(name))
        encodedValues: List[bytes] = []
        for v in values:
            if isinstance(v, str):
                _v = v.encode('utf8')
            else:
                _v = v
            encodedValues.append(_sanitizeLinearWhitespace(_v))
        self._rawHeaders[_name] = encodedValues

    def addRawHeader(self, name: Union[str, bytes], value: Union[str, bytes]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a new raw value for the given header.\n\n        @param name: The name of the header for which to set the value.\n\n        @param value: The value to set for the named header.\n        '
        if not isinstance(name, (bytes, str)):
            raise TypeError(f'Header name is an instance of {type(name)!r}, not bytes or str')
        if not isinstance(value, (bytes, str)):
            raise TypeError('Header value is an instance of %r, not bytes or str' % (type(value),))
        self._rawHeaders.setdefault(_sanitizeLinearWhitespace(self._encodeName(name)), []).append(_sanitizeLinearWhitespace(value.encode('utf8') if isinstance(value, str) else value))

    @overload
    def getRawHeaders(self, name: AnyStr) -> Optional[Sequence[AnyStr]]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def getRawHeaders(self, name: AnyStr, default: _T) -> Union[Sequence[AnyStr], _T]:
        if False:
            i = 10
            return i + 15
        ...

    def getRawHeaders(self, name: AnyStr, default: Optional[_T]=None) -> Union[Sequence[AnyStr], Optional[_T]]:
        if False:
            while True:
                i = 10
        '\n        Returns a sequence of headers matching the given name as the raw string\n        given.\n\n        @param name: The name of the HTTP header to get the values of.\n\n        @param default: The value to return if no header with the given C{name}\n            exists.\n\n        @return: If the named header is present, a sequence of its\n            values.  Otherwise, C{default}.\n        '
        encodedName = self._encodeName(name)
        values = self._rawHeaders.get(encodedName, [])
        if not values:
            return default
        if isinstance(name, str):
            return [v.decode('utf8') for v in values]
        return values

    def getAllRawHeaders(self) -> Iterator[Tuple[bytes, Sequence[bytes]]]:
        if False:
            i = 10
            return i + 15
        '\n        Return an iterator of key, value pairs of all headers contained in this\n        object, as L{bytes}.  The keys are capitalized in canonical\n        capitalization.\n        '
        for (k, v) in self._rawHeaders.items():
            yield (self._canonicalNameCaps(k), v)

    def _canonicalNameCaps(self, name: bytes) -> bytes:
        if False:
            i = 10
            return i + 15
        '\n        Return the canonical name for the given header.\n\n        @param name: The all-lowercase header name to capitalize in its\n            canonical form.\n\n        @return: The canonical name of the header.\n        '
        return self._caseMappings.get(name, _dashCapitalize(name))
__all__ = ['Headers']