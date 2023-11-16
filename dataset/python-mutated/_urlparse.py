"""
An implementation of `urlparse` that provides URL validation and normalization
as described by RFC3986.

We rely on this implementation rather than the one in Python's stdlib, because:

* It provides more complete URL validation.
* It properly differentiates between an empty querystring and an absent querystring,
  to distinguish URLs with a trailing '?'.
* It handles scheme, hostname, port, and path normalization.
* It supports IDNA hostnames, normalizing them to their encoded form.
* The API supports passing individual components, as well as the complete URL string.

Previously we relied on the excellent `rfc3986` package to handle URL parsing and
validation, but this module provides a simpler alternative, with less indirection
required.
"""
import ipaddress
import re
import typing
import idna
from ._exceptions import InvalidURL
MAX_URL_LENGTH = 65536
UNRESERVED_CHARACTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'
SUB_DELIMS = "!$&'()*+,;="
PERCENT_ENCODED_REGEX = re.compile('%[A-Fa-f0-9]{2}')
URL_REGEX = re.compile('(?:(?P<scheme>{scheme}):)?(?://(?P<authority>{authority}))?(?P<path>{path})(?:\\?(?P<query>{query}))?(?:#(?P<fragment>{fragment}))?'.format(scheme='([a-zA-Z][a-zA-Z0-9+.-]*)?', authority='[^/?#]*', path='[^?#]*', query='[^#]*', fragment='.*'))
AUTHORITY_REGEX = re.compile('(?:(?P<userinfo>{userinfo})@)?(?P<host>{host}):?(?P<port>{port})?'.format(userinfo='[^@]*', host='(\\[.*\\]|[^:]*)', port='.*'))
COMPONENT_REGEX = {'scheme': re.compile('([a-zA-Z][a-zA-Z0-9+.-]*)?'), 'authority': re.compile('[^/?#]*'), 'path': re.compile('[^?#]*'), 'query': re.compile('[^#]*'), 'fragment': re.compile('.*'), 'userinfo': re.compile('[^@]*'), 'host': re.compile('(\\[.*\\]|[^:]*)'), 'port': re.compile('.*')}
IPv4_STYLE_HOSTNAME = re.compile('^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$')
IPv6_STYLE_HOSTNAME = re.compile('^\\[.*\\]$')

class ParseResult(typing.NamedTuple):
    scheme: str
    userinfo: str
    host: str
    port: typing.Optional[int]
    path: str
    query: typing.Optional[str]
    fragment: typing.Optional[str]

    @property
    def authority(self) -> str:
        if False:
            return 10
        return ''.join([f'{self.userinfo}@' if self.userinfo else '', f'[{self.host}]' if ':' in self.host else self.host, f':{self.port}' if self.port is not None else ''])

    @property
    def netloc(self) -> str:
        if False:
            i = 10
            return i + 15
        return ''.join([f'[{self.host}]' if ':' in self.host else self.host, f':{self.port}' if self.port is not None else ''])

    def copy_with(self, **kwargs: typing.Optional[str]) -> 'ParseResult':
        if False:
            print('Hello World!')
        if not kwargs:
            return self
        defaults = {'scheme': self.scheme, 'authority': self.authority, 'path': self.path, 'query': self.query, 'fragment': self.fragment}
        defaults.update(kwargs)
        return urlparse('', **defaults)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        authority = self.authority
        return ''.join([f'{self.scheme}:' if self.scheme else '', f'//{authority}' if authority else '', self.path, f'?{self.query}' if self.query is not None else '', f'#{self.fragment}' if self.fragment is not None else ''])

def urlparse(url: str='', **kwargs: typing.Optional[str]) -> ParseResult:
    if False:
        return 10
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURL('URL too long')
    if any((char.isascii() and (not char.isprintable()) for char in url)):
        raise InvalidURL('Invalid non-printable ASCII character in URL')
    if 'port' in kwargs:
        port = kwargs['port']
        kwargs['port'] = str(port) if isinstance(port, int) else port
    if 'netloc' in kwargs:
        netloc = kwargs.pop('netloc') or ''
        (kwargs['host'], _, kwargs['port']) = netloc.partition(':')
    if 'username' in kwargs or 'password' in kwargs:
        username = quote(kwargs.pop('username', '') or '')
        password = quote(kwargs.pop('password', '') or '')
        kwargs['userinfo'] = f'{username}:{password}' if password else username
    if 'raw_path' in kwargs:
        raw_path = kwargs.pop('raw_path') or ''
        (kwargs['path'], seperator, kwargs['query']) = raw_path.partition('?')
        if not seperator:
            kwargs['query'] = None
    if 'host' in kwargs:
        host = kwargs.get('host') or ''
        if ':' in host and (not (host.startswith('[') and host.endswith(']'))):
            kwargs['host'] = f'[{host}]'
    for (key, value) in kwargs.items():
        if value is not None:
            if len(value) > MAX_URL_LENGTH:
                raise InvalidURL(f"URL component '{key}' too long")
            if any((char.isascii() and (not char.isprintable()) for char in value)):
                raise InvalidURL(f"Invalid non-printable ASCII character in URL component '{key}'")
            if not COMPONENT_REGEX[key].fullmatch(value):
                raise InvalidURL(f"Invalid URL component '{key}'")
    url_match = URL_REGEX.match(url)
    assert url_match is not None
    url_dict = url_match.groupdict()
    scheme = kwargs.get('scheme', url_dict['scheme']) or ''
    authority = kwargs.get('authority', url_dict['authority']) or ''
    path = kwargs.get('path', url_dict['path']) or ''
    query = kwargs.get('query', url_dict['query'])
    fragment = kwargs.get('fragment', url_dict['fragment'])
    authority_match = AUTHORITY_REGEX.match(authority)
    assert authority_match is not None
    authority_dict = authority_match.groupdict()
    userinfo = kwargs.get('userinfo', authority_dict['userinfo']) or ''
    host = kwargs.get('host', authority_dict['host']) or ''
    port = kwargs.get('port', authority_dict['port'])
    parsed_scheme: str = scheme.lower()
    parsed_userinfo: str = quote(userinfo, safe=SUB_DELIMS + ':')
    parsed_host: str = encode_host(host)
    parsed_port: typing.Optional[int] = normalize_port(port, scheme)
    has_scheme = parsed_scheme != ''
    has_authority = parsed_userinfo != '' or parsed_host != '' or parsed_port is not None
    validate_path(path, has_scheme=has_scheme, has_authority=has_authority)
    if has_authority:
        path = normalize_path(path)
    parsed_path: str = quote(path, safe=SUB_DELIMS + ':/[]@')
    parsed_query: typing.Optional[str] = None if query is None else quote(query, safe=SUB_DELIMS + ':?[]@')
    parsed_fragment: typing.Optional[str] = None if fragment is None else quote(fragment, safe=SUB_DELIMS + ':/?#[]@')
    return ParseResult(parsed_scheme, parsed_userinfo, parsed_host, parsed_port, parsed_path, parsed_query, parsed_fragment)

def encode_host(host: str) -> str:
    if False:
        print('Hello World!')
    if not host:
        return ''
    elif IPv4_STYLE_HOSTNAME.match(host):
        try:
            ipaddress.IPv4Address(host)
        except ipaddress.AddressValueError:
            raise InvalidURL(f'Invalid IPv4 address: {host!r}')
        return host
    elif IPv6_STYLE_HOSTNAME.match(host):
        try:
            ipaddress.IPv6Address(host[1:-1])
        except ipaddress.AddressValueError:
            raise InvalidURL(f'Invalid IPv6 address: {host!r}')
        return host[1:-1]
    elif host.isascii():
        return quote(host.lower(), safe=SUB_DELIMS)
    try:
        return idna.encode(host.lower()).decode('ascii')
    except idna.IDNAError:
        raise InvalidURL(f'Invalid IDNA hostname: {host!r}')

def normalize_port(port: typing.Optional[typing.Union[str, int]], scheme: str) -> typing.Optional[int]:
    if False:
        print('Hello World!')
    if port is None or port == '':
        return None
    try:
        port_as_int = int(port)
    except ValueError:
        raise InvalidURL(f'Invalid port: {port!r}')
    default_port = {'ftp': 21, 'http': 80, 'https': 443, 'ws': 80, 'wss': 443}.get(scheme)
    if port_as_int == default_port:
        return None
    return port_as_int

def validate_path(path: str, has_scheme: bool, has_authority: bool) -> None:
    if False:
        while True:
            i = 10
    '\n    Path validation rules that depend on if the URL contains a scheme or authority component.\n\n    See https://datatracker.ietf.org/doc/html/rfc3986.html#section-3.3\n    '
    if has_authority:
        if path and (not path.startswith('/')):
            raise InvalidURL("For absolute URLs, path must be empty or begin with '/'")
    else:
        if path.startswith('//'):
            raise InvalidURL("URLs with no authority component cannot have a path starting with '//'")
        if path.startswith(':') and (not has_scheme):
            raise InvalidURL("URLs with no scheme component cannot have a path starting with ':'")

def normalize_path(path: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Drop "." and ".." segments from a URL path.\n\n    For example:\n\n        normalize_path("/path/./to/somewhere/..") == "/path/to"\n    '
    components = path.split('/')
    output: typing.List[str] = []
    for component in components:
        if component == '.':
            pass
        elif component == '..':
            if output and output != ['']:
                output.pop()
        else:
            output.append(component)
    return '/'.join(output)

def percent_encode(char: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Replace a single character with the percent-encoded representation.\n\n    Characters outside the ASCII range are represented with their a percent-encoded\n    representation of their UTF-8 byte sequence.\n\n    For example:\n\n        percent_encode(" ") == "%20"\n    '
    return ''.join([f'%{byte:02x}' for byte in char.encode('utf-8')]).upper()

def is_safe(string: str, safe: str='/') -> bool:
    if False:
        print('Hello World!')
    '\n    Determine if a given string is already quote-safe.\n    '
    NON_ESCAPED_CHARS = UNRESERVED_CHARACTERS + safe + '%'
    for char in string:
        if char not in NON_ESCAPED_CHARS:
            return False
    return string.count('%') == len(PERCENT_ENCODED_REGEX.findall(string))

def quote(string: str, safe: str='/') -> str:
    if False:
        return 10
    '\n    Use percent-encoding to quote a string if required.\n    '
    if is_safe(string, safe=safe):
        return string
    NON_ESCAPED_CHARS = UNRESERVED_CHARACTERS + safe
    return ''.join([char if char in NON_ESCAPED_CHARS else percent_encode(char) for char in string])

def urlencode(items: typing.List[typing.Tuple[str, str]]) -> str:
    if False:
        for i in range(10):
            print('nop')
    return '&'.join([quote(k, safe='') + '=' + quote(v, safe='') for (k, v) in items])