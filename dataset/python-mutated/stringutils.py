import itertools
import re
import secrets
import string
from typing import Any, Iterable, Optional, Tuple
from netaddr import valid_ipv6
from synapse.api.errors import Codes, SynapseError
_string_with_symbols = string.digits + string.ascii_letters + '.,;:^&*-_+=#~@'
CLIENT_SECRET_REGEX = re.compile('^[0-9a-zA-Z\\.=_\\-]+$')
MXC_REGEX = re.compile('^mxc://([^/]+)/([^/#?]+)$')

def random_string(length: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Generate a cryptographically secure string of random letters.\n\n    Drawn from the characters: `a-z` and `A-Z`\n    '
    return ''.join((secrets.choice(string.ascii_letters) for _ in range(length)))

def random_string_with_symbols(length: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Generate a cryptographically secure string of random letters/numbers/symbols.\n\n    Drawn from the characters: `a-z`, `A-Z`, `0-9`, and `.,;:^&*-_+=#~@`\n    '
    return ''.join((secrets.choice(_string_with_symbols) for _ in range(length)))

def is_ascii(s: bytes) -> bool:
    if False:
        return 10
    try:
        s.decode('ascii').encode('ascii')
    except UnicodeError:
        return False
    return True

def assert_valid_client_secret(client_secret: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Validate that a given string matches the client_secret defined by the spec'
    if len(client_secret) <= 0 or len(client_secret) > 255 or CLIENT_SECRET_REGEX.match(client_secret) is None:
        raise SynapseError(400, 'Invalid client_secret parameter', errcode=Codes.INVALID_PARAM)

def parse_server_name(server_name: str) -> Tuple[str, Optional[int]]:
    if False:
        return 10
    'Split a server name into host/port parts.\n\n    Args:\n        server_name: server name to parse\n\n    Returns:\n        host/port parts.\n\n    Raises:\n        ValueError if the server name could not be parsed.\n    '
    try:
        if server_name and server_name[-1] == ']':
            return (server_name, None)
        domain_port = server_name.rsplit(':', 1)
        domain = domain_port[0]
        port = int(domain_port[1]) if domain_port[1:] else None
        return (domain, port)
    except Exception:
        raise ValueError("Invalid server name '%s'" % server_name)
VALID_HOST_REGEX = re.compile('\\A[0-9a-zA-Z-]+(?:\\.[0-9a-zA-Z-]+)*\\Z')

def parse_and_validate_server_name(server_name: str) -> Tuple[str, Optional[int]]:
    if False:
        return 10
    'Split a server name into host/port parts and do some basic validation.\n\n    Args:\n        server_name: server name to parse\n\n    Returns:\n        host/port parts.\n\n    Raises:\n        ValueError if the server name could not be parsed.\n    '
    (host, port) = parse_server_name(server_name)
    if host and host[0] == '[':
        if host[-1] != ']':
            raise ValueError("Mismatched [...] in server name '%s'" % (server_name,))
        ipv6_address = host[1:-1]
        if not ipv6_address or not valid_ipv6(ipv6_address):
            raise ValueError("Server name '%s' is not a valid IPv6 address" % (server_name,))
    elif not VALID_HOST_REGEX.match(host):
        raise ValueError("Server name '%s' has an invalid format" % (server_name,))
    return (host, port)

def valid_id_server_location(id_server: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Check whether an identity server location, such as the one passed as the\n    `id_server` parameter to `/_matrix/client/r0/account/3pid/bind`, is valid.\n\n    A valid identity server location consists of a valid hostname and optional\n    port number, optionally followed by any number of `/` delimited path\n    components, without any fragment or query string parts.\n\n    Args:\n        id_server: identity server location string to validate\n\n    Returns:\n        True if valid, False otherwise.\n    '
    components = id_server.split('/', 1)
    host = components[0]
    try:
        parse_and_validate_server_name(host)
    except ValueError:
        return False
    if len(components) < 2:
        return True
    path = components[1]
    return '#' not in path and '?' not in path

def parse_and_validate_mxc_uri(mxc: str) -> Tuple[str, Optional[int], str]:
    if False:
        print('Hello World!')
    'Parse the given string as an MXC URI\n\n    Checks that the "server name" part is a valid server name\n\n    Args:\n        mxc: the (alleged) MXC URI to be checked\n    Returns:\n        hostname, port, media id\n    Raises:\n        ValueError if the URI cannot be parsed\n    '
    m = MXC_REGEX.match(mxc)
    if not m:
        raise ValueError('mxc URI %r did not match expected format' % (mxc,))
    server_name = m.group(1)
    media_id = m.group(2)
    (host, port) = parse_and_validate_server_name(server_name)
    return (host, port, media_id)

def shortstr(iterable: Iterable, maxitems: int=5) -> str:
    if False:
        i = 10
        return i + 15
    'If iterable has maxitems or fewer, return the stringification of a list\n    containing those items.\n\n    Otherwise, return the stringification of a list with the first maxitems items,\n    followed by "...".\n\n    Args:\n        iterable: iterable to truncate\n        maxitems: number of items to return before truncating\n    '
    items = list(itertools.islice(iterable, maxitems + 1))
    if len(items) <= maxitems:
        return str(items)
    return '[' + ', '.join((repr(r) for r in items[:maxitems])) + ', ...]'

def strtobool(val: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    "Convert a string representation of truth to True or False\n\n    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values\n    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if\n    'val' is anything else.\n\n    This is lifted from distutils.util.strtobool, with the exception that it actually\n    returns a bool, rather than an int.\n    "
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError('invalid truth value %r' % (val,))
_BASE62 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def base62_encode(num: int, minwidth: int=1) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Encode a number using base62\n\n    Args:\n        num: number to be encoded\n        minwidth: width to pad to, if the number is small\n    '
    res = ''
    while num:
        (num, rem) = divmod(num, 62)
        res = _BASE62[rem] + res
    pad = '0' * (minwidth - len(res))
    return pad + res

def non_null_str_or_none(val: Any) -> Optional[str]:
    if False:
        print('Hello World!')
    'Check that the arg is a string containing no null (U+0000) codepoints.\n\n    If so, returns the given string unmodified; otherwise, returns None.\n    '
    return val if isinstance(val, str) and '\x00' not in val else None