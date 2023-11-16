"""URI utilities.

This module provides utility functions to parse, encode, decode, and
otherwise manipulate a URI. These functions are not available directly
in the `falcon` module, and so must be explicitly imported::

    from falcon import uri

    name, port = uri.parse_host('example.org:8080')
"""
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple, TYPE_CHECKING
from typing import Union
from falcon.constants import PYPY
try:
    from falcon.cyutil.uri import decode as _cy_decode, parse_query_string as _cy_parse_query_string
except ImportError:
    _cy_decode = None
    _cy_parse_query_string = None
_UNRESERVED = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'
_DELIMITERS = ":/?#[]@!$&'()*+,;="
_ALL_ALLOWED = _UNRESERVED + _DELIMITERS
_HEX_DIGITS = '0123456789ABCDEFabcdef'
_HEX_TO_BYTE = {(a + b).encode(): bytes([int(a + b, 16)]) for a in _HEX_DIGITS for b in _HEX_DIGITS}

def _create_char_encoder(allowed_chars: str) -> Callable[[int], str]:
    if False:
        return 10
    lookup = {}
    for code_point in range(256):
        if chr(code_point) in allowed_chars:
            encoded_char = chr(code_point)
        else:
            encoded_char = '%{0:02X}'.format(code_point)
        lookup[code_point] = encoded_char
    return lookup.__getitem__

def _create_str_encoder(is_value: bool, check_is_escaped: bool=False) -> Callable[[str], str]:
    if False:
        return 10
    allowed_chars = _UNRESERVED if is_value else _ALL_ALLOWED
    allowed_chars_plus_percent = allowed_chars + '%'
    encode_char = _create_char_encoder(allowed_chars)

    def encoder(uri: str) -> str:
        if False:
            print('Hello World!')
        if not uri.rstrip(allowed_chars):
            return uri
        if check_is_escaped and (not uri.rstrip(allowed_chars_plus_percent)):
            tokens = uri.split('%')
            for token in tokens[1:]:
                hex_octet = token[:2]
                if not len(hex_octet) == 2:
                    break
                if not (hex_octet[0] in _HEX_DIGITS and hex_octet[1] in _HEX_DIGITS):
                    break
            else:
                return uri
        encoded_uri = uri.encode()
        return ''.join(map(encode_char, encoded_uri))
    return encoder
encode = _create_str_encoder(False)
encode.__name__ = 'encode'
encode.__doc__ = 'Encodes a full or relative URI according to RFC 3986.\n\nRFC 3986 defines a set of "unreserved" characters as well as a\nset of "reserved" characters used as delimiters. This function escapes\nall other "disallowed" characters by percent-encoding them.\n\nNote:\n    This utility is faster in the average case than the similar\n    `quote` function found in ``urlib``. It also strives to be easier\n    to use by assuming a sensible default of allowed characters.\n\nArgs:\n    uri (str): URI or part of a URI to encode.\n\nReturns:\n    str: An escaped version of `uri`, where all disallowed characters\n    have been percent-encoded.\n\n'
encode_value = _create_str_encoder(True)
encode_value.__name__ = 'encode_value'
encode_value.__doc__ = 'Encodes a value string according to RFC 3986.\n\nDisallowed characters are percent-encoded in a way that models\n``urllib.parse.quote(safe="~")``. However, the Falcon function is faster\nin the average case than the similar `quote` function found in urlib.\nIt also strives to be easier to use by assuming a sensible default\nof allowed characters.\n\nAll reserved characters are lumped together into a single set of\n"delimiters", and everything in that set is escaped.\n\nNote:\n    RFC 3986 defines a set of "unreserved" characters as well as a\n    set of "reserved" characters used as delimiters.\n\nArgs:\n    uri (str): URI fragment to encode. It is assumed not to cross delimiter\n        boundaries, and so any reserved URI delimiter characters\n        included in it will be percent-encoded.\n\nReturns:\n    str: An escaped version of `uri`, where all disallowed characters\n    have been percent-encoded.\n\n'
encode_check_escaped = _create_str_encoder(False, True)
encode_check_escaped.__name__ = 'encode_check_escaped'
encode_check_escaped.__doc__ = 'Encodes a full or relative URI according to RFC 3986.\n\nRFC 3986 defines a set of "unreserved" characters as well as a\nset of "reserved" characters used as delimiters. This function escapes\nall other "disallowed" characters by percent-encoding them unless they\nappear to have been previously encoded. For example, ``\'%26\'`` will not be\nencoded again as it follows the format of an encoded value.\n\nNote:\n    This utility is faster in the average case than the similar\n    `quote` function found in ``urlib``. It also strives to be easier\n    to use by assuming a sensible default of allowed characters.\n\nArgs:\n    uri (str): URI or part of a URI to encode.\n\nReturns:\n    str: An escaped version of `uri`, where all disallowed characters\n    have been percent-encoded.\n\n'
encode_value_check_escaped = _create_str_encoder(True, True)
encode_value_check_escaped.__name__ = 'encode_value_check_escaped'
encode_value_check_escaped.__doc__ = 'Encodes a value string according to RFC 3986.\n\nRFC 3986 defines a set of "unreserved" characters as well as a\nset of "reserved" characters used as delimiters. Disallowed characters\nare percent-encoded in a way that models ``urllib.parse.quote(safe="~")``\nunless they appear to have been previously encoded. For example, ``\'%26\'``\nwill not be encoded again as it follows the format of an encoded value.\n\nAll reserved characters are lumped together into a single set of\n"delimiters", and everything in that set is escaped.\n\nNote:\n    This utility is faster in the average case than the similar\n    `quote` function found in ``urlib``. It also strives to be easier\n    to use by assuming a sensible default of allowed characters.\n\nArgs:\n    uri (str): URI fragment to encode. It is assumed not to cross delimiter\n        boundaries, and so any reserved URI delimiter characters\n        included in it will be percent-encoded.\n\nReturns:\n    str: An escaped version of `uri`, where all disallowed characters\n    have been percent-encoded.\n\n'

def _join_tokens_bytearray(tokens: List[bytes]) -> str:
    if False:
        i = 10
        return i + 15
    decoded_uri = bytearray(tokens[0])
    for token in tokens[1:]:
        token_partial = token[:2]
        try:
            decoded_uri += _HEX_TO_BYTE[token_partial] + token[2:]
        except KeyError:
            decoded_uri += b'%' + token
    return decoded_uri.decode('utf-8', 'replace')

def _join_tokens_list(tokens: List[bytes]) -> str:
    if False:
        for i in range(10):
            print('nop')
    decoded = tokens[:1]
    skip = True
    for token in tokens:
        if skip:
            skip = False
            continue
        token_partial = token[:2]
        try:
            decoded.append(_HEX_TO_BYTE[token_partial] + token[2:])
        except KeyError:
            decoded.append(b'%' + token)
    return b''.join(decoded).decode('utf-8', 'replace')
_join_tokens = _join_tokens_list if PYPY else _join_tokens_bytearray

def decode(encoded_uri: str, unquote_plus: bool=True) -> str:
    if False:
        return 10
    "Decode percent-encoded characters in a URI or query string.\n\n    This function models the behavior of `urllib.parse.unquote_plus`,\n    albeit in a faster, more straightforward manner.\n\n    Args:\n        encoded_uri (str): An encoded URI (full or partial).\n\n    Keyword Arguments:\n        unquote_plus (bool): Set to ``False`` to retain any plus ('+')\n            characters in the given string, rather than converting them to\n            spaces (default ``True``). Typically you should set this\n            to ``False`` when decoding any part of a URI other than the\n            query string.\n\n    Returns:\n        str: A decoded URL. If the URL contains escaped non-ASCII\n        characters, UTF-8 is assumed per RFC 3986.\n\n    "
    decoded_uri = encoded_uri
    if '+' in decoded_uri and unquote_plus:
        decoded_uri = decoded_uri.replace('+', ' ')
    if '%' not in decoded_uri:
        return decoded_uri
    reencoded_uri = decoded_uri.encode()
    tokens = reencoded_uri.split(b'%')
    if len(tokens) < 8:
        reencoded_uri = tokens[0]
        for token in tokens[1:]:
            token_partial = token[:2]
            try:
                reencoded_uri += _HEX_TO_BYTE[token_partial] + token[2:]
            except KeyError:
                reencoded_uri += b'%' + token
        return reencoded_uri.decode('utf-8', 'replace')
    return _join_tokens(tokens)

def parse_query_string(query_string: str, keep_blank: bool=False, csv: bool=True) -> Dict[str, Union[str, List[str]]]:
    if False:
        print('Hello World!')
    "Parse a query string into a dict.\n\n    Query string parameters are assumed to use standard form-encoding. Only\n    parameters with values are returned. For example, given 'foo=bar&flag',\n    this function would ignore 'flag' unless the `keep_blank_qs_values` option\n    is set.\n\n    Note:\n        In addition to the standard HTML form-based method for specifying\n        lists by repeating a given param multiple times, Falcon supports\n        a more compact form in which the param may be given a single time\n        but set to a ``list`` of comma-separated elements (e.g., 'foo=a,b,c').\n\n        When using this format, all commas uri-encoded will not be treated by\n        Falcon as a delimiter. If the client wants to send a value as a list,\n        it must not encode the commas with the values.\n\n        The two different ways of specifying lists may not be mixed in\n        a single query string for the same parameter.\n\n    Args:\n        query_string (str): The query string to parse.\n        keep_blank (bool): Set to ``True`` to return fields even if\n            they do not have a value (default ``False``). For comma-separated\n            values, this option also determines whether or not empty elements\n            in the parsed list are retained.\n        csv: Set to ``False`` in order to disable splitting query\n            parameters on ``,`` (default ``True``). Depending on the user agent,\n            encoding lists as multiple occurrences of the same parameter might\n            be preferable. In this case, setting `parse_qs_csv` to ``False``\n            will cause the framework to treat commas as literal characters in\n            each occurring parameter value.\n\n    Returns:\n        dict: A dictionary of (*name*, *value*) pairs, one per query\n        parameter. Note that *value* may be a single ``str``, or a\n        ``list`` of ``str``.\n\n    Raises:\n        TypeError: `query_string` was not a ``str``.\n\n    "
    params: dict = {}
    is_encoded = '+' in query_string or '%' in query_string
    for field in query_string.split('&'):
        (k, _, v) = field.partition('=')
        if not v and (not keep_blank or not k):
            continue
        if is_encoded:
            k = decode(k)
        if k in params:
            old_value = params[k]
            if csv and ',' in v:
                values = v.split(',')
                if not keep_blank:
                    additional_values = [decode(element) for element in values if element]
                else:
                    additional_values = [decode(element) for element in values]
                if isinstance(old_value, list):
                    old_value.extend(additional_values)
                else:
                    additional_values.insert(0, old_value)
                    params[k] = additional_values
            else:
                if is_encoded:
                    v = decode(v)
                if isinstance(old_value, list):
                    old_value.append(v)
                else:
                    params[k] = [old_value, v]
        elif csv and ',' in v:
            values = v.split(',')
            if not keep_blank:
                params[k] = [decode(element) for element in values if element]
            else:
                params[k] = [decode(element) for element in values]
        elif is_encoded:
            params[k] = decode(v)
        else:
            params[k] = v
    return params

def parse_host(host: str, default_port: Optional[int]=None) -> Tuple[str, Optional[int]]:
    if False:
        i = 10
        return i + 15
    "Parse a canonical 'host:port' string into parts.\n\n    Parse a host string (which may or may not contain a port) into\n    parts, taking into account that the string may contain\n    either a domain name or an IP address. In the latter case,\n    both IPv4 and IPv6 addresses are supported.\n\n    Args:\n        host (str): Host string to parse, optionally containing a\n            port number.\n\n    Keyword Arguments:\n        default_port (int): Port number to return when the host string\n            does not contain one (default ``None``).\n\n    Returns:\n        tuple: A parsed (*host*, *port*) tuple from the given\n        host string, with the port converted to an ``int``.\n        If the host string does not specify a port, `default_port` is\n        used instead.\n\n    "
    if host.startswith('['):
        pos = host.rfind(']:')
        if pos != -1:
            return (host[1:pos], int(host[pos + 2:]))
        else:
            return (host[1:-1], default_port)
    pos = host.rfind(':')
    if pos == -1 or pos != host.find(':'):
        return (host, default_port)
    (name, _, port) = host.partition(':')
    return (name, int(port))

def unquote_string(quoted: str) -> str:
    if False:
        return 10
    'Unquote an RFC 7320 "quoted-string".\n\n    Args:\n        quoted (str): Original quoted string\n\n    Returns:\n        str: unquoted string\n\n    Raises:\n        TypeError: `quoted` was not a ``str``.\n    '
    if len(quoted) < 2:
        return quoted
    elif quoted[0] != '"' or quoted[-1] != '"':
        return quoted
    tmp_quoted = quoted[1:-1]
    if '\\' not in tmp_quoted:
        return tmp_quoted
    elif '\\\\' not in tmp_quoted:
        return tmp_quoted.replace('\\', '')
    else:
        return '\\'.join([q.replace('\\', '') for q in tmp_quoted.split('\\\\')])
if not TYPE_CHECKING:
    decode = _cy_decode or decode
    parse_query_string = _cy_parse_query_string or parse_query_string
__all__ = ['decode', 'encode', 'encode_value', 'encode_check_escaped', 'encode_value_check_escaped', 'parse_host', 'parse_query_string', 'unquote_string']