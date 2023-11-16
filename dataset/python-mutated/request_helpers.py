"""Utilities for the Request class."""
from http import cookies as http_cookies
import re
from falcon.stream import Body
from falcon.stream import BoundedStream
from falcon.util import ETag
_COOKIE_NAME_RESERVED_CHARS = re.compile('[\x00-\x1f\x7f-ÿ()<>@,;:\\\\"/[\\]?={} \t]')
_ENTITY_TAG_PATTERN = re.compile('([Ww]/)?"([^"]*)"')

def parse_cookie_header(header_value):
    if False:
        print('Hello World!')
    'Parse a Cookie header value into a dict of named values.\n\n    (See also: RFC 6265, Section 5.4)\n\n    Args:\n        header_value (str): Value of a Cookie header\n\n    Returns:\n        dict: Map of cookie names to a list of all cookie values found in the\n        header for that name. If a cookie is specified more than once in the\n        header, the order of the values will be preserved.\n    '
    cookies = {}
    for token in header_value.split(';'):
        (name, __, value) = token.partition('=')
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        if _COOKIE_NAME_RESERVED_CHARS.search(name):
            continue
        if len(value) > 2 and value[0] == '"' and (value[-1] == '"'):
            value = http_cookies._unquote(value)
        if name in cookies:
            cookies[name].append(value)
        else:
            cookies[name] = [value]
    return cookies

def header_property(wsgi_name):
    if False:
        for i in range(10):
            print('nop')
    "Create a read-only header property.\n\n    Args:\n        wsgi_name (str): Case-sensitive name of the header as it would\n            appear in the WSGI environ ``dict`` (i.e., 'HTTP_*')\n\n    Returns:\n        A property instance than can be assigned to a class variable.\n\n    "

    def fget(self):
        if False:
            while True:
                i = 10
        try:
            return self.env[wsgi_name] or None
        except KeyError:
            return None
    return property(fget)

def _parse_etags(etag_str):
    if False:
        return 10
    "Parse a string containing one or more HTTP entity-tags.\n\n    The string is assumed to be formatted as defined for a precondition\n    header, and may contain either a single ETag, or multiple comma-separated\n    ETags. The string may also contain a '*' character, in order to indicate\n    that any ETag should match the precondition.\n\n    (See also: RFC 7232, Section 3)\n\n    Args:\n        etag_str (str): An ASCII header value to parse ETags from. ETag values\n            within may be prefixed by ``W/`` to indicate that the weak comparison\n            function should be used.\n\n    Returns:\n        list: A list of unquoted ETags or ``['*']`` if all ETags should be\n        matched. If the string to be parse is empty, or contains only\n        whitespace, ``None`` will be returned instead.\n\n    "
    etag_str = etag_str.strip()
    if not etag_str:
        return None
    if etag_str == '*':
        return [etag_str]
    if ',' not in etag_str:
        return [ETag.loads(etag_str)]
    etags = []
    for (weak, value) in _ENTITY_TAG_PATTERN.findall(etag_str):
        t = ETag(value)
        t.is_weak = bool(weak)
        etags.append(t)
    return etags or None