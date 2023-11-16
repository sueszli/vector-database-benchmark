import re
import string
from falcon.util.uri import unquote_string
_TCHAR = string.digits + string.ascii_letters + "!#$%&'*+.^_`|~-"
_TOKEN = '[{tchar}]+'.format(tchar=_TCHAR)
_QDTEXT = '[{0}]'.format(''.join((chr(c) for c in (9, 32, 33) + tuple(range(35, 127)))))
_QUOTED_PAIR = '\\\\[\\t !-~]'
_QUOTED_STRING = '"(?:{quoted_pair}|{qdtext})*"'.format(qdtext=_QDTEXT, quoted_pair=_QUOTED_PAIR)
_FORWARDED_PAIR = '({token})=({token}|{quoted_string})'.format(token=_TOKEN, quoted_string=_QUOTED_STRING)
_QUOTED_PAIR_REPLACE_RE = re.compile('\\\\([\\t !-~])')
_FORWARDED_PAIR_RE = re.compile(_FORWARDED_PAIR)

class Forwarded:
    """Represents a parsed Forwarded header.

    (See also: RFC 7239, Section 4)

    Attributes:
        src (str): The value of the "for" parameter, or
            ``None`` if the parameter is absent. Identifies the
            node making the request to the proxy.
        dest (str): The value of the "by" parameter, or
            ``None`` if the parameter is absent. Identifies the
            client-facing interface of the proxy.
        host (str): The value of the "host" parameter, or
            ``None`` if the parameter is absent. Provides the host
            request header field as received by the proxy.
        scheme (str): The value of the "proto" parameter, or
            ``None`` if the parameter is absent. Indicates the
            protocol that was used to make the request to
            the proxy.
    """
    __slots__ = ('src', 'dest', 'host', 'scheme')

    def __init__(self):
        if False:
            return 10
        self.src = None
        self.dest = None
        self.host = None
        self.scheme = None

def _parse_forwarded_header(forwarded):
    if False:
        for i in range(10):
            print('nop')
    "Parse the value of a Forwarded header.\n\n    Makes an effort to parse Forwarded headers as specified by RFC 7239:\n\n    - It checks that every value has valid syntax in general as specified\n      in section 4: either a 'token' or a 'quoted-string'.\n    - It un-escapes found escape sequences.\n    - It does NOT validate 'by' and 'for' contents as specified in section\n      6.\n    - It does NOT validate 'host' contents (Host ABNF).\n    - It does NOT validate 'proto' contents for valid URI scheme names.\n\n    Arguments:\n        forwarded (str): Value of a Forwarded header\n\n    Returns:\n        list: Sequence of Forwarded instances, representing each forwarded-element\n        in the header, in the same order as they appeared in the header.\n    "
    elements = []
    pos = 0
    end = len(forwarded)
    need_separator = False
    parsed_element = None
    while 0 <= pos < end:
        match = _FORWARDED_PAIR_RE.match(forwarded, pos)
        if match is not None:
            if need_separator:
                pos = forwarded.find(',', pos)
            else:
                pos += len(match.group(0))
                need_separator = True
                (name, value) = match.groups()
                name = name.lower()
                if value[0] == '"':
                    value = unquote_string(value)
                if not parsed_element:
                    parsed_element = Forwarded()
                if name == 'by':
                    parsed_element.dest = value
                elif name == 'for':
                    parsed_element.src = value
                elif name == 'host':
                    parsed_element.host = value
                elif name == 'proto':
                    parsed_element.scheme = value.lower()
        elif forwarded[pos] == ',':
            need_separator = False
            pos += 1
            if parsed_element:
                elements.append(parsed_element)
                parsed_element = None
        elif forwarded[pos] == ';':
            need_separator = False
            pos += 1
        elif forwarded[pos] in ' \t':
            pos += 1
        else:
            pos = forwarded.find(',', pos)
    if parsed_element:
        elements.append(parsed_element)
    return elements