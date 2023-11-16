"""DNS TTL conversion."""
from typing import Union
import dns.exception
MAX_TTL = 2 ** 32 - 1

class BadTTL(dns.exception.SyntaxError):
    """DNS TTL value is not well-formed."""

def from_text(text: str) -> int:
    if False:
        while True:
            i = 10
    "Convert the text form of a TTL to an integer.\n\n    The BIND 8 units syntax for TTLs (e.g. '1w6d4h3m10s') is supported.\n\n    *text*, a ``str``, the textual TTL.\n\n    Raises ``dns.ttl.BadTTL`` if the TTL is not well-formed.\n\n    Returns an ``int``.\n    "
    if text.isdigit():
        total = int(text)
    elif len(text) == 0:
        raise BadTTL
    else:
        total = 0
        current = 0
        need_digit = True
        for c in text:
            if c.isdigit():
                current *= 10
                current += int(c)
                need_digit = False
            else:
                if need_digit:
                    raise BadTTL
                c = c.lower()
                if c == 'w':
                    total += current * 604800
                elif c == 'd':
                    total += current * 86400
                elif c == 'h':
                    total += current * 3600
                elif c == 'm':
                    total += current * 60
                elif c == 's':
                    total += current
                else:
                    raise BadTTL("unknown unit '%s'" % c)
                current = 0
                need_digit = True
        if not current == 0:
            raise BadTTL('trailing integer')
    if total < 0 or total > MAX_TTL:
        raise BadTTL('TTL should be between 0 and 2**32 - 1 (inclusive)')
    return total

def make(value: Union[int, str]) -> int:
    if False:
        return 10
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        return dns.ttl.from_text(value)
    else:
        raise ValueError('cannot convert value to TTL')