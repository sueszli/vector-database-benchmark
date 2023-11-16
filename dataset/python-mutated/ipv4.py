"""IPv4 helper functions."""
import struct
from typing import Union
import dns.exception

def inet_ntoa(address: bytes) -> str:
    if False:
        i = 10
        return i + 15
    'Convert an IPv4 address in binary form to text form.\n\n    *address*, a ``bytes``, the IPv4 address in binary form.\n\n    Returns a ``str``.\n    '
    if len(address) != 4:
        raise dns.exception.SyntaxError
    return '%u.%u.%u.%u' % (address[0], address[1], address[2], address[3])

def inet_aton(text: Union[str, bytes]) -> bytes:
    if False:
        while True:
            i = 10
    'Convert an IPv4 address in text form to binary form.\n\n    *text*, a ``str`` or ``bytes``, the IPv4 address in textual form.\n\n    Returns a ``bytes``.\n    '
    if not isinstance(text, bytes):
        btext = text.encode()
    else:
        btext = text
    parts = btext.split(b'.')
    if len(parts) != 4:
        raise dns.exception.SyntaxError
    for part in parts:
        if not part.isdigit():
            raise dns.exception.SyntaxError
        if len(part) > 1 and part[0] == ord('0'):
            raise dns.exception.SyntaxError
    try:
        b = [int(part) for part in parts]
        return struct.pack('BBBB', *b)
    except Exception:
        raise dns.exception.SyntaxError