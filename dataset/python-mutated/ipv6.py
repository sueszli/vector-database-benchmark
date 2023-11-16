"""IPv6 helper functions."""
import binascii
import re
from typing import List, Union
import dns.exception
import dns.ipv4
_leading_zero = re.compile('0+([0-9a-f]+)')

def inet_ntoa(address: bytes) -> str:
    if False:
        i = 10
        return i + 15
    "Convert an IPv6 address in binary form to text form.\n\n    *address*, a ``bytes``, the IPv6 address in binary form.\n\n    Raises ``ValueError`` if the address isn't 16 bytes long.\n    Returns a ``str``.\n    "
    if len(address) != 16:
        raise ValueError('IPv6 addresses are 16 bytes long')
    hex = binascii.hexlify(address)
    chunks = []
    i = 0
    l = len(hex)
    while i < l:
        chunk = hex[i:i + 4].decode()
        m = _leading_zero.match(chunk)
        if m is not None:
            chunk = m.group(1)
        chunks.append(chunk)
        i += 4
    best_start = 0
    best_len = 0
    start = -1
    last_was_zero = False
    for i in range(8):
        if chunks[i] != '0':
            if last_was_zero:
                end = i
                current_len = end - start
                if current_len > best_len:
                    best_start = start
                    best_len = current_len
                last_was_zero = False
        elif not last_was_zero:
            start = i
            last_was_zero = True
    if last_was_zero:
        end = 8
        current_len = end - start
        if current_len > best_len:
            best_start = start
            best_len = current_len
    if best_len > 1:
        if best_start == 0 and (best_len == 6 or (best_len == 5 and chunks[5] == 'ffff')):
            if best_len == 6:
                prefix = '::'
            else:
                prefix = '::ffff:'
            thex = prefix + dns.ipv4.inet_ntoa(address[12:])
        else:
            thex = ':'.join(chunks[:best_start]) + '::' + ':'.join(chunks[best_start + best_len:])
    else:
        thex = ':'.join(chunks)
    return thex
_v4_ending = re.compile(b'(.*):(\\d+\\.\\d+\\.\\d+\\.\\d+)$')
_colon_colon_start = re.compile(b'::.*')
_colon_colon_end = re.compile(b'.*::$')

def inet_aton(text: Union[str, bytes], ignore_scope: bool=False) -> bytes:
    if False:
        i = 10
        return i + 15
    'Convert an IPv6 address in text form to binary form.\n\n    *text*, a ``str``, the IPv6 address in textual form.\n\n    *ignore_scope*, a ``bool``.  If ``True``, a scope will be ignored.\n    If ``False``, the default, it is an error for a scope to be present.\n\n    Returns a ``bytes``.\n    '
    if not isinstance(text, bytes):
        btext = text.encode()
    else:
        btext = text
    if ignore_scope:
        parts = btext.split(b'%')
        l = len(parts)
        if l == 2:
            btext = parts[0]
        elif l > 2:
            raise dns.exception.SyntaxError
    if btext == b'':
        raise dns.exception.SyntaxError
    elif btext.endswith(b':') and (not btext.endswith(b'::')):
        raise dns.exception.SyntaxError
    elif btext.startswith(b':') and (not btext.startswith(b'::')):
        raise dns.exception.SyntaxError
    elif btext == b'::':
        btext = b'0::'
    m = _v4_ending.match(btext)
    if m is not None:
        b = dns.ipv4.inet_aton(m.group(2))
        btext = '{}:{:02x}{:02x}:{:02x}{:02x}'.format(m.group(1).decode(), b[0], b[1], b[2], b[3]).encode()
    m = _colon_colon_start.match(btext)
    if m is not None:
        btext = btext[1:]
    else:
        m = _colon_colon_end.match(btext)
        if m is not None:
            btext = btext[:-1]
    chunks = btext.split(b':')
    l = len(chunks)
    if l > 8:
        raise dns.exception.SyntaxError
    seen_empty = False
    canonical: List[bytes] = []
    for c in chunks:
        if c == b'':
            if seen_empty:
                raise dns.exception.SyntaxError
            seen_empty = True
            for _ in range(0, 8 - l + 1):
                canonical.append(b'0000')
        else:
            lc = len(c)
            if lc > 4:
                raise dns.exception.SyntaxError
            if lc != 4:
                c = b'0' * (4 - lc) + c
            canonical.append(c)
    if l < 8 and (not seen_empty):
        raise dns.exception.SyntaxError
    btext = b''.join(canonical)
    try:
        return binascii.unhexlify(btext)
    except (binascii.Error, TypeError):
        raise dns.exception.SyntaxError
_mapped_prefix = b'\x00' * 10 + b'\xff\xff'

def is_mapped(address: bytes) -> bool:
    if False:
        return 10
    'Is the specified address a mapped IPv4 address?\n\n    *address*, a ``bytes`` is an IPv6 address in binary form.\n\n    Returns a ``bool``.\n    '
    return address.startswith(_mapped_prefix)