"""
Private support for parsing textual addresses.

"""
from __future__ import absolute_import, division, print_function
import binascii
import re
import struct
from gevent.resolver import hostname_types

class AddressSyntaxError(ValueError):
    pass

def _ipv4_inet_aton(text):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert an IPv4 address in text form to binary struct.\n\n    *text*, a ``text``, the IPv4 address in textual form.\n\n    Returns a ``binary``.\n    '
    if not isinstance(text, bytes):
        text = text.encode()
    parts = text.split(b'.')
    if len(parts) != 4:
        raise AddressSyntaxError(text)
    for part in parts:
        if not part.isdigit():
            raise AddressSyntaxError
        if len(part) > 1 and part[0] == '0':
            raise AddressSyntaxError(text)
    try:
        ints = [int(part) for part in parts]
        return struct.pack('BBBB', *ints)
    except:
        raise AddressSyntaxError(text)

def _ipv6_inet_aton(text, _v4_ending=re.compile(b'(.*):(\\d+\\.\\d+\\.\\d+\\.\\d+)$'), _colon_colon_start=re.compile(b'::.*'), _colon_colon_end=re.compile(b'.*::$')):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert an IPv6 address in text form to binary form.\n\n    *text*, a ``text``, the IPv6 address in textual form.\n\n    Returns a ``binary``.\n    '
    if not isinstance(text, bytes):
        text = text.encode()
    if text == b'::':
        text = b'0::'
    m = _v4_ending.match(text)
    if not m is None:
        b = bytearray(_ipv4_inet_aton(m.group(2)))
        text = u'{}:{:02x}{:02x}:{:02x}{:02x}'.format(m.group(1).decode(), b[0], b[1], b[2], b[3]).encode()
    m = _colon_colon_start.match(text)
    if not m is None:
        text = text[1:]
    else:
        m = _colon_colon_end.match(text)
        if not m is None:
            text = text[:-1]
    chunks = text.split(b':')
    l = len(chunks)
    if l > 8:
        raise SyntaxError
    seen_empty = False
    canonical = []
    for c in chunks:
        if c == b'':
            if seen_empty:
                raise AddressSyntaxError(text)
            seen_empty = True
            for _ in range(0, 8 - l + 1):
                canonical.append(b'0000')
        else:
            lc = len(c)
            if lc > 4:
                raise AddressSyntaxError(text)
            if lc != 4:
                c = b'0' * (4 - lc) + c
            canonical.append(c)
    if l < 8 and (not seen_empty):
        raise AddressSyntaxError(text)
    text = b''.join(canonical)
    try:
        return binascii.unhexlify(text)
    except (binascii.Error, TypeError):
        raise AddressSyntaxError(text)

def _is_addr(host, parse=_ipv4_inet_aton):
    if False:
        for i in range(10):
            print('nop')
    if not host or not isinstance(host, hostname_types):
        return False
    try:
        parse(host)
    except AddressSyntaxError:
        return False
    return True
is_ipv4_addr = _is_addr

def is_ipv6_addr(host):
    if False:
        print('Hello World!')
    if host and isinstance(host, hostname_types):
        s = '%' if isinstance(host, str) else b'%'
        host = host.split(s, 1)[0]
    return _is_addr(host, _ipv6_inet_aton)