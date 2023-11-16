"""
Convert IPv6 addresses between textual representation and binary.

These functions are missing when python is compiled
without IPv6 support, on Windows for instance.
"""
import socket
import re
import binascii
from scapy.compat import plain_str, hex_bytes, bytes_encode, bytes_hex
from typing import Union
_IP6_ZEROS = re.compile('(?::|^)(0(?::0)+)(?::|$)')
_INET6_PTON_EXC = socket.error('illegal IP address string passed to inet_pton')

def _inet6_pton(addr):
    if False:
        print('Hello World!')
    'Convert an IPv6 address from text representation into binary form,\nused when socket.inet_pton is not available.\n\n    '
    joker_pos = None
    result = b''
    addr = plain_str(addr)
    if addr == '::':
        return b'\x00' * 16
    if addr.startswith('::'):
        addr = addr[1:]
    if addr.endswith('::'):
        addr = addr[:-1]
    parts = addr.split(':')
    nparts = len(parts)
    for (i, part) in enumerate(parts):
        if not part:
            if joker_pos is None:
                joker_pos = len(result)
            else:
                raise _INET6_PTON_EXC
        elif i + 1 == nparts and '.' in part:
            if part.count('.') != 3:
                raise _INET6_PTON_EXC
            try:
                result += socket.inet_aton(part)
            except socket.error:
                raise _INET6_PTON_EXC
        else:
            try:
                result += hex_bytes(part.rjust(4, '0'))
            except (binascii.Error, TypeError):
                raise _INET6_PTON_EXC
    if joker_pos is not None:
        if len(result) == 16:
            raise _INET6_PTON_EXC
        result = result[:joker_pos] + b'\x00' * (16 - len(result)) + result[joker_pos:]
    if len(result) != 16:
        raise _INET6_PTON_EXC
    return result
_INET_PTON = {socket.AF_INET: socket.inet_aton, socket.AF_INET6: _inet6_pton}

def inet_pton(af, addr):
    if False:
        while True:
            i = 10
    'Convert an IP address from text representation into binary form.'
    addr = plain_str(addr)
    try:
        if not socket.has_ipv6:
            raise AttributeError
        return socket.inet_pton(af, addr)
    except AttributeError:
        try:
            return _INET_PTON[af](addr)
        except KeyError:
            raise socket.error('Address family not supported by protocol')

def _inet6_ntop(addr):
    if False:
        for i in range(10):
            print('nop')
    'Convert an IPv6 address from binary form into text representation,\nused when socket.inet_pton is not available.\n\n    '
    if len(addr) != 16:
        raise ValueError('invalid length of packed IP address string')
    address = ':'.join((plain_str(bytes_hex(addr[idx:idx + 2])).lstrip('0') or '0' for idx in range(0, 16, 2)))
    try:
        match = max(_IP6_ZEROS.finditer(address), key=lambda m: m.end(1) - m.start(1))
        return '{}::{}'.format(address[:match.start()], address[match.end():])
    except ValueError:
        return address
_INET_NTOP = {socket.AF_INET: socket.inet_ntoa, socket.AF_INET6: _inet6_ntop}

def inet_ntop(af, addr):
    if False:
        while True:
            i = 10
    'Convert an IP address from binary form into text representation.'
    addr = bytes_encode(addr)
    try:
        if not socket.has_ipv6:
            raise AttributeError
        return socket.inet_ntop(af, addr)
    except AttributeError:
        try:
            return _INET_NTOP[af](addr)
        except KeyError:
            raise ValueError('unknown address family %d' % af)