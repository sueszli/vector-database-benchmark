from __future__ import unicode_literals, division, absolute_import, print_function
import socket
import struct
from ._errors import unwrap
from ._types import byte_cls, bytes_to_list, str_cls, type_name

def inet_ntop(address_family, packed_ip):
    if False:
        for i in range(10):
            print('nop')
    '\n    Windows compatibility shim for socket.inet_ntop().\n\n    :param address_family:\n        socket.AF_INET for IPv4 or socket.AF_INET6 for IPv6\n\n    :param packed_ip:\n        A byte string of the network form of an IP address\n\n    :return:\n        A unicode string of the IP address\n    '
    if address_family not in set([socket.AF_INET, socket.AF_INET6]):
        raise ValueError(unwrap('\n            address_family must be socket.AF_INET (%s) or socket.AF_INET6 (%s),\n            not %s\n            ', repr(socket.AF_INET), repr(socket.AF_INET6), repr(address_family)))
    if not isinstance(packed_ip, byte_cls):
        raise TypeError(unwrap('\n            packed_ip must be a byte string, not %s\n            ', type_name(packed_ip)))
    required_len = 4 if address_family == socket.AF_INET else 16
    if len(packed_ip) != required_len:
        raise ValueError(unwrap('\n            packed_ip must be %d bytes long - is %d\n            ', required_len, len(packed_ip)))
    if address_family == socket.AF_INET:
        return '%d.%d.%d.%d' % tuple(bytes_to_list(packed_ip))
    octets = struct.unpack(b'!HHHHHHHH', packed_ip)
    runs_of_zero = {}
    longest_run = 0
    zero_index = None
    for (i, octet) in enumerate(octets + (-1,)):
        if octet != 0:
            if zero_index is not None:
                length = i - zero_index
                if length not in runs_of_zero:
                    runs_of_zero[length] = zero_index
                longest_run = max(longest_run, length)
                zero_index = None
        elif zero_index is None:
            zero_index = i
    hexed = [hex(o)[2:] for o in octets]
    if longest_run < 2:
        return ':'.join(hexed)
    zero_start = runs_of_zero[longest_run]
    zero_end = zero_start + longest_run
    return ':'.join(hexed[:zero_start]) + '::' + ':'.join(hexed[zero_end:])

def inet_pton(address_family, ip_string):
    if False:
        return 10
    '\n    Windows compatibility shim for socket.inet_ntop().\n\n    :param address_family:\n        socket.AF_INET for IPv4 or socket.AF_INET6 for IPv6\n\n    :param ip_string:\n        A unicode string of an IP address\n\n    :return:\n        A byte string of the network form of the IP address\n    '
    if address_family not in set([socket.AF_INET, socket.AF_INET6]):
        raise ValueError(unwrap('\n            address_family must be socket.AF_INET (%s) or socket.AF_INET6 (%s),\n            not %s\n            ', repr(socket.AF_INET), repr(socket.AF_INET6), repr(address_family)))
    if not isinstance(ip_string, str_cls):
        raise TypeError(unwrap('\n            ip_string must be a unicode string, not %s\n            ', type_name(ip_string)))
    if address_family == socket.AF_INET:
        octets = ip_string.split('.')
        error = len(octets) != 4
        if not error:
            ints = []
            for o in octets:
                o = int(o)
                if o > 255 or o < 0:
                    error = True
                    break
                ints.append(o)
        if error:
            raise ValueError(unwrap('\n                ip_string must be a dotted string with four integers in the\n                range of 0 to 255, got %s\n                ', repr(ip_string)))
        return struct.pack(b'!BBBB', *ints)
    error = False
    omitted = ip_string.count('::')
    if omitted > 1:
        error = True
    elif omitted == 0:
        octets = ip_string.split(':')
        error = len(octets) != 8
    else:
        (begin, end) = ip_string.split('::')
        begin_octets = begin.split(':')
        end_octets = end.split(':')
        missing = 8 - len(begin_octets) - len(end_octets)
        octets = begin_octets + ['0'] * missing + end_octets
    if not error:
        ints = []
        for o in octets:
            o = int(o, 16)
            if o > 65535 or o < 0:
                error = True
                break
            ints.append(o)
        return struct.pack(b'!HHHHHHHH', *ints)
    raise ValueError(unwrap('\n        ip_string must be a valid ipv6 string, got %s\n        ', repr(ip_string)))