"""
MIT License

Copyright (c) 2018 Samuel Meuli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from ipaddress import ip_address

def anonymize_ip(address, ipv4_mask='255.255.255.0', ipv6_mask='ffff:ffff:ffff:0000:0000:0000:0000:0000'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Anonymize the provided IPv4 or IPv6 address by setting parts of the\n    address to 0\n    :param str|int address: IP address to be anonymized\n    :param str ipv4_mask: Mask that defines which parts of an IPv4 address are\n    set to 0 (default: "255.255.255.0")\n    :param str ipv6_mask: Mask that defines which parts of an IPv6 address are\n    set to 0 (default: "ffff:ffff:ffff:0000:0000:0000:0000:0000")\n    :return: Anonymized IP address\n    :rtype: str\n    '
    address_packed = ip_address(str(address)).packed
    address_len = len(address_packed)
    if address_len == 4:
        ipv4_mask_packed = ip_address(ipv4_mask).packed
        __validate_ipv4_mask(ipv4_mask_packed)
        return __apply_mask(address_packed, ipv4_mask_packed, 4)
    elif address_len == 16:
        ipv6_mask_packed = ip_address(ipv6_mask).packed
        __validate_ipv6_mask(ipv6_mask_packed)
        return __apply_mask(address_packed, ipv6_mask_packed, 16)
    else:
        raise ValueError('Address does not consist of 4 (IPv4) or 16 (IPv6) octets')

def __apply_mask(address_packed, mask_packed, nr_bytes):
    if False:
        print('Hello World!')
    '\n    Perform a bitwise AND operation on all corresponding bytes between the\n    mask and the provided address. Mask parts set to 0 will become 0 in the\n    anonymized IP address as well\n    :param bytes address_packed: Binary representation of the IP address to\n    be anonymized\n    :param bytes mask_packed: Binary representation of the corresponding IP\n    address mask\n    :param int nr_bytes: Number of bytes in the address (4 for IPv4, 16 for\n    IPv6)\n    :return: Anonymized IP address\n    :rtype: str\n    '
    address_ints = [b for b in iter(address_packed)]
    mask_ints = [b for b in iter(mask_packed)]
    anon_packed = bytearray()
    for i in range(0, nr_bytes):
        anon_packed.append(mask_ints[i] & address_ints[i])
    return str(ip_address(bytes(anon_packed)))

def __validate_ipv4_mask(mask_packed):
    if False:
        print('Hello World!')
    for byte in iter(mask_packed):
        if byte != 0 and byte != 255:
            raise ValueError('ipv4_mask must only contain numbers 0 or 255')
    if mask_packed == b'\x00\x00\x00\x00':
        raise ValueError('ipv4_mask cannot be set to "0.0.0.0" (all anonymized addresses will be 0.0.0.0)')
    if mask_packed == b'\xff\xff\xff\xff':
        raise ValueError('ipv4_mask cannot be set to "255.255.255.255" (addresses will not be anonymized)')

def __validate_ipv6_mask(mask_packed):
    if False:
        print('Hello World!')
    for byte in iter(mask_packed):
        if byte != 0 and byte != 255:
            raise ValueError('ipv6_mask must only contain numbers 0 or ffff')
    if mask_packed == b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00':
        raise ValueError('ipv6_mask cannot be set to "0000:0000:0000:0000:0000:0000:0000:0000" (all anonymized addresses will be 0.0.0.0)')
    if mask_packed == b'\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff':
        raise ValueError('ipv6_mask cannot be set to "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff" (addresses will not be anonymized)')