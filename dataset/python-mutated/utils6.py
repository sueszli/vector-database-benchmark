"""
Utility functions for IPv6.
"""
import socket
import struct
import time
from scapy.config import conf
from scapy.base_classes import Net
from scapy.data import IPV6_ADDR_GLOBAL, IPV6_ADDR_LINKLOCAL, IPV6_ADDR_SITELOCAL, IPV6_ADDR_LOOPBACK, IPV6_ADDR_UNICAST, IPV6_ADDR_MULTICAST, IPV6_ADDR_6TO4, IPV6_ADDR_UNSPECIFIED
from scapy.utils import strxor
from scapy.compat import orb, chb
from scapy.pton_ntop import inet_pton, inet_ntop
from scapy.volatile import RandMAC, RandBin
from scapy.error import warning, Scapy_Exception
from functools import reduce, cmp_to_key
from typing import Iterator, List, Optional, Tuple, Union, cast

def construct_source_candidate_set(addr, plen, laddr):
    if False:
        while True:
            i = 10
    '\n    Given all addresses assigned to a specific interface (\'laddr\' parameter),\n    this function returns the "candidate set" associated with \'addr/plen\'.\n\n    Basically, the function filters all interface addresses to keep only those\n    that have the same scope as provided prefix.\n\n    This is on this list of addresses that the source selection mechanism\n    will then be performed to select the best source address associated\n    with some specific destination that uses this prefix.\n    '

    def cset_sort(x, y):
        if False:
            i = 10
            return i + 15
        x_global = 0
        if in6_isgladdr(x):
            x_global = 1
        y_global = 0
        if in6_isgladdr(y):
            y_global = 1
        res = y_global - x_global
        if res != 0 or y_global != 1:
            return res
        if not in6_isaddr6to4(x):
            return -1
        return -res
    cset = iter([])
    if in6_isgladdr(addr) or in6_isuladdr(addr):
        cset = (x for x in laddr if x[1] == IPV6_ADDR_GLOBAL)
    elif in6_islladdr(addr):
        cset = (x for x in laddr if x[1] == IPV6_ADDR_LINKLOCAL)
    elif in6_issladdr(addr):
        cset = (x for x in laddr if x[1] == IPV6_ADDR_SITELOCAL)
    elif in6_ismaddr(addr):
        if in6_ismnladdr(addr):
            cset = (x for x in [('::1', 16, conf.loopback_name)])
        elif in6_ismgladdr(addr):
            cset = (x for x in laddr if x[1] == IPV6_ADDR_GLOBAL)
        elif in6_ismlladdr(addr):
            cset = (x for x in laddr if x[1] == IPV6_ADDR_LINKLOCAL)
        elif in6_ismsladdr(addr):
            cset = (x for x in laddr if x[1] == IPV6_ADDR_SITELOCAL)
    elif addr == '::' and plen == 0:
        cset = (x for x in laddr if x[1] == IPV6_ADDR_GLOBAL)
    addrs = [x[0] for x in cset]
    addrs.sort(key=cmp_to_key(cset_sort))
    return addrs

def get_source_addr_from_candidate_set(dst, candidate_set):
    if False:
        while True:
            i = 10
    '\n    This function implement a limited version of source address selection\n    algorithm defined in section 5 of RFC 3484. The format is very different\n    from that described in the document because it operates on a set\n    of candidate source address for some specific route.\n    '

    def scope_cmp(a, b):
        if False:
            return 10
        '\n        Given two addresses, returns -1, 0 or 1 based on comparison of\n        their scope\n        '
        scope_mapper = {IPV6_ADDR_GLOBAL: 4, IPV6_ADDR_SITELOCAL: 3, IPV6_ADDR_LINKLOCAL: 2, IPV6_ADDR_LOOPBACK: 1}
        sa = in6_getscope(a)
        if sa == -1:
            sa = IPV6_ADDR_LOOPBACK
        sb = in6_getscope(b)
        if sb == -1:
            sb = IPV6_ADDR_LOOPBACK
        sa = scope_mapper[sa]
        sb = scope_mapper[sb]
        if sa == sb:
            return 0
        if sa > sb:
            return 1
        return -1

    def rfc3484_cmp(source_a, source_b):
        if False:
            return 10
        '\n        The function implements a limited version of the rules from Source\n        Address selection algorithm defined section of RFC 3484.\n        '
        if source_a == dst:
            return 1
        if source_b == dst:
            return 1
        tmp = scope_cmp(source_a, source_b)
        if tmp == -1:
            if scope_cmp(source_a, dst) == -1:
                return 1
            else:
                return -1
        elif tmp == 1:
            if scope_cmp(source_b, dst) == -1:
                return 1
            else:
                return -1
        tmp1 = in6_get_common_plen(source_a, dst)
        tmp2 = in6_get_common_plen(source_b, dst)
        if tmp1 > tmp2:
            return 1
        elif tmp2 > tmp1:
            return -1
        return 0
    if not candidate_set:
        return ''
    candidate_set.sort(key=cmp_to_key(rfc3484_cmp), reverse=True)
    return candidate_set[0]

def in6_getAddrType(addr):
    if False:
        print('Hello World!')
    naddr = inet_pton(socket.AF_INET6, addr)
    paddr = inet_ntop(socket.AF_INET6, naddr)
    addrType = 0
    if orb(naddr[0]) & 224 == 32:
        addrType = IPV6_ADDR_UNICAST | IPV6_ADDR_GLOBAL
        if naddr[:2] == b' \x02':
            addrType |= IPV6_ADDR_6TO4
    elif orb(naddr[0]) == 255:
        addrScope = paddr[3]
        if addrScope == '2':
            addrType = IPV6_ADDR_LINKLOCAL | IPV6_ADDR_MULTICAST
        elif addrScope == 'e':
            addrType = IPV6_ADDR_GLOBAL | IPV6_ADDR_MULTICAST
        else:
            addrType = IPV6_ADDR_GLOBAL | IPV6_ADDR_MULTICAST
    elif orb(naddr[0]) == 254 and int(paddr[2], 16) & 12 == 8:
        addrType = IPV6_ADDR_UNICAST | IPV6_ADDR_LINKLOCAL
    elif paddr == '::1':
        addrType = IPV6_ADDR_LOOPBACK
    elif paddr == '::':
        addrType = IPV6_ADDR_UNSPECIFIED
    else:
        addrType = IPV6_ADDR_GLOBAL | IPV6_ADDR_UNICAST
    return addrType

def in6_mactoifaceid(mac, ulbit=None):
    if False:
        while True:
            i = 10
    "\n    Compute the interface ID in modified EUI-64 format associated\n    to the Ethernet address provided as input.\n    value taken by U/L bit in the interface identifier is basically\n    the reversed value of that in given MAC address it can be forced\n    to a specific value by using optional 'ulbit' parameter.\n    "
    if len(mac) != 17:
        raise ValueError('Invalid MAC')
    m = ''.join(mac.split(':'))
    if len(m) != 12:
        raise ValueError('Invalid MAC')
    first = int(m[0:2], 16)
    if ulbit is None or not (ulbit == 0 or ulbit == 1):
        ulbit = [1, 0, 0][first & 2]
    ulbit *= 2
    first_b = '%.02x' % (first & 253 | ulbit)
    eui64 = first_b + m[2:4] + ':' + m[4:6] + 'FF:FE' + m[6:8] + ':' + m[8:12]
    return eui64.upper()

def in6_ifaceidtomac(ifaceid_s):
    if False:
        print('Hello World!')
    '\n    Extract the mac address from provided iface ID. Iface ID is provided\n    in printable format ("XXXX:XXFF:FEXX:XXXX", eventually compressed). None\n    is returned on error.\n    '
    try:
        ifaceid = inet_pton(socket.AF_INET6, '::' + ifaceid_s)[8:16]
    except Exception:
        return None
    if ifaceid[3:5] != b'\xff\xfe':
        return None
    first = struct.unpack('B', ifaceid[:1])[0]
    ulbit = 2 * [1, '-', 0][first & 2]
    first = struct.pack('B', first & 253 | ulbit)
    oui = first + ifaceid[1:3]
    end = ifaceid[5:]
    mac_bytes = ['%.02x' % orb(x) for x in list(oui + end)]
    return ':'.join(mac_bytes)

def in6_addrtomac(addr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract the mac address from provided address. None is returned\n    on error.\n    '
    mask = inet_pton(socket.AF_INET6, '::ffff:ffff:ffff:ffff')
    x = in6_and(mask, inet_pton(socket.AF_INET6, addr))
    ifaceid = inet_ntop(socket.AF_INET6, x)[2:]
    return in6_ifaceidtomac(ifaceid)

def in6_addrtovendor(addr):
    if False:
        print('Hello World!')
    '\n    Extract the MAC address from a modified EUI-64 constructed IPv6\n    address provided and use the IANA oui.txt file to get the vendor.\n    The database used for the conversion is the one loaded by Scapy\n    from a Wireshark installation if discovered in a well-known\n    location. None is returned on error, "UNKNOWN" if the vendor is\n    unknown.\n    '
    mac = in6_addrtomac(addr)
    if mac is None or not conf.manufdb:
        return None
    res = conf.manufdb._get_manuf(mac)
    if len(res) == 17 and res.count(':') != 5:
        res = 'UNKNOWN'
    return res

def in6_getLinkScopedMcastAddr(addr, grpid=None, scope=2):
    if False:
        print('Hello World!')
    "\n    Generate a Link-Scoped Multicast Address as described in RFC 4489.\n    Returned value is in printable notation.\n\n    'addr' parameter specifies the link-local address to use for generating\n    Link-scoped multicast address IID.\n\n    By default, the function returns a ::/96 prefix (aka last 32 bits of\n    returned address are null). If a group id is provided through 'grpid'\n    parameter, last 32 bits of the address are set to that value (accepted\n    formats : b'\x124Vx' or '12345678' or 0x12345678 or 305419896).\n\n    By default, generated address scope is Link-Local (2). That value can\n    be modified by passing a specific 'scope' value as an argument of the\n    function. RFC 4489 only authorizes scope values <= 2. Enforcement\n    is performed by the function (None will be returned).\n\n    If no link-local address can be used to generate the Link-Scoped IPv6\n    Multicast address, or if another error occurs, None is returned.\n    "
    if scope not in [0, 1, 2]:
        return None
    try:
        if not in6_islladdr(addr):
            return None
        baddr = inet_pton(socket.AF_INET6, addr)
    except Exception:
        warning('in6_getLinkScopedMcastPrefix(): Invalid address provided')
        return None
    iid = baddr[8:]
    if grpid is None:
        b_grpid = b'\x00\x00\x00\x00'
    else:
        b_grpid = b''
        if isinstance(grpid, (str, bytes)):
            try:
                if isinstance(grpid, str) and len(grpid) == 8:
                    i_grpid = int(grpid, 16) & 4294967295
                elif isinstance(grpid, bytes) and len(grpid) == 4:
                    i_grpid = struct.unpack('!I', grpid)[0]
                else:
                    raise ValueError
            except Exception:
                warning('in6_getLinkScopedMcastPrefix(): Invalid group id provided')
                return None
        elif isinstance(grpid, int):
            i_grpid = grpid
        else:
            warning('in6_getLinkScopedMcastPrefix(): Invalid group id provided')
            return None
        b_grpid = struct.pack('!I', i_grpid)
    flgscope = struct.pack('B', 255 & (3 << 4 | scope))
    plen = b'\xff'
    res = b'\x00'
    a = b'\xff' + flgscope + res + plen + iid + b_grpid
    return inet_ntop(socket.AF_INET6, a)

def in6_get6to4Prefix(addr):
    if False:
        return 10
    '\n    Returns the /48 6to4 prefix associated with provided IPv4 address\n    On error, None is returned. No check is performed on public/private\n    status of the address\n    '
    try:
        baddr = inet_pton(socket.AF_INET, addr)
        return inet_ntop(socket.AF_INET6, b' \x02' + baddr + b'\x00' * 10)
    except Exception:
        return None

def in6_6to4ExtractAddr(addr):
    if False:
        while True:
            i = 10
    '\n    Extract IPv4 address embedded in 6to4 address. Passed address must be\n    a 6to4 address. None is returned on error.\n    '
    try:
        baddr = inet_pton(socket.AF_INET6, addr)
    except Exception:
        return None
    if baddr[:2] != b' \x02':
        return None
    return inet_ntop(socket.AF_INET, baddr[2:6])

def in6_getLocalUniquePrefix():
    if False:
        print('Hello World!')
    '\n    Returns a pseudo-randomly generated Local Unique prefix. Function\n    follows recommendation of Section 3.2.2 of RFC 4193 for prefix\n    generation.\n    '
    tod = time.time()
    i = int(tod)
    j = int((tod - i) * 2 ** 32)
    btod = struct.pack('!II', i, j)
    mac = RandMAC()
    eui64 = inet_pton(socket.AF_INET6, '::' + in6_mactoifaceid(str(mac)))[8:]
    import hashlib
    globalid = hashlib.sha1(btod + eui64).digest()[:5]
    return inet_ntop(socket.AF_INET6, b'\xfd' + globalid + b'\x00' * 10)

def in6_getRandomizedIfaceId(ifaceid, previous=None):
    if False:
        return 10
    '\n    Implements the interface ID generation algorithm described in RFC 3041.\n    The function takes the Modified EUI-64 interface identifier generated\n    as described in RFC 4291 and an optional previous history value (the\n    first element of the output of this function). If no previous interface\n    identifier is provided, a random one is generated. The function returns\n    a tuple containing the randomized interface identifier and the history\n    value (for possible future use). Input and output values are provided in\n    a "printable" format as depicted below.\n\n    ex::\n        >>> in6_getRandomizedIfaceId(\'20b:93ff:feeb:2d3\')\n        (\'4c61:76ff:f46a:a5f3\', \'d006:d540:db11:b092\')\n        >>> in6_getRandomizedIfaceId(\'20b:93ff:feeb:2d3\',\n                                     previous=\'d006:d540:db11:b092\')\n        (\'fe97:46fe:9871:bd38\', \'eeed:d79c:2e3f:62e\')\n    '
    s = b''
    if previous is None:
        b_previous = bytes(RandBin(8))
    else:
        b_previous = inet_pton(socket.AF_INET6, '::' + previous)[8:]
    s = inet_pton(socket.AF_INET6, '::' + ifaceid)[8:] + b_previous
    import hashlib
    s = hashlib.md5(s).digest()
    (s1, s2) = (s[:8], s[8:])
    s1 = chb(orb(s1[0]) & ~4) + s1[1:]
    bs1 = inet_ntop(socket.AF_INET6, b'\xff' * 8 + s1)[20:]
    bs2 = inet_ntop(socket.AF_INET6, b'\xff' * 8 + s2)[20:]
    return (bs1, bs2)
_rfc1924map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '!', '#', '$', '%', '&', '(', ')', '*', '+', '-', ';', '<', '=', '>', '?', '@', '^', '_', '`', '{', '|', '}', '~']

def in6_ctop(addr):
    if False:
        return 10
    '\n    Convert an IPv6 address in Compact Representation Notation\n    (RFC 1924) to printable representation ;-)\n    Returns None on error.\n    '
    if len(addr) != 20 or not reduce(lambda x, y: x and y, [x in _rfc1924map for x in addr]):
        return None
    i = 0
    for c in addr:
        j = _rfc1924map.index(c)
        i = 85 * i + j
    res = []
    for j in range(4):
        res.append(struct.pack('!I', i % 2 ** 32))
        i = i // 2 ** 32
    res.reverse()
    return inet_ntop(socket.AF_INET6, b''.join(res))

def in6_ptoc(addr):
    if False:
        print('Hello World!')
    '\n    Converts an IPv6 address in printable representation to RFC\n    1924 Compact Representation ;-)\n    Returns None on error.\n    '
    try:
        d = struct.unpack('!IIII', inet_pton(socket.AF_INET6, addr))
    except Exception:
        return None
    rem = 0
    m = [2 ** 96, 2 ** 64, 2 ** 32, 1]
    for i in range(4):
        rem += d[i] * m[i]
    res = []
    while rem:
        res.append(_rfc1924map[rem % 85])
        rem = rem // 85
    res.reverse()
    return ''.join(res)

def in6_isaddr6to4(x):
    if False:
        print('Hello World!')
    '\n    Return True if provided address (in printable format) is a 6to4\n    address (being in 2002::/16).\n    '
    bx = inet_pton(socket.AF_INET6, x)
    return bx[:2] == b' \x02'
conf.teredoPrefix = '2001::'
conf.teredoServerPort = 3544

def in6_isaddrTeredo(x):
    if False:
        return 10
    '\n    Return True if provided address is a Teredo, meaning it is under\n    the /32 conf.teredoPrefix prefix value (by default, 2001::).\n    Otherwise, False is returned. Address must be passed in printable\n    format.\n    '
    our = inet_pton(socket.AF_INET6, x)[0:4]
    teredoPrefix = inet_pton(socket.AF_INET6, conf.teredoPrefix)[0:4]
    return teredoPrefix == our

def teredoAddrExtractInfo(x):
    if False:
        i = 10
        return i + 15
    '\n    Extract information from a Teredo address. Return value is\n    a 4-tuple made of IPv4 address of Teredo server, flag value (int),\n    mapped address (non obfuscated) and mapped port (non obfuscated).\n    No specific checks are performed on passed address.\n    '
    addr = inet_pton(socket.AF_INET6, x)
    server = inet_ntop(socket.AF_INET, addr[4:8])
    flag = struct.unpack('!H', addr[8:10])[0]
    mappedport = struct.unpack('!H', strxor(addr[10:12], b'\xff' * 2))[0]
    mappedaddr = inet_ntop(socket.AF_INET, strxor(addr[12:16], b'\xff' * 4))
    return (server, flag, mappedaddr, mappedport)

def in6_iseui64(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return True if provided address has an interface identifier part\n    created in modified EUI-64 format (meaning it matches ``*::*:*ff:fe*:*``).\n    Otherwise, False is returned. Address must be passed in printable\n    format.\n    '
    eui64 = inet_pton(socket.AF_INET6, '::ff:fe00:0')
    bx = in6_and(inet_pton(socket.AF_INET6, x), eui64)
    return bx == eui64

def in6_isanycast(x):
    if False:
        for i in range(10):
            print('nop')
    if in6_iseui64(x):
        s = '::fdff:ffff:ffff:ff80'
        packed_x = inet_pton(socket.AF_INET6, x)
        packed_s = inet_pton(socket.AF_INET6, s)
        x_and_s = in6_and(packed_x, packed_s)
        return x_and_s == packed_s
    else:
        warning('in6_isanycast(): TODO not EUI-64')
        return False

def _in6_bitops(xa1, xa2, operator=0):
    if False:
        return 10
    a1 = struct.unpack('4I', xa1)
    a2 = struct.unpack('4I', xa2)
    fop = [lambda x, y: x | y, lambda x, y: x & y, lambda x, y: x ^ y]
    ret = map(fop[operator % len(fop)], a1, a2)
    return b''.join((struct.pack('I', x) for x in ret))

def in6_or(a1, a2):
    if False:
        print('Hello World!')
    '\n    Provides a bit to bit OR of provided addresses. They must be\n    passed in network format. Return value is also an IPv6 address\n    in network format.\n    '
    return _in6_bitops(a1, a2, 0)

def in6_and(a1, a2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Provides a bit to bit AND of provided addresses. They must be\n    passed in network format. Return value is also an IPv6 address\n    in network format.\n    '
    return _in6_bitops(a1, a2, 1)

def in6_xor(a1, a2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Provides a bit to bit XOR of provided addresses. They must be\n    passed in network format. Return value is also an IPv6 address\n    in network format.\n    '
    return _in6_bitops(a1, a2, 2)

def in6_cidr2mask(m):
    if False:
        i = 10
        return i + 15
    "\n    Return the mask (bitstring) associated with provided length\n    value. For instance if function is called on 48, return value is\n    b'ÿÿÿÿÿÿ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'.\n\n    "
    if m > 128 or m < 0:
        raise Scapy_Exception('value provided to in6_cidr2mask outside [0, 128] domain (%d)' % m)
    t = []
    for i in range(0, 4):
        t.append(max(0, 2 ** 32 - 2 ** (32 - min(32, m))))
        m -= 32
    return b''.join((struct.pack('!I', x) for x in t))

def in6_getnsma(a):
    if False:
        print('Hello World!')
    '\n    Return link-local solicited-node multicast address for given\n    address. Passed address must be provided in network format.\n    Returned value is also in network format.\n    '
    r = in6_and(a, inet_pton(socket.AF_INET6, '::ff:ffff'))
    r = in6_or(inet_pton(socket.AF_INET6, 'ff02::1:ff00:0'), r)
    return r

def in6_getnsmac(a):
    if False:
        return 10
    '\n    Return the multicast mac address associated with provided\n    IPv6 address. Passed address must be in network format.\n    '
    ba = struct.unpack('16B', a)[-4:]
    mac = '33:33:'
    mac += ':'.join(('%.2x' % x for x in ba))
    return mac

def in6_getha(prefix):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the anycast address associated with all home agents on a given\n    subnet.\n    '
    r = in6_and(inet_pton(socket.AF_INET6, prefix), in6_cidr2mask(64))
    r = in6_or(r, inet_pton(socket.AF_INET6, '::fdff:ffff:ffff:fffe'))
    return inet_ntop(socket.AF_INET6, r)

def in6_ptop(str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Normalizes IPv6 addresses provided in printable format, returning the\n    same address in printable format. (2001:0db8:0:0::1 -> 2001:db8::1)\n    '
    return inet_ntop(socket.AF_INET6, inet_pton(socket.AF_INET6, str))

def in6_isincluded(addr, prefix, plen):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns True when 'addr' belongs to prefix/plen. False otherwise.\n    "
    temp = inet_pton(socket.AF_INET6, addr)
    pref = in6_cidr2mask(plen)
    zero = inet_pton(socket.AF_INET6, prefix)
    return zero == in6_and(temp, pref)

def in6_isllsnmaddr(str):
    if False:
        print('Hello World!')
    '\n    Return True if provided address is a link-local solicited node\n    multicast address, i.e. belongs to ff02::1:ff00:0/104. False is\n    returned otherwise.\n    '
    temp = in6_and(b'\xff' * 13 + b'\x00' * 3, inet_pton(socket.AF_INET6, str))
    temp2 = b'\xff\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\xff\x00\x00\x00'
    return temp == temp2

def in6_isdocaddr(str):
    if False:
        i = 10
        return i + 15
    '\n    Returns True if provided address in printable format belongs to\n    2001:db8::/32 address space reserved for documentation (as defined\n    in RFC 3849).\n    '
    return in6_isincluded(str, '2001:db8::', 32)

def in6_islladdr(str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns True if provided address in printable format belongs to\n    _allocated_ link-local unicast address space (fe80::/10)\n    '
    return in6_isincluded(str, 'fe80::', 10)

def in6_issladdr(str):
    if False:
        i = 10
        return i + 15
    '\n    Returns True if provided address in printable format belongs to\n    _allocated_ site-local address space (fec0::/10). This prefix has\n    been deprecated, address being now reserved by IANA. Function\n    will remain for historic reasons.\n    '
    return in6_isincluded(str, 'fec0::', 10)

def in6_isuladdr(str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns True if provided address in printable format belongs to\n    Unique local address space (fc00::/7).\n    '
    return in6_isincluded(str, 'fc00::', 7)

def in6_isgladdr(str):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns True if provided address in printable format belongs to\n    _allocated_ global address space (2000::/3). Please note that,\n    Unique Local addresses (FC00::/7) are not part of global address\n    space, and won't match.\n    "
    return in6_isincluded(str, '2000::', 3)

def in6_ismaddr(str):
    if False:
        return 10
    '\n    Returns True if provided address in printable format belongs to\n    allocated Multicast address space (ff00::/8).\n    '
    return in6_isincluded(str, 'ff00::', 8)

def in6_ismnladdr(str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns True if address belongs to node-local multicast address\n    space (ff01::/16) as defined in RFC\n    '
    return in6_isincluded(str, 'ff01::', 16)

def in6_ismgladdr(str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns True if address belongs to global multicast address\n    space (ff0e::/16).\n    '
    return in6_isincluded(str, 'ff0e::', 16)

def in6_ismlladdr(str):
    if False:
        return 10
    '\n    Returns True if address belongs to link-local multicast address\n    space (ff02::/16)\n    '
    return in6_isincluded(str, 'ff02::', 16)

def in6_ismsladdr(str):
    if False:
        return 10
    '\n    Returns True if address belongs to site-local multicast address\n    space (ff05::/16). Site local address space has been deprecated.\n    Function remains for historic reasons.\n    '
    return in6_isincluded(str, 'ff05::', 16)

def in6_isaddrllallnodes(str):
    if False:
        while True:
            i = 10
    '\n    Returns True if address is the link-local all-nodes multicast\n    address (ff02::1).\n    '
    return inet_pton(socket.AF_INET6, 'ff02::1') == inet_pton(socket.AF_INET6, str)

def in6_isaddrllallservers(str):
    if False:
        return 10
    '\n    Returns True if address is the link-local all-servers multicast\n    address (ff02::2).\n    '
    return inet_pton(socket.AF_INET6, 'ff02::2') == inet_pton(socket.AF_INET6, str)

def in6_getscope(addr):
    if False:
        i = 10
        return i + 15
    '\n    Returns the scope of the address.\n    '
    if in6_isgladdr(addr) or in6_isuladdr(addr):
        scope = IPV6_ADDR_GLOBAL
    elif in6_islladdr(addr):
        scope = IPV6_ADDR_LINKLOCAL
    elif in6_issladdr(addr):
        scope = IPV6_ADDR_SITELOCAL
    elif in6_ismaddr(addr):
        if in6_ismgladdr(addr):
            scope = IPV6_ADDR_GLOBAL
        elif in6_ismlladdr(addr):
            scope = IPV6_ADDR_LINKLOCAL
        elif in6_ismsladdr(addr):
            scope = IPV6_ADDR_SITELOCAL
        elif in6_ismnladdr(addr):
            scope = IPV6_ADDR_LOOPBACK
        else:
            scope = -1
    elif addr == '::1':
        scope = IPV6_ADDR_LOOPBACK
    else:
        scope = -1
    return scope

def in6_get_common_plen(a, b):
    if False:
        print('Hello World!')
    '\n    Return common prefix length of IPv6 addresses a and b.\n    '

    def matching_bits(byte1, byte2):
        if False:
            while True:
                i = 10
        for i in range(8):
            cur_mask = 128 >> i
            if byte1 & cur_mask != byte2 & cur_mask:
                return i
        return 8
    tmpA = inet_pton(socket.AF_INET6, a)
    tmpB = inet_pton(socket.AF_INET6, b)
    for i in range(16):
        mbits = matching_bits(orb(tmpA[i]), orb(tmpB[i]))
        if mbits != 8:
            return 8 * i + mbits
    return 128

def in6_isvalid(address):
    if False:
        print('Hello World!')
    "Return True if 'address' is a valid IPv6 address string, False\n       otherwise."
    try:
        inet_pton(socket.AF_INET6, address)
        return True
    except Exception:
        return False

class Net6(Net):
    """Network object from an IP address or hostname and mask"""
    name = 'Net6'
    family = socket.AF_INET6
    max_mask = 128

    @classmethod
    def ip2int(cls, addr):
        if False:
            for i in range(10):
                print('nop')
        (val1, val2) = struct.unpack('!QQ', inet_pton(socket.AF_INET6, cls.name2addr(addr)))
        return cast(int, (val1 << 64) + val2)

    @staticmethod
    def int2ip(val):
        if False:
            while True:
                i = 10
        return inet_ntop(socket.AF_INET6, struct.pack('!QQ', val >> 64, val & 18446744073709551615))