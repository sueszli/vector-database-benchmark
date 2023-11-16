"""
Common customizations for all Unix-like operating systems other than Linux
"""
import os
import re
import socket
import struct
from fcntl import ioctl
import scapy.config
import scapy.utils
from scapy.config import conf
from scapy.consts import FREEBSD, NETBSD, OPENBSD, SOLARIS
from scapy.error import log_runtime, warning
from scapy.interfaces import network_name, NetworkInterface
from scapy.pton_ntop import inet_pton
from scapy.utils6 import in6_getscope, construct_source_candidate_set
from scapy.utils6 import in6_isvalid, in6_ismlladdr, in6_ismnladdr
from typing import List, Optional, Tuple, Union, cast

def get_if(iff, cmd):
    if False:
        return 10
    'Ease SIOCGIF* ioctl calls'
    iff = network_name(iff)
    sck = socket.socket()
    try:
        return ioctl(sck, cmd, struct.pack('16s16x', iff.encode('utf8')))
    finally:
        sck.close()

def get_if_raw_hwaddr(iff, siocgifhwaddr=None):
    if False:
        while True:
            i = 10
    'Get the raw MAC address of a local interface.\n\n    This function uses SIOCGIFHWADDR calls, therefore only works\n    on some distros.\n\n    :param iff: the network interface name as a string\n    :returns: the corresponding raw MAC address\n    '
    if siocgifhwaddr is None:
        from scapy.arch import SIOCGIFHWADDR
        siocgifhwaddr = SIOCGIFHWADDR
    return cast('Tuple[int, bytes]', struct.unpack('16xH6s8x', get_if(iff, siocgifhwaddr)))

def _guess_iface_name(netif):
    if False:
        for i in range(10):
            print('nop')
    '\n    We attempt to guess the name of interfaces that are truncated from the\n    output of ifconfig -l.\n    If there is only one possible candidate matching the interface name then we\n    return it.\n    If there are none or more, then we return None.\n    '
    with os.popen('%s -l' % conf.prog.ifconfig) as fdesc:
        ifaces = fdesc.readline().strip().split(' ')
    matches = [iface for iface in ifaces if iface.startswith(netif)]
    if len(matches) == 1:
        return matches[0]
    return None

def read_routes():
    if False:
        print('Hello World!')
    'Return a list of IPv4 routes than can be used by Scapy.\n\n    This function parses netstat.\n    '
    if SOLARIS:
        f = os.popen('netstat -rvn -f inet')
    elif FREEBSD:
        f = os.popen('netstat -rnW -f inet')
    else:
        f = os.popen('netstat -rn -f inet')
    ok = 0
    mtu_present = False
    prio_present = False
    refs_present = False
    use_present = False
    routes = []
    pending_if = []
    for line in f.readlines():
        if not line:
            break
        line = line.strip().lower()
        if line.find('----') >= 0:
            continue
        if not ok:
            if line.find('destination') >= 0:
                ok = 1
                mtu_present = 'mtu' in line
                prio_present = 'prio' in line
                refs_present = 'ref' in line
                use_present = 'use' in line or 'nhop' in line
            continue
        if not line:
            break
        rt = line.split()
        if SOLARIS:
            (dest_, netmask_, gw, netif) = rt[:4]
            flg = rt[4 + mtu_present + refs_present]
        else:
            (dest_, gw, flg) = rt[:3]
            locked = OPENBSD and rt[6] == 'l'
            offset = mtu_present + prio_present + refs_present + locked
            offset += use_present
            netif = rt[3 + offset]
        if flg.find('lc') >= 0:
            continue
        elif dest_ == 'default':
            dest = 0
            netmask = 0
        elif SOLARIS:
            dest = scapy.utils.atol(dest_)
            netmask = scapy.utils.atol(netmask_)
        else:
            if '/' in dest_:
                (dest_, netmask_) = dest_.split('/')
                netmask = scapy.utils.itom(int(netmask_))
            else:
                netmask = scapy.utils.itom((dest_.count('.') + 1) * 8)
            dest_ += '.0' * (3 - dest_.count('.'))
            dest = scapy.utils.atol(dest_)
        metric = 1
        if 'g' not in flg:
            gw = '0.0.0.0'
        if netif is not None:
            from scapy.arch import get_if_addr
            try:
                ifaddr = get_if_addr(netif)
                if ifaddr == '0.0.0.0':
                    guessed_netif = _guess_iface_name(netif)
                    if guessed_netif is not None:
                        ifaddr = get_if_addr(guessed_netif)
                        netif = guessed_netif
                    else:
                        log_runtime.info('Could not guess partial interface name: %s', netif)
                routes.append((dest, netmask, gw, netif, ifaddr, metric))
            except OSError:
                raise
        else:
            pending_if.append((dest, netmask, gw))
    f.close()
    for (dest, netmask, gw) in pending_if:
        gw_l = scapy.utils.atol(gw)
        (max_rtmask, gw_if, gw_if_addr) = (0, None, None)
        for (rtdst, rtmask, _, rtif, rtaddr, _) in routes[:]:
            if gw_l & rtmask == rtdst:
                if rtmask >= max_rtmask:
                    max_rtmask = rtmask
                    gw_if = rtif
                    gw_if_addr = rtaddr
        metric = 1
        if gw_if and gw_if_addr:
            routes.append((dest, netmask, gw, gw_if, gw_if_addr, metric))
        else:
            warning('Did not find output interface to reach gateway %s', gw)
    return routes

def _in6_getifaddr(ifname):
    if False:
        print('Hello World!')
    '\n    Returns a list of IPv6 addresses configured on the interface ifname.\n    '
    try:
        f = os.popen('%s %s' % (conf.prog.ifconfig, ifname))
    except OSError:
        log_runtime.warning('Failed to execute ifconfig.')
        return []
    ret = []
    for line in f:
        if 'inet6' in line:
            addr = line.rstrip().split(None, 2)[1]
        else:
            continue
        if '%' in line:
            addr = addr.split('%', 1)[0]
        try:
            inet_pton(socket.AF_INET6, addr)
        except (socket.error, ValueError):
            continue
        scope = in6_getscope(addr)
        ret.append((addr, scope, ifname))
    f.close()
    return ret

def in6_getifaddr():
    if False:
        while True:
            i = 10
    "\n    Returns a list of 3-tuples of the form (addr, scope, iface) where\n    'addr' is the address of scope 'scope' associated to the interface\n    'iface'.\n\n    This is the list of all addresses of all interfaces available on\n    the system.\n    "
    if OPENBSD or SOLARIS:
        if SOLARIS:
            cmd = '%s -a6'
        else:
            cmd = '%s'
        try:
            f = os.popen(cmd % conf.prog.ifconfig)
        except OSError:
            log_runtime.warning('Failed to execute ifconfig.')
            return []
        splitted_line = []
        for line in f:
            if 'flags' in line:
                iface = line.split()[0].rstrip(':')
                splitted_line.append(iface)
    else:
        try:
            f = os.popen('%s -l' % conf.prog.ifconfig)
        except OSError:
            log_runtime.warning('Failed to execute ifconfig.')
            return []
        splitted_line = f.readline().rstrip().split()
    ret = []
    for i in splitted_line:
        ret += _in6_getifaddr(i)
    f.close()
    return ret

def read_routes6():
    if False:
        print('Hello World!')
    'Return a list of IPv6 routes than can be used by Scapy.\n\n    This function parses netstat.\n    '
    fd_netstat = os.popen('netstat -rn -f inet6')
    lifaddr = in6_getifaddr()
    if not lifaddr:
        fd_netstat.close()
        return []
    got_header = False
    mtu_present = False
    prio_present = False
    routes = []
    for line in fd_netstat.readlines():
        if not got_header:
            if 'Destination' == line[:11]:
                got_header = True
                mtu_present = 'Mtu' in line
                prio_present = 'Prio' in line
            continue
        splitted_line = line.split()
        if OPENBSD or NETBSD:
            index = 5 + mtu_present + prio_present
            if len(splitted_line) < index:
                warning('Not enough columns in route entry !')
                continue
            (destination, next_hop, flags) = splitted_line[:3]
            dev = splitted_line[index]
        else:
            if len(splitted_line) < 4:
                warning('Not enough columns in route entry !')
                continue
            (destination, next_hop, flags, dev) = splitted_line[:4]
        metric = 1
        if 'U' not in flags:
            continue
        if 'R' in flags:
            continue
        if 'm' in flags:
            continue
        if 'link' in next_hop:
            next_hop = '::'
        destination_plen = 128
        if '%' in destination:
            (destination, dev) = destination.split('%')
            if '/' in dev:
                (dev, destination_plen) = dev.split('/')
        if '%' in next_hop:
            (next_hop, dev) = next_hop.split('%')
        if not in6_isvalid(next_hop):
            next_hop = '::'
        if destination == 'default':
            (destination, destination_plen) = ('::', 0)
        elif '/' in destination:
            (destination, destination_plen) = destination.split('/')
        if '/' in dev:
            (dev, destination_plen) = dev.split('/')
        if not in6_isvalid(destination):
            warning('Invalid destination IPv6 address in route entry !')
            continue
        try:
            destination_plen = int(destination_plen)
        except Exception:
            warning('Invalid IPv6 prefix length in route entry !')
            continue
        if in6_ismlladdr(destination) or in6_ismnladdr(destination):
            continue
        if conf.loopback_name in dev:
            cset = ['::1']
            next_hop = '::'
        else:
            devaddrs = (x for x in lifaddr if x[2] == dev)
            cset = construct_source_candidate_set(destination, destination_plen, devaddrs)
        if len(cset):
            routes.append((destination, destination_plen, next_hop, dev, cset, metric))
    fd_netstat.close()
    return routes

def read_nameservers() -> List[str]:
    if False:
        return 10
    'Return the nameservers configured by the OS\n    '
    try:
        with open('/etc/resolv.conf', 'r') as fd:
            return re.findall('nameserver\\s+([^\\s]+)', fd.read())
    except FileNotFoundError:
        warning("Could not retrieve the OS's nameserver !")
        return []