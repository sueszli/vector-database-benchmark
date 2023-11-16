"""
Routing and network interface handling for IPv6.
"""
import socket
from scapy.config import conf
from scapy.interfaces import resolve_iface, NetworkInterface
from scapy.utils6 import in6_ptop, in6_cidr2mask, in6_and, in6_islladdr, in6_ismlladdr, in6_isincluded, in6_isgladdr, in6_isaddr6to4, in6_ismaddr, construct_source_candidate_set, get_source_addr_from_candidate_set
from scapy.arch import read_routes6, in6_getifaddr
from scapy.pton_ntop import inet_pton, inet_ntop
from scapy.error import warning, log_loading
from scapy.utils import pretty_list
from typing import Any, Dict, List, Optional, Set, Tuple, Union

class Route6:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.resync()
        self.invalidate_cache()

    def invalidate_cache(self):
        if False:
            for i in range(10):
                print('nop')
        self.cache = {}

    def flush(self):
        if False:
            i = 10
            return i + 15
        self.invalidate_cache()
        self.ipv6_ifaces = set()
        self.routes = []

    def resync(self):
        if False:
            return 10
        self.invalidate_cache()
        self.routes = read_routes6()
        self.ipv6_ifaces = set()
        for route in self.routes:
            self.ipv6_ifaces.add(route[3])
        if self.routes == []:
            log_loading.info('No IPv6 support in kernel')

    def __repr__(self):
        if False:
            print('Hello World!')
        rtlst = []
        for (net, msk, gw, iface, cset, metric) in self.routes:
            if_repr = resolve_iface(iface).description
            rtlst.append(('%s/%i' % (net, msk), gw, if_repr, cset, str(metric)))
        return pretty_list(rtlst, [('Destination', 'Next Hop', 'Iface', 'Src candidates', 'Metric')], sortBy=1)

    def make_route(self, dst, gw=None, dev=None):
        if False:
            while True:
                i = 10
        "Internal function : create a route for 'dst' via 'gw'.\n        "
        (prefix, plen_b) = (dst.split('/') + ['128'])[:2]
        plen = int(plen_b)
        if gw is None:
            gw = '::'
        if dev is None:
            (dev, ifaddr_uniq, x) = self.route(gw)
            ifaddr = [ifaddr_uniq]
        else:
            lifaddr = in6_getifaddr()
            devaddrs = (x for x in lifaddr if x[2] == dev)
            ifaddr = construct_source_candidate_set(prefix, plen, devaddrs)
        self.ipv6_ifaces.add(dev)
        return (prefix, plen, gw, dev, ifaddr, 1)

    def add(self, *args, **kargs):
        if False:
            while True:
                i = 10
        'Ex:\n        add(dst="2001:db8:cafe:f000::/56")\n        add(dst="2001:db8:cafe:f000::/56", gw="2001:db8:cafe::1")\n        add(dst="2001:db8:cafe:f000::/64", gw="2001:db8:cafe::1", dev="eth0")\n        '
        self.invalidate_cache()
        self.routes.append(self.make_route(*args, **kargs))

    def remove_ipv6_iface(self, iface):
        if False:
            i = 10
            return i + 15
        "\n        Remove the network interface 'iface' from the list of interfaces\n        supporting IPv6.\n        "
        if not all((r[3] == iface for r in conf.route6.routes)):
            try:
                self.ipv6_ifaces.remove(iface)
            except KeyError:
                pass

    def delt(self, dst, gw=None):
        if False:
            i = 10
            return i + 15
        ' Ex:\n        delt(dst="::/0")\n        delt(dst="2001:db8:cafe:f000::/56")\n        delt(dst="2001:db8:cafe:f000::/56", gw="2001:db8:deca::1")\n        '
        tmp = dst + '/128'
        (dst, plen_b) = tmp.split('/')[:2]
        dst = in6_ptop(dst)
        plen = int(plen_b)
        to_del = [x for x in self.routes if in6_ptop(x[0]) == dst and x[1] == plen]
        if gw:
            gw = in6_ptop(gw)
            to_del = [x for x in self.routes if in6_ptop(x[2]) == gw]
        if len(to_del) == 0:
            warning('No matching route found')
        elif len(to_del) > 1:
            warning('Found more than one match. Aborting.')
        else:
            i = self.routes.index(to_del[0])
            self.invalidate_cache()
            self.remove_ipv6_iface(self.routes[i][3])
            del self.routes[i]

    def ifchange(self, iff, addr):
        if False:
            i = 10
            return i + 15
        (the_addr, the_plen_b) = (addr.split('/') + ['128'])[:2]
        the_plen = int(the_plen_b)
        naddr = inet_pton(socket.AF_INET6, the_addr)
        nmask = in6_cidr2mask(the_plen)
        the_net = inet_ntop(socket.AF_INET6, in6_and(nmask, naddr))
        for (i, route) in enumerate(self.routes):
            (net, plen, gw, iface, _, metric) = route
            if iface != iff:
                continue
            self.ipv6_ifaces.add(iface)
            if gw == '::':
                self.routes[i] = (the_net, the_plen, gw, iface, [the_addr], metric)
            else:
                self.routes[i] = (net, plen, gw, iface, [the_addr], metric)
        self.invalidate_cache()
        conf.netcache.in6_neighbor.flush()

    def ifdel(self, iff):
        if False:
            while True:
                i = 10
        " removes all route entries that uses 'iff' interface. "
        new_routes = []
        for rt in self.routes:
            if rt[3] != iff:
                new_routes.append(rt)
        self.invalidate_cache()
        self.routes = new_routes
        self.remove_ipv6_iface(iff)

    def ifadd(self, iff, addr):
        if False:
            print('Hello World!')
        "\n        Add an interface 'iff' with provided address into routing table.\n\n        Ex: ifadd('eth0', '2001:bd8:cafe:1::1/64') will add following entry into  # noqa: E501\n            Scapy6 internal routing table:\n\n            Destination           Next Hop  iface  Def src @           Metric\n            2001:bd8:cafe:1::/64  ::        eth0   2001:bd8:cafe:1::1  1\n\n            prefix length value can be omitted. In that case, a value of 128\n            will be used.\n        "
        (addr, plen_b) = (addr.split('/') + ['128'])[:2]
        addr = in6_ptop(addr)
        plen = int(plen_b)
        naddr = inet_pton(socket.AF_INET6, addr)
        nmask = in6_cidr2mask(plen)
        prefix = inet_ntop(socket.AF_INET6, in6_and(nmask, naddr))
        self.invalidate_cache()
        self.routes.append((prefix, plen, '::', iff, [addr], 1))
        self.ipv6_ifaces.add(iff)

    def route(self, dst='', dev=None, verbose=conf.verb):
        if False:
            print('Hello World!')
        "\n        Provide best route to IPv6 destination address, based on Scapy\n        internal routing table content.\n\n        When a set of address is passed (e.g. ``2001:db8:cafe:*::1-5``) an\n        address of the set is used. Be aware of that behavior when using\n        wildcards in upper parts of addresses !\n\n        If 'dst' parameter is a FQDN, name resolution is performed and result\n        is used.\n\n        if optional 'dev' parameter is provided a specific interface, filtering\n        is performed to limit search to route associated to that interface.\n        "
        dst = dst or '::/0'
        dst = dst.split('/')[0]
        savedst = dst
        dst = dst.replace('*', '0')
        idx = dst.find('-')
        while idx >= 0:
            m = (dst[idx:] + ':').find(':')
            dst = dst[:idx] + dst[idx + m:]
            idx = dst.find('-')
        try:
            inet_pton(socket.AF_INET6, dst)
        except socket.error:
            dst = socket.getaddrinfo(savedst, None, socket.AF_INET6)[0][-1][0]
        if dev is None and (in6_islladdr(dst) or in6_ismlladdr(dst)):
            dev = conf.iface
            if dev not in self.ipv6_ifaces and self.ipv6_ifaces:
                tmp_routes = [route for route in self.routes if route[3] != conf.iface]
                default_routes = [route for route in tmp_routes if (route[0], route[1]) == ('::', 0)]
                ll_routes = [route for route in tmp_routes if (route[0], route[1]) == ('fe80::', 64)]
                if default_routes:
                    dev = default_routes[0][3]
                elif ll_routes:
                    dev = ll_routes[0][3]
                else:
                    dev = conf.loopback_name
                warning('The conf.iface interface (%s) does not support IPv6! Using %s instead for routing!' % (conf.iface, dev))
        k = dst
        if dev is not None:
            k = dst + '%%' + dev
        if k in self.cache:
            return self.cache[k]
        paths = []
        for (p, plen, gw, iface, cset, me) in self.routes:
            if dev is not None and iface != dev:
                continue
            if in6_isincluded(dst, p, plen):
                paths.append((plen, me, (iface, cset, gw)))
            elif in6_ismlladdr(dst) and in6_islladdr(p) and in6_islladdr(cset[0]):
                paths.append((plen, me, (iface, cset, gw)))
        if not paths:
            if dst == '::1':
                return (conf.loopback_name, '::1', '::')
            else:
                if verbose:
                    warning('No route found for IPv6 destination %s (no default route?)', dst)
                return (conf.loopback_name, '::', '::')
        paths.sort(key=lambda x: (-x[0], x[1]))
        best_plen = (paths[0][0], paths[0][1])
        paths = [x for x in paths if (x[0], x[1]) == best_plen]
        res = []
        for path in paths:
            tmp_c = path[2]
            srcaddr = get_source_addr_from_candidate_set(dst, tmp_c[1])
            if srcaddr is not None:
                res.append((path[0], path[1], (tmp_c[0], srcaddr, tmp_c[2])))
        if res == []:
            warning("Found a route for IPv6 destination '%s', but no possible source address.", dst)
            return (conf.loopback_name, '::', '::')
        if len(res) > 1:
            tmp = []
            if in6_isgladdr(dst) and in6_isaddr6to4(dst):
                tmp = [x for x in res if in6_isaddr6to4(x[2][1])]
            elif in6_ismaddr(dst) or in6_islladdr(dst):
                tmp = [x for x in res if x[2][0] == conf.iface]
            if tmp:
                res = tmp
        k = dst
        if dev is not None:
            k = dst + '%%' + dev
        self.cache[k] = res[0][2]
        return res[0][2]
conf.route6 = Route6()