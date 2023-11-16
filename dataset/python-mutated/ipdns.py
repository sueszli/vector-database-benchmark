__author__ = 'Fail2Ban Developers, Alexander Koeppe, Serg G. Brester, Yaroslav Halchenko'
__copyright__ = 'Copyright (c) 2004-2016 Fail2ban Developers'
__license__ = 'GPL'
import socket
import struct
import re
from .utils import Utils
from ..helpers import getLogger
logSys = getLogger(__name__)

def asip(ip):
    if False:
        for i in range(10):
            print('nop')
    'A little helper to guarantee ip being an IPAddr instance'
    if isinstance(ip, IPAddr):
        return ip
    return IPAddr(ip)

def getfqdn(name=''):
    if False:
        i = 10
        return i + 15
    'Get fully-qualified hostname of given host, thereby resolve of an external\n\tIPs and name will be preferred before the local domain (or a loopback), see gh-2438\n\t'
    try:
        name = name or socket.gethostname()
        names = (ai[3] for ai in socket.getaddrinfo(name, None, 0, socket.SOCK_DGRAM, 0, socket.AI_CANONNAME) if ai[3])
        if names:
            pref = name + '.'
            first = None
            for ai in names:
                if ai.startswith(pref):
                    return ai
                if not first:
                    first = ai
            return first
    except socket.error:
        pass
    return socket.getfqdn(name)

class DNSUtils:
    CACHE_nameToIp = Utils.Cache(maxCount=1000, maxTime=5 * 60)
    CACHE_ipToName = Utils.Cache(maxCount=1000, maxTime=5 * 60)

    @staticmethod
    def dnsToIp(dns):
        if False:
            return 10
        ' Convert a DNS into an IP address using the Python socket module.\n\t\t\tThanks to Kevin Drapel.\n\t\t'
        ips = DNSUtils.CACHE_nameToIp.get(dns)
        if ips is not None:
            return ips
        ips = set()
        saveerr = None
        for fam in (socket.AF_INET, socket.AF_INET6) if DNSUtils.IPv6IsAllowed() else (socket.AF_INET,):
            try:
                for result in socket.getaddrinfo(dns, None, fam, 0, socket.IPPROTO_TCP):
                    if len(result) < 4 or not len(result[4]):
                        continue
                    ip = IPAddr(str(result[4][0]), IPAddr._AF2FAM(fam))
                    if ip.isValid:
                        ips.add(ip)
            except Exception as e:
                saveerr = e
        if not ips and saveerr:
            logSys.warning('Unable to find a corresponding IP address for %s: %s', dns, saveerr)
        DNSUtils.CACHE_nameToIp.set(dns, ips)
        return ips

    @staticmethod
    def ipToName(ip):
        if False:
            return 10
        v = DNSUtils.CACHE_ipToName.get(ip, ())
        if v != ():
            return v
        try:
            v = socket.gethostbyaddr(ip)[0]
        except socket.error as e:
            logSys.debug('Unable to find a name for the IP %s: %s', ip, e)
            v = None
        DNSUtils.CACHE_ipToName.set(ip, v)
        return v

    @staticmethod
    def textToIp(text, useDns):
        if False:
            print('Hello World!')
        ' Return the IP of DNS found in a given text.\n\t\t'
        ipList = set()
        plainIP = IPAddr.searchIP(text)
        if plainIP is not None:
            ip = IPAddr(plainIP)
            if ip.isValid:
                ipList.add(ip)
        if useDns in ('yes', 'warn') and (not ipList):
            ip = DNSUtils.dnsToIp(text)
            ipList.update(ip)
            if ip and useDns == 'warn':
                logSys.warning('Determined IP using DNS Lookup: %s = %s', text, ipList)
        return ipList

    @staticmethod
    def getHostname(fqdn=True):
        if False:
            i = 10
            return i + 15
        'Get short hostname or fully-qualified hostname of host self'
        key = ('self', 'hostname', fqdn)
        name = DNSUtils.CACHE_ipToName.get(key)
        if name is not None:
            return name
        name = ''
        for hostname in (getfqdn, socket.gethostname) if fqdn else (socket.gethostname, getfqdn):
            try:
                name = hostname()
                break
            except Exception as e:
                logSys.warning('Retrieving own hostnames failed: %s', e)
        DNSUtils.CACHE_ipToName.set(key, name)
        return name
    _getSelfNames_key = ('self', 'dns')

    @staticmethod
    def getSelfNames():
        if False:
            i = 10
            return i + 15
        'Get own host names of self'
        names = DNSUtils.CACHE_ipToName.get(DNSUtils._getSelfNames_key)
        if names is not None:
            return names
        names = set(['localhost', DNSUtils.getHostname(False), DNSUtils.getHostname(True)]) - set([''])
        DNSUtils.CACHE_ipToName.set(DNSUtils._getSelfNames_key, names)
        return names
    _getNetIntrfIPs_key = ('netintrf', 'ips')

    @staticmethod
    def getNetIntrfIPs():
        if False:
            i = 10
            return i + 15
        'Get own IP addresses of self'
        ips = DNSUtils.CACHE_nameToIp.get(DNSUtils._getNetIntrfIPs_key)
        if ips is not None:
            return ips
        try:
            ips = IPAddrSet([a for (ni, a) in DNSUtils._NetworkInterfacesAddrs()])
        except:
            ips = IPAddrSet()
        DNSUtils.CACHE_nameToIp.set(DNSUtils._getNetIntrfIPs_key, ips)
        return ips
    _getSelfIPs_key = ('self', 'ips')

    @staticmethod
    def getSelfIPs():
        if False:
            for i in range(10):
                print('nop')
        'Get own IP addresses of self'
        ips = DNSUtils.CACHE_nameToIp.get(DNSUtils._getSelfIPs_key)
        if ips is not None:
            return ips
        ips = IPAddrSet(DNSUtils.getNetIntrfIPs())
        for hostname in DNSUtils.getSelfNames():
            try:
                ips |= IPAddrSet(DNSUtils.dnsToIp(hostname))
            except Exception as e:
                logSys.warning('Retrieving own IPs of %s failed: %s', hostname, e)
        DNSUtils.CACHE_nameToIp.set(DNSUtils._getSelfIPs_key, ips)
        return ips
    _IPv6IsAllowed = None

    @staticmethod
    def _IPv6IsSupportedBySystem():
        if False:
            return 10
        if not socket.has_ipv6:
            return False
        try:
            with open('/proc/sys/net/ipv6/conf/all/disable_ipv6', 'rb') as f:
                return not int(f.read())
        except:
            pass
        s = None
        try:
            s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            s.bind(('', 0))
            return True
        except Exception as e:
            if hasattr(e, 'errno'):
                import errno
                if e.errno < 0 or e.errno in (errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT):
                    return False
                if e.errno in (errno.EADDRINUSE, errno.EACCES):
                    return True
        finally:
            if s:
                s.close()
        return None

    @staticmethod
    def setIPv6IsAllowed(value):
        if False:
            for i in range(10):
                print('nop')
        DNSUtils._IPv6IsAllowed = value
        logSys.debug('IPv6 is %s', ('on' if value else 'off') if value is not None else 'auto')
        return value
    _IPv6IsAllowed_key = ('self', 'ipv6-allowed')

    @staticmethod
    def IPv6IsAllowed():
        if False:
            while True:
                i = 10
        if DNSUtils._IPv6IsAllowed is not None:
            return DNSUtils._IPv6IsAllowed
        v = DNSUtils.CACHE_nameToIp.get(DNSUtils._IPv6IsAllowed_key)
        if v is not None:
            return v
        v = DNSUtils._IPv6IsSupportedBySystem()
        if v is None:
            ips = DNSUtils.getNetIntrfIPs()
            if not ips:
                DNSUtils._IPv6IsAllowed = True
                try:
                    ips = DNSUtils.getSelfIPs()
                finally:
                    DNSUtils._IPv6IsAllowed = None
            v = any((':' in ip.ntoa for ip in ips))
        DNSUtils.CACHE_nameToIp.set(DNSUtils._IPv6IsAllowed_key, v)
        return v

class IPAddr(object):
    """Encapsulate functionality for IPv4 and IPv6 addresses
	"""
    IP_4_RE = '(?:\\d{1,3}\\.){3}\\d{1,3}'
    IP_6_RE = '(?:[0-9a-fA-F]{1,4}::?|:){1,7}(?:[0-9a-fA-F]{1,4}|(?<=:):)'
    IP_4_6_CRE = re.compile('^(?:(?P<IPv4>%s)|\\[?(?P<IPv6>%s)\\]?)$' % (IP_4_RE, IP_6_RE))
    IP_W_CIDR_CRE = re.compile('^(%s|%s)/(?:(\\d+)|(%s|%s))$' % (IP_4_RE, IP_6_RE, IP_4_RE, IP_6_RE))
    IP6_4COMPAT = None
    __slots__ = ('_family', '_addr', '_plen', '_maskplen', '_raw')
    CACHE_OBJ = Utils.Cache(maxCount=10000, maxTime=5 * 60)
    CIDR_RAW = -2
    CIDR_UNSPEC = -1
    FAM_IPv4 = CIDR_RAW - socket.AF_INET
    FAM_IPv6 = CIDR_RAW - socket.AF_INET6

    @staticmethod
    def _AF2FAM(v):
        if False:
            for i in range(10):
                print('nop')
        return IPAddr.CIDR_RAW - v

    def __new__(cls, ipstr, cidr=CIDR_UNSPEC):
        if False:
            for i in range(10):
                print('nop')
        if cidr == IPAddr.CIDR_UNSPEC and isinstance(ipstr, (tuple, list)):
            cidr = IPAddr.CIDR_RAW
        if cidr == IPAddr.CIDR_RAW:
            ip = super(IPAddr, cls).__new__(cls)
            ip.__init(ipstr, cidr)
            return ip
        args = (ipstr, cidr)
        ip = IPAddr.CACHE_OBJ.get(args)
        if ip is not None:
            return ip
        if cidr == IPAddr.CIDR_UNSPEC:
            (ipstr, cidr) = IPAddr.__wrap_ipstr(ipstr)
            args = (ipstr, cidr)
            if cidr != IPAddr.CIDR_UNSPEC:
                ip = IPAddr.CACHE_OBJ.get(args)
                if ip is not None:
                    return ip
        ip = super(IPAddr, cls).__new__(cls)
        ip.__init(ipstr, cidr)
        if ip._family != IPAddr.CIDR_RAW:
            IPAddr.CACHE_OBJ.set(args, ip)
        return ip

    @staticmethod
    def __wrap_ipstr(ipstr):
        if False:
            for i in range(10):
                print('nop')
        if len(ipstr) > 2 and ipstr[0] == '[' and (ipstr[-1] == ']'):
            ipstr = ipstr[1:-1]
        if '/' not in ipstr:
            return (ipstr, IPAddr.CIDR_UNSPEC)
        s = IPAddr.IP_W_CIDR_CRE.match(ipstr)
        if s is None:
            return (ipstr, IPAddr.CIDR_UNSPEC)
        s = list(s.groups())
        if s[2]:
            s[1] = IPAddr.masktoplen(s[2])
        del s[2]
        try:
            s[1] = int(s[1])
        except ValueError:
            return (ipstr, IPAddr.CIDR_UNSPEC)
        return s

    def __init(self, ipstr, cidr=CIDR_UNSPEC):
        if False:
            while True:
                i = 10
        ' initialize IP object by converting IP address string\n\t\t\tto binary to integer\n\t\t'
        self._family = socket.AF_UNSPEC
        self._addr = 0
        self._plen = 0
        self._maskplen = None
        self._raw = ipstr
        if cidr != IPAddr.CIDR_RAW:
            if cidr is not None and cidr < IPAddr.CIDR_RAW:
                family = [IPAddr.CIDR_RAW - cidr]
            else:
                family = [socket.AF_INET, socket.AF_INET6]
            for family in family:
                try:
                    binary = socket.inet_pton(family, ipstr)
                    self._family = family
                    break
                except socket.error:
                    continue
            if self._family == socket.AF_INET:
                (self._addr,) = struct.unpack('!L', binary)
                self._plen = 32
                if cidr is not None and cidr >= 0:
                    mask = ~(4294967295 >> cidr)
                    self._addr &= mask
                    self._plen = cidr
            elif self._family == socket.AF_INET6:
                (hi, lo) = struct.unpack('!QQ', binary)
                self._addr = hi << 64 | lo
                self._plen = 128
                if cidr is not None and cidr >= 0:
                    mask = ~(340282366920938463463374607431768211455 >> cidr)
                    self._addr &= mask
                    self._plen = cidr
                elif self.isInNet(IPAddr.IP6_4COMPAT):
                    self._addr = lo & 4294967295
                    self._family = socket.AF_INET
                    self._plen = 32
        else:
            self._family = IPAddr.CIDR_RAW

    def __repr__(self):
        if False:
            while True:
                i = 10
        return repr(self.ntoa)

    def __str__(self):
        if False:
            return 10
        return self.ntoa if isinstance(self.ntoa, str) else str(self.ntoa)

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        "IPAddr pickle-handler, that simply wraps IPAddr to the str\n\n\t\tReturns a string as instance to be pickled, because fail2ban-client can't\n\t\tunserialize IPAddr objects\n\t\t"
        return (str, (self.ntoa,))

    @property
    def addr(self):
        if False:
            print('Hello World!')
        return self._addr

    @property
    def family(self):
        if False:
            while True:
                i = 10
        return self._family
    FAM2STR = {socket.AF_INET: 'inet4', socket.AF_INET6: 'inet6'}

    @property
    def familyStr(self):
        if False:
            i = 10
            return i + 15
        return IPAddr.FAM2STR.get(self._family)

    @property
    def plen(self):
        if False:
            return 10
        return self._plen

    @property
    def raw(self):
        if False:
            for i in range(10):
                print('nop')
        "The raw address\n\n\t\tShould only be set to a non-empty string if prior address\n\t\tconversion wasn't possible\n\t\t"
        return self._raw

    @property
    def isValid(self):
        if False:
            return 10
        'Either the object corresponds to a valid IP address\n\t\t'
        return self._family != socket.AF_UNSPEC

    @property
    def isSingle(self):
        if False:
            while True:
                i = 10
        'Returns whether the object is a single IP address (not DNS and subnet)\n\t\t'
        return self._plen == {socket.AF_INET: 32, socket.AF_INET6: 128}.get(self._family, -1000)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if self._family == IPAddr.CIDR_RAW and (not isinstance(other, IPAddr)):
            return self._raw == other
        if not isinstance(other, IPAddr):
            if other is None:
                return False
            other = IPAddr(other)
        if self._family != other._family:
            return False
        if self._family == socket.AF_UNSPEC:
            return self._raw == other._raw
        return self._addr == other._addr and self._plen == other._plen

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self == other

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        if self._family == IPAddr.CIDR_RAW and (not isinstance(other, IPAddr)):
            return self._raw < other
        if not isinstance(other, IPAddr):
            if other is None:
                return False
            other = IPAddr(other)
        return self._family < other._family or self._addr < other._addr

    def __add__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, IPAddr):
            other = IPAddr(other)
        return '%s%s' % (self, other)

    def __radd__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, IPAddr):
            other = IPAddr(other)
        return '%s%s' % (other, self)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.ntoa)

    @property
    def hexdump(self):
        if False:
            for i in range(10):
                print('nop')
        'Hex representation of the IP address (for debug purposes)\n\t\t'
        if self._family == socket.AF_INET:
            return '%08x' % self._addr
        elif self._family == socket.AF_INET6:
            return '%032x' % self._addr
        else:
            return ''

    @property
    def ntoa(self):
        if False:
            for i in range(10):
                print('nop')
        ' represent IP object as text like the deprecated\n\t\t\tC pendant inet.ntoa but address family independent\n\t\t'
        add = ''
        if self.isIPv4:
            binary = struct.pack('!L', self._addr)
            if self._plen and self._plen < 32:
                add = '/%d' % self._plen
        elif self.isIPv6:
            hi = self._addr >> 64
            lo = self._addr & 18446744073709551615
            binary = struct.pack('!QQ', hi, lo)
            if self._plen and self._plen < 128:
                add = '/%d' % self._plen
        else:
            return self._raw
        return socket.inet_ntop(self._family, binary) + add

    def getPTR(self, suffix=None):
        if False:
            print('Hello World!')
        ' return the DNS PTR string of the provided IP address object\n\n\t\t\tIf "suffix" is provided it will be appended as the second and top\n\t\t\tlevel reverse domain.\n\t\t\tIf omitted it is implicitly set to the second and top level reverse\n\t\t\tdomain of the according IP address family\n\t\t'
        if self.isIPv4:
            exploded_ip = self.ntoa.split('.')
            if suffix is None:
                suffix = 'in-addr.arpa.'
        elif self.isIPv6:
            exploded_ip = self.hexdump
            if suffix is None:
                suffix = 'ip6.arpa.'
        else:
            return ''
        return '%s.%s' % ('.'.join(reversed(exploded_ip)), suffix)

    def getHost(self):
        if False:
            i = 10
            return i + 15
        'Return the host name (DNS) of the provided IP address object\n\t\t'
        return DNSUtils.ipToName(self.ntoa)

    @property
    def isIPv4(self):
        if False:
            for i in range(10):
                print('nop')
        'Either the IP object is of address family AF_INET\n\t\t'
        return self.family == socket.AF_INET

    @property
    def isIPv6(self):
        if False:
            while True:
                i = 10
        'Either the IP object is of address family AF_INET6\n\t\t'
        return self.family == socket.AF_INET6

    def isInNet(self, net):
        if False:
            i = 10
            return i + 15
        'Return either the IP object is in the provided network\n\t\t'
        if not net.isValid and net.raw != '':
            return self in DNSUtils.dnsToIp(net.raw)
        if self.family != net.family:
            return False
        if self.isIPv4:
            mask = ~(4294967295 >> net.plen)
        elif self.isIPv6:
            mask = ~(340282366920938463463374607431768211455 >> net.plen)
        else:
            return False
        return self.addr & mask == net.addr

    def contains(self, ip):
        if False:
            print('Hello World!')
        'Return whether the object (as network) contains given IP\n\t\t'
        return isinstance(ip, IPAddr) and (ip == self or ip.isInNet(self))

    def __contains__(self, ip):
        if False:
            for i in range(10):
                print('nop')
        return self.contains(ip)

    def __getMaskMap():
        if False:
            for i in range(10):
                print('nop')
        m6 = (1 << 128) - 1
        m4 = (1 << 32) - 1
        mmap = {m6: 128, m4: 32, 0: 0}
        m = 0
        for i in range(0, 128):
            m |= 1 << i
            if i < 32:
                mmap[m ^ m4] = 32 - 1 - i
            mmap[m ^ m6] = 128 - 1 - i
        return mmap
    MAP_ADDR2MASKPLEN = __getMaskMap()

    @property
    def maskplen(self):
        if False:
            while True:
                i = 10
        mplen = 0
        if self._maskplen is not None:
            return self._maskplen
        mplen = IPAddr.MAP_ADDR2MASKPLEN.get(self._addr)
        if mplen is None:
            raise ValueError('invalid mask %r, no plen representation' % (str(self),))
        self._maskplen = mplen
        return mplen

    @staticmethod
    def masktoplen(mask):
        if False:
            return 10
        'Convert mask string to prefix length\n\n\t\tTo be used only for IPv4 masks\n\t\t'
        return IPAddr(mask).maskplen

    @staticmethod
    def searchIP(text):
        if False:
            for i in range(10):
                print('nop')
        'Search if text is an IP address, and return it if so, else None\n\t\t'
        match = IPAddr.IP_4_6_CRE.match(text)
        if not match:
            return None
        ipstr = match.group('IPv4')
        if ipstr is not None and ipstr != '':
            return ipstr
        return match.group('IPv6')
IPAddr.IP6_4COMPAT = IPAddr('::ffff:0:0', 96)

class IPAddrSet(set):
    hasSubNet = False

    def __init__(self, ips=[]):
        if False:
            for i in range(10):
                print('nop')
        ips2 = set()
        for ip in ips:
            if not isinstance(ip, IPAddr):
                ip = IPAddr(ip)
            ips2.add(ip)
            self.hasSubNet |= not ip.isSingle
        set.__init__(self, ips2)

    def add(self, ip):
        if False:
            return 10
        if not isinstance(ip, IPAddr):
            ip = IPAddr(ip)
        self.hasSubNet |= not ip.isSingle
        set.add(self, ip)

    def __contains__(self, ip):
        if False:
            i = 10
            return i + 15
        if not isinstance(ip, IPAddr):
            ip = IPAddr(ip)
        return set.__contains__(self, ip) or (self.hasSubNet and any((n.contains(ip) for n in self)))

def _NetworkInterfacesAddrs(withMask=False):
    if False:
        while True:
            i = 10
    try:
        from ctypes import Structure, Union, POINTER, pointer, get_errno, cast, c_ushort, c_byte, c_void_p, c_char_p, c_uint, c_int, c_uint16, c_uint32
        import ctypes.util
        import ctypes

        class struct_sockaddr(Structure):
            _fields_ = [('sa_family', c_ushort), ('sa_data', c_byte * 14)]

        class struct_sockaddr_in(Structure):
            _fields_ = [('sin_family', c_ushort), ('sin_port', c_uint16), ('sin_addr', c_byte * 4)]

        class struct_sockaddr_in6(Structure):
            _fields_ = [('sin6_family', c_ushort), ('sin6_port', c_uint16), ('sin6_flowinfo', c_uint32), ('sin6_addr', c_byte * 16), ('sin6_scope_id', c_uint32)]

        class union_ifa_ifu(Union):
            _fields_ = [('ifu_broadaddr', POINTER(struct_sockaddr)), ('ifu_dstaddr', POINTER(struct_sockaddr))]

        class struct_ifaddrs(Structure):
            pass
        struct_ifaddrs._fields_ = [('ifa_next', POINTER(struct_ifaddrs)), ('ifa_name', c_char_p), ('ifa_flags', c_uint), ('ifa_addr', POINTER(struct_sockaddr)), ('ifa_netmask', POINTER(struct_sockaddr)), ('ifa_ifu', union_ifa_ifu), ('ifa_data', c_void_p)]
        libc = ctypes.CDLL(ctypes.util.find_library('c') or '')
        if not libc.getifaddrs:
            raise NotImplementedError('libc.getifaddrs is not available')

        def ifap_iter(ifap):
            if False:
                i = 10
                return i + 15
            ifa = ifap.contents
            while True:
                yield ifa
                if not ifa.ifa_next:
                    break
                ifa = ifa.ifa_next.contents

        def getfamaddr(ifa, withMask=False):
            if False:
                i = 10
                return i + 15
            sa = ifa.ifa_addr.contents
            fam = sa.sa_family
            if fam == socket.AF_INET:
                sa = cast(pointer(sa), POINTER(struct_sockaddr_in)).contents
                addr = socket.inet_ntop(fam, sa.sin_addr)
                if withMask:
                    nm = ifa.ifa_netmask.contents
                    if nm is not None and nm.sa_family == socket.AF_INET:
                        nm = cast(pointer(nm), POINTER(struct_sockaddr_in)).contents
                        addr += '/' + socket.inet_ntop(fam, nm.sin_addr)
                return IPAddr(addr)
            elif fam == socket.AF_INET6:
                sa = cast(pointer(sa), POINTER(struct_sockaddr_in6)).contents
                addr = socket.inet_ntop(fam, sa.sin6_addr)
                if withMask:
                    nm = ifa.ifa_netmask.contents
                    if nm is not None and nm.sa_family == socket.AF_INET6:
                        nm = cast(pointer(nm), POINTER(struct_sockaddr_in6)).contents
                        addr += '/' + socket.inet_ntop(fam, nm.sin6_addr)
                return IPAddr(addr)
            return None

        def _NetworkInterfacesAddrs(withMask=False):
            if False:
                for i in range(10):
                    print('nop')
            ifap = POINTER(struct_ifaddrs)()
            result = libc.getifaddrs(pointer(ifap))
            if result != 0:
                raise OSError(get_errno())
            del result
            try:
                for ifa in ifap_iter(ifap):
                    name = ifa.ifa_name.decode('UTF-8')
                    addr = getfamaddr(ifa, withMask)
                    if addr:
                        yield (name, addr)
            finally:
                libc.freeifaddrs(ifap)
    except Exception as e:
        _init_error = NotImplementedError(e)

        def _NetworkInterfacesAddrs():
            if False:
                print('Hello World!')
            raise _init_error
    DNSUtils._NetworkInterfacesAddrs = staticmethod(_NetworkInterfacesAddrs)
    return _NetworkInterfacesAddrs(withMask)
DNSUtils._NetworkInterfacesAddrs = staticmethod(_NetworkInterfacesAddrs)