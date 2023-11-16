from __future__ import absolute_import, print_function, division
import sys
import time
from _socket import error
from _socket import gaierror
from _socket import herror
from _socket import NI_NUMERICSERV
from _socket import AF_INET
from _socket import AF_INET6
from _socket import AF_UNSPEC
from _socket import EAI_NONAME
from _socket import EAI_FAMILY
import socket
from gevent.resolver import AbstractResolver
from gevent.resolver._hostsfile import HostsFile
from gevent.builtins import __import__ as g_import
from gevent._compat import string_types
from gevent._compat import iteritems
from gevent._config import config
__all__ = ['Resolver']

def _patch_dns():
    if False:
        i = 10
        return i + 15
    from gevent._patcher import import_patched as importer
    extras = {'dns': ('rdata', 'resolver', 'rdtypes'), 'dns.rdtypes': ('IN', 'ANY'), 'dns.rdtypes.IN': ('A', 'AAAA'), 'dns.rdtypes.ANY': ('SOA', 'PTR')}

    def extra_all(mod_name):
        if False:
            print('Hello World!')
        return extras.get(mod_name, ())

    def after_import_hook(dns):
        if False:
            for i in range(10):
                print('nop')
        rdata = dns.rdata
        get_rdata_class = rdata.get_rdata_class
        try:
            rdclass_values = list(dns.rdataclass.RdataClass)
        except AttributeError:
            rdclass_values = dns.rdataclass._by_value
        try:
            rdtype_values = list(dns.rdatatype.RdataType)
        except AttributeError:
            rdtype_values = dns.rdatatype._by_value
        for rdclass in rdclass_values:
            for rdtype in rdtype_values:
                get_rdata_class(rdclass, rdtype)
    patcher = importer('dns', extra_all, after_import_hook)
    top = patcher.module

    def _no_dynamic_imports(name):
        if False:
            print('Hello World!')
        raise ValueError(name)
    top.rdata.__import__ = _no_dynamic_imports
    return top
dns = _patch_dns()
resolver = dns.resolver
dTimeout = dns.resolver.Timeout

def _getaddrinfo(host=None, service=None, family=AF_UNSPEC, socktype=0, proto=0, flags=0, _orig_gai=resolver._getaddrinfo, _exc_clear=getattr(sys, 'exc_clear', lambda : None)):
    if False:
        print('Hello World!')
    if flags & (socket.AI_ADDRCONFIG | socket.AI_V4MAPPED) != 0:
        raise socket.gaierror(socket.EAI_SYSTEM)
    res = _orig_gai(host, service, family, socktype, proto, flags)
    _exc_clear()
    return res
resolver._getaddrinfo = _getaddrinfo
HOSTS_TTL = 300.0

class _HostsAnswer(dns.resolver.Answer):

    def __init__(self, qname, rdtype, rdclass, rrset, raise_on_no_answer=True):
        if False:
            print('Hello World!')
        self.response = None
        self.qname = qname
        self.rdtype = rdtype
        self.rdclass = rdclass
        self.canonical_name = qname
        if not rrset and raise_on_no_answer:
            raise dns.resolver.NoAnswer()
        self.rrset = rrset
        self.expiration = time.time() + rrset.ttl if hasattr(rrset, 'ttl') else 0

class _HostsResolver(object):
    """
    Class to parse the hosts file
    """

    def __init__(self, fname=None, interval=HOSTS_TTL):
        if False:
            return 10
        self.hosts_file = HostsFile(fname)
        self.interval = interval
        self._last_load = 0

    def query(self, qname, rdtype=dns.rdatatype.A, rdclass=dns.rdataclass.IN, tcp=False, source=None, raise_on_no_answer=True):
        if False:
            return 10
        now = time.time()
        hosts_file = self.hosts_file
        if self._last_load + self.interval < now:
            self._last_load = now
            hosts_file.load()
        rdclass = dns.rdataclass.IN
        if isinstance(qname, string_types):
            name = qname
            qname = dns.name.from_text(qname)
        else:
            name = str(qname)
        name = name.lower()
        rrset = dns.rrset.RRset(qname, rdclass, rdtype)
        rrset.ttl = self._last_load + self.interval - now
        if rdtype == dns.rdatatype.A:
            mapping = hosts_file.v4
            kind = dns.rdtypes.IN.A.A
        elif rdtype == dns.rdatatype.AAAA:
            mapping = hosts_file.v6
            kind = dns.rdtypes.IN.AAAA.AAAA
        elif rdtype == dns.rdatatype.CNAME:
            mapping = hosts_file.aliases
            kind = lambda c, t, addr: dns.rdtypes.ANY.CNAME.CNAME(c, t, dns.name.from_text(addr))
        elif rdtype == dns.rdatatype.PTR:
            mapping = hosts_file.reverse
            kind = lambda c, t, addr: dns.rdtypes.ANY.PTR.PTR(c, t, dns.name.from_text(addr))
        addr = mapping.get(name)
        if not addr and qname.is_absolute():
            addr = mapping.get(name[:-1])
        if addr:
            rrset.add(kind(rdclass, rdtype, addr))
        return _HostsAnswer(qname, rdtype, rdclass, rrset, raise_on_no_answer)

    def getaliases(self, hostname):
        if False:
            print('Hello World!')
        aliases = self.hosts_file.aliases
        result = []
        if hostname in aliases:
            cannon = aliases[hostname]
        else:
            cannon = hostname
        result.append(cannon)
        for (alias, cname) in iteritems(aliases):
            if cannon == cname:
                result.append(alias)
        result.remove(hostname)
        return result

class _DualResolver(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.hosts_resolver = _HostsResolver()
        self.network_resolver = resolver.get_default_resolver()
        self.network_resolver.cache = resolver.LRUCache()

    def query(self, qname, rdtype=dns.rdatatype.A, rdclass=dns.rdataclass.IN, tcp=False, source=None, raise_on_no_answer=True, _hosts_rdtypes=(dns.rdatatype.A, dns.rdatatype.AAAA, dns.rdatatype.PTR)):
        if False:
            i = 10
            return i + 15
        if qname is None:
            qname = '0.0.0.0'
        if not isinstance(qname, string_types):
            if isinstance(qname, bytes):
                qname = qname.decode('idna')
        if isinstance(qname, string_types):
            qname = dns.name.from_text(qname, None)
        if isinstance(rdtype, string_types):
            rdtype = dns.rdatatype.from_text(rdtype)
        if rdclass == dns.rdataclass.IN and rdtype in _hosts_rdtypes:
            try:
                answer = self.hosts_resolver.query(qname, rdtype, raise_on_no_answer=False)
            except Exception:
                from gevent import get_hub
                get_hub().handle_error(self, *sys.exc_info())
            else:
                if answer.rrset:
                    return answer
        return self.network_resolver.query(qname, rdtype, rdclass, tcp, source, raise_on_no_answer=raise_on_no_answer)

def _family_to_rdtype(family):
    if False:
        return 10
    if family == socket.AF_INET:
        rdtype = dns.rdatatype.A
    elif family == socket.AF_INET6:
        rdtype = dns.rdatatype.AAAA
    else:
        raise socket.gaierror(socket.EAI_FAMILY, 'Address family not supported')
    return rdtype

class Resolver(AbstractResolver):
    """
    An *experimental* resolver that uses `dnspython`_.

    This is typically slower than the default threaded resolver
    (unless there's a cache hit, in which case it can be much faster).
    It is usually much faster than the c-ares resolver. It tends to
    scale well as more concurrent resolutions are attempted.

    Under Python 2, if the ``idna`` package is installed, this
    resolver can resolve Unicode host names that the system resolver
    cannot.

    .. note::

        This **does not** use dnspython's default resolver object, or share any
        classes with ``import dns``. A separate copy of the objects is imported to
        be able to function in a non monkey-patched process. The documentation for the resolver
        object still applies.

        The resolver that we use is available as the :attr:`resolver` attribute
        of this object (typically ``gevent.get_hub().resolver.resolver``).

    .. caution::

        Many of the same caveats about DNS results apply here as are documented
        for :class:`gevent.resolver.ares.Resolver`. In addition, the handling of
        symbolic scope IDs in IPv6 addresses passed to ``getaddrinfo`` exhibits
        some differences.

        On PyPy, ``getnameinfo`` can produce results when CPython raises
        ``socket.error``, and gevent's DNSPython resolver also
        raises ``socket.error``.

    .. caution::

        This resolver is experimental. It may be removed or modified in
        the future. As always, feedback is welcome.

    .. versionadded:: 1.3a2

    .. versionchanged:: 20.5.0
       The errors raised are now much more consistent with those
       raised by the standard library resolvers.

       Handling of localhost and broadcast names is now more consistent.

    .. _dnspython: http://www.dnspython.org
    """

    def __init__(self, hub=None):
        if False:
            return 10
        if resolver._resolver is None:
            _resolver = resolver._resolver = _DualResolver()
            if config.resolver_nameservers:
                _resolver.network_resolver.nameservers[:] = config.resolver_nameservers
            if config.resolver_timeout:
                _resolver.network_resolver.lifetime = config.resolver_timeout
        assert isinstance(resolver._resolver, _DualResolver)
        self._resolver = resolver._resolver

    @property
    def resolver(self):
        if False:
            print('Hello World!')
        '\n        The dnspython resolver object we use.\n\n        This object has several useful attributes that can be used to\n        adjust the behaviour of the DNS system:\n\n        * ``cache`` is a :class:`dns.resolver.LRUCache`. Its maximum size\n          can be configured by calling :meth:`resolver.cache.set_max_size`\n        * ``nameservers`` controls which nameservers to talk to\n        * ``lifetime`` configures a timeout for each individual query.\n        '
        return self._resolver.network_resolver

    def close(self):
        if False:
            while True:
                i = 10
        pass

    def _getaliases(self, hostname, family):
        if False:
            return 10
        if not isinstance(hostname, str):
            if isinstance(hostname, bytes):
                hostname = hostname.decode('idna')
        aliases = self._resolver.hosts_resolver.getaliases(hostname)
        net_resolver = self._resolver.network_resolver
        rdtype = _family_to_rdtype(family)
        while 1:
            try:
                ans = net_resolver.query(hostname, dns.rdatatype.CNAME, rdtype)
            except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers):
                break
            except dTimeout:
                break
            except AttributeError as ex:
                if hostname is None or isinstance(hostname, int):
                    raise TypeError(ex)
                raise
            else:
                aliases.extend((str(rr.target) for rr in ans.rrset))
                hostname = ans[0].target
        return aliases

    def _getaddrinfo(self, host_bytes, port, family, socktype, proto, flags):
        if False:
            return 10
        if not isinstance(host_bytes, str):
            host_bytes = host_bytes.decode(self.HOSTNAME_ENCODING)
        if host_bytes == 'ff02::1de:c0:face:8D':
            host_bytes = 'ff02::1de:c0:face:8d'
        if family == AF_UNSPEC:
            try:
                return _getaddrinfo(host_bytes, port, family, socktype, proto, flags)
            except gaierror:
                try:
                    return _getaddrinfo(host_bytes, port, AF_INET6, socktype, proto, flags)
                except gaierror:
                    return _getaddrinfo(host_bytes, port, AF_INET, socktype, proto, flags)
        else:
            try:
                return _getaddrinfo(host_bytes, port, family, socktype, proto, flags)
            except gaierror as ex:
                if ex.args[0] == EAI_NONAME and family not in self._KNOWN_ADDR_FAMILIES:
                    ex.args = (EAI_FAMILY, self.EAI_FAMILY_MSG)
                    ex.errno = EAI_FAMILY
                raise

    def _getnameinfo(self, address_bytes, port, sockaddr, flags):
        if False:
            print('Hello World!')
        try:
            return resolver._getnameinfo(sockaddr, flags)
        except error:
            if not flags:
                return resolver._getnameinfo(sockaddr, NI_NUMERICSERV)

    def _gethostbyaddr(self, ip_address_bytes):
        if False:
            i = 10
            return i + 15
        try:
            return resolver._gethostbyaddr(ip_address_bytes)
        except gaierror as ex:
            if ex.args[0] == EAI_NONAME:
                raise herror(1, 'Unknown host')
            raise
    getnameinfo = AbstractResolver.fixup_gaierror(AbstractResolver.getnameinfo)
    gethostbyaddr = AbstractResolver.fixup_gaierror(AbstractResolver.gethostbyaddr)
    gethostbyname_ex = AbstractResolver.fixup_gaierror(AbstractResolver.gethostbyname_ex)
    getaddrinfo = AbstractResolver.fixup_gaierror(AbstractResolver.getaddrinfo)