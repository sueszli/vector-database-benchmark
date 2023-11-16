"""
c-ares based hostname resolver.
"""
from __future__ import absolute_import, print_function, division
import os
import warnings
from _socket import gaierror
from _socket import herror
from _socket import error
from _socket import EAI_NONAME
from gevent._compat import text_type
from gevent._compat import integer_types
from gevent.hub import Waiter
from gevent.hub import get_hub
from gevent.socket import AF_UNSPEC
from gevent.socket import AF_INET
from gevent.socket import AF_INET6
from gevent.socket import SOCK_DGRAM
from gevent.socket import SOCK_STREAM
from gevent.socket import SOL_TCP
from gevent.socket import SOL_UDP
from gevent._config import config
from gevent._config import AresSettingMixin
from .cares import channel, InvalidIP
from . import _lookup_port as lookup_port
from . import AbstractResolver
__all__ = ['Resolver']

class Resolver(AbstractResolver):
    """
    Implementation of the resolver API using the `c-ares`_ library.

    This implementation uses the c-ares library to handle name
    resolution. c-ares is natively asynchronous at the socket level
    and so integrates well into gevent's event loop.

    In comparison to :class:`gevent.resolver_thread.Resolver` (which
    delegates to the native system resolver), the implementation is
    much more complex. In addition, there have been reports of it not
    properly honoring certain system configurations (for example, the
    order in which IPv4 and IPv6 results are returned may not match
    the threaded resolver). However, because it does not use threads,
    it may scale better for applications that make many lookups.

    There are some known differences from the system resolver.

    - ``gethostbyname_ex`` and ``gethostbyaddr`` may return
      different for the ``aliaslist`` tuple member. (Sometimes the
      same, sometimes in a different order, sometimes a different
      alias altogether.)

    - ``gethostbyname_ex`` may return the ``ipaddrlist`` in a
      different order.

    - ``getaddrinfo`` does not return ``SOCK_RAW`` results.

    - ``getaddrinfo`` may return results in a different order.

    - Handling of ``.local`` (mDNS) names may be different, even
      if they are listed in the hosts file.

    - c-ares will not resolve ``broadcasthost``, even if listed in
      the hosts file prior to 2020-04-30.

    - This implementation may raise ``gaierror(4)`` where the
      system implementation would raise ``herror(1)`` or vice versa,
      with different error numbers. However, after 2020-04-30, this should be
      much reduced.

    - The results for ``localhost`` may be different. In
      particular, some system resolvers will return more results
      from ``getaddrinfo`` than c-ares does, such as SOCK_DGRAM
      results, and c-ares may report more ips on a multi-homed
      host.

    - The system implementation may return some names fully qualified, where
      this implementation returns only the host name. This appears to be
      the case only with entries found in ``/etc/hosts``.

    - c-ares supports a limited set of flags for ``getnameinfo`` and
      ``getaddrinfo``; unknown flags are ignored. System-specific flags
      such as ``AI_V4MAPPED_CFG`` are not supported.

    - ``getaddrinfo`` may return canonical names even without the ``AI_CANONNAME``
      being set.

    .. caution::

        This module is considered extremely experimental on PyPy, and
        due to its implementation in cython, it may be slower. It may also lead to
        interpreter crashes.

    .. versionchanged:: 1.5.0
       This version of gevent typically embeds c-ares 1.15.0 or newer. In
       that version of c-ares, domains ending in ``.onion`` `are never
       resolved <https://github.com/c-ares/c-ares/issues/196>`_ or even
       sent to the DNS server.

    .. versionchanged:: 20.5.0
       ``getaddrinfo`` is now implemented using the native c-ares function
       from c-ares 1.16 or newer.

    .. versionchanged:: 20.5.0
       Now ``herror`` and ``gaierror`` are raised more consistently with
       the standard library resolver, and have more consistent errno values.

       Handling of localhost and broadcast names is now more consistent.

    .. versionchanged:: 22.10.1
       Now has a ``__del__`` method that warns if the object is destroyed
       without being properly closed.

    .. _c-ares: http://c-ares.haxx.se
    """
    cares_class = channel

    def __init__(self, hub=None, use_environ=True, **kwargs):
        if False:
            i = 10
            return i + 15
        AbstractResolver.__init__(self)
        if hub is None:
            hub = get_hub()
        self.hub = hub
        if use_environ:
            for setting in config.settings.values():
                if isinstance(setting, AresSettingMixin):
                    value = setting.get()
                    if value is not None:
                        kwargs.setdefault(setting.kwarg_name, value)
        self.cares = self.cares_class(hub.loop, **kwargs)
        self.pid = os.getpid()
        self.params = kwargs
        self.fork_watcher = hub.loop.fork(ref=False)
        self.fork_watcher.start(self._on_fork)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<gevent.resolver_ares.Resolver at 0x%x ares=%r>' % (id(self), self.cares)

    def _on_fork(self):
        if False:
            for i in range(10):
                print('nop')
        pid = os.getpid()
        if pid != self.pid:
            self.hub.loop.run_callback(self.cares.destroy)
            self.cares = self.cares_class(self.hub.loop, **self.params)
            self.pid = pid

    def close(self):
        if False:
            i = 10
            return i + 15
        AbstractResolver.close(self)
        if self.cares is not None:
            self.hub.loop.run_callback(self.cares.destroy)
            self.cares = None
        self.fork_watcher.stop()

    def __del__(self):
        if False:
            print('Hello World!')
        if self.cares is not None:
            warnings.warn('cares Resolver destroyed while not closed', ResourceWarning)
            self.close()

    def _gethostbyname_ex(self, hostname_bytes, family):
        if False:
            return 10
        while True:
            ares = self.cares
            try:
                waiter = Waiter(self.hub)
                ares.gethostbyname(waiter, hostname_bytes, family)
                result = waiter.get()
                if not result[-1]:
                    raise herror(EAI_NONAME, self.EAI_NONAME_MSG)
                return result
            except herror as ex:
                if ares is self.cares:
                    if ex.args[0] == 1:
                        raise gaierror(EAI_NONAME, self.EAI_NONAME_MSG)
                    raise

    def _lookup_port(self, port, socktype):
        if False:
            for i in range(10):
                print('nop')
        return lookup_port(port, socktype)

    def __getaddrinfo(self, host, port, family=0, socktype=0, proto=0, flags=0, fill_in_type_proto=True):
        if False:
            while True:
                i = 10
        '\n        Returns a list ``(family, socktype, proto, canonname, sockaddr)``\n\n        :raises gaierror: If no results are found.\n        '
        if isinstance(host, text_type):
            host = host.encode('idna')
        if isinstance(port, text_type):
            port = port.encode('ascii')
        elif isinstance(port, integer_types):
            if port == 0:
                port = None
            else:
                port = str(port).encode('ascii')
        waiter = Waiter(self.hub)
        self.cares.getaddrinfo(waiter, host, port, family, socktype, proto, flags)
        result = waiter.get()
        if not result:
            raise gaierror(EAI_NONAME, self.EAI_NONAME_MSG)
        if fill_in_type_proto:
            if socktype:
                hard_type_proto = [(socktype, SOL_TCP if socktype == SOCK_STREAM else SOL_UDP)]
            elif proto:
                hard_type_proto = [(SOCK_STREAM if proto == SOL_TCP else SOCK_DGRAM, proto)]
            else:
                hard_type_proto = [(SOCK_STREAM, SOL_TCP), (SOCK_DGRAM, SOL_UDP)]
            result = [(rfamily, hard_type if not rtype else rtype, hard_proto if not rproto else rproto, rcanon, raddr) for (rfamily, rtype, rproto, rcanon, raddr) in result for (hard_type, hard_proto) in hard_type_proto]
        return result

    def _getaddrinfo(self, host_bytes, port, family, socktype, proto, flags):
        if False:
            while True:
                i = 10
        while True:
            ares = self.cares
            try:
                return self.__getaddrinfo(host_bytes, port, family, socktype, proto, flags)
            except gaierror:
                if ares is self.cares:
                    raise

    def __gethostbyaddr(self, ip_address):
        if False:
            print('Hello World!')
        waiter = Waiter(self.hub)
        try:
            self.cares.gethostbyaddr(waiter, ip_address)
            return waiter.get()
        except InvalidIP:
            result = self._getaddrinfo(ip_address, None, family=AF_UNSPEC, socktype=SOCK_DGRAM, proto=0, flags=0)
            if not result:
                raise
            _ip_address = result[0][-1][0]
            if isinstance(_ip_address, text_type):
                _ip_address = _ip_address.encode('ascii')
            if _ip_address == ip_address:
                raise
            waiter.clear()
            self.cares.gethostbyaddr(waiter, _ip_address)
            return waiter.get()

    def _gethostbyaddr(self, ip_address_bytes):
        if False:
            while True:
                i = 10
        while True:
            ares = self.cares
            try:
                return self.__gethostbyaddr(ip_address_bytes)
            except herror:
                if ares is self.cares:
                    raise

    def __getnameinfo(self, hostname, port, sockaddr, flags):
        if False:
            for i in range(10):
                print('nop')
        result = self.__getaddrinfo(hostname, port, family=AF_UNSPEC, socktype=SOCK_DGRAM, proto=0, flags=0, fill_in_type_proto=False)
        if len(result) != 1:
            raise error('sockaddr resolved to multiple addresses')
        (family, _socktype, _proto, _name, address) = result[0]
        if family == AF_INET:
            if len(sockaddr) != 2:
                raise error('IPv4 sockaddr must be 2 tuple')
        elif family == AF_INET6:
            address = address[:2] + sockaddr[2:]
        waiter = Waiter(self.hub)
        self.cares.getnameinfo(waiter, address, flags)
        (node, service) = waiter.get()
        if service is None:
            err = gaierror(EAI_NONAME, self.EAI_NONAME_MSG)
            err.errno = EAI_NONAME
            raise err
        return (node, service or '0')

    def _getnameinfo(self, address_bytes, port, sockaddr, flags):
        if False:
            return 10
        while True:
            ares = self.cares
            try:
                return self.__getnameinfo(address_bytes, port, sockaddr, flags)
            except gaierror:
                if ares is self.cares:
                    raise