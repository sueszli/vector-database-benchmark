"""Bridges between the Twisted package and Tornado.
"""
import socket
import sys
import twisted.internet.abstract
import twisted.internet.asyncioreactor
from twisted.internet.defer import Deferred
from twisted.python import failure
import twisted.names.cache
import twisted.names.client
import twisted.names.hosts
import twisted.names.resolve
from tornado.concurrent import Future, future_set_exc_info
from tornado.escape import utf8
from tornado import gen
from tornado.netutil import Resolver
import typing
if typing.TYPE_CHECKING:
    from typing import Generator, Any, List, Tuple

class TwistedResolver(Resolver):
    """Twisted-based asynchronous resolver.

    This is a non-blocking and non-threaded resolver.  It is
    recommended only when threads cannot be used, since it has
    limitations compared to the standard ``getaddrinfo``-based
    `~tornado.netutil.Resolver` and
    `~tornado.netutil.DefaultExecutorResolver`.  Specifically, it returns at
    most one result, and arguments other than ``host`` and ``family``
    are ignored.  It may fail to resolve when ``family`` is not
    ``socket.AF_UNSPEC``.

    Requires Twisted 12.1 or newer.

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    .. deprecated:: 6.2
       This class is deprecated and will be removed in Tornado 7.0. Use the default
       thread-based resolver instead.
    """

    def initialize(self) -> None:
        if False:
            return 10
        self.reactor = twisted.internet.asyncioreactor.AsyncioSelectorReactor()
        host_resolver = twisted.names.hosts.Resolver('/etc/hosts')
        cache_resolver = twisted.names.cache.CacheResolver(reactor=self.reactor)
        real_resolver = twisted.names.client.Resolver('/etc/resolv.conf', reactor=self.reactor)
        self.resolver = twisted.names.resolve.ResolverChain([host_resolver, cache_resolver, real_resolver])

    @gen.coroutine
    def resolve(self, host: str, port: int, family: int=0) -> 'Generator[Any, Any, List[Tuple[int, Any]]]':
        if False:
            i = 10
            return i + 15
        if twisted.internet.abstract.isIPAddress(host):
            resolved = host
            resolved_family = socket.AF_INET
        elif twisted.internet.abstract.isIPv6Address(host):
            resolved = host
            resolved_family = socket.AF_INET6
        else:
            deferred = self.resolver.getHostByName(utf8(host))
            fut = Future()
            deferred.addBoth(fut.set_result)
            resolved = (yield fut)
            if isinstance(resolved, failure.Failure):
                try:
                    resolved.raiseException()
                except twisted.names.error.DomainError as e:
                    raise IOError(e)
            elif twisted.internet.abstract.isIPAddress(resolved):
                resolved_family = socket.AF_INET
            elif twisted.internet.abstract.isIPv6Address(resolved):
                resolved_family = socket.AF_INET6
            else:
                resolved_family = socket.AF_UNSPEC
        if family != socket.AF_UNSPEC and family != resolved_family:
            raise Exception('Requested socket family %d but got %d' % (family, resolved_family))
        result = [(typing.cast(int, resolved_family), (resolved, port))]
        return result

def install() -> None:
    if False:
        print('Hello World!')
    'Install ``AsyncioSelectorReactor`` as the default Twisted reactor.\n\n    .. deprecated:: 5.1\n\n       This function is provided for backwards compatibility; code\n       that does not require compatibility with older versions of\n       Tornado should use\n       ``twisted.internet.asyncioreactor.install()`` directly.\n\n    .. versionchanged:: 6.0.3\n\n       In Tornado 5.x and before, this function installed a reactor\n       based on the Tornado ``IOLoop``. When that reactor\n       implementation was removed in Tornado 6.0.0, this function was\n       removed as well. It was restored in Tornado 6.0.3 using the\n       ``asyncio`` reactor instead.\n\n    '
    from twisted.internet.asyncioreactor import install
    install()
if hasattr(gen.convert_yielded, 'register'):

    @gen.convert_yielded.register(Deferred)
    def _(d: Deferred) -> Future:
        if False:
            for i in range(10):
                print('nop')
        f = Future()

        def errback(failure: failure.Failure) -> None:
            if False:
                while True:
                    i = 10
            try:
                failure.raiseException()
                raise Exception('errback called without error')
            except:
                future_set_exc_info(f, sys.exc_info())
        d.addCallbacks(f.set_result, errback)
        return f