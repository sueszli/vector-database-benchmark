"""
IPv6-aware hostname resolution.

@see: L{IHostnameResolver}
"""
from socket import AF_INET, AF_INET6, AF_UNSPEC, SOCK_DGRAM, SOCK_STREAM, AddressFamily, SocketKind, gaierror, getaddrinfo
from typing import TYPE_CHECKING, Callable, List, NoReturn, Optional, Sequence, Tuple, Type, Union
from zope.interface import implementer
from twisted.internet._idna import _idnaBytes
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import IAddress, IHostnameResolver, IHostResolution, IReactorThreads, IResolutionReceiver, IResolverSimple
from twisted.internet.threads import deferToThreadPool
from twisted.logger import Logger
from twisted.python.compat import nativeString
if TYPE_CHECKING:
    from twisted.python.threadpool import ThreadPool

@implementer(IHostResolution)
class HostResolution:
    """
    The in-progress resolution of a given hostname.
    """

    def __init__(self, name: str):
        if False:
            i = 10
            return i + 15
        '\n        Create a L{HostResolution} with the given name.\n        '
        self.name = name

    def cancel(self) -> NoReturn:
        if False:
            while True:
                i = 10
        raise NotImplementedError()
_any = frozenset([IPv4Address, IPv6Address])
_typesToAF = {frozenset([IPv4Address]): AF_INET, frozenset([IPv6Address]): AF_INET6, _any: AF_UNSPEC}
_afToType = {AF_INET: IPv4Address, AF_INET6: IPv6Address}
_transportToSocket = {'TCP': SOCK_STREAM, 'UDP': SOCK_DGRAM}
_socktypeToType = {SOCK_STREAM: 'TCP', SOCK_DGRAM: 'UDP'}
_GETADDRINFO_RESULT = List[Tuple[AddressFamily, SocketKind, int, str, Union[Tuple[str, int], Tuple[str, int, int, int]]]]

@implementer(IHostnameResolver)
class GAIResolver:
    """
    L{IHostnameResolver} implementation that resolves hostnames by calling
    L{getaddrinfo} in a thread.
    """

    def __init__(self, reactor: IReactorThreads, getThreadPool: Optional[Callable[[], 'ThreadPool']]=None, getaddrinfo: Callable[[str, int, int, int], _GETADDRINFO_RESULT]=getaddrinfo):
        if False:
            i = 10
            return i + 15
        "\n        Create a L{GAIResolver}.\n\n        @param reactor: the reactor to schedule result-delivery on\n        @type reactor: L{IReactorThreads}\n\n        @param getThreadPool: a function to retrieve the thread pool to use for\n            scheduling name resolutions.  If not supplied, the use the given\n            C{reactor}'s thread pool.\n        @type getThreadPool: 0-argument callable returning a\n            L{twisted.python.threadpool.ThreadPool}\n\n        @param getaddrinfo: a reference to the L{getaddrinfo} to use - mainly\n            parameterized for testing.\n        @type getaddrinfo: callable with the same signature as L{getaddrinfo}\n        "
        self._reactor = reactor
        self._getThreadPool = reactor.getThreadPool if getThreadPool is None else getThreadPool
        self._getaddrinfo = getaddrinfo

    def resolveHostName(self, resolutionReceiver: IResolutionReceiver, hostName: str, portNumber: int=0, addressTypes: Optional[Sequence[Type[IAddress]]]=None, transportSemantics: str='TCP') -> IHostResolution:
        if False:
            while True:
                i = 10
        '\n        See L{IHostnameResolver.resolveHostName}\n\n        @param resolutionReceiver: see interface\n\n        @param hostName: see interface\n\n        @param portNumber: see interface\n\n        @param addressTypes: see interface\n\n        @param transportSemantics: see interface\n\n        @return: see interface\n        '
        pool = self._getThreadPool()
        addressFamily = _typesToAF[_any if addressTypes is None else frozenset(addressTypes)]
        socketType = _transportToSocket[transportSemantics]

        def get() -> _GETADDRINFO_RESULT:
            if False:
                i = 10
                return i + 15
            try:
                return self._getaddrinfo(hostName, portNumber, addressFamily, socketType)
            except gaierror:
                return []
        d = deferToThreadPool(self._reactor, pool, get)
        resolution = HostResolution(hostName)
        resolutionReceiver.resolutionBegan(resolution)

        @d.addCallback
        def deliverResults(result: _GETADDRINFO_RESULT) -> None:
            if False:
                i = 10
                return i + 15
            for (family, socktype, proto, cannoname, sockaddr) in result:
                addrType = _afToType[family]
                resolutionReceiver.addressResolved(addrType(_socktypeToType.get(socktype, 'TCP'), *sockaddr))
            resolutionReceiver.resolutionComplete()
        return resolution

@implementer(IHostnameResolver)
class SimpleResolverComplexifier:
    """
    A converter from L{IResolverSimple} to L{IHostnameResolver}.
    """
    _log = Logger()

    def __init__(self, simpleResolver: IResolverSimple):
        if False:
            print('Hello World!')
        '\n        Construct a L{SimpleResolverComplexifier} with an L{IResolverSimple}.\n        '
        self._simpleResolver = simpleResolver

    def resolveHostName(self, resolutionReceiver: IResolutionReceiver, hostName: str, portNumber: int=0, addressTypes: Optional[Sequence[Type[IAddress]]]=None, transportSemantics: str='TCP') -> IHostResolution:
        if False:
            i = 10
            return i + 15
        '\n        See L{IHostnameResolver.resolveHostName}\n\n        @param resolutionReceiver: see interface\n\n        @param hostName: see interface\n\n        @param portNumber: see interface\n\n        @param addressTypes: see interface\n\n        @param transportSemantics: see interface\n\n        @return: see interface\n        '
        try:
            hostName_bytes = hostName.encode('ascii')
        except UnicodeEncodeError:
            hostName_bytes = _idnaBytes(hostName)
        hostName = nativeString(hostName_bytes)
        resolution = HostResolution(hostName)
        resolutionReceiver.resolutionBegan(resolution)
        self._simpleResolver.getHostByName(hostName).addCallback(lambda address: resolutionReceiver.addressResolved(IPv4Address('TCP', address, portNumber))).addErrback(lambda error: None if error.check(DNSLookupError) else self._log.failure('while looking up {name} with {resolver}', error, name=hostName, resolver=self._simpleResolver)).addCallback(lambda nothing: resolutionReceiver.resolutionComplete())
        return resolution

@implementer(IResolutionReceiver)
class FirstOneWins:
    """
    An L{IResolutionReceiver} which fires a L{Deferred} with its first result.
    """

    def __init__(self, deferred: 'Deferred[str]'):
        if False:
            return 10
        '\n        @param deferred: The L{Deferred} to fire when the first resolution\n            result arrives.\n        '
        self._deferred = deferred
        self._resolved = False

    def resolutionBegan(self, resolution: IHostResolution) -> None:
        if False:
            print('Hello World!')
        '\n        See L{IResolutionReceiver.resolutionBegan}\n\n        @param resolution: See L{IResolutionReceiver.resolutionBegan}\n        '
        self._resolution = resolution

    def addressResolved(self, address: IAddress) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        See L{IResolutionReceiver.addressResolved}\n\n        @param address: See L{IResolutionReceiver.addressResolved}\n        '
        if self._resolved:
            return
        self._resolved = True
        assert isinstance(address, IPv4Address)
        self._deferred.callback(address.host)

    def resolutionComplete(self) -> None:
        if False:
            return 10
        '\n        See L{IResolutionReceiver.resolutionComplete}\n        '
        if self._resolved:
            return
        self._deferred.errback(DNSLookupError(self._resolution.name))

@implementer(IResolverSimple)
class ComplexResolverSimplifier:
    """
    A converter from L{IHostnameResolver} to L{IResolverSimple}
    """

    def __init__(self, nameResolver: IHostnameResolver):
        if False:
            return 10
        '\n        Create a L{ComplexResolverSimplifier} with an L{IHostnameResolver}.\n\n        @param nameResolver: The L{IHostnameResolver} to use.\n        '
        self._nameResolver = nameResolver

    def getHostByName(self, name: str, timeouts: Sequence[int]=()) -> 'Deferred[str]':
        if False:
            print('Hello World!')
        '\n        See L{IResolverSimple.getHostByName}\n\n        @param name: see L{IResolverSimple.getHostByName}\n\n        @param timeouts: see L{IResolverSimple.getHostByName}\n\n        @return: see L{IResolverSimple.getHostByName}\n        '
        result: 'Deferred[str]' = Deferred()
        self._nameResolver.resolveHostName(FirstOneWins(result), name, 0, [IPv4Address])
        return result