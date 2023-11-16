from socket import AF_INET, AF_INET6, AF_UNSPEC, SOCK_DGRAM, SOCK_STREAM, AddressFamily, SocketKind, gaierror, getaddrinfo
from typing import TYPE_CHECKING, Callable, List, NoReturn, Optional, Sequence, Tuple, Type, Union
from zope.interface import implementer
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.interfaces import IAddress, IHostnameResolver, IHostResolution, IReactorThreads, IResolutionReceiver
from twisted.internet.threads import deferToThreadPool
if TYPE_CHECKING:
    from twisted.python.runtime import platform
    if platform.supportsThreads():
        from twisted.python.threadpool import ThreadPool
    else:
        ThreadPool = object

@implementer(IHostResolution)
class HostResolution:
    """
    The in-progress resolution of a given hostname.
    """

    def __init__(self, name: str):
        if False:
            print('Hello World!')
        '\n        Create a L{HostResolution} with the given name.\n        '
        self.name = name

    def cancel(self) -> NoReturn:
        if False:
            return 10
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
        "\n        Create a L{GAIResolver}.\n        @param reactor: the reactor to schedule result-delivery on\n        @type reactor: L{IReactorThreads}\n        @param getThreadPool: a function to retrieve the thread pool to use for\n            scheduling name resolutions.  If not supplied, the use the given\n            C{reactor}'s thread pool.\n        @type getThreadPool: 0-argument callable returning a\n            L{twisted.python.threadpool.ThreadPool}\n        @param getaddrinfo: a reference to the L{getaddrinfo} to use - mainly\n            parameterized for testing.\n        @type getaddrinfo: callable with the same signature as L{getaddrinfo}\n        "
        self._reactor = reactor
        self._getThreadPool = reactor.getThreadPool if getThreadPool is None else getThreadPool
        self._getaddrinfo = getaddrinfo

    def resolveHostName(self, resolutionReceiver: IResolutionReceiver, hostName: str, portNumber: int=0, addressTypes: Optional[Sequence[Type[IAddress]]]=None, transportSemantics: str='TCP') -> IHostResolution:
        if False:
            while True:
                i = 10
        '\n        See L{IHostnameResolver.resolveHostName}\n        @param resolutionReceiver: see interface\n        @param hostName: see interface\n        @param portNumber: see interface\n        @param addressTypes: see interface\n        @param transportSemantics: see interface\n        @return: see interface\n        '
        pool = self._getThreadPool()
        addressFamily = _typesToAF[_any if addressTypes is None else frozenset(addressTypes)]
        socketType = _transportToSocket[transportSemantics]

        def get() -> _GETADDRINFO_RESULT:
            if False:
                while True:
                    i = 10
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
                while True:
                    i = 10
            for (family, socktype, _proto, _cannoname, sockaddr) in result:
                addrType = _afToType[family]
                resolutionReceiver.addressResolved(addrType(_socktypeToType.get(socktype, 'TCP'), *sockaddr))
            resolutionReceiver.resolutionComplete()
        return resolution