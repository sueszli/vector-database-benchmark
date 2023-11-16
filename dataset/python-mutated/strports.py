"""
Construct listening port services from a simple string description.

@see: L{twisted.internet.endpoints.serverFromString}
@see: L{twisted.internet.endpoints.clientFromString}
"""
from typing import Optional, cast
from twisted.application.internet import StreamServerEndpointService
from twisted.internet import endpoints, interfaces

def _getReactor() -> interfaces.IReactorCore:
    if False:
        while True:
            i = 10
    from twisted.internet import reactor
    return cast(interfaces.IReactorCore, reactor)

def service(description: str, factory: interfaces.IProtocolFactory, reactor: Optional[interfaces.IReactorCore]=None) -> StreamServerEndpointService:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the service corresponding to a description.\n\n    @param description: The description of the listening port, in the syntax\n        described by L{twisted.internet.endpoints.serverFromString}.\n    @type description: C{str}\n\n    @param factory: The protocol factory which will build protocols for\n        connections to this service.\n    @type factory: L{twisted.internet.interfaces.IProtocolFactory}\n\n    @rtype: C{twisted.application.service.IService}\n    @return: the service corresponding to a description of a reliable stream\n        server.\n\n    @see: L{twisted.internet.endpoints.serverFromString}\n    '
    if reactor is None:
        reactor = _getReactor()
    svc = StreamServerEndpointService(endpoints.serverFromString(reactor, description), factory)
    svc._raiseSynchronously = True
    return svc

def listen(description: str, factory: interfaces.IProtocolFactory) -> interfaces.IListeningPort:
    if False:
        print('Hello World!')
    '\n    Listen on a port corresponding to a description.\n\n    @param description: The description of the connecting port, in the syntax\n        described by L{twisted.internet.endpoints.serverFromString}.\n    @type description: L{str}\n\n    @param factory: The protocol factory which will build protocols on\n        connection.\n    @type factory: L{twisted.internet.interfaces.IProtocolFactory}\n\n    @rtype: L{twisted.internet.interfaces.IListeningPort}\n    @return: the port corresponding to a description of a reliable virtual\n        circuit server.\n\n    @see: L{twisted.internet.endpoints.serverFromString}\n    '
    from twisted.internet import reactor
    (name, args, kw) = endpoints._parseServer(description, factory)
    return cast(interfaces.IListeningPort, getattr(reactor, 'listen' + name)(*args, **kw))
__all__ = ['service', 'listen']