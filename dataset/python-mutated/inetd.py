"""
Twisted inetd.

Maintainer: Andrew Bennetts

Future Plans: Bugfixes.  Specifically for UDP and Sun-RPC, which don't work
correctly yet.
"""
import os
from twisted.internet import fdesc, process, reactor
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.protocols import wire
internalProtocols = {'echo': wire.Echo, 'chargen': wire.Chargen, 'discard': wire.Discard, 'daytime': wire.Daytime, 'time': wire.Time}

class InetdProtocol(Protocol):
    """Forks a child process on connectionMade, passing the socket as fd 0."""

    def connectionMade(self):
        if False:
            print('Hello World!')
        sockFD = self.transport.fileno()
        childFDs = {0: sockFD, 1: sockFD}
        if self.factory.stderrFile:
            childFDs[2] = self.factory.stderrFile.fileno()
        fdesc.setBlocking(sockFD)
        if 2 in childFDs:
            fdesc.setBlocking(childFDs[2])
        service = self.factory.service
        uid = service.user
        gid = service.group
        if uid == os.getuid():
            uid = None
        if gid == os.getgid():
            gid = None
        process.Process(None, service.program, service.programArgs, os.environ, None, None, uid, gid, childFDs)
        reactor.removeReader(self.transport)
        reactor.removeWriter(self.transport)

class InetdFactory(ServerFactory):
    protocol = InetdProtocol
    stderrFile = None

    def __init__(self, service):
        if False:
            print('Hello World!')
        self.service = service