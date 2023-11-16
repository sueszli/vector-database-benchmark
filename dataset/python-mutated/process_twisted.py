"""A process that reads from stdin and out using Twisted."""
import os
import sys
pos = os.path.abspath(sys.argv[0]).find(os.sep + 'Twisted')
if pos != -1:
    sys.path.insert(0, os.path.abspath(sys.argv[0])[:pos + 8])
sys.path.insert(0, os.curdir)
from zope.interface import implementer
from twisted.internet import interfaces
from twisted.python import log
log.startLogging(sys.stderr)
from twisted.internet import protocol, reactor, stdio

@implementer(interfaces.IHalfCloseableProtocol)
class Echo(protocol.Protocol):

    def connectionMade(self):
        if False:
            while True:
                i = 10
        print('connection made')

    def dataReceived(self, data):
        if False:
            print('Hello World!')
        self.transport.write(data)

    def readConnectionLost(self):
        if False:
            return 10
        print('readConnectionLost')
        self.transport.loseConnection()

    def writeConnectionLost(self):
        if False:
            return 10
        print('writeConnectionLost')

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        print('connectionLost', reason)
        reactor.stop()
stdio.StandardIO(Echo())
reactor.run()