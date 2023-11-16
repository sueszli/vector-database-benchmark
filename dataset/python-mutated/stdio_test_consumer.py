"""
Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_consumer} to test
that process transports implement IConsumer properly.
"""
import sys
from twisted.internet import protocol, stdio
from twisted.protocols import basic
from twisted.python import log, reflect

def failed(err):
    if False:
        i = 10
        return i + 15
    log.startLogging(sys.stderr)
    log.err(err)

class ConsumerChild(protocol.Protocol):

    def __init__(self, junkPath):
        if False:
            while True:
                i = 10
        self.junkPath = junkPath

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        d = basic.FileSender().beginFileTransfer(open(self.junkPath, 'rb'), self.transport)
        d.addErrback(failed)
        d.addCallback(lambda ign: self.transport.loseConnection())

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        reactor.stop()
if __name__ == '__main__':
    reflect.namedAny(sys.argv[1]).install()
    from twisted.internet import reactor
    stdio.StandardIO(ConsumerChild(sys.argv[2]))
    reactor.run()