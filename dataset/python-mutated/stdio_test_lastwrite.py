"""
Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_lastWriteReceived}
to test that L{os.write} can be reliably used after
L{twisted.internet.stdio.StandardIO} has finished.
"""
import sys
from twisted.internet.protocol import Protocol
from twisted.internet.stdio import StandardIO
from twisted.python.reflect import namedAny

class LastWriteChild(Protocol):

    def __init__(self, reactor, magicString):
        if False:
            return 10
        self.reactor = reactor
        self.magicString = magicString

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        self.transport.write(self.magicString)
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        self.reactor.stop()

def main(reactor, magicString):
    if False:
        return 10
    p = LastWriteChild(reactor, magicString.encode('ascii'))
    StandardIO(p)
    reactor.run()
if __name__ == '__main__':
    namedAny(sys.argv[1]).install()
    from twisted.internet import reactor
    main(reactor, sys.argv[2])