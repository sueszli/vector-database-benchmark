"""
Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_writeSequence} to test
that ITransport.writeSequence() works for process transports.
"""
import sys
from twisted.internet import protocol, stdio
from twisted.python import reflect

class WriteSequenceChild(protocol.Protocol):

    def connectionMade(self):
        if False:
            return 10
        self.transport.writeSequence([b'o', b'k', b'!'])
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        reactor.stop()
if __name__ == '__main__':
    reflect.namedAny(sys.argv[1]).install()
    from twisted.internet import reactor
    stdio.StandardIO(WriteSequenceChild())
    reactor.run()