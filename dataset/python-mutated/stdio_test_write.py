"""
Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_write} to test that
ITransport.write() works for process transports.
"""
import sys
from twisted.internet import protocol, stdio
from twisted.python import reflect

class WriteChild(protocol.Protocol):

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self.transport.write(b'o')
        self.transport.write(b'k')
        self.transport.write(b'!')
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        reactor.stop()
if __name__ == '__main__':
    reflect.namedAny(sys.argv[1]).install()
    from twisted.internet import reactor
    stdio.StandardIO(WriteChild())
    reactor.run()