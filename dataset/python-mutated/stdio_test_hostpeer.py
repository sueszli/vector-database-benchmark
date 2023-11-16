"""
Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_hostAndPeer} to test
that ITransport.getHost() and ITransport.getPeer() work for process transports.
"""
import sys
from twisted.internet import protocol, stdio
from twisted.python import reflect

class HostPeerChild(protocol.Protocol):

    def connectionMade(self):
        if False:
            while True:
                i = 10
        self.transport.write(b'\n'.join([str(self.transport.getHost()).encode('ascii'), str(self.transport.getPeer()).encode('ascii')]))
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        reactor.stop()
if __name__ == '__main__':
    reflect.namedAny(sys.argv[1]).install()
    from twisted.internet import reactor
    stdio.StandardIO(HostPeerChild())
    reactor.run()