"""
Main program for the child process run by
L{twisted.test.test_stdio.StandardInputOutputTests.test_loseConnection} to
test that ITransport.loseConnection() works for process transports.
"""
import sys
from twisted.internet import protocol, stdio
from twisted.internet.error import ConnectionDone
from twisted.python import log, reflect

class LoseConnChild(protocol.Protocol):
    exitCode = 0

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        '\n        Check that C{reason} is a L{Failure} wrapping a L{ConnectionDone}\n        instance and stop the reactor.  If C{reason} is wrong for some reason,\n        log something about that in C{self.errorLogFile} and make sure the\n        process exits with a non-zero status.\n        '
        try:
            try:
                reason.trap(ConnectionDone)
            except BaseException:
                log.err(None, 'Problem with reason passed to connectionLost')
                self.exitCode = 1
        finally:
            reactor.stop()
if __name__ == '__main__':
    reflect.namedAny(sys.argv[1]).install()
    log.startLogging(open(sys.argv[2], 'wb'))
    from twisted.internet import reactor
    protocolLoseConnChild = LoseConnChild()
    stdio.StandardIO(protocolLoseConnChild)
    reactor.run()
    sys.exit(protocolLoseConnChild.exitCode)