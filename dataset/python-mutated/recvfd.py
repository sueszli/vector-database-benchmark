"""
Client-side of an example for sending file descriptors between processes over
UNIX sockets.  This client connects to a server listening on a UNIX socket and
waits for one file descriptor to arrive over the connection.  It displays the
name of the file and the first 80 bytes it contains, then exits.

To run this example, run this program with one argument: a path giving the UNIX
socket the server side of this example is already listening on.  For example:

    $ python recvfd.py /tmp/sendfd.sock

See sendfd.py for the server side of this example.
"""
if __name__ == '__main__':
    import recvfd
    raise SystemExit(recvfd.main())
import os
import sys
from zope.interface import implementer
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.endpoints import UNIXClientEndpoint
from twisted.internet.interfaces import IFileDescriptorReceiver
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineOnlyReceiver
from twisted.python.filepath import FilePath
from twisted.python.log import startLogging

@implementer(IFileDescriptorReceiver)
class ReceiveFDProtocol(LineOnlyReceiver):
    descriptor = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.whenDisconnected = Deferred()

    def fileDescriptorReceived(self, descriptor):
        if False:
            while True:
                i = 10
        self.descriptor = descriptor

    def lineReceived(self, line):
        if False:
            print('Hello World!')
        if self.descriptor is None:
            print(f'Received {line} without receiving descriptor!')
        else:
            data = os.read(self.descriptor, 80)
            print(f'Received {line} from the server.')
            print(f'First 80 bytes are:\n{data}\n')
        os.close(self.descriptor)
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            print('Hello World!')
        self.whenDisconnected.callback(None)

def main():
    if False:
        while True:
            i = 10
    address = FilePath(sys.argv[1])
    startLogging(sys.stdout)
    factory = Factory()
    factory.protocol = ReceiveFDProtocol
    factory.quiet = True
    endpoint = UNIXClientEndpoint(reactor, address.path)
    connected = endpoint.connect(factory)

    def succeeded(client):
        if False:
            for i in range(10):
                print('nop')
        return client.whenDisconnected

    def failed(reason):
        if False:
            while True:
                i = 10
        print('Could not connect:', reason.getErrorMessage())

    def disconnected(ignored):
        if False:
            i = 10
            return i + 15
        reactor.stop()
    connected.addCallbacks(succeeded, failed)
    connected.addCallback(disconnected)
    reactor.run()