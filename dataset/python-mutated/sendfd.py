"""
Server-side of an example for sending file descriptors between processes over
UNIX sockets.  This server accepts connections on a UNIX socket and sends one
file descriptor to them, along with the name of the file it is associated with.

To run this example, run this program with two arguments: a path giving a UNIX
socket to listen on (must not exist) and a path to a file to send to clients
which connect (must exist).  For example:

    $ python sendfd.py /tmp/sendfd.sock /etc/motd

It will listen for client connections until stopped (eg, using Control-C).  Most
interesting behavior happens on the client side.

See recvfd.py for the client side of this example.
"""
if __name__ == '__main__':
    import sendfd
    raise SystemExit(sendfd.main())
import sys
from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineOnlyReceiver
from twisted.python.filepath import FilePath
from twisted.python.log import startLogging

class SendFDProtocol(LineOnlyReceiver):

    def connectionMade(self):
        if False:
            print('Hello World!')
        self.fObj = self.factory.content.open()
        self.transport.sendFileDescriptor(self.fObj.fileno())
        encoding = sys.getfilesystemencoding()
        self.sendLine(self.factory.content.path.encode(encoding))
        self.timeoutCall = reactor.callLater(60, self.transport.loseConnection)

    def connectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        self.fObj.close()
        self.fObj = None
        if self.timeoutCall.active():
            self.timeoutCall.cancel()
            self.timeoutCall = None

def main():
    if False:
        return 10
    address = FilePath(sys.argv[1])
    content = FilePath(sys.argv[2])
    if address.exists():
        raise SystemExit('Cannot listen on an existing path')
    if not content.isfile():
        raise SystemExit('Content file must exist')
    startLogging(sys.stdout)
    serverFactory = Factory()
    serverFactory.content = content
    serverFactory.protocol = SendFDProtocol
    reactor.listenUNIX(address.path, serverFactory)
    reactor.run()