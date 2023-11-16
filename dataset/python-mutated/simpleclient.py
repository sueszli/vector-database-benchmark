"""
An example client. Run simpleserv.py first before running this.
"""
from twisted.internet import protocol, reactor

class EchoClient(protocol.Protocol):
    """Once connected, send a message, then print the result."""

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self.transport.write(b'hello, world!')

    def dataReceived(self, data):
        if False:
            while True:
                i = 10
        'As soon as any data is received, write it back.'
        print('Server said:', data)
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            return 10
        print('connection lost')

class EchoFactory(protocol.ClientFactory):
    protocol = EchoClient

    def clientConnectionFailed(self, connector, reason):
        if False:
            print('Hello World!')
        print('Connection failed - goodbye!')
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        if False:
            while True:
                i = 10
        print('Connection lost - goodbye!')
        reactor.stop()

def main():
    if False:
        return 10
    f = EchoFactory()
    reactor.connectTCP('localhost', 8000, f)
    reactor.run()
if __name__ == '__main__':
    main()