from twisted.internet import protocol, reactor

class Echo(protocol.Protocol):
    """This is just about the simplest possible protocol"""

    def dataReceived(self, data):
        if False:
            print('Hello World!')
        'As soon as any data is received, write it back.'
        self.transport.write(data)

def main():
    if False:
        print('Hello World!')
    'This runs the protocol on port 8000'
    factory = protocol.ServerFactory()
    factory.protocol = Echo
    reactor.listenTCP(8000, factory)
    reactor.run()
if __name__ == '__main__':
    main()