from twisted.internet import reactor
from twisted.internet.protocol import Factory, Protocol

class Echo(Protocol):

    def dataReceived(self, data):
        if False:
            i = 10
            return i + 15
        '\n        As soon as any data is received, write it back.\n        '
        self.transport.write(data)

def main():
    if False:
        while True:
            i = 10
    f = Factory()
    f.protocol = Echo
    reactor.listenTCP(8000, f)
    reactor.run()
if __name__ == '__main__':
    main()