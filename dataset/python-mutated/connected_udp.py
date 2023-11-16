from twisted.internet import reactor
from twisted.internet.protocol import DatagramProtocol

class Helloer(DatagramProtocol):

    def startProtocol(self):
        if False:
            while True:
                i = 10
        host = '192.168.1.1'
        port = 1234
        self.transport.connect(host, port)
        print('now we can only send to host %s port %d' % (host, port))
        self.transport.write(b'hello')

    def datagramReceived(self, data, addr):
        if False:
            while True:
                i = 10
        print(f'received {data!r} from {addr}')

    def connectionRefused(self):
        if False:
            print('Hello World!')
        print('No one listening')
reactor.listenUDP(0, Helloer())
reactor.run()