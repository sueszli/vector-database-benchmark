from twisted.internet import reactor
from twisted.internet.protocol import DatagramProtocol

class MulticastPingPong(DatagramProtocol):

    def startProtocol(self):
        if False:
            return 10
        '\n        Called after protocol has started listening.\n        '
        self.transport.setTTL(5)
        self.transport.joinGroup('228.0.0.5')

    def datagramReceived(self, datagram, address):
        if False:
            print('Hello World!')
        print(f'Datagram {repr(datagram)} received from {repr(address)}')
        if datagram == b'Client: Ping' or datagram == 'Client: Ping':
            self.transport.write(b'Server: Pong', address)
reactor.listenMulticast(9999, MulticastPingPong(), listenMultiple=True)
reactor.run()