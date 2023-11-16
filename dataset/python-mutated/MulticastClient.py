from twisted.internet import reactor
from twisted.internet.protocol import DatagramProtocol

class MulticastPingClient(DatagramProtocol):

    def startProtocol(self):
        if False:
            i = 10
            return i + 15
        self.transport.joinGroup('228.0.0.5')
        self.transport.write(b'Client: Ping', ('228.0.0.5', 9999))

    def datagramReceived(self, datagram, address):
        if False:
            for i in range(10):
                print('nop')
        print(f'Datagram {repr(datagram)} received from {repr(address)}')
reactor.listenMulticast(9999, MulticastPingClient(), listenMultiple=True)
reactor.run()