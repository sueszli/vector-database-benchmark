from twisted.internet import reactor
from twisted.internet.protocol import DatagramProtocol

class EchoUDP(DatagramProtocol):

    def datagramReceived(self, datagram, address):
        if False:
            while True:
                i = 10
        self.transport.write(datagram, address)

def main():
    if False:
        for i in range(10):
            print('nop')
    reactor.listenUDP(8000, EchoUDP())
    reactor.run()
if __name__ == '__main__':
    main()