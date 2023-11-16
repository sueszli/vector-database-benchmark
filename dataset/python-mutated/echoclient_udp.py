from twisted.internet import reactor
from twisted.internet.protocol import DatagramProtocol

class EchoClientDatagramProtocol(DatagramProtocol):
    strings = [b'Hello, world!', b'What a fine day it is.', b'Bye-bye!']

    def startProtocol(self):
        if False:
            while True:
                i = 10
        self.transport.connect('127.0.0.1', 8000)
        self.sendDatagram()

    def sendDatagram(self):
        if False:
            return 10
        if len(self.strings):
            datagram = self.strings.pop(0)
            self.transport.write(datagram)
        else:
            reactor.stop()

    def datagramReceived(self, datagram, host):
        if False:
            while True:
                i = 10
        print('Datagram received: ', repr(datagram))
        self.sendDatagram()

def main():
    if False:
        print('Hello World!')
    protocol = EchoClientDatagramProtocol()
    reactor.listenUDP(0, protocol)
    reactor.run()
if __name__ == '__main__':
    main()