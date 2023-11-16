import socket
from twisted.internet import reactor
from twisted.internet.protocol import DatagramProtocol

class Echo(DatagramProtocol):

    def datagramReceived(self, data, addr):
        if False:
            for i in range(10):
                print('nop')
        print(f'received {data!r} from {addr}')
        self.transport.write(data, addr)
portSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
portSocket.setblocking(False)
portSocket.bind(('127.0.0.1', 9999))
port = reactor.adoptDatagramPort(portSocket.fileno(), socket.AF_INET, Echo())
portSocket.close()
reactor.run()