"""The most basic chat protocol possible.

run me with twistd -y chatserver.py, and then connect with multiple
telnet clients to port 1025
"""
from twisted.protocols import basic

class MyChat(basic.LineReceiver):

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        print('Got new client!')
        self.factory.clients.append(self)

    def connectionLost(self, reason):
        if False:
            return 10
        print('Lost a client!')
        self.factory.clients.remove(self)

    def lineReceived(self, line):
        if False:
            for i in range(10):
                print('nop')
        print('received', repr(line))
        for c in self.factory.clients:
            c.message(line)

    def message(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.transport.write(message + b'\n')
from twisted.application import internet, service
from twisted.internet import protocol
factory = protocol.ServerFactory()
factory.protocol = MyChat
factory.clients = []
application = service.Application('chatserver')
internet.TCPServer(1025, factory).setServiceParent(application)