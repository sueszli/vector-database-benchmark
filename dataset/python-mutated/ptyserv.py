"""
A PTY server that spawns a shell upon connection.

Run this example by typing in:
> python ptyserv.py

Telnet to the server once you start it by typing in: 
> telnet localhost 5823
"""
from twisted.internet import protocol, reactor

class FakeTelnet(protocol.Protocol):
    commandToRun = ['/bin/sh']
    dirToRunIn = '/tmp'

    def connectionMade(self):
        if False:
            return 10
        print('connection made')
        self.propro = ProcessProtocol(self)
        reactor.spawnProcess(self.propro, self.commandToRun[0], self.commandToRun, {}, self.dirToRunIn, usePTY=1)

    def dataReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.propro.transport.write(data)

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        print('connection lost')
        self.propro.tranport.loseConnection()

class ProcessProtocol(protocol.ProcessProtocol):

    def __init__(self, pr):
        if False:
            for i in range(10):
                print('nop')
        self.pr = pr

    def outReceived(self, data):
        if False:
            i = 10
            return i + 15
        self.pr.transport.write(data)

    def processEnded(self, reason):
        if False:
            return 10
        print('protocol connection lost')
        self.pr.transport.loseConnection()
f = protocol.Factory()
f.protocol = FakeTelnet
reactor.listenTCP(5823, f)
reactor.run()