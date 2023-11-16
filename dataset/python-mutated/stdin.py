"""
An example of reading a line at a time from standard input
without blocking the reactor.
"""
from os import linesep
from twisted.internet import stdio
from twisted.protocols import basic

class Echo(basic.LineReceiver):
    delimiter = linesep.encode('ascii')

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self.transport.write(b'>>> ')

    def lineReceived(self, line):
        if False:
            i = 10
            return i + 15
        self.sendLine(b'Echo: ' + line)
        self.transport.write(b'>>> ')

def main():
    if False:
        for i in range(10):
            print('nop')
    stdio.StandardIO(Echo())
    from twisted.internet import reactor
    reactor.run()
if __name__ == '__main__':
    main()