"""
Example using stdio, Deferreds, LineReceiver and twisted.web.client.

Note that the WebCheckerCommandProtocol protocol could easily be used in e.g.
a telnet server instead; see the comments for details.

Based on an example by Abe Fettig.
"""
from twisted.internet import reactor, stdio
from twisted.protocols import basic
from twisted.web import client

class WebCheckerCommandProtocol(basic.LineReceiver):
    delimiter = b'\n'

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        self.sendLine(b"Web checker console. Type 'help' for help.")

    def lineReceived(self, line):
        if False:
            return 10
        if not line:
            return
        line = line.decode('ascii')
        commandParts = line.split()
        command = commandParts[0].lower()
        args = commandParts[1:]
        try:
            method = getattr(self, 'do_' + command)
        except AttributeError:
            self.sendLine(b'Error: no such command.')
        else:
            try:
                method(*args)
            except Exception as e:
                self.sendLine(b'Error: ' + str(e).encode('ascii'))

    def do_help(self, command=None):
        if False:
            return 10
        'help [command]: List commands, or show help on the given command'
        if command:
            doc = getattr(self, 'do_' + command).__doc__
            self.sendLine(doc.encode('ascii'))
        else:
            commands = [cmd[3:].encode('ascii') for cmd in dir(self) if cmd.startswith('do_')]
            self.sendLine(b'Valid commands: ' + b' '.join(commands))

    def do_quit(self):
        if False:
            return 10
        'quit: Quit this session'
        self.sendLine(b'Goodbye.')
        self.transport.loseConnection()

    def do_check(self, url):
        if False:
            for i in range(10):
                print('nop')
        'check <url>: Attempt to download the given web page'
        url = url.encode('ascii')
        client.Agent(reactor).request(b'GET', url).addCallback(client.readBody).addCallback(self.__checkSuccess).addErrback(self.__checkFailure)

    def __checkSuccess(self, pageData):
        if False:
            for i in range(10):
                print('nop')
        msg = f'Success: got {len(pageData)} bytes.'
        self.sendLine(msg.encode('ascii'))

    def __checkFailure(self, failure):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Failure: ' + failure.getErrorMessage()
        self.sendLine(msg.encode('ascii'))

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        reactor.stop()
if __name__ == '__main__':
    stdio.StandardIO(WebCheckerCommandProtocol())
    reactor.run()