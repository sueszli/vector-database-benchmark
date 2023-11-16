import getpass
import os
import struct
import sys
from twisted.conch.ssh import channel, common, connection, keys, transport, userauth
from twisted.internet import defer, protocol, reactor
from twisted.python import log
'\nExample of using a simple SSH client.\n\nIt will try to authenticate with a SSH key or ask for a password.\n\nRe-using a private key is dangerous, generate one.\nFor this example you can use:\n\n$ ckeygen -t rsa -f ssh-keys/client_rsa\n'
USER = b'user'
HOST = 'localhost'
PORT = 5022
SERVER_FINGERPRINT = b'55:55:66:24:6b:03:0e:f1:ec:f8:66:c3:51:df:27:4b'
CLIENT_RSA_PUBLIC = 'ssh-keys/client_rsa.pub'
CLIENT_RSA_PRIVATE = 'ssh-keys/client_rsa'

class SimpleTransport(transport.SSHClientTransport):

    def verifyHostKey(self, hostKey, fingerprint):
        if False:
            print('Hello World!')
        print('Server host key fingerprint: %s' % fingerprint)
        if SERVER_FINGERPRINT == fingerprint:
            return defer.succeed(True)
        else:
            print('Bad host key. Expecting: %s' % SERVER_FINGERPRINT)
            return defer.fail(Exception('Bad server key'))

    def connectionSecure(self):
        if False:
            while True:
                i = 10
        self.requestService(SimpleUserAuth(USER, SimpleConnection()))

class SimpleUserAuth(userauth.SSHUserAuthClient):

    def getPassword(self):
        if False:
            i = 10
            return i + 15
        return defer.succeed(getpass.getpass(f"{USER}@{HOST}'s password: "))

    def getGenericAnswers(self, name, instruction, questions):
        if False:
            return 10
        print(name)
        print(instruction)
        answers = []
        for (prompt, echo) in questions:
            if echo:
                answer = input(prompt)
            else:
                answer = getpass.getpass(prompt)
            answers.append(answer)
        return defer.succeed(answers)

    def getPublicKey(self):
        if False:
            i = 10
            return i + 15
        if not CLIENT_RSA_PUBLIC or not os.path.exists(CLIENT_RSA_PUBLIC) or self.lastPublicKey:
            return
        return keys.Key.fromFile(filename=CLIENT_RSA_PUBLIC)

    def getPrivateKey(self):
        if False:
            return 10
        '\n        A deferred can also be returned.\n        '
        return defer.succeed(keys.Key.fromFile(CLIENT_RSA_PRIVATE))

class SimpleConnection(connection.SSHConnection):

    def serviceStarted(self):
        if False:
            print('Hello World!')
        self.openChannel(TrueChannel(2 ** 16, 2 ** 15, self))
        self.openChannel(FalseChannel(2 ** 16, 2 ** 15, self))
        self.openChannel(CatChannel(2 ** 16, 2 ** 15, self))

class TrueChannel(channel.SSHChannel):
    name = b'session'

    def openFailed(self, reason):
        if False:
            for i in range(10):
                print('nop')
        print('true failed', reason)

    def channelOpen(self, ignoredData):
        if False:
            return 10
        self.conn.sendRequest(self, 'exec', common.NS('true'))

    def request_exit_status(self, data):
        if False:
            print('Hello World!')
        status = struct.unpack('>L', data)[0]
        print('true status was: %s' % status)
        self.loseConnection()

class FalseChannel(channel.SSHChannel):
    name = b'session'

    def openFailed(self, reason):
        if False:
            return 10
        print('false failed', reason)

    def channelOpen(self, ignoredData):
        if False:
            i = 10
            return i + 15
        self.conn.sendRequest(self, 'exec', common.NS('false'))

    def request_exit_status(self, data):
        if False:
            return 10
        status = struct.unpack('>L', data)[0]
        print('false status was: %s' % status)
        self.loseConnection()

class CatChannel(channel.SSHChannel):
    name = b'session'

    def openFailed(self, reason):
        if False:
            return 10
        print('echo failed', reason)

    def channelOpen(self, ignoredData):
        if False:
            while True:
                i = 10
        self.data = b''
        d = self.conn.sendRequest(self, 'exec', common.NS('cat'), wantReply=1)
        d.addCallback(self._cbRequest)

    def _cbRequest(self, ignored):
        if False:
            for i in range(10):
                print('nop')
        self.write(b'hello conch\n')
        self.conn.sendEOF(self)

    def dataReceived(self, data):
        if False:
            while True:
                i = 10
        self.data += data

    def closed(self):
        if False:
            print('Hello World!')
        print('got data from cat: %s' % repr(self.data))
        self.loseConnection()
        reactor.stop()
log.startLogging(sys.stdout)
protocol.ClientCreator(reactor, SimpleTransport).connectTCP(HOST, PORT)
reactor.run()