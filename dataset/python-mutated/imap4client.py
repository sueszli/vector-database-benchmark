"""
Simple IMAP4 client which displays the subjects of all messages in a
particular mailbox.
"""
import sys
from twisted.internet import defer, endpoints, protocol, ssl, stdio
from twisted.mail import imap4
from twisted.protocols import basic
from twisted.python import log, util
try:
    raw_input
except NameError:
    raw_input = input

class TrivialPrompter(basic.LineReceiver):
    from os import linesep as delimiter
    delimiter = delimiter.encode('utf-8')
    promptDeferred = None

    def prompt(self, msg):
        if False:
            i = 10
            return i + 15
        assert self.promptDeferred is None
        self.display(msg)
        self.promptDeferred = defer.Deferred()
        return self.promptDeferred

    def display(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.transport.write(msg.encode('utf-8'))

    def lineReceived(self, line):
        if False:
            while True:
                i = 10
        if self.promptDeferred is None:
            return
        (d, self.promptDeferred) = (self.promptDeferred, None)
        d.callback(line.decode('utf-8'))

class SimpleIMAP4Client(imap4.IMAP4Client):
    """
    A client with callbacks for greeting messages from an IMAP server.
    """
    greetDeferred = None

    def serverGreeting(self, caps):
        if False:
            print('Hello World!')
        self.serverCapabilities = caps
        if self.greetDeferred is not None:
            (d, self.greetDeferred) = (self.greetDeferred, None)
            d.callback(self)

class SimpleIMAP4ClientFactory(protocol.ClientFactory):
    usedUp = False
    protocol = SimpleIMAP4Client

    def __init__(self, username, onConn):
        if False:
            for i in range(10):
                print('nop')
        self.username = username
        self.onConn = onConn

    def buildProtocol(self, addr):
        if False:
            for i in range(10):
                print('nop')
        "\n        Initiate the protocol instance. Since we are building a simple IMAP\n        client, we don't bother checking what capabilities the server has. We\n        just add all the authenticators twisted.mail has.  Note: Gmail no\n        longer uses any of the methods below, it's been using XOAUTH since\n        2010.\n        "
        assert not self.usedUp
        self.usedUp = True
        p = self.protocol()
        p.factory = self
        p.greetDeferred = self.onConn
        p.registerAuthenticator(imap4.PLAINAuthenticator(self.username))
        p.registerAuthenticator(imap4.LOGINAuthenticator(self.username))
        p.registerAuthenticator(imap4.CramMD5ClientAuthenticator(self.username))
        return p

    def clientConnectionFailed(self, connector, reason):
        if False:
            while True:
                i = 10
        (d, self.onConn) = (self.onConn, None)
        d.errback(reason)

def cbServerGreeting(proto, username, password):
    if False:
        return 10
    '\n    Initial callback - invoked after the server sends us its greet message.\n    '
    tp = TrivialPrompter()
    stdio.StandardIO(tp)
    proto.prompt = tp.prompt
    proto.display = tp.display
    return proto.authenticate(password).addCallback(cbAuthentication, proto).addErrback(ebAuthentication, proto, username, password)

def ebConnection(reason):
    if False:
        return 10
    '\n    Fallback error-handler. If anything goes wrong, log it and quit.\n    '
    log.startLogging(sys.stdout)
    log.err(reason)
    return reason

def cbAuthentication(result, proto):
    if False:
        return 10
    '\n    Callback after authentication has succeeded.\n\n    Lists a bunch of mailboxes.\n    '
    return proto.list('', '*').addCallback(cbMailboxList, proto)

def ebAuthentication(failure, proto, username, password):
    if False:
        while True:
            i = 10
    '\n    Errback invoked when authentication fails.\n\n    If it failed because no SASL mechanisms match, offer the user the choice\n    of logging in insecurely.\n\n    If you are trying to connect to your Gmail account, you will be here!\n    '
    failure.trap(imap4.NoSupportedAuthentication)
    return proto.prompt('No secure authentication available. Login insecurely? (y/N) ').addCallback(cbInsecureLogin, proto, username, password)

def cbInsecureLogin(result, proto, username, password):
    if False:
        i = 10
        return i + 15
    '\n    Callback for "insecure-login" prompt.\n    '
    if result.lower() == 'y':
        return proto.login(username, password).addCallback(cbAuthentication, proto)
    return defer.fail(Exception('Login failed for security reasons.'))

def cbMailboxList(result, proto):
    if False:
        print('Hello World!')
    '\n    Callback invoked when a list of mailboxes has been retrieved.\n    '
    result = [e[2] for e in result]
    s = '\n'.join(['%d. %s' % (n + 1, m) for (n, m) in zip(range(len(result)), result)])
    if not s:
        return defer.fail(Exception('No mailboxes exist on server!'))
    return proto.prompt(s + '\nWhich mailbox? [1] ').addCallback(cbPickMailbox, proto, result)

def cbPickMailbox(result, proto, mboxes):
    if False:
        return 10
    '\n    When the user selects a mailbox, "examine" it.\n    '
    mbox = mboxes[int(result or '1') - 1]
    return proto.examine(mbox).addCallback(cbExamineMbox, proto)

def cbExamineMbox(result, proto):
    if False:
        while True:
            i = 10
    '\n    Callback invoked when examine command completes.\n\n    Retrieve the subject header of every message in the mailbox.\n    '
    return proto.fetchSpecific('1:*', headerType='HEADER.FIELDS', headerArgs=['SUBJECT']).addCallback(cbFetch, proto)

def cbFetch(result, proto):
    if False:
        i = 10
        return i + 15
    '\n    Finally, display headers.\n    '
    if result:
        keys = sorted(result)
        for k in keys:
            proto.display(f'{k} {result[k][0][2]}')
    else:
        print('Hey, an empty mailbox!')
    return proto.logout()

def cbClose(result):
    if False:
        print('Hello World!')
    '\n    Close the connection when we finish everything.\n    '
    from twisted.internet import reactor
    reactor.stop()

def main():
    if False:
        while True:
            i = 10
    hostname = raw_input('IMAP4 Server Hostname: ')
    port = raw_input('IMAP4 Server Port (the default is 143, 993 uses SSL): ')
    username = raw_input('IMAP4 Username: ').encode('ascii')
    password = util.getPassword('IMAP4 Password: ').encode('ascii')
    onConn = defer.Deferred().addCallback(cbServerGreeting, username, password).addErrback(ebConnection).addBoth(cbClose)
    factory = SimpleIMAP4ClientFactory(username, onConn)
    if not port:
        port = 143
    else:
        port = int(port)
    from twisted.internet import reactor
    endpoint = endpoints.HostnameEndpoint(reactor, hostname, port)
    if port == 993:
        if isinstance(hostname, bytes):
            hostname = hostname.decode('utf-8')
        contextFactory = ssl.optionsForClientTLS(hostname=hostname)
        endpoint = endpoints.wrapClientTLS(contextFactory, endpoint)
    endpoint.connect(factory)
    reactor.run()
if __name__ == '__main__':
    main()