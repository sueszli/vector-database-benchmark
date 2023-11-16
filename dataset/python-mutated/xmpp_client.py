"""
A very simple twisted xmpp-client (Jabber ID)

To run the script:
$ python xmpp_client.py <jid> <secret>
"""
import sys
from twisted.internet.defer import Deferred
from twisted.internet.task import react
from twisted.names.srvconnect import SRVConnector
from twisted.words.protocols.jabber import client, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish

class Client:

    def __init__(self, reactor, jid, secret):
        if False:
            while True:
                i = 10
        self.reactor = reactor
        f = client.XMPPClientFactory(jid, secret)
        f.addBootstrap(xmlstream.STREAM_CONNECTED_EVENT, self.connected)
        f.addBootstrap(xmlstream.STREAM_END_EVENT, self.disconnected)
        f.addBootstrap(xmlstream.STREAM_AUTHD_EVENT, self.authenticated)
        f.addBootstrap(xmlstream.INIT_FAILED_EVENT, self.init_failed)
        connector = SRVConnector(reactor, 'xmpp-client', jid.host, f, defaultPort=5222)
        connector.connect()
        self.finished = Deferred()

    def rawDataIn(self, buf):
        if False:
            print('Hello World!')
        print('RECV: %r' % buf)

    def rawDataOut(self, buf):
        if False:
            for i in range(10):
                print('nop')
        print('SEND: %r' % buf)

    def connected(self, xs):
        if False:
            i = 10
            return i + 15
        print('Connected.')
        self.xmlstream = xs
        xs.rawDataInFn = self.rawDataIn
        xs.rawDataOutFn = self.rawDataOut

    def disconnected(self, reason):
        if False:
            i = 10
            return i + 15
        print('Disconnected.')
        print(reason)
        self.finished.callback(None)

    def authenticated(self, xs):
        if False:
            for i in range(10):
                print('nop')
        print('Authenticated.')
        presence = domish.Element((None, 'presence'))
        xs.send(presence)
        self.reactor.callLater(5, xs.sendFooter)

    def init_failed(self, failure):
        if False:
            i = 10
            return i + 15
        print('Initialization failed.')
        print(failure)
        self.xmlstream.sendFooter()

def main(reactor, jid, secret):
    if False:
        for i in range(10):
            print('nop')
    '\n    Connect to the given Jabber ID and return a L{Deferred} which will be\n    called back when the connection is over.\n\n    @param reactor: The reactor to use for the connection.\n    @param jid: A L{JID} to connect to.\n    @param secret: A C{str}\n    '
    return Client(reactor, JID(jid), secret).finished
if __name__ == '__main__':
    react(main, sys.argv[1:])