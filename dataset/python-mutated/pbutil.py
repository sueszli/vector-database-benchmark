"""Base classes handy for use with PB clients.
"""
from twisted.internet import protocol
from twisted.python import log
from twisted.spread import pb
from twisted.spread.pb import PBClientFactory
from buildbot.util import bytes2unicode

class NewCredPerspective(pb.Avatar):

    def attached(self, mind):
        if False:
            print('Hello World!')
        return self

    def detached(self, mind):
        if False:
            i = 10
            return i + 15
        pass

class ReconnectingPBClientFactory(PBClientFactory, protocol.ReconnectingClientFactory):
    """Reconnecting client factory for PB brokers.

    Like PBClientFactory, but if the connection fails or is lost, the factory
    will attempt to reconnect.

    Instead of using f.getRootObject (which gives a Deferred that can only
    be fired once), override the gotRootObject method.

    Instead of using the newcred f.login (which is also one-shot), call
    f.startLogin() with the credentials and client, and override the
    gotPerspective method.

    Instead of using the oldcred f.getPerspective (also one-shot), call
    f.startGettingPerspective() with the same arguments, and override
    gotPerspective.

    gotRootObject and gotPerspective will be called each time the object is
    received (once per successful connection attempt). You will probably want
    to use obj.notifyOnDisconnect to find out when the connection is lost.

    If an authorization error occurs, failedToGetPerspective() will be
    invoked.

    To use me, subclass, then hand an instance to a connector (like
    TCPClient).
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._doingLogin = False
        self._doingGetPerspective = False

    def clientConnectionFailed(self, connector, reason):
        if False:
            while True:
                i = 10
        super().clientConnectionFailed(connector, reason)
        if self.continueTrying:
            self.connector = connector
            self.retry()

    def clientConnectionLost(self, connector, reason):
        if False:
            i = 10
            return i + 15
        super().clientConnectionLost(connector, reason, reconnecting=True)
        RCF = protocol.ReconnectingClientFactory
        RCF.clientConnectionLost(self, connector, reason)

    def clientConnectionMade(self, broker):
        if False:
            while True:
                i = 10
        self.resetDelay()
        super().clientConnectionMade(broker)
        if self._doingLogin:
            self.doLogin(self._root)
        if self._doingGetPerspective:
            self.doGetPerspective(self._root)
        self.gotRootObject(self._root)

    def getPerspective(self, *args):
        if False:
            while True:
                i = 10
        raise RuntimeError('getPerspective is one-shot: use startGettingPerspective instead')

    def startGettingPerspective(self, username, password, serviceName, perspectiveName=None, client=None):
        if False:
            for i in range(10):
                print('nop')
        self._doingGetPerspective = True
        if perspectiveName is None:
            perspectiveName = username
        self._oldcredArgs = (username, password, serviceName, perspectiveName, client)

    def doGetPerspective(self, root):
        if False:
            while True:
                i = 10
        (username, password, serviceName, perspectiveName, client) = self._oldcredArgs
        d = self._cbAuthIdentity(root, username, password)
        d.addCallback(self._cbGetPerspective, serviceName, perspectiveName, client)
        d.addCallbacks(self.gotPerspective, self.failedToGetPerspective)

    def login(self, *args):
        if False:
            while True:
                i = 10
        raise RuntimeError('login is one-shot: use startLogin instead')

    def startLogin(self, credentials, client=None):
        if False:
            while True:
                i = 10
        self._credentials = credentials
        self._client = client
        self._doingLogin = True

    def doLogin(self, root):
        if False:
            while True:
                i = 10
        d = self._cbSendUsername(root, self._credentials.username, self._credentials.password, self._client)
        d.addCallbacks(self.gotPerspective, self.failedToGetPerspective)

    def gotPerspective(self, perspective):
        if False:
            i = 10
            return i + 15
        'The remote avatar or perspective (obtained each time this factory\n        connects) is now available.'

    def gotRootObject(self, root):
        if False:
            for i in range(10):
                print('nop')
        'The remote root object (obtained each time this factory connects)\n        is now available. This method will be called each time the connection\n        is established and the object reference is retrieved.'

    def failedToGetPerspective(self, why):
        if False:
            while True:
                i = 10
        'The login process failed, most likely because of an authorization\n        failure (bad password), but it is also possible that we lost the new\n        connection before we managed to send our credentials.\n        '
        log.msg('ReconnectingPBClientFactory.failedToGetPerspective')
        if why.check(pb.PBConnectionLost):
            log.msg('we lost the brand-new connection')
            return
        self.stopTrying()
        log.err(why)

def decode(data, encoding='utf-8', errors='strict'):
    if False:
        return 10
    'We need to convert a dictionary where keys and values\n    are bytes, to unicode strings.  This happens when a\n    Python 2 worker sends a dictionary back to a Python 3 master.\n    '
    data_type = type(data)
    if data_type == bytes:
        return bytes2unicode(data, encoding, errors)
    if data_type in (dict, list, tuple):
        if data_type == dict:
            data = data.items()
        return data_type(map(decode, data))
    return data