"""
insults/SSH integration support.

@author: Jp Calderone
"""
from typing import Dict
from zope.interface import implementer
from twisted.conch import avatar, error as econch, interfaces as iconch
from twisted.conch.insults import insults
from twisted.conch.ssh import factory, session
from twisted.python import components

class _Glue:
    """
    A feeble class for making one attribute look like another.

    This should be replaced with a real class at some point, probably.
    Try not to write new code that uses it.
    """

    def __init__(self, **kw):
        if False:
            return 10
        self.__dict__.update(kw)

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        raise AttributeError(self.name, 'has no attribute', name)

class TerminalSessionTransport:

    def __init__(self, proto, chainedProtocol, avatar, width, height):
        if False:
            while True:
                i = 10
        self.proto = proto
        self.avatar = avatar
        self.chainedProtocol = chainedProtocol
        protoSession = self.proto.session
        self.proto.makeConnection(_Glue(write=self.chainedProtocol.dataReceived, loseConnection=lambda : avatar.conn.sendClose(protoSession), name='SSH Proto Transport'))

        def loseConnection():
            if False:
                for i in range(10):
                    print('nop')
            self.proto.loseConnection()
        self.chainedProtocol.makeConnection(_Glue(write=self.proto.write, loseConnection=loseConnection, name='Chained Proto Transport'))
        self.chainedProtocol.terminalProtocol.terminalSize(width, height)

@implementer(iconch.ISession)
class TerminalSession(components.Adapter):
    transportFactory = TerminalSessionTransport
    chainedProtocolFactory = insults.ServerProtocol

    def getPty(self, term, windowSize, attrs):
        if False:
            return 10
        (self.height, self.width) = windowSize[:2]

    def openShell(self, proto):
        if False:
            for i in range(10):
                print('nop')
        self.transportFactory(proto, self.chainedProtocolFactory(), iconch.IConchUser(self.original), self.width, self.height)

    def execCommand(self, proto, cmd):
        if False:
            for i in range(10):
                print('nop')
        raise econch.ConchError('Cannot execute commands')

    def windowChanged(self, newWindowSize):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TerminalSession.windowChanged')

    def eofReceived(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Unimplemented: TerminalSession.eofReceived')

    def closed(self):
        if False:
            return 10
        pass

class TerminalUser(avatar.ConchUser, components.Adapter):

    def __init__(self, original, avatarId):
        if False:
            while True:
                i = 10
        components.Adapter.__init__(self, original)
        avatar.ConchUser.__init__(self)
        self.channelLookup[b'session'] = session.SSHSession

class TerminalRealm:
    userFactory = TerminalUser
    sessionFactory = TerminalSession
    transportFactory = TerminalSessionTransport
    chainedProtocolFactory = insults.ServerProtocol

    def _getAvatar(self, avatarId):
        if False:
            while True:
                i = 10
        comp = components.Componentized()
        user = self.userFactory(comp, avatarId)
        sess = self.sessionFactory(comp)
        sess.transportFactory = self.transportFactory
        sess.chainedProtocolFactory = self.chainedProtocolFactory
        comp.setComponent(iconch.IConchUser, user)
        comp.setComponent(iconch.ISession, sess)
        return user

    def __init__(self, transportFactory=None):
        if False:
            i = 10
            return i + 15
        if transportFactory is not None:
            self.transportFactory = transportFactory

    def requestAvatar(self, avatarId, mind, *interfaces):
        if False:
            print('Hello World!')
        for i in interfaces:
            if i is iconch.IConchUser:
                return (iconch.IConchUser, self._getAvatar(avatarId), lambda : None)
        raise NotImplementedError()

class ConchFactory(factory.SSHFactory):
    publicKeys: Dict[bytes, bytes] = {}
    privateKeys: Dict[bytes, bytes] = {}

    def __init__(self, portal):
        if False:
            while True:
                i = 10
        self.portal = portal