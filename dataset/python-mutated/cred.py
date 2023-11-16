import sys
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import protocol
from twisted.protocols import basic
from twisted.python import log

class IProtocolUser(Interface):

    def getPrivileges():
        if False:
            print('Hello World!')
        'Return a list of privileges this user has.'

    def logout():
        if False:
            return 10
        'Cleanup per-login resources allocated to this avatar'

@implementer(IProtocolUser)
class AnonymousUser:

    def getPrivileges(self):
        if False:
            while True:
                i = 10
        return [1, 2, 3]

    def logout(self):
        if False:
            i = 10
            return i + 15
        print('Cleaning up anonymous user resources')

@implementer(IProtocolUser)
class RegularUser:

    def getPrivileges(self):
        if False:
            return 10
        return [1, 2, 3, 5, 6]

    def logout(self):
        if False:
            i = 10
            return i + 15
        print('Cleaning up regular user resources')

@implementer(IProtocolUser)
class Administrator:

    def getPrivileges(self):
        if False:
            print('Hello World!')
        return range(50)

    def logout(self):
        if False:
            print('Hello World!')
        print('Cleaning up administrator resources')

class Protocol(basic.LineReceiver):
    user = None
    portal = None
    avatar = None
    logout = None

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self.sendLine(b'Login with USER <name> followed by PASS <password> or ANON')
        self.sendLine(b'Check privileges with PRIVS')

    def connectionLost(self, reason):
        if False:
            return 10
        if self.logout:
            self.logout()
            self.avatar = None
            self.logout = None

    def lineReceived(self, line):
        if False:
            for i in range(10):
                print('nop')
        f = getattr(self, 'cmd_' + line.decode('ascii').upper().split()[0])
        if f:
            try:
                f(*line.split()[1:])
            except TypeError:
                self.sendLine(b'Wrong number of arguments.')
            except BaseException:
                self.sendLine(b'Server error (probably your fault)')

    def cmd_ANON(self):
        if False:
            i = 10
            return i + 15
        if self.portal:
            self.portal.login(credentials.Anonymous(), None, IProtocolUser).addCallbacks(self._cbLogin, self._ebLogin)
        else:
            self.sendLine(b'DENIED')

    def cmd_USER(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.user = name
        self.sendLine(b'Alright.  Now PASS?')

    def cmd_PASS(self, password):
        if False:
            i = 10
            return i + 15
        if not self.user:
            self.sendLine(b'USER required before PASS')
        elif self.portal:
            self.portal.login(credentials.UsernamePassword(self.user, password), None, IProtocolUser).addCallbacks(self._cbLogin, self._ebLogin)
        else:
            self.sendLine(b'DENIED')

    def cmd_PRIVS(self):
        if False:
            for i in range(10):
                print('nop')
        self.sendLine(b'You have the following privileges: ')
        self.sendLine(b' '.join([str(priv).encode('ascii') for priv in self.avatar.getPrivileges()]))

    def _cbLogin(self, result):
        if False:
            print('Hello World!')
        (interface, avatar, logout) = result
        assert interface is IProtocolUser
        self.avatar = avatar
        self.logout = logout
        self.sendLine(b'Login successful.  Available commands: PRIVS')

    def _ebLogin(self, failure):
        if False:
            for i in range(10):
                print('nop')
        failure.trap(error.UnauthorizedLogin)
        self.sendLine(b'Login denied!  Go away.')

class ServerFactory(protocol.ServerFactory):
    protocol = Protocol

    def __init__(self, portal):
        if False:
            i = 10
            return i + 15
        self.portal = portal

    def buildProtocol(self, addr):
        if False:
            i = 10
            return i + 15
        p = protocol.ServerFactory.buildProtocol(self, addr)
        p.portal = self.portal
        return p

@implementer(portal.IRealm)
class Realm:

    def requestAvatar(self, avatarId, mind, *interfaces):
        if False:
            while True:
                i = 10
        if IProtocolUser in interfaces:
            if avatarId == checkers.ANONYMOUS:
                av = AnonymousUser()
            elif avatarId.isupper():
                av = Administrator()
            else:
                av = RegularUser()
            return (IProtocolUser, av, av.logout)
        raise NotImplementedError('Only IProtocolUser interface is supported by this realm')

def main():
    if False:
        i = 10
        return i + 15
    r = Realm()
    p = portal.Portal(r)
    c = checkers.InMemoryUsernamePasswordDatabaseDontUse()
    c.addUser(b'auser', b'thepass')
    c.addUser(b'SECONDUSER', b'secret')
    p.registerChecker(c)
    p.registerChecker(checkers.AllowAnonymousAccess())
    f = ServerFactory(p)
    log.startLogging(sys.stdout)
    from twisted.internet import reactor
    reactor.listenTCP(4738, f)
    reactor.run()
if __name__ == '__main__':
    main()