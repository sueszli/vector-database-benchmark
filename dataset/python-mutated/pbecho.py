if __name__ == '__main__':
    from pbecho import main
    raise SystemExit(main())
from zope.interface import implementer
from twisted.cred.portal import IRealm
from twisted.spread import pb

class DefinedError(pb.Error):
    pass

class SimplePerspective(pb.Avatar):

    def perspective_echo(self, text):
        if False:
            print('Hello World!')
        print('echoing', text)
        return text

    def perspective_error(self):
        if False:
            print('Hello World!')
        raise DefinedError('exception!')

    def logout(self):
        if False:
            i = 10
            return i + 15
        print(self, 'logged out')

@implementer(IRealm)
class SimpleRealm:

    def requestAvatar(self, avatarId, mind, *interfaces):
        if False:
            return 10
        if pb.IPerspective in interfaces:
            avatar = SimplePerspective()
            return (pb.IPerspective, avatar, avatar.logout)
        else:
            raise NotImplementedError('no interface')

def main():
    if False:
        return 10
    from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
    from twisted.cred.portal import Portal
    from twisted.internet import reactor
    portal = Portal(SimpleRealm())
    checker = InMemoryUsernamePasswordDatabaseDontUse()
    checker.addUser('guest', 'guest')
    portal.registerChecker(checker)
    reactor.listenTCP(pb.portno, pb.PBServerFactory(portal))
    reactor.run()