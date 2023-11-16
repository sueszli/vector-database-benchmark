from zope.interface import implementer
from twisted.cred import checkers, portal
from twisted.internet import reactor
from twisted.spread import pb

class MyPerspective(pb.Avatar):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.name = name

    def perspective_foo(self, arg):
        if False:
            while True:
                i = 10
        print('I am', self.name, 'perspective_foo(', arg, ') called on', self)

@implementer(portal.IRealm)
class MyRealm:

    def requestAvatar(self, avatarId, mind, *interfaces):
        if False:
            while True:
                i = 10
        if pb.IPerspective not in interfaces:
            raise NotImplementedError
        return (pb.IPerspective, MyPerspective(avatarId), lambda : None)
p = portal.Portal(MyRealm())
c = checkers.InMemoryUsernamePasswordDatabaseDontUse(user1='pass1', user2='pass2')
p.registerChecker(c)
reactor.listenTCP(8800, pb.PBServerFactory(p))
reactor.run()