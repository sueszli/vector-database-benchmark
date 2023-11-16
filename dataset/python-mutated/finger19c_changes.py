import os
import pwd
from twisted.internet import defer, protocol, reactor, utils

@implementer(IFingerService)
class LocalFingerService(service.Service):

    def getUser(self, user):
        if False:
            for i in range(10):
                print('nop')
        user = user.strip()
        try:
            entry = pwd.getpwnam(user)
        except KeyError:
            return defer.succeed('No such user')
        try:
            f = open(os.path.join(entry[5], '.plan'))
        except OSError:
            return defer.succeed('No such user')
        with f:
            data = f.read()
        data = data.strip()
        return defer.succeed(data)

    def getUsers(self):
        if False:
            print('Hello World!')
        return defer.succeed([])
f = LocalFingerService()