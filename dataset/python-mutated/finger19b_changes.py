import pwd
from twisted.internet import defer, protocol, reactor, utils

@implementer(IFingerService)
class LocalFingerService(service.Service):

    def getUser(self, user):
        if False:
            for i in range(10):
                print('nop')
        return utils.getProcessOutput('finger', [user])

    def getUsers(self):
        if False:
            i = 10
            return i + 15
        return defer.succeed([])
f = LocalFingerService()