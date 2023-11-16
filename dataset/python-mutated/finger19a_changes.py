class IFingerSetterService(Interface):

    def setUser(user, status):
        if False:
            print('Hello World!')
        "Set the user's status to something"

@implementer(IFingerService, IFingerSetterService)
class MemoryFingerService(service.Service):

    def __init__(self, users):
        if False:
            return 10
        self.users = users

    def getUser(self, user):
        if False:
            return 10
        return defer.succeed(self.users.get(user, b'No such user'))

    def getUsers(self):
        if False:
            print('Hello World!')
        return defer.succeed(list(self.users.keys()))

    def setUser(self, user, status):
        if False:
            print('Hello World!')
        self.users[user] = status
f = MemoryFingerService({b'moshez': b'Happy and well'})
serviceCollection = service.IServiceCollection(application)
strports.service('tcp:1079:interface=127.0.0.1', IFingerSetterFactory(f)).setServiceParent(serviceCollection)