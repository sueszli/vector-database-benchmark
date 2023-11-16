from twisted.internet import defer
from buildbot.util import service

class FakeMsgManager(service.AsyncMultiService):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.setName('fake-msgmanager')
        self._registrations = []
        self._unregistrations = []

    def register(self, portstr, username, password, pfactory):
        if False:
            while True:
                i = 10
        if (portstr, username) not in self._registrations:
            reg = FakeRegistration(self, portstr, username)
            self._registrations.append((portstr, username, password))
            return defer.succeed(reg)
        else:
            raise KeyError(f"username '{username}' is already registered on port {portstr}")

    def _unregister(self, portstr, username):
        if False:
            i = 10
            return i + 15
        self._unregistrations.append((portstr, username))
        return defer.succeed(None)

class FakeRegistration:

    def __init__(self, msgmanager, portstr, username):
        if False:
            print('Hello World!')
        self._portstr = portstr
        self._username = username
        self._msgmanager = msgmanager

    def unregister(self):
        if False:
            return 10
        self._msgmanager._unregister(self._portstr, self._username)