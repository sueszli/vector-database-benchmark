from unittest import mock
from twisted.internet import defer

class PBManagerMixin:

    def setUpPBChangeSource(self):
        if False:
            while True:
                i = 10
        'Set up a fake self.pbmanager.'
        self.registrations = []
        self.unregistrations = []
        pbm = self.pbmanager = mock.Mock()
        pbm.register = self._fake_register

    def _fake_register(self, portstr, username, password, factory):
        if False:
            for i in range(10):
                print('nop')
        reg = mock.Mock()

        def unregister():
            if False:
                for i in range(10):
                    print('nop')
            self.unregistrations.append((portstr, username, password))
            return defer.succeed(None)
        reg.unregister = unregister
        self.registrations.append((portstr, username, password))
        return reg

    def assertNotRegistered(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.registrations, [])

    def assertNotUnregistered(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.unregistrations, [])

    def assertRegistered(self, portstr, username, password):
        if False:
            print('Hello World!')
        for (ps, un, pw) in self.registrations:
            if ps == portstr and username == un and (pw == password):
                return
        self.fail(f'not registered: {repr(portstr, username, password)} not in {self.registrations}')

    def assertUnregistered(self, portstr, username, password):
        if False:
            i = 10
            return i + 15
        for (ps, un, pw) in self.unregistrations:
            if ps == portstr and username == un and (pw == password):
                return
        self.fail('still registered')