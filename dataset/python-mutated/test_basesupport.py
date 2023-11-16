from twisted.internet import defer, error
from twisted.trial import unittest
from twisted.words.im import basesupport

class DummyAccount(basesupport.AbstractAccount):
    """
    An account object that will do nothing when asked to start to log on.
    """
    loginHasFailed = False
    loginCallbackCalled = False

    def _startLogOn(self, *args):
        if False:
            while True:
                i = 10
        '\n        Set self.loginDeferred to the same as the deferred returned, allowing a\n        testcase to .callback or .errback.\n\n        @return: A deferred.\n        '
        self.loginDeferred = defer.Deferred()
        return self.loginDeferred

    def _loginFailed(self, result):
        if False:
            return 10
        self.loginHasFailed = True
        return basesupport.AbstractAccount._loginFailed(self, result)

    def _cb_logOn(self, result):
        if False:
            for i in range(10):
                print('nop')
        self.loginCallbackCalled = True
        return basesupport.AbstractAccount._cb_logOn(self, result)

class DummyUI:
    """
    Provide just the interface required to be passed to AbstractAccount.logOn.
    """
    clientRegistered = False

    def registerAccountClient(self, result):
        if False:
            i = 10
            return i + 15
        self.clientRegistered = True

class ClientMsgTests(unittest.TestCase):

    def makeUI(self):
        if False:
            return 10
        return DummyUI()

    def makeAccount(self):
        if False:
            print('Hello World!')
        return DummyAccount('la', False, 'la', None, 'localhost', 6667)

    def test_connect(self):
        if False:
            print('Hello World!')
        '\n        Test that account.logOn works, and it calls the right callback when a\n        connection is established.\n        '
        account = self.makeAccount()
        ui = self.makeUI()
        d = account.logOn(ui)
        account.loginDeferred.callback(None)

        def check(result):
            if False:
                for i in range(10):
                    print('nop')
            self.assertFalse(account.loginHasFailed, "Login shouldn't have failed")
            self.assertTrue(account.loginCallbackCalled, 'We should be logged in')
        d.addCallback(check)
        return d

    def test_failedConnect(self):
        if False:
            while True:
                i = 10
        '\n        Test that account.logOn works, and it calls the right callback when a\n        connection is established.\n        '
        account = self.makeAccount()
        ui = self.makeUI()
        d = account.logOn(ui)
        account.loginDeferred.errback(Exception())

        def err(reason):
            if False:
                return 10
            self.assertTrue(account.loginHasFailed, 'Login should have failed')
            self.assertFalse(account.loginCallbackCalled, "We shouldn't be logged in")
            self.assertTrue(not ui.clientRegistered, "Client shouldn't be registered in the UI")
        cb = lambda r: self.assertTrue(False, "Shouldn't get called back")
        d.addCallbacks(cb, err)
        return d

    def test_alreadyConnecting(self):
        if False:
            while True:
                i = 10
        '\n        Test that it can fail sensibly when someone tried to connect before\n        we did.\n        '
        account = self.makeAccount()
        ui = self.makeUI()
        account.logOn(ui)
        self.assertRaises(error.ConnectError, account.logOn, ui)