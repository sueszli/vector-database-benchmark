from twisted.internet import defer
from twisted.trial import unittest
from buildbot import util
from buildbot.changes import base
from buildbot.changes.manager import ChangeManager
from buildbot.test.fake import fakemaster
from buildbot.test.fake.web import FakeRequest
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util.warnings import assertProducesWarnings
from buildbot.warnings import DeprecatedApiWarning
from buildbot.www import change_hook

class TestPollingChangeHook(TestReactorMixin, unittest.TestCase):

    class Subclass(base.ReconfigurablePollingChangeSource):
        pollInterval = None
        called = False

        def poll(self):
            if False:
                i = 10
                return i + 15
            self.called = True

    class OldstyleSubclass(base.PollingChangeSource):
        pollInterval = None
        called = False

        def poll(self):
            if False:
                i = 10
                return i + 15
            self.called = True

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_test_reactor()

    @defer.inlineCallbacks
    def setUpRequest(self, args, options=True, activate=True, poller_cls=Subclass):
        if False:
            return 10
        self.request = FakeRequest(args=args)
        self.request.uri = b'/change_hook/poller'
        self.request.method = b'GET'
        www = self.request.site.master.www
        self.master = master = self.request.site.master = fakemaster.make_master(self, wantData=True)
        master.www = www
        yield self.master.startService()
        self.changeHook = change_hook.ChangeHookResource(dialects={'poller': options}, master=master)
        master.change_svc = ChangeManager()
        yield master.change_svc.setServiceParent(master)
        self.changesrc = poller_cls(21, name=b'example')
        yield self.changesrc.setServiceParent(master.change_svc)
        self.otherpoller = poller_cls(22, name=b'otherpoller')
        yield self.otherpoller.setServiceParent(master.change_svc)
        anotherchangesrc = base.ChangeSource(name=b'notapoller')
        anotherchangesrc.setName('notapoller')
        yield anotherchangesrc.setServiceParent(master.change_svc)
        yield self.request.test_render(self.changeHook)
        yield util.asyncSleep(0.1)

    def tearDown(self):
        if False:
            return 10
        return self.master.stopService()

    @defer.inlineCallbacks
    def test_no_args(self):
        if False:
            i = 10
            return i + 15
        yield self.setUpRequest({})
        self.assertEqual(self.request.written, b'no change found')
        self.assertEqual(self.changesrc.called, True)
        self.assertEqual(self.otherpoller.called, True)

    @defer.inlineCallbacks
    def test_no_poller(self):
        if False:
            i = 10
            return i + 15
        yield self.setUpRequest({b'poller': [b'nosuchpoller']})
        expected = b'Could not find pollers: nosuchpoller'
        self.assertEqual(self.request.written, expected)
        self.request.setResponseCode.assert_called_with(400, expected)
        self.assertEqual(self.changesrc.called, False)
        self.assertEqual(self.otherpoller.called, False)

    @defer.inlineCallbacks
    def test_invalid_poller(self):
        if False:
            while True:
                i = 10
        yield self.setUpRequest({b'poller': [b'notapoller']})
        expected = b'Could not find pollers: notapoller'
        self.assertEqual(self.request.written, expected)
        self.request.setResponseCode.assert_called_with(400, expected)
        self.assertEqual(self.changesrc.called, False)
        self.assertEqual(self.otherpoller.called, False)

    @defer.inlineCallbacks
    def test_trigger_poll(self):
        if False:
            return 10
        yield self.setUpRequest({b'poller': [b'example']})
        self.assertEqual(self.request.written, b'no change found')
        self.assertEqual(self.changesrc.called, True)
        self.assertEqual(self.otherpoller.called, False)

    @defer.inlineCallbacks
    def test_allowlist_deny(self):
        if False:
            return 10
        yield self.setUpRequest({b'poller': [b'otherpoller']}, options={b'allowed': [b'example']})
        expected = b'Could not find pollers: otherpoller'
        self.assertEqual(self.request.written, expected)
        self.request.setResponseCode.assert_called_with(400, expected)
        self.assertEqual(self.changesrc.called, False)
        self.assertEqual(self.otherpoller.called, False)

    @defer.inlineCallbacks
    def test_allowlist_allow(self):
        if False:
            return 10
        yield self.setUpRequest({b'poller': [b'example']}, options={b'allowed': [b'example']})
        self.assertEqual(self.request.written, b'no change found')
        self.assertEqual(self.changesrc.called, True)
        self.assertEqual(self.otherpoller.called, False)

    @defer.inlineCallbacks
    def test_allowlist_all(self):
        if False:
            i = 10
            return i + 15
        yield self.setUpRequest({}, options={b'allowed': [b'example']})
        self.assertEqual(self.request.written, b'no change found')
        self.assertEqual(self.changesrc.called, True)
        self.assertEqual(self.otherpoller.called, False)

    @defer.inlineCallbacks
    def test_trigger_old_poller(self):
        if False:
            print('Hello World!')
        with assertProducesWarnings(DeprecatedApiWarning, num_warnings=2, message_pattern='use ReconfigurablePollingChangeSource'):
            yield self.setUpRequest({b'poller': [b'example']}, poller_cls=self.OldstyleSubclass)
        self.assertEqual(self.request.written, b'no change found')
        self.assertEqual(self.changesrc.called, True)
        self.assertEqual(self.otherpoller.called, False)