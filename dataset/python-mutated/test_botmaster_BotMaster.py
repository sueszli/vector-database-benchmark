from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot import config
from buildbot.process import factory
from buildbot.process.botmaster import BotMaster
from buildbot.process.results import CANCELLED
from buildbot.process.results import RETRY
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin

class TestCleanShutdown(TestReactorMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantData=True)
        self.botmaster = BotMaster()
        yield self.botmaster.setServiceParent(self.master)
        self.botmaster.startService()

    def assertReactorStopped(self, _=None):
        if False:
            return 10
        self.assertTrue(self.reactor.stop_called)

    def assertReactorNotStopped(self, _=None):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(self.reactor.stop_called)

    def makeFakeBuild(self, waitedFor=False):
        if False:
            while True:
                i = 10
        self.fake_builder = builder = mock.Mock()
        self.build_deferred = defer.Deferred()
        request = mock.Mock()
        request.waitedFor = waitedFor
        build = mock.Mock()
        build.stopBuild = self.stopFakeBuild
        build.waitUntilFinished.return_value = self.build_deferred
        build.requests = [request]
        builder.building = [build]
        self.botmaster.builders = mock.Mock()
        self.botmaster.builders.values.return_value = [builder]

    def stopFakeBuild(self, reason, results):
        if False:
            i = 10
            return i + 15
        self.reason = reason
        self.results = results
        self.finishFakeBuild()
        return defer.succeed(None)

    def finishFakeBuild(self):
        if False:
            print('Hello World!')
        self.fake_builder.building = []
        self.build_deferred.callback(None)

    def test_shutdown_idle(self):
        if False:
            i = 10
            return i + 15
        "Test that the master shuts down when it's idle"
        self.botmaster.cleanShutdown()
        self.assertReactorStopped()

    def test_shutdown_busy(self):
        if False:
            return 10
        'Test that the master shuts down after builds finish'
        self.makeFakeBuild()
        self.botmaster.cleanShutdown()
        self.assertReactorNotStopped()
        self.botmaster.cleanShutdown()
        self.finishFakeBuild()
        self.assertReactorStopped()

    def test_shutdown_busy_quick(self):
        if False:
            return 10
        'Test that the master shuts down after builds finish'
        self.makeFakeBuild()
        self.botmaster.cleanShutdown(quickMode=True)
        self.assertReactorStopped()
        self.assertEqual(self.results, RETRY)

    def test_shutdown_busy_quick_cancelled(self):
        if False:
            print('Hello World!')
        'Test that the master shuts down after builds finish'
        self.makeFakeBuild(waitedFor=True)
        self.botmaster.cleanShutdown(quickMode=True)
        self.assertReactorStopped()
        self.assertEqual(self.results, CANCELLED)

    def test_shutdown_cancel_not_shutting_down(self):
        if False:
            while True:
                i = 10
        'Test that calling cancelCleanShutdown when none is in progress\n        works'
        self.botmaster.cancelCleanShutdown()

    def test_shutdown_cancel(self):
        if False:
            print('Hello World!')
        'Test that we can cancel a shutdown'
        self.makeFakeBuild()
        self.botmaster.cleanShutdown()
        self.assertReactorNotStopped()
        self.assertFalse(self.botmaster.brd.running)
        self.botmaster.cancelCleanShutdown()
        self.finishFakeBuild()
        self.assertReactorNotStopped()
        self.assertTrue(self.botmaster.brd.running)

class TestBotMaster(TestReactorMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            while True:
                i = 10
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantMq=True, wantData=True)
        self.master.mq = self.master.mq
        self.master.botmaster.disownServiceParent()
        self.botmaster = BotMaster()
        yield self.botmaster.setServiceParent(self.master)
        self.new_config = mock.Mock()
        self.botmaster.startService()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        return self.botmaster.stopService()

    @defer.inlineCallbacks
    def test_reconfigServiceWithBuildbotConfig(self):
        if False:
            print('Hello World!')
        self.patch(self.botmaster, 'reconfigProjects', mock.Mock(side_effect=lambda c: defer.succeed(None)))
        self.patch(self.botmaster, 'reconfigServiceBuilders', mock.Mock(side_effect=lambda c: defer.succeed(None)))
        self.patch(self.botmaster, 'maybeStartBuildsForAllBuilders', mock.Mock())
        new_config = mock.Mock()
        yield self.botmaster.reconfigServiceWithBuildbotConfig(new_config)
        self.botmaster.reconfigServiceBuilders.assert_called_with(new_config)
        self.botmaster.reconfigProjects.assert_called_with(new_config)
        self.assertTrue(self.botmaster.maybeStartBuildsForAllBuilders.called)

    @defer.inlineCallbacks
    def test_reconfigServiceBuilders_add_remove(self):
        if False:
            while True:
                i = 10
        bc = config.BuilderConfig(name='bldr', factory=factory.BuildFactory(), workername='f')
        self.new_config.builders = [bc]
        yield self.botmaster.reconfigServiceBuilders(self.new_config)
        bldr = self.botmaster.builders['bldr']
        self.assertIdentical(bldr.parent, self.botmaster)
        self.assertIdentical(bldr.master, self.master)
        self.assertEqual(self.botmaster.builderNames, ['bldr'])
        self.new_config.builders = []
        yield self.botmaster.reconfigServiceBuilders(self.new_config)
        self.assertIdentical(bldr.parent, None)
        self.assertIdentical(bldr.master, None)
        self.assertEqual(self.botmaster.builders, {})
        self.assertEqual(self.botmaster.builderNames, [])

    def test_maybeStartBuildsForBuilder(self):
        if False:
            i = 10
            return i + 15
        brd = self.botmaster.brd = mock.Mock()
        self.botmaster.maybeStartBuildsForBuilder('frank')
        brd.maybeStartBuildsOn.assert_called_once_with(['frank'])

    def test_maybeStartBuildsForWorker(self):
        if False:
            return 10
        brd = self.botmaster.brd = mock.Mock()
        b1 = mock.Mock(name='frank')
        b1.name = 'frank'
        b2 = mock.Mock(name='larry')
        b2.name = 'larry'
        self.botmaster.getBuildersForWorker = mock.Mock(return_value=[b1, b2])
        self.botmaster.maybeStartBuildsForWorker('centos')
        self.botmaster.getBuildersForWorker.assert_called_once_with('centos')
        brd.maybeStartBuildsOn.assert_called_once_with(['frank', 'larry'])

    def test_maybeStartBuildsForAll(self):
        if False:
            while True:
                i = 10
        brd = self.botmaster.brd = mock.Mock()
        self.botmaster.builderNames = ['frank', 'larry']
        self.botmaster.maybeStartBuildsForAllBuilders()
        brd.maybeStartBuildsOn.assert_called_once_with(['frank', 'larry'])