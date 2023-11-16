import re
from unittest import mock
from twisted.internet import defer
from twisted.internet import reactor
from twisted.trial import unittest
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS
from buildbot.reporters import words
from buildbot.test import fakedb
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.util import datetime2epoch

class ContactMixin(TestReactorMixin):
    botClass = words.StatusBot
    channelClass = words.Channel
    contactClass = words.Contact
    USER = 'me'
    CHANNEL = '#buildbot'
    BUILDER_NAMES = ['builder1', 'builder2']
    BUILDER_IDS = [23, 45]

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.patch(reactor, 'callLater', self.reactor.callLater)
        self.patch(reactor, 'seconds', self.reactor.seconds)
        self.patch(reactor, 'stop', self.reactor.stop)
        self.master = fakemaster.make_master(self, wantMq=True, wantData=True, wantDb=True)
        for (builderid, name) in zip(self.BUILDER_IDS, self.BUILDER_NAMES):
            self.master.db.builders.addTestBuilder(builderid=builderid, name=name)
        self.bot = self.botClass(notify_events={'success': 1, 'failure': 1})
        self.bot.channelClass = self.channelClass
        self.bot.contactClass = self.contactClass
        self.bot.nickname = 'nick'
        self.missing_workers = set()
        self.subscribed = False
        self.bot.parent = self
        self.bot.master.botmaster = mock.Mock(name='StatusBot-instance.master.botmaster')
        self.bot.master.botmaster.shuttingDown = False

        def cleanShutdown():
            if False:
                for i in range(10):
                    print('nop')
            self.bot.master.botmaster.shuttingDown = True
        self.bot.master.botmaster.cleanShutdown = cleanShutdown

        def cancelCleanShutdown():
            if False:
                i = 10
                return i + 15
            self.bot.master.botmaster.shuttingDown = False
        self.bot.master.botmaster.cancelCleanShutdown = cancelCleanShutdown
        self.contact = self.contactClass(user=self.USER, channel=self.bot.getChannel(self.CHANNEL))
        yield self.contact.channel.setServiceParent(self.master)
        yield self.master.startService()

    def patch_send(self):
        if False:
            print('Hello World!')
        self.sent = []

        def send(msg):
            if False:
                while True:
                    i = 10
            if not isinstance(msg, (list, tuple)):
                msg = (msg,)
            for m in msg:
                self.sent.append(m)
        self.contact.channel.send = send

    @defer.inlineCallbacks
    def do_test_command(self, command, args='', contact=None, clock_ticks=None, exp_usage=True, exp_UsageError=False, shuttingDown=False, **kwargs):
        if False:
            print('Hello World!')
        if contact is None:
            contact = self.contact
        cmd = getattr(contact, 'command_' + command.upper())
        if exp_usage:
            self.assertTrue(hasattr(cmd, 'usage'))
        self.patch_send()
        self.bot.master.botmaster.shuttingDown = shuttingDown
        if exp_UsageError:
            try:
                yield cmd(args, **kwargs)
            except words.UsageError:
                return
            else:
                self.fail('no UsageError')
        else:
            yield cmd(args, **kwargs)
        if clock_ticks:
            self.reactor.pump(clock_ticks)

    def setupSomeBuilds(self):
        if False:
            for i in range(10):
                print('nop')
        self.master.db.insert_test_data([fakedb.Build(id=13, masterid=88, workerid=13, builderid=self.BUILDER_IDS[0], buildrequestid=82, number=3), fakedb.Build(id=14, masterid=88, workerid=13, builderid=self.BUILDER_IDS[0], buildrequestid=83, number=4), fakedb.Build(id=15, masterid=88, workerid=13, builderid=self.BUILDER_IDS[1], buildrequestid=84, number=5), fakedb.Build(id=16, masterid=88, workerid=13, builderid=self.BUILDER_IDS[0], buildrequestid=85, number=6)])
        self.master.db.builds.finishBuild(buildid=14, results=SUCCESS)

    def setup_multi_builders(self):
        if False:
            print('Hello World!')
        self.master.db.insert_test_data([fakedb.Worker(id=1, name='linux1', info={}), fakedb.Worker(id=2, name='linux2', info={}), fakedb.BuilderMaster(id=4012, masterid=13, builderid=self.BUILDER_IDS[0]), fakedb.BuilderMaster(id=4013, masterid=13, builderid=self.BUILDER_IDS[1]), fakedb.ConfiguredWorker(id=14013, workerid=2, buildermasterid=4012), fakedb.ConfiguredWorker(id=14013, workerid=1, buildermasterid=4013)])

class TestContact(ContactMixin, unittest.TestCase):

    def test_channel_service(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.contact.channel.running)
        self.contact.channel.stopService()

    @defer.inlineCallbacks
    def test_command_notify0(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.do_test_command('notify', exp_UsageError=True)
        yield self.do_test_command('notify', args='invalid arg', exp_UsageError=True)
        yield self.do_test_command('notify', args='on')
        self.assertEqual(self.sent, ['The following events are being notified: finished, started.'])
        yield self.do_test_command('notify', args='off')
        self.assertEqual(self.sent, ['No events are being notified.'])
        yield self.do_test_command('notify', args='on started')
        self.assertEqual(self.sent, ['The following events are being notified: started.'])
        yield self.do_test_command('notify', args='off started')
        self.assertEqual(self.sent, ['No events are being notified.'])
        yield self.assertFailure(self.do_test_command('notify', args='off finished'), KeyError)
        yield self.do_test_command('notify', args='list')
        self.assertEqual(self.sent, ['No events are being notified.'])

    @defer.inlineCallbacks
    def notify_build_test(self, notify_args):
        if False:
            return 10
        self.bot.tags = None
        yield self.test_command_watch_builder0()
        yield self.do_test_command('notify', args=notify_args)
        buildStarted = self.contact.channel.subscribed[0].callback
        buildFinished = self.contact.channel.subscribed[1].callback
        for buildid in (13, 14, 16):
            self.master.db.builds.finishBuild(buildid=buildid, results=SUCCESS)
            build = (yield self.master.db.builds.getBuild(buildid))
            buildStarted('somekey', build)
            buildFinished('somekey', build)

    def test_command_notify_build_started(self):
        if False:
            print('Hello World!')
        self.notify_build_test('on started')

    def test_command_notify_build_finished(self):
        if False:
            while True:
                i = 10
        self.notify_build_test('on finished')

    def test_command_notify_build_better(self):
        if False:
            print('Hello World!')
        self.notify_build_test('on better')

    def test_command_notify_build_worse(self):
        if False:
            i = 10
            return i + 15
        self.notify_build_test('on worse')

    def test_command_notify_build_problem(self):
        if False:
            return 10
        self.notify_build_test('on problem')

    def test_command_notify_build_recovery(self):
        if False:
            i = 10
            return i + 15
        self.notify_build_test('on recovery')

    def test_command_notify_build_started_finished(self):
        if False:
            for i in range(10):
                print('nop')
        self.notify_build_test('on')

    @defer.inlineCallbacks
    def test_notify_missing_worker(self):
        if False:
            print('Hello World!')
        self.patch_send()
        yield self.do_test_command('notify', args='on worker')
        missing_worker = self.contact.channel.subscribed[2].callback
        missing_worker((None, None, 'missing'), {'workerid': 1, 'name': 'work', 'last_connection': 'sometime'})
        self.assertEquals(self.sent[1], 'Worker `work` is missing. It was seen last on sometime.')
        self.assertIn(1, self.contact.channel.missing_workers)

    @defer.inlineCallbacks
    def test_notify_worker_is_back(self):
        if False:
            for i in range(10):
                print('nop')
        self.patch_send()
        yield self.do_test_command('notify', args='on worker')
        self.contact.channel.missing_workers.add(1)
        missing_worker = self.contact.channel.subscribed[2].callback
        missing_worker((None, None, 'connected'), {'workerid': 1, 'name': 'work', 'last_connection': 'sometime'})
        self.assertEquals(self.sent[1], 'Worker `work` is back online.')
        self.assertNotIn(1, self.contact.channel.missing_workers)

    @defer.inlineCallbacks
    def test_command_help_noargs(self):
        if False:
            i = 10
            return i + 15
        yield self.do_test_command('help')
        self.assertIn('help - ', '\n'.join(self.sent))

    @defer.inlineCallbacks
    def test_command_help_arg(self):
        if False:
            print('Hello World!')
        self.contact.command_FOO = lambda : None
        self.contact.command_FOO.usage = 'foo - bar'
        yield self.do_test_command('help', args='foo')
        self.assertIn('Usage: foo - bar', self.sent[0])

    @defer.inlineCallbacks
    def test_command_help_no_usage(self):
        if False:
            print('Hello World!')
        self.contact.command_FOO = lambda : None
        yield self.do_test_command('help', args='foo')
        self.assertIn('No usage info for', self.sent[0])

    @defer.inlineCallbacks
    def test_command_help_dict_command(self):
        if False:
            for i in range(10):
                print('nop')
        self.contact.command_FOO = lambda : None
        self.contact.command_FOO.usage = {None: 'foo - bar'}
        yield self.do_test_command('help', args='foo')
        self.assertIn('Usage: foo - bar', self.sent[0])

    @defer.inlineCallbacks
    def test_command_help_dict_command_no_usage(self):
        if False:
            while True:
                i = 10
        self.contact.command_FOO = lambda : None
        self.contact.command_FOO.usage = {}
        yield self.do_test_command('help', args='foo')
        self.assertIn("No usage info for 'foo'", self.sent[0])

    @defer.inlineCallbacks
    def test_command_help_dict_command_arg(self):
        if False:
            for i in range(10):
                print('nop')
        self.contact.command_FOO = lambda : None
        self.contact.command_FOO.usage = {'this': 'foo this - bar'}
        yield self.do_test_command('help', args='foo this')
        self.assertIn('Usage: foo this - bar', self.sent[0])

    @defer.inlineCallbacks
    def test_command_help_dict_command_arg_no_usage(self):
        if False:
            for i in range(10):
                print('nop')
        self.contact.command_FOO = lambda : None
        self.contact.command_FOO.usage = {('this', 'first'): 'foo this first - bar'}
        yield self.do_test_command('help', args='foo this')
        self.assertIn("No usage info for 'foo' 'this'", self.sent[0])

    @defer.inlineCallbacks
    def test_command_help_dict_command_arg_subarg(self):
        if False:
            i = 10
            return i + 15
        self.contact.command_FOO = lambda : None
        self.contact.command_FOO.usage = {('this', 'first'): 'foo this first - bar'}
        yield self.do_test_command('help', args='foo this first')
        self.assertIn('Usage: foo this first - bar', self.sent[0])

    @defer.inlineCallbacks
    def test_command_help_dict_command_arg_subarg_no_usage(self):
        if False:
            i = 10
            return i + 15
        self.contact.command_FOO = lambda : None
        self.contact.command_FOO.usage = {None: 'foo - bar', 'this': 'foo this - bar', ('this', 'first'): 'foo this first - bar'}
        yield self.do_test_command('help', args='foo this missing')
        self.assertIn("No usage info for 'foo' 'this' 'missing'", self.sent[0])

    @defer.inlineCallbacks
    def test_command_help_nosuch(self):
        if False:
            print('Hello World!')
        yield self.do_test_command('help', args='foo', exp_UsageError=True)

    @defer.inlineCallbacks
    def test_command_shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.do_test_command('shutdown', exp_UsageError=True)
        self.assertEqual(self.bot.master.botmaster.shuttingDown, False)

    @defer.inlineCallbacks
    def test_command_shutdown_check_running(self):
        if False:
            print('Hello World!')
        yield self.do_test_command('shutdown', args='check', shuttingDown=False)
        self.assertEqual(self.bot.master.botmaster.shuttingDown, False)
        self.assertIn('buildbot is running', self.sent[0])

    @defer.inlineCallbacks
    def test_command_shutdown_check_shutting_down(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.do_test_command('shutdown', args='check', shuttingDown=True)
        self.assertEqual(self.bot.master.botmaster.shuttingDown, True)
        self.assertIn('buildbot is shutting down', self.sent[0])

    @defer.inlineCallbacks
    def test_command_shutdown_start(self):
        if False:
            print('Hello World!')
        yield self.do_test_command('shutdown', args='start', shuttingDown=False)
        self.assertEqual(self.bot.master.botmaster.shuttingDown, True)

    @defer.inlineCallbacks
    def test_command_shutdown_stop(self):
        if False:
            while True:
                i = 10
        yield self.do_test_command('shutdown', args='stop', shuttingDown=True)
        self.assertEqual(self.bot.master.botmaster.shuttingDown, False)

    @defer.inlineCallbacks
    def test_command_shutdown_now(self):
        if False:
            i = 10
            return i + 15
        yield self.do_test_command('shutdown', args='now')
        self.assertEqual(self.bot.master.botmaster.shuttingDown, False)
        self.assertTrue(self.reactor.stop_called)

    @defer.inlineCallbacks
    def test_command_source(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.do_test_command('source')
        self.assertIn('My source', self.sent[0])

    @defer.inlineCallbacks
    def test_command_commands(self):
        if False:
            print('Hello World!')
        yield self.do_test_command('commands')
        self.assertIn('Buildbot commands', self.sent[0])

    @defer.inlineCallbacks
    def test_command_hello(self):
        if False:
            i = 10
            return i + 15
        yield self.do_test_command('hello', exp_usage=False)
        self.assertIn(self.sent[0], words.GREETINGS)

    @defer.inlineCallbacks
    def test_command_list(self):
        if False:
            print('Hello World!')
        yield self.do_test_command('list', exp_UsageError=True)

    @defer.inlineCallbacks
    def test_command_list_builders(self):
        if False:
            return 10
        yield self.do_test_command('list', args='all builders')
        self.assertEqual(len(self.sent), 1)
        for builder in self.BUILDER_NAMES:
            self.assertIn(f'{builder} [offline]', self.sent[0])

    @defer.inlineCallbacks
    def test_command_list_workers(self):
        if False:
            print('Hello World!')
        workers = ['worker1', 'worker2']
        for worker in workers:
            self.master.db.workers.db.insert_test_data([fakedb.Worker(name=worker)])
        yield self.do_test_command('list', args='all workers')
        self.assertEqual(len(self.sent), 1)
        for worker in workers:
            self.assertIn(f'{worker} [offline]', self.sent[0])

    @defer.inlineCallbacks
    def test_command_list_workers_online(self):
        if False:
            while True:
                i = 10
        self.setup_multi_builders()
        self.master.db.insert_test_data([fakedb.ConnectedWorker(id=113, masterid=13, workerid=1)])
        yield self.do_test_command('list', args='all workers')
        self.assertEqual(len(self.sent), 1)
        self.assertNotIn('linux1 [disconnected]', self.sent[0])
        self.assertIn('linux2 [disconnected]', self.sent[0])

    @defer.inlineCallbacks
    def test_command_list_changes(self):
        if False:
            for i in range(10):
                print('nop')
        self.master.db.workers.db.insert_test_data([fakedb.Change()])
        yield self.do_test_command('list', args='2 changes')
        self.assertEqual(len(self.sent), 1)

    @defer.inlineCallbacks
    def test_command_list_builders_not_connected(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_multi_builders()
        yield self.do_test_command('list', args='all builders')
        self.assertEqual(len(self.sent), 1)
        self.assertIn(f'{self.BUILDER_NAMES[0]} [offline]', self.sent[0])
        self.assertIn(f'{self.BUILDER_NAMES[1]} [offline]', self.sent[0])

    @defer.inlineCallbacks
    def test_command_list_builders_connected(self):
        if False:
            print('Hello World!')
        self.setup_multi_builders()
        self.master.db.insert_test_data([fakedb.ConnectedWorker(id=113, masterid=13, workerid=1)])
        yield self.do_test_command('list', args='all builders')
        self.assertEqual(len(self.sent), 1)
        self.assertIn(f'{self.BUILDER_NAMES[0]} [offline]', self.sent[0])
        self.assertNotIn(f'{self.BUILDER_NAMES[1]} [offline]', self.sent[0])

    @defer.inlineCallbacks
    def test_command_status(self):
        if False:
            while True:
                i = 10
        yield self.do_test_command('status')

    @defer.inlineCallbacks
    def test_command_status_online(self):
        if False:
            while True:
                i = 10
        self.setup_multi_builders()
        self.master.db.insert_test_data([fakedb.ConfiguredWorker(id=14012, workerid=1, buildermasterid=4013), fakedb.ConnectedWorker(id=114, masterid=13, workerid=1)])
        self.setupSomeBuilds()
        self.master.db.builds.finishBuild(buildid=13, results=FAILURE)
        self.master.db.builds.finishBuild(buildid=15, results=SUCCESS)
        self.master.db.builds.finishBuild(buildid=16, results=FAILURE)
        yield self.do_test_command('status')

    @defer.inlineCallbacks
    def test_command_status_all(self):
        if False:
            while True:
                i = 10
        yield self.do_test_command('status', args='all')

    @defer.inlineCallbacks
    def test_command_status_builder0_offline(self):
        if False:
            while True:
                i = 10
        yield self.do_test_command('status', args=self.BUILDER_NAMES[0])
        self.assertEqual(self.sent, [f'`{self.BUILDER_NAMES[0]}`: offline'])

    @defer.inlineCallbacks
    def test_command_status_builder0_running(self):
        if False:
            print('Hello World!')
        self.setupSomeBuilds()
        yield self.do_test_command('status', args=self.BUILDER_NAMES[0])
        self.assertEqual(len(self.sent), 1)
        self.assertIn('`builder1`: running', self.sent[0])
        self.assertRegex(self.sent[0], ' build \\[#3\\].* \\(no current step\\)')
        self.assertRegex(self.sent[0], ' build \\[#6\\].* \\(no current step\\)')

    @defer.inlineCallbacks
    def test_command_status_bogus(self):
        if False:
            i = 10
            return i + 15
        yield self.do_test_command('status', args='bogus_builder', exp_UsageError=True)

    def sub_seconds(self, strings):
        if False:
            print('Hello World!')
        return [re.sub('\\d seconds|a moment', 'N seconds', s) for s in strings]

    @defer.inlineCallbacks
    def test_command_last(self):
        if False:
            print('Hello World!')
        self.setupSomeBuilds()
        self.setup_multi_builders()
        self.master.db.insert_test_data([fakedb.ConnectedWorker(id=113, masterid=13, workerid=2)])
        yield self.do_test_command('last')
        self.assertEqual(len(self.sent), 1)
        sent = self.sub_seconds(self.sent)
        self.assertIn('`builder1`: last build completed successfully (N seconds ago)', sent)

    @defer.inlineCallbacks
    def test_command_last_all(self):
        if False:
            while True:
                i = 10
        self.setupSomeBuilds()
        yield self.do_test_command('last', args='all')
        self.assertEqual(len(self.sent), 1)
        sent = self.sub_seconds(self.sent)
        self.assertIn('`builder1`: last build completed successfully (N seconds ago)', sent[0])
        self.assertIn('`builder2`: no builds run since last restart', sent[0])

    @defer.inlineCallbacks
    def test_command_last_builder_bogus(self):
        if False:
            i = 10
            return i + 15
        yield self.do_test_command('last', args='BOGUS', exp_UsageError=True)

    @defer.inlineCallbacks
    def test_command_last_builder0(self):
        if False:
            while True:
                i = 10
        self.setupSomeBuilds()
        yield self.do_test_command('last', args=self.BUILDER_NAMES[0])
        self.assertEqual(len(self.sent), 1)
        sent = self.sub_seconds(self.sent)
        self.assertIn('`builder1`: last build completed successfully (N seconds ago)', sent)

    @defer.inlineCallbacks
    def test_command_last_builder1(self):
        if False:
            return 10
        self.setupSomeBuilds()
        yield self.do_test_command('last', args=self.BUILDER_NAMES[1])
        self.assertEqual(len(self.sent), 1)
        self.assertIn('`builder2`: no builds run since last restart', self.sent)

    @defer.inlineCallbacks
    def test_command_watch(self):
        if False:
            print('Hello World!')
        yield self.do_test_command('watch', exp_UsageError=True)

    @defer.inlineCallbacks
    def test_command_watch_builder0_no_builds(self):
        if False:
            print('Hello World!')
        yield self.do_test_command('watch', args=self.BUILDER_NAMES[0])
        self.assertEqual(len(self.sent), 1)
        self.assertIn('There are no currently running builds.', self.sent[0])

    @defer.inlineCallbacks
    def test_command_watch_builder0(self):
        if False:
            for i in range(10):
                print('nop')
        self.setupSomeBuilds()
        yield self.do_test_command('watch', args=self.BUILDER_NAMES[0])
        self.assertEqual(len(self.sent), 2)
        self.assertIn('Watching build [#3](http://localhost:8080/#/builders/23/builds/3) of `builder1` until it finishes...', self.sent)
        self.assertIn('Watching build [#6](http://localhost:8080/#/builders/23/builds/6) of `builder1` until it finishes...', self.sent)

    @defer.inlineCallbacks
    def test_command_watch_builder0_get_notifications(self):
        if False:
            print('Hello World!')
        self.bot.tags = None
        yield self.test_command_watch_builder0()
        del self.sent[:]
        yield self.sendBuildFinishedMessage(16)
        self.assertEqual(len(self.sent), 1)
        self.assertIn('Build [#6](http://localhost:8080/#/builders/23/builds/6) of `builder1` completed successfully.', self.sent)

    @defer.inlineCallbacks
    def test_command_watch_builder1(self):
        if False:
            i = 10
            return i + 15
        self.setupSomeBuilds()
        yield self.do_test_command('watch', args=self.BUILDER_NAMES[0])
        self.assertEqual(len(self.sent), 2)
        self.assertIn('Watching build [#3](http://localhost:8080/#/builders/23/builds/3) of `builder1` until it finishes...', self.sent)
        self.assertIn('Watching build [#6](http://localhost:8080/#/builders/23/builds/6) of `builder1` until it finishes...', self.sent)

    @defer.inlineCallbacks
    def sendBuildFinishedMessage(self, buildid, results=0):
        if False:
            for i in range(10):
                print('nop')
        self.master.db.builds.finishBuild(buildid=buildid, results=SUCCESS)
        build = (yield self.master.db.builds.getBuild(buildid))
        self.master.mq.callConsumer(('builds', str(buildid), 'complete'), {'buildid': buildid, 'number': build['number'], 'builderid': build['builderid'], 'buildrequestid': build['buildrequestid'], 'workerid': build['workerid'], 'masterid': build['masterid'], 'started_at': datetime2epoch(build['started_at']), 'complete': True, 'complete_at': datetime2epoch(build['complete_at']), 'state_string': '', 'results': results})

    @defer.inlineCallbacks
    def test_command_stop(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.do_test_command('stop', exp_UsageError=True)

    @defer.inlineCallbacks
    def test_command_stop_bogus_builder(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.do_test_command('stop', args="build BOGUS 'i have a reason'", exp_UsageError=True)

    @defer.inlineCallbacks
    def test_command_stop_builder0_no_builds(self):
        if False:
            return 10
        yield self.do_test_command('stop', args=f"build {self.BUILDER_NAMES[0]} 'i have a reason'")
        self.assertEqual(len(self.sent), 1)
        self.assertIn('no build is', self.sent[0])

    @defer.inlineCallbacks
    def test_command_stop_builder0_1_builds(self):
        if False:
            while True:
                i = 10
        self.setupSomeBuilds()
        yield self.do_test_command('stop', args=f"build {self.BUILDER_NAMES[0]} 'i have a reason'")
        self.assertEqual(len(self.sent), 2)
        self.assertRegex(self.sent[0], 'Build \\[#[36]\\].* of `builder1` interrupted')
        self.assertRegex(self.sent[1], 'Build \\[#[63]\\].* of `builder1` interrupted')

    @defer.inlineCallbacks
    def test_command_force_no_args(self):
        if False:
            i = 10
            return i + 15
        yield self.do_test_command('force', exp_UsageError=True)

    @defer.inlineCallbacks
    def test_command_force_wrong_first_arg(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.do_test_command('force', args='notbuild', exp_UsageError=True)

    @defer.inlineCallbacks
    def test_command_force_build_no_args(self):
        if False:
            print('Hello World!')
        yield self.do_test_command('force', args='build', exp_UsageError=True)

    @defer.inlineCallbacks
    def test_command_force(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.do_test_command('force', args=f'build --branch BRANCH1 --revision REV1 --props=PROP1=VALUE1 {self.BUILDER_NAMES[0]} REASON')

    @defer.inlineCallbacks
    def test_handleMessage_short_command(self):
        if False:
            for i in range(10):
                print('nop')
        self.contact.command_TESTY = mock.Mock()
        yield self.contact.handleMessage('testy')
        self.contact.command_TESTY.assert_called_with('')

    @defer.inlineCallbacks
    def test_handleMessage_long_command(self):
        if False:
            i = 10
            return i + 15
        self.contact.command_TESTY = mock.Mock()
        yield self.contact.handleMessage('testy   westy boo')
        self.contact.command_TESTY.assert_called_with('westy boo')

    @defer.inlineCallbacks
    def test_handleMessage_excited(self):
        if False:
            return 10
        self.patch_send()
        yield self.contact.handleMessage('hi!')
        self.assertEqual(len(self.sent), 1)

    @defer.inlineCallbacks
    def test_handleMessage_exception(self):
        if False:
            while True:
                i = 10
        self.patch_send()

        def command_TESTY(msg):
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError('FAIL')
        self.contact.command_TESTY = command_TESTY
        yield self.contact.handleMessage('testy boom')
        self.assertEqual(self.sent, ['Something bad happened (see logs)'])
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)

    @defer.inlineCallbacks
    def test_handleMessage_UsageError(self):
        if False:
            print('Hello World!')
        self.patch_send()

        def command_TESTY(msg):
            if False:
                print('Hello World!')
            raise words.UsageError('oh noes')
        self.contact.command_TESTY = command_TESTY
        yield self.contact.handleMessage('testy boom')
        self.assertEqual(self.sent, ['oh noes'])

    @defer.inlineCallbacks
    def test_unclosed_quote(self):
        if False:
            return 10
        yield self.do_test_command('list', args="args'", exp_UsageError=True)
        yield self.do_test_command('status', args="args'", exp_UsageError=True)
        yield self.do_test_command('notify', args="args'", exp_UsageError=True)
        yield self.do_test_command('watch', args="args'", exp_UsageError=True)
        yield self.do_test_command('force', args="args'", exp_UsageError=True)
        yield self.do_test_command('stop', args="args'", exp_UsageError=True)
        yield self.do_test_command('last', args="args'", exp_UsageError=True)
        yield self.do_test_command('help', args="args'", exp_UsageError=True)

    @defer.inlineCallbacks
    def test_buildStarted(self):
        if False:
            for i in range(10):
                print('nop')
        self.setupSomeBuilds()
        self.patch_send()
        build = (yield self.master.db.builds.getBuild(13))
        self.bot.tags = None
        self.contact.channel.notify_for = lambda _: True
        self.contact.useRevisions = False
        self.contact.channel.buildStarted(build)
        self.assertEqual(self.sent.pop(), 'Build [#3](http://localhost:8080/#/builders/23/builds/3) of `builder1` started.')

    def test_getCommandMethod_authz_default(self):
        if False:
            i = 10
            return i + 15
        self.bot.authz = words.StatusBot.expand_authz(None)
        meth = self.contact.getCommandMethod('shutdown')
        self.assertEqual(meth, self.contact.access_denied)
    authz1 = {'force': ['me'], 'shutdown': ['notme', 'someone'], ('dance', 'notify'): True, '': False}

    def test_getCommandMethod_explicit_allow(self):
        if False:
            i = 10
            return i + 15
        self.bot.authz = words.StatusBot.expand_authz(self.authz1)
        meth = self.contact.getCommandMethod('force')
        self.assertNotEqual(meth, self.contact.access_denied)

    def test_getCommandMethod_explicit_disallow(self):
        if False:
            while True:
                i = 10
        self.bot.authz = words.StatusBot.expand_authz(self.authz1)
        meth = self.contact.getCommandMethod('shutdown')
        self.assertEqual(meth, self.contact.access_denied)

    def test_getCommandMethod_explicit_multi(self):
        if False:
            return 10
        self.bot.authz = words.StatusBot.expand_authz(self.authz1)
        self.assertIn('DANCE', self.bot.authz)
        meth = self.contact.getCommandMethod('dance')
        self.assertNotEqual(meth, self.contact.access_denied)

    def test_getCommandMethod_explicit_default(self):
        if False:
            print('Hello World!')
        self.bot.authz = words.StatusBot.expand_authz(self.authz1)
        meth = self.contact.getCommandMethod('help')
        self.assertEqual(meth, self.contact.access_denied)
    authz2 = {'shutdown': False, '': False, '*': True}

    def test_getCommandMethod_exclamation(self):
        if False:
            print('Hello World!')
        self.bot.authz = words.StatusBot.expand_authz(self.authz2)
        meth = self.contact.getCommandMethod('help')
        self.assertNotEqual(meth, self.contact.access_denied)

    def test_getCommandMethod_exclamation_override(self):
        if False:
            i = 10
            return i + 15
        self.bot.authz = words.StatusBot.expand_authz(self.authz2)
        meth = self.contact.getCommandMethod('shutdown')
        self.assertEqual(meth, self.contact.access_denied)

    def test_access_denied(self):
        if False:
            for i in range(10):
                print('nop')
        self.patch_send()
        self.contact.access_denied()
        self.assertIn('not pass', self.sent[0])

    @defer.inlineCallbacks
    def test_bot_loadState(self):
        if False:
            print('Hello World!')
        boid = (yield self.bot._get_object_id())
        self.master.db.insert_test_data([fakedb.ObjectState(objectid=boid, name='notify_events', value_json='[["#channel1", ["warnings"]]]')])
        yield self.bot.loadState()
        self.assertEqual(self.bot.channels['#channel1'].notify_events, {'warnings'})