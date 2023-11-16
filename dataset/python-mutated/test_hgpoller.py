import os
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.changes import hgpoller
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.runprocess import ExpectMasterShell
from buildbot.test.runprocess import MasterRunProcessMixin
from buildbot.test.util import changesource
ENVIRON_2116_KEY = 'TEST_THAT_ENVIRONMENT_GETS_PASSED_TO_SUBPROCESSES'
LINESEP_BYTES = os.linesep.encode('ascii')
PATHSEP_BYTES = os.pathsep.encode('ascii')

class TestHgPollerBase(MasterRunProcessMixin, changesource.ChangeSourceMixin, TestReactorMixin, unittest.TestCase):
    usetimestamps = True
    branches = None
    bookmarks = None

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.setup_master_run_process()
        yield self.setUpChangeSource()
        os.environ[ENVIRON_2116_KEY] = 'TRUE'
        yield self.setUpChangeSource()
        self.remote_repo = 'ssh://example.com/foo/baz'
        self.remote_hgweb = 'http://example.com/foo/baz/rev/{}'
        self.repo_ready = True

        def _isRepositoryReady():
            if False:
                while True:
                    i = 10
            return self.repo_ready
        self.poller = hgpoller.HgPoller(self.remote_repo, usetimestamps=self.usetimestamps, workdir='/some/dir', branches=self.branches, bookmarks=self.bookmarks, revlink=lambda branch, revision: self.remote_hgweb.format(revision))
        yield self.poller.setServiceParent(self.master)
        self.poller._isRepositoryReady = _isRepositoryReady
        yield self.master.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            while True:
                i = 10
        yield self.master.stopService()
        yield self.tearDownChangeSource()

    @defer.inlineCallbacks
    def check_current_rev(self, wished, branch='default'):
        if False:
            for i in range(10):
                print('nop')
        rev = (yield self.poller._getCurrentRev(branch))
        self.assertEqual(rev, str(wished))

class TestHgPollerBranches(TestHgPollerBase):
    branches = ['one', 'two']

    @defer.inlineCallbacks
    def test_poll_initial(self):
        if False:
            for i in range(10):
                print('nop')
        self.expect_commands(ExpectMasterShell(['hg', 'pull', '-b', 'one', '-b', 'two', 'ssh://example.com/foo/baz']).workdir('/some/dir'), ExpectMasterShell(['hg', 'heads', 'one', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'73591'), ExpectMasterShell(['hg', 'heads', 'two', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'22341'))
        yield self.poller.poll()
        self.assertEqual(len(self.master.data.updates.changesAdded), 0)
        yield self.check_current_rev(73591, 'one')
        yield self.check_current_rev(22341, 'two')

    @defer.inlineCallbacks
    def test_poll_regular(self):
        if False:
            print('Hello World!')
        self.expect_commands(ExpectMasterShell(['hg', 'pull', '-b', 'one', '-b', 'two', 'ssh://example.com/foo/baz']).workdir('/some/dir'), ExpectMasterShell(['hg', 'heads', 'one', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'6' + LINESEP_BYTES), ExpectMasterShell(['hg', 'log', '-r', '4::6', '--template={rev}:{node}\\n']).workdir('/some/dir').stdout(LINESEP_BYTES.join([b'4:1aaa5', b'6:784bd'])), ExpectMasterShell(['hg', 'log', '-r', '784bd', '--template={date|hgdate}' + os.linesep + '{author}' + os.linesep + "{files % '{file}" + os.pathsep + "'}" + os.linesep + '{desc|strip}']).workdir('/some/dir').stdout(LINESEP_BYTES.join([b'1273258009.0 -7200', b'Joe Test <joetest@example.org>', b'file1 file2', b'Comment', b''])), ExpectMasterShell(['hg', 'heads', 'two', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'3' + LINESEP_BYTES))
        yield self.poller._setCurrentRev(3, 'two')
        yield self.poller._setCurrentRev(4, 'one')
        yield self.poller.poll()
        yield self.check_current_rev(6, 'one')
        self.assertEqual(len(self.master.data.updates.changesAdded), 1)
        change = self.master.data.updates.changesAdded[0]
        self.assertEqual(change['revision'], '784bd')
        self.assertEqual(change['revlink'], 'http://example.com/foo/baz/rev/784bd')
        self.assertEqual(change['comments'], 'Comment')

class TestHgPollerBookmarks(TestHgPollerBase):
    bookmarks = ['one', 'two']

    @defer.inlineCallbacks
    def test_poll_initial(self):
        if False:
            while True:
                i = 10
        self.expect_commands(ExpectMasterShell(['hg', 'pull', '-B', 'one', '-B', 'two', 'ssh://example.com/foo/baz']).workdir('/some/dir'), ExpectMasterShell(['hg', 'heads', 'one', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'73591'), ExpectMasterShell(['hg', 'heads', 'two', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'22341'))
        yield self.poller.poll()
        self.assertEqual(len(self.master.data.updates.changesAdded), 0)
        yield self.check_current_rev(73591, 'one')
        yield self.check_current_rev(22341, 'two')

    @defer.inlineCallbacks
    def test_poll_regular(self):
        if False:
            return 10
        self.expect_commands(ExpectMasterShell(['hg', 'pull', '-B', 'one', '-B', 'two', 'ssh://example.com/foo/baz']).workdir('/some/dir'), ExpectMasterShell(['hg', 'heads', 'one', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'6' + LINESEP_BYTES), ExpectMasterShell(['hg', 'log', '-r', '4::6', '--template={rev}:{node}\\n']).workdir('/some/dir').stdout(LINESEP_BYTES.join([b'4:1aaa5', b'6:784bd'])), ExpectMasterShell(['hg', 'log', '-r', '784bd', '--template={date|hgdate}' + os.linesep + '{author}' + os.linesep + "{files % '{file}" + os.pathsep + "'}" + os.linesep + '{desc|strip}']).workdir('/some/dir').stdout(LINESEP_BYTES.join([b'1273258009.0 -7200', b'Joe Test <joetest@example.org>', b'file1 file2', b'Comment', b''])), ExpectMasterShell(['hg', 'heads', 'two', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'3' + LINESEP_BYTES))
        yield self.poller._setCurrentRev(3, 'two')
        yield self.poller._setCurrentRev(4, 'one')
        yield self.poller.poll()
        yield self.check_current_rev(6, 'one')
        self.assertEqual(len(self.master.data.updates.changesAdded), 1)
        change = self.master.data.updates.changesAdded[0]
        self.assertEqual(change['revision'], '784bd')
        self.assertEqual(change['comments'], 'Comment')

class TestHgPoller(TestHgPollerBase):

    def tearDown(self):
        if False:
            print('Hello World!')
        del os.environ[ENVIRON_2116_KEY]
        return self.tearDownChangeSource()

    def gpoFullcommandPattern(self, commandName, *expected_args):
        if False:
            for i in range(10):
                print('nop')
        'Match if the command is commandName and arg list start as expected.\n\n        This allows to test a bit more if expected GPO are issued, be it\n        by obscure failures due to the result not being given.\n        '

        def matchesSubcommand(bin, given_args, **kwargs):
            if False:
                i = 10
                return i + 15
            return bin == commandName and tuple(given_args[:len(expected_args)]) == expected_args
        return matchesSubcommand

    def test_describe(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertSubstring('HgPoller', self.poller.describe())

    def test_name(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.remote_repo, self.poller.name)
        other = hgpoller.HgPoller(self.remote_repo, name='MyName', workdir='/some/dir')
        self.assertEqual('MyName', other.name)
        other = hgpoller.HgPoller(self.remote_repo, branches=['b1', 'b2'], workdir='/some/dir')
        self.assertEqual(self.remote_repo + '_b1_b2', other.name)

    def test_hgbin_default(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.poller.hgbin, 'hg')

    @defer.inlineCallbacks
    def test_poll_initial(self):
        if False:
            for i in range(10):
                print('nop')
        self.repo_ready = False
        expected_env = {ENVIRON_2116_KEY: 'TRUE'}
        self.add_run_process_expect_env(expected_env)
        self.expect_commands(ExpectMasterShell(['hg', 'init', '/some/dir']), ExpectMasterShell(['hg', 'pull', '-b', 'default', 'ssh://example.com/foo/baz']).workdir('/some/dir'), ExpectMasterShell(['hg', 'heads', 'default', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'73591'))
        yield self.poller.poll()
        self.assertEqual(len(self.master.data.updates.changesAdded), 0)
        yield self.check_current_rev(73591)

    @defer.inlineCallbacks
    def test_poll_several_heads(self):
        if False:
            while True:
                i = 10
        self.expect_commands(ExpectMasterShell(['hg', 'pull', '-b', 'default', 'ssh://example.com/foo/baz']).workdir('/some/dir'), ExpectMasterShell(['hg', 'heads', 'default', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'5' + LINESEP_BYTES + b'6' + LINESEP_BYTES))
        yield self.poller._setCurrentRev(3)
        yield self.poller.poll()
        yield self.check_current_rev(3)

    @defer.inlineCallbacks
    def test_poll_regular(self):
        if False:
            return 10
        self.expect_commands(ExpectMasterShell(['hg', 'pull', '-b', 'default', 'ssh://example.com/foo/baz']).workdir('/some/dir'), ExpectMasterShell(['hg', 'heads', 'default', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'5' + LINESEP_BYTES), ExpectMasterShell(['hg', 'log', '-r', '4::5', '--template={rev}:{node}\\n']).workdir('/some/dir').stdout(LINESEP_BYTES.join([b'4:1aaa5', b'5:784bd'])), ExpectMasterShell(['hg', 'log', '-r', '784bd', '--template={date|hgdate}' + os.linesep + '{author}' + os.linesep + "{files % '{file}" + os.pathsep + "'}" + os.linesep + '{desc|strip}']).workdir('/some/dir').stdout(LINESEP_BYTES.join([b'1273258009.0 -7200', b'Joe Test <joetest@example.org>', b'file1 file2', b'Comment for rev 5', b''])))
        yield self.poller._setCurrentRev(4)
        yield self.poller.poll()
        yield self.check_current_rev(5)
        self.assertEqual(len(self.master.data.updates.changesAdded), 1)
        change = self.master.data.updates.changesAdded[0]
        self.assertEqual(change['revision'], '784bd')
        self.assertEqual(change['comments'], 'Comment for rev 5')

    @defer.inlineCallbacks
    def test_poll_force_push(self):
        if False:
            while True:
                i = 10
        self.expect_commands(ExpectMasterShell(['hg', 'pull', '-b', 'default', 'ssh://example.com/foo/baz']).workdir('/some/dir'), ExpectMasterShell(['hg', 'heads', 'default', '--template={rev}' + os.linesep]).workdir('/some/dir').stdout(b'5' + LINESEP_BYTES), ExpectMasterShell(['hg', 'log', '-r', '4::5', '--template={rev}:{node}\\n']).workdir('/some/dir').stdout(b''), ExpectMasterShell(['hg', 'log', '-r', '5', '--template={rev}:{node}\\n']).workdir('/some/dir').stdout(LINESEP_BYTES.join([b'5:784bd'])), ExpectMasterShell(['hg', 'log', '-r', '784bd', '--template={date|hgdate}' + os.linesep + '{author}' + os.linesep + "{files % '{file}" + os.pathsep + "'}" + os.linesep + '{desc|strip}']).workdir('/some/dir').stdout(LINESEP_BYTES.join([b'1273258009.0 -7200', b'Joe Test <joetest@example.org>', b'file1 file2', b'Comment for rev 5', b''])))
        yield self.poller._setCurrentRev(4)
        yield self.poller.poll()
        yield self.check_current_rev(5)
        self.assertEqual(len(self.master.data.updates.changesAdded), 1)
        change = self.master.data.updates.changesAdded[0]
        self.assertEqual(change['revision'], '784bd')
        self.assertEqual(change['comments'], 'Comment for rev 5')

class HgPollerNoTimestamp(TestHgPoller):
    """ Test HgPoller() without parsing revision commit timestamp """
    usetimestamps = False