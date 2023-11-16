from __future__ import absolute_import
from __future__ import print_function
import os
from twisted.internet import defer
from twisted.trial import unittest
from buildbot_worker.commands import shell
from buildbot_worker.test.fake.runprocess import Expect
from buildbot_worker.test.util.command import CommandTestMixin

class TestWorkerShellCommand(CommandTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setUpCommand()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tearDownCommand()

    @defer.inlineCallbacks
    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        workdir = os.path.join(self.basedir, 'workdir')
        self.make_command(shell.WorkerShellCommand, {'command': ['echo', 'hello'], 'workdir': workdir})
        self.patch_runprocess(Expect(['echo', 'hello'], self.basedir_workdir).update('header', 'headers').update('stdout', 'hello\n').update('rc', 0).exit(0))
        yield self.run_command()
        self.assertUpdates([('header', 'headers'), ('stdout', 'hello\n'), ('rc', 0)], self.protocol_command.show())