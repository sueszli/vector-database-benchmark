from __future__ import absolute_import
from __future__ import print_function
from twisted.internet import defer
from twisted.trial import unittest
from buildbot_worker.commands.base import Command
from buildbot_worker.test.util.command import CommandTestMixin

class DummyCommand(Command):

    def setup(self, args):
        if False:
            while True:
                i = 10
        self.setup_done = True
        self.interrupted = False
        self.started = False

    def start(self):
        if False:
            print('Hello World!')
        self.started = True
        data = []
        for (key, value) in self.args.items():
            data.append((key, value))
        self.sendStatus(data)
        self.cmd_deferred = defer.Deferred()
        return self.cmd_deferred

    def interrupt(self):
        if False:
            while True:
                i = 10
        self.interrupted = True
        self.finishCommand()

    def finishCommand(self):
        if False:
            return 10
        d = self.cmd_deferred
        self.cmd_deferred = None
        d.callback(None)

    def failCommand(self):
        if False:
            i = 10
            return i + 15
        d = self.cmd_deferred
        self.cmd_deferred = None
        d.errback(RuntimeError('forced failure'))

class DummyArgsCommand(DummyCommand):
    requiredArgs = ['workdir']

class TestDummyCommand(CommandTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setUpCommand()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.tearDownCommand()

    def assertState(self, setup_done, running, started, interrupted, msg=None):
        if False:
            while True:
                i = 10
        self.assertEqual({'setup_done': self.cmd.setup_done, 'running': self.cmd.running, 'started': self.cmd.started, 'interrupted': self.cmd.interrupted}, {'setup_done': setup_done, 'running': running, 'started': started, 'interrupted': interrupted}, msg)

    def test_run(self):
        if False:
            i = 10
            return i + 15
        cmd = self.make_command(DummyCommand, {'stdout': 'yay'})
        self.assertState(True, False, False, False, 'setup called by constructor')
        d = self.run_command()
        self.assertState(True, True, True, False, 'started and running both set')
        cmd.finishCommand()

        def check(_):
            if False:
                for i in range(10):
                    print('nop')
            self.assertState(True, False, True, False, 'started and not running when done')
        d.addCallback(check)

        def checkresult(_):
            if False:
                while True:
                    i = 10
            self.assertUpdates([('stdout', 'yay')], 'updates processed')
        d.addCallback(checkresult)
        return d

    def test_run_failure(self):
        if False:
            return 10
        cmd = self.make_command(DummyCommand, {})
        self.assertState(True, False, False, False, 'setup called by constructor')
        d = self.run_command()
        self.assertState(True, True, True, False, 'started and running both set')
        cmd.failCommand()

        def check(_):
            if False:
                print('Hello World!')
            self.assertState(True, False, True, False, 'started and not running when done')
        d.addErrback(check)

        def checkresult(_):
            if False:
                print('Hello World!')
            self.assertUpdates([], 'updates processed')
        d.addCallback(checkresult)
        return d

    def test_run_interrupt(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = self.make_command(DummyCommand, {})
        self.assertState(True, False, False, False, 'setup called by constructor')
        d = self.run_command()
        self.assertState(True, True, True, False, 'started and running both set')
        cmd.doInterrupt()
        self.assertTrue(cmd.interrupted)

        def check(_):
            if False:
                print('Hello World!')
            self.assertState(True, False, True, True, 'finishes with interrupted set')
        d.addCallback(check)
        return d

    def test_required_args(self):
        if False:
            print('Hello World!')
        self.make_command(DummyArgsCommand, {'workdir': '.'})
        try:
            self.make_command(DummyArgsCommand, {'stdout': 'boo'})
        except ValueError:
            return
        self.fail('Command was supposed to raise ValueError when missing args')