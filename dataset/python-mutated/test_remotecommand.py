from unittest import mock
from twisted.trial import unittest
from buildbot.process import remotecommand
from buildbot.test.fake import logfile
from buildbot.test.util import interfaces
from buildbot.test.util.warnings import assertNotProducesWarnings
from buildbot.warnings import DeprecatedApiWarning

class TestRemoteShellCommand(unittest.TestCase):

    def test_obfuscated_arguments(self):
        if False:
            while True:
                i = 10
        command = ['echo', ('obfuscated', 'real', 'fake'), 'test', ('obfuscated', 'real2', 'fake2'), ('not obfuscated', 'a', 'b'), 'obfuscated', ('obfuscated', 'test'), ('obfuscated', '1', '2', '3')]
        cmd = remotecommand.RemoteShellCommand('build', command)
        self.assertEqual(cmd.command, command)
        self.assertEqual(cmd.fake_command, ['echo', 'fake', 'test', 'fake2', ('not obfuscated', 'a', 'b'), 'obfuscated', ('obfuscated', 'test'), ('obfuscated', '1', '2', '3')])

    def test_not_obfuscated_arguments(self):
        if False:
            return 10
        command = 'echo test'
        cmd = remotecommand.RemoteShellCommand('build', command)
        self.assertEqual(cmd.command, command)
        self.assertEqual(cmd.fake_command, command)

class Tests(interfaces.InterfaceTests, unittest.TestCase):

    def makeRemoteCommand(self, stdioLogName='stdio'):
        if False:
            print('Hello World!')
        return remotecommand.RemoteCommand('ping', {'arg': 'val'}, stdioLogName=stdioLogName)

    def test_signature_RemoteCommand_constructor(self):
        if False:
            return 10

        @self.assertArgSpecMatches(remotecommand.RemoteCommand.__init__)
        def __init__(self, remote_command, args, ignore_updates=False, collectStdout=False, collectStderr=False, decodeRC=None, stdioLogName='stdio'):
            if False:
                while True:
                    i = 10
            pass

    def test_signature_RemoteShellCommand_constructor(self):
        if False:
            i = 10
            return i + 15

        @self.assertArgSpecMatches(remotecommand.RemoteShellCommand.__init__)
        def __init__(self, workdir, command, env=None, want_stdout=1, want_stderr=1, timeout=20 * 60, maxTime=None, sigtermTime=None, logfiles=None, usePTY=None, logEnviron=True, collectStdout=False, collectStderr=False, interruptSignal=None, initialStdin=None, decodeRC=None, stdioLogName='stdio'):
            if False:
                return 10
            pass

    def test_signature_run(self):
        if False:
            print('Hello World!')
        cmd = self.makeRemoteCommand()

        @self.assertArgSpecMatches(cmd.run)
        def run(self, step, conn, builder_name):
            if False:
                for i in range(10):
                    print('nop')
            pass

    def test_signature_useLog(self):
        if False:
            i = 10
            return i + 15
        cmd = self.makeRemoteCommand()

        @self.assertArgSpecMatches(cmd.useLog)
        def useLog(self, log_, closeWhenFinished=False, logfileName=None):
            if False:
                for i in range(10):
                    print('nop')
            pass

    def test_signature_useLogDelayed(self):
        if False:
            i = 10
            return i + 15
        cmd = self.makeRemoteCommand()

        @self.assertArgSpecMatches(cmd.useLogDelayed)
        def useLogDelayed(self, logfileName, activateCallBack, closeWhenFinished=False):
            if False:
                print('Hello World!')
            pass

    def test_signature_interrupt(self):
        if False:
            print('Hello World!')
        cmd = self.makeRemoteCommand()

        @self.assertArgSpecMatches(cmd.interrupt)
        def useLogDelayed(self, why):
            if False:
                for i in range(10):
                    print('nop')
            pass

    def test_signature_didFail(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = self.makeRemoteCommand()

        @self.assertArgSpecMatches(cmd.didFail)
        def useLogDelayed(self):
            if False:
                i = 10
                return i + 15
            pass

    def test_signature_logs(self):
        if False:
            return 10
        cmd = self.makeRemoteCommand()
        self.assertIsInstance(cmd.logs, dict)

    def test_signature_active(self):
        if False:
            print('Hello World!')
        cmd = self.makeRemoteCommand()
        self.assertIsInstance(cmd.active, bool)

    def test_RemoteShellCommand_constructor(self):
        if False:
            while True:
                i = 10
        remotecommand.RemoteShellCommand('wkdir', 'some-command')

    def test_notStdioLog(self):
        if False:
            for i in range(10):
                print('nop')
        logname = 'notstdio'
        cmd = self.makeRemoteCommand(stdioLogName=logname)
        log = logfile.FakeLogFile(logname)
        cmd.useLog(log)
        cmd.addStdout('some stdout')
        self.assertEqual(log.stdout, 'some stdout')
        cmd.addStderr('some stderr')
        self.assertEqual(log.stderr, 'some stderr')
        cmd.addHeader('some header')
        self.assertEqual(log.header, 'some header')

    def test_RemoteShellCommand_usePTY_on_worker_2_16(self):
        if False:
            i = 10
            return i + 15
        cmd = remotecommand.RemoteShellCommand('workdir', 'shell')

        def workerVersion(command, oldversion=None):
            if False:
                i = 10
                return i + 15
            return '2.16'

        def workerVersionIsOlderThan(command, minversion):
            if False:
                i = 10
                return i + 15
            return ['2', '16'] < minversion.split('.')
        step = mock.Mock()
        step.workerVersionIsOlderThan = workerVersionIsOlderThan
        step.workerVersion = workerVersion
        conn = mock.Mock()
        conn.remoteStartCommand = mock.Mock(return_value=None)
        cmd.run(step, conn, 'builder')
        self.assertEqual(cmd.args['usePTY'], 'slave-config')

class TestWorkerTransition(unittest.TestCase):

    def test_RemoteShellCommand_usePTY(self):
        if False:
            i = 10
            return i + 15
        with assertNotProducesWarnings(DeprecatedApiWarning):
            cmd = remotecommand.RemoteShellCommand('workdir', 'command')
        self.assertTrue(cmd.args['usePTY'] is None)
        with assertNotProducesWarnings(DeprecatedApiWarning):
            cmd = remotecommand.RemoteShellCommand('workdir', 'command', usePTY=True)
        self.assertTrue(cmd.args['usePTY'])
        with assertNotProducesWarnings(DeprecatedApiWarning):
            cmd = remotecommand.RemoteShellCommand('workdir', 'command', usePTY=False)
        self.assertFalse(cmd.args['usePTY'])