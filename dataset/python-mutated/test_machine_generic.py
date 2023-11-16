import os
from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.machine.generic import LocalWakeAction
from buildbot.machine.generic import LocalWOLAction
from buildbot.machine.generic import RemoteSshSuspendAction
from buildbot.machine.generic import RemoteSshWakeAction
from buildbot.machine.generic import RemoteSshWOLAction
from buildbot.test.fake.private_tempdir import MockPrivateTemporaryDirectory
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.runprocess import ExpectMasterShell
from buildbot.test.runprocess import MasterRunProcessMixin
from buildbot.test.util import config

class FakeManager:

    def __init__(self, reactor, basedir=None):
        if False:
            while True:
                i = 10
        self.master = mock.Mock()
        self.master.basedir = basedir
        self.master.reactor = reactor

    def renderSecrets(self, args):
        if False:
            return 10
        return defer.succeed(args)

class TestActions(MasterRunProcessMixin, config.ConfigErrorsMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        self.setup_master_run_process()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @defer.inlineCallbacks
    def test_local_wake_action(self):
        if False:
            return 10
        self.expect_commands(ExpectMasterShell(['cmd', 'arg1', 'arg2']).exit(1), ExpectMasterShell(['cmd', 'arg1', 'arg2']).exit(0))
        manager = FakeManager(self.reactor)
        action = LocalWakeAction(['cmd', 'arg1', 'arg2'])
        self.assertFalse((yield action.perform(manager)))
        self.assertTrue((yield action.perform(manager)))
        self.assert_all_commands_ran()

    def test_local_wake_action_command_not_list(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesConfigError('command parameter must be a list'):
            LocalWakeAction('not-list')

    @defer.inlineCallbacks
    def test_local_wol_action(self):
        if False:
            return 10
        self.expect_commands(ExpectMasterShell(['wol', '00:11:22:33:44:55']).exit(1), ExpectMasterShell(['wakeonlan', '00:11:22:33:44:55']).exit(0))
        manager = FakeManager(self.reactor)
        action = LocalWOLAction('00:11:22:33:44:55', wolBin='wol')
        self.assertFalse((yield action.perform(manager)))
        action = LocalWOLAction('00:11:22:33:44:55')
        self.assertTrue((yield action.perform(manager)))
        self.assert_all_commands_ran()

    @mock.patch('buildbot.util.private_tempdir.PrivateTemporaryDirectory', new_callable=MockPrivateTemporaryDirectory)
    @mock.patch('buildbot.util.misc.writeLocalFile')
    @defer.inlineCallbacks
    def test_remote_ssh_wake_action_no_keys(self, write_local_file_mock, temp_dir_mock):
        if False:
            i = 10
            return i + 15
        self.expect_commands(ExpectMasterShell(['ssh', '-o', 'BatchMode=yes', 'remote_host', 'remotebin', 'arg1']).exit(1), ExpectMasterShell(['ssh', '-o', 'BatchMode=yes', 'remote_host', 'remotebin', 'arg1']).exit(0))
        manager = FakeManager(self.reactor)
        action = RemoteSshWakeAction('remote_host', ['remotebin', 'arg1'])
        self.assertFalse((yield action.perform(manager)))
        self.assertTrue((yield action.perform(manager)))
        self.assert_all_commands_ran()
        self.assertEqual(temp_dir_mock.dirs, [])
        write_local_file_mock.assert_not_called()

    @mock.patch('buildbot.util.private_tempdir.PrivateTemporaryDirectory', new_callable=MockPrivateTemporaryDirectory)
    @mock.patch('buildbot.util.misc.writeLocalFile')
    @defer.inlineCallbacks
    def test_remote_ssh_wake_action_with_keys(self, write_local_file_mock, temp_dir_mock):
        if False:
            return 10
        temp_dir_path = os.path.join('path-to-master', 'ssh-@@@')
        ssh_key_path = os.path.join(temp_dir_path, 'ssh-key')
        ssh_known_hosts_path = os.path.join(temp_dir_path, 'ssh-known-hosts')
        self.expect_commands(ExpectMasterShell(['ssh', '-o', 'BatchMode=yes', '-i', ssh_key_path, '-o', f'UserKnownHostsFile={ssh_known_hosts_path}', 'remote_host', 'remotebin', 'arg1']).exit(0))
        manager = FakeManager(self.reactor, 'path-to-master')
        action = RemoteSshWakeAction('remote_host', ['remotebin', 'arg1'], sshKey='ssh_key', sshHostKey='ssh_host_key')
        self.assertTrue((yield action.perform(manager)))
        self.assert_all_commands_ran()
        self.assertEqual(temp_dir_mock.dirs, [(temp_dir_path, 448)])
        self.assertSequenceEqual(write_local_file_mock.call_args_list, [mock.call(ssh_key_path, 'ssh_key', mode=256), mock.call(ssh_known_hosts_path, '* ssh_host_key')])

    def test_remote_ssh_wake_action_sshBin_not_str(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesConfigError('sshBin parameter must be a string'):
            RemoteSshWakeAction('host', ['cmd'], sshBin=123)

    def test_remote_ssh_wake_action_host_not_str(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesConfigError('host parameter must be a string'):
            RemoteSshWakeAction(123, ['cmd'])

    def test_remote_ssh_wake_action_command_not_list(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesConfigError('remoteCommand parameter must be a list'):
            RemoteSshWakeAction('host', 'cmd')

    @mock.patch('buildbot.util.private_tempdir.PrivateTemporaryDirectory', new_callable=MockPrivateTemporaryDirectory)
    @mock.patch('buildbot.util.misc.writeLocalFile')
    @defer.inlineCallbacks
    def test_remote_ssh_wol_action_no_keys(self, write_local_file_mock, temp_dir_mock):
        if False:
            while True:
                i = 10
        self.expect_commands(ExpectMasterShell(['ssh', '-o', 'BatchMode=yes', 'remote_host', 'wakeonlan', '00:11:22:33:44:55']).exit(0), ExpectMasterShell(['ssh', '-o', 'BatchMode=yes', 'remote_host', 'wolbin', '00:11:22:33:44:55']).exit(0))
        manager = FakeManager(self.reactor)
        action = RemoteSshWOLAction('remote_host', '00:11:22:33:44:55')
        self.assertTrue((yield action.perform(manager)))
        action = RemoteSshWOLAction('remote_host', '00:11:22:33:44:55', wolBin='wolbin')
        self.assertTrue((yield action.perform(manager)))
        self.assert_all_commands_ran()
        self.assertEqual(temp_dir_mock.dirs, [])
        write_local_file_mock.assert_not_called()

    @mock.patch('buildbot.util.private_tempdir.PrivateTemporaryDirectory', new_callable=MockPrivateTemporaryDirectory)
    @mock.patch('buildbot.util.misc.writeLocalFile')
    @defer.inlineCallbacks
    def test_remote_ssh_suspend_action_no_keys(self, write_local_file_mock, temp_dir_mock):
        if False:
            return 10
        self.expect_commands(ExpectMasterShell(['ssh', '-o', 'BatchMode=yes', 'remote_host', 'systemctl', 'suspend']).exit(0), ExpectMasterShell(['ssh', '-o', 'BatchMode=yes', 'remote_host', 'dosuspend', 'arg1']).exit(0))
        manager = FakeManager(self.reactor)
        action = RemoteSshSuspendAction('remote_host')
        self.assertTrue((yield action.perform(manager)))
        action = RemoteSshSuspendAction('remote_host', remoteCommand=['dosuspend', 'arg1'])
        self.assertTrue((yield action.perform(manager)))
        self.assert_all_commands_ran()
        self.assertEqual(temp_dir_mock.dirs, [])
        write_local_file_mock.assert_not_called()