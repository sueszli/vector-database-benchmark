from __future__ import absolute_import
import unittest2
from st2common.models.system.paramiko_command_action import ParamikoRemoteCommandAction
__all__ = ['ParamikoRemoteCommandActionTestCase']

class ParamikoRemoteCommandActionTestCase(unittest2.TestCase):

    def test_get_command_string_no_env_vars(self):
        if False:
            while True:
                i = 10
        cmd_action = ParamikoRemoteCommandActionTestCase._get_test_command_action('echo boo bah baz')
        ex = 'cd /tmp && echo boo bah baz'
        self.assertEqual(cmd_action.get_full_command_string(), ex)
        cmd_action.sudo = True
        ex = "sudo -E -- bash -c 'cd /tmp && echo boo bah baz'"
        self.assertEqual(cmd_action.get_full_command_string(), ex)
        cmd_action = ParamikoRemoteCommandActionTestCase._get_test_command_action('"/t/space stuff.sh"')
        ex = 'cd /tmp && "/t/space stuff.sh"'
        self.assertEqual(cmd_action.get_full_command_string(), ex)
        cmd_action = ParamikoRemoteCommandActionTestCase._get_test_command_action('echo boo bah baz')
        cmd_action.sudo = True
        cmd_action.sudo_password = 'sudo pass'
        ex = "set +o history ; echo -e 'sudo pass\n' | sudo -S -E -- bash -c 'cd /tmp && echo boo bah baz'"
        self.assertEqual(cmd_action.get_full_command_string(), ex)

    def test_get_command_string_with_env_vars(self):
        if False:
            return 10
        cmd_action = ParamikoRemoteCommandActionTestCase._get_test_command_action('echo boo bah baz')
        cmd_action.env_vars = {'FOO': 'BAR', 'BAR': 'BEET CAFE'}
        ex = "export BAR='BEET CAFE' " + 'FOO=BAR' + ' && cd /tmp && echo boo bah baz'
        self.assertEqual(cmd_action.get_full_command_string(), ex)
        cmd_action.sudo = True
        ex = 'sudo -E -- bash -c ' + "'export FOO=BAR " + 'BAR=\'"\'"\'BEET CAFE\'"\'"\'' + " && cd /tmp && echo boo bah baz'"
        ex = 'sudo -E -- bash -c ' + '\'export BAR=\'"\'"\'BEET CAFE\'"\'"\' ' + 'FOO=BAR' + " && cd /tmp && echo boo bah baz'"
        self.assertEqual(cmd_action.get_full_command_string(), ex)
        cmd_action.sudo = True
        cmd_action.sudo_password = 'sudo pass'
        ex = "set +o history ; echo -e 'sudo pass\n' | sudo -S -E -- bash -c " + '\'export BAR=\'"\'"\'BEET CAFE\'"\'"\' ' + 'FOO=BAR HISTFILE=/dev/null HISTSIZE=0' + " && cd /tmp && echo boo bah baz'"
        self.assertEqual(cmd_action.get_full_command_string(), ex)

    def test_get_command_string_no_user(self):
        if False:
            for i in range(10):
                print('nop')
        cmd_action = ParamikoRemoteCommandActionTestCase._get_test_command_action('echo boo bah baz')
        cmd_action.user = None
        ex = 'cd /tmp && echo boo bah baz'
        self.assertEqual(cmd_action.get_full_command_string(), ex)
        cmd = 'bash "/tmp/stuff space.sh"'
        cmd_action = ParamikoRemoteCommandActionTestCase._get_test_command_action(cmd)
        cmd_action.user = None
        ex = 'cd /tmp && bash "/tmp/stuff space.sh"'
        self.assertEqual(cmd_action.get_full_command_string(), ex)

    def test_get_command_string_no_user_env_vars(self):
        if False:
            return 10
        cmd_action = ParamikoRemoteCommandActionTestCase._get_test_command_action('echo boo bah baz')
        cmd_action.user = None
        cmd_action.env_vars = {'FOO': 'BAR'}
        ex = 'export FOO=BAR && cd /tmp && echo boo bah baz'
        self.assertEqual(cmd_action.get_full_command_string(), ex)

    @staticmethod
    def _get_test_command_action(command):
        if False:
            for i in range(10):
                print('nop')
        cmd_action = ParamikoRemoteCommandAction('fixtures.remote_command', '55ce39d532ed3543aecbe71d', command=command, env_vars={}, on_behalf_user='svetlana', user='estee', password=None, private_key='---PRIVATE-KEY---', hosts='127.0.0.1', parallel=True, sudo=False, timeout=None, cwd='/tmp')
        return cmd_action