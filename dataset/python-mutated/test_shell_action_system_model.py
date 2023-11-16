from __future__ import absolute_import
import pwd
import os
import copy
from collections import OrderedDict
import unittest2
from st2common.models.system.action import ShellCommandAction
from st2common.models.system.action import ShellScriptAction
from st2common.models.db.action import ActionDB
from st2common.util import param as param_utils
from st2common.logging.formatters import MASKED_ATTRIBUTE_VALUE
from local_runner.local_shell_script_runner import LocalShellScriptRunner
from tests.fixtures.local_runner.fixture import FIXTURE_DIR as LOCAL_RUNNER_FIXTURE_DIR
LOGGED_USER_USERNAME = pwd.getpwuid(os.getuid())[0]
__all__ = ['ShellCommandActionTestCase', 'ShellScriptActionTestCase']

class ShellCommandActionTestCase(unittest2.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._base_kwargs = {'name': 'test action', 'action_exec_id': '1', 'command': 'ls -la', 'env_vars': {}, 'timeout': None}

    def test_user_argument(self):
        if False:
            return 10
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = LOGGED_USER_USERNAME
        action = ShellCommandAction(**kwargs)
        command = action.get_full_command_string()
        self.assertEqual(command, 'ls -la')
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = 'mauser'
        action = ShellCommandAction(**kwargs)
        command = action.get_full_command_string()
        self.assertEqual(command, "sudo -E -H -u mauser -- bash -c 'ls -la'")
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['sudo_password'] = 'sudopass'
        kwargs['user'] = 'mauser'
        action = ShellCommandAction(**kwargs)
        command = action.get_full_command_string()
        expected_command = "sudo -S -E -H -u mauser -- bash -c 'ls -la'"
        self.assertEqual(command, expected_command)
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = True
        kwargs['user'] = 'mauser'
        action = ShellCommandAction(**kwargs)
        command = action.get_full_command_string()
        self.assertEqual(command, "sudo -E -- bash -c 'ls -la'")
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = True
        kwargs['user'] = 'mauser'
        kwargs['sudo_password'] = 'sudopass'
        action = ShellCommandAction(**kwargs)
        command = action.get_full_command_string()
        expected_command = "sudo -S -E -- bash -c 'ls -la'"
        self.assertEqual(command, expected_command)

class ShellScriptActionTestCase(unittest2.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._base_kwargs = {'name': 'test action', 'action_exec_id': '1', 'script_local_path_abs': '/tmp/foo.sh', 'named_args': {}, 'positional_args': [], 'env_vars': {}, 'timeout': None}

    def _get_fixture(self, name):
        if False:
            while True:
                i = 10
        path = os.path.join(LOCAL_RUNNER_FIXTURE_DIR, name)
        with open(path, 'r') as fp:
            content = fp.read().strip()
        return content

    def test_user_argument(self):
        if False:
            return 10
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = LOGGED_USER_USERNAME
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        self.assertEqual(command, '/tmp/foo.sh')
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = 'mauser'
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        self.assertEqual(command, 'sudo -E -H -u mauser -- bash -c /tmp/foo.sh')
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = 'mauser'
        kwargs['sudo_password'] = 'sudopass'
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        expected_command = 'sudo -S -E -H -u mauser -- bash -c /tmp/foo.sh'
        self.assertEqual(command, expected_command)
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = 'mauser'
        kwargs['sudo_password'] = '$udo p\'as"sss'
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        expected_command = 'sudo -S -E -H -u mauser -- bash -c /tmp/foo.sh'
        self.assertEqual(command, expected_command)
        command = action.get_sanitized_full_command_string()
        expected_command = "echo -e '%s\n' | sudo -S -E -H -u mauser -- bash -c /tmp/foo.sh" % MASKED_ATTRIBUTE_VALUE
        self.assertEqual(command, expected_command)
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = True
        kwargs['user'] = 'mauser'
        kwargs['sudo_password'] = 'sudopass'
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        expected_command = 'sudo -S -E -- bash -c /tmp/foo.sh'
        self.assertEqual(command, expected_command)

    def test_command_construction_with_parameters(self):
        if False:
            while True:
                i = 10
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = LOGGED_USER_USERNAME
        kwargs['named_args'] = OrderedDict([('key1', 'value1'), ('key2', 'value2')])
        kwargs['positional_args'] = []
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        self.assertEqual(command, '/tmp/foo.sh key1=value1 key2=value2')
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = True
        kwargs['sudo_password'] = 'sudopass'
        kwargs['user'] = LOGGED_USER_USERNAME
        kwargs['named_args'] = OrderedDict([('key1', 'value1'), ('key2', 'value2')])
        kwargs['positional_args'] = []
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        expected = "sudo -S -E -- bash -c '/tmp/foo.sh key1=value1 key2=value2'"
        self.assertEqual(command, expected)
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = 'mauser'
        kwargs['named_args'] = OrderedDict([('key1', 'value1'), ('key2', 'value2')])
        kwargs['positional_args'] = []
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        expected = "sudo -E -H -u mauser -- bash -c '/tmp/foo.sh key1=value1 key2=value2'"
        self.assertEqual(command, expected)
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['sudo_password'] = 'sudopass'
        kwargs['user'] = 'mauser'
        kwargs['named_args'] = OrderedDict([('key1', 'value1'), ('key2', 'value2')])
        kwargs['positional_args'] = []
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        expected = "sudo -S -E -H -u mauser -- bash -c '/tmp/foo.sh key1=value1 key2=value2'"
        self.assertEqual(command, expected)
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = LOGGED_USER_USERNAME
        kwargs['named_args'] = {}
        kwargs['positional_args'] = ['ein', 'zwei', 'drei', 'mamma mia', 'foo\nbar']
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        self.assertEqual(command, "/tmp/foo.sh ein zwei drei 'mamma mia' 'foo\nbar'")
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = 'mauser'
        kwargs['named_args'] = {}
        kwargs['positional_args'] = ['ein', 'zwei', 'drei', 'mamma mia']
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        ex = 'sudo -E -H -u mauser -- bash -c \'/tmp/foo.sh ein zwei drei \'"\'"\'mamma mia\'"\'"\'\''
        self.assertEqual(command, ex)
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = LOGGED_USER_USERNAME
        kwargs['named_args'] = OrderedDict([('key1', 'value1'), ('key2', 'value2'), ('key3', 'value 3')])
        kwargs['positional_args'] = ['ein', 'zwei', 'drei']
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        exp = "/tmp/foo.sh key1=value1 key2=value2 key3='value 3' ein zwei drei"
        self.assertEqual(command, exp)
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = 'mauser'
        kwargs['named_args'] = OrderedDict([('key1', 'value1'), ('key2', 'value2'), ('key3', 'value 3')])
        kwargs['positional_args'] = ['ein', 'zwei', 'drei']
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        expected = 'sudo -E -H -u mauser -- bash -c \'/tmp/foo.sh key1=value1 key2=value2 key3=\'"\'"\'value 3\'"\'"\' ein zwei drei\''
        self.assertEqual(command, expected)

    def test_named_parameter_escaping(self):
        if False:
            print('Hello World!')
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = LOGGED_USER_USERNAME
        kwargs['named_args'] = OrderedDict([('key1', 'value foo bar'), ('key2', 'value "bar" foo'), ('key3', 'date ; whoami'), ('key4', '"date ; whoami"')])
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        expected = self._get_fixture('escaping_test_command_1.txt')
        self.assertEqual(command, expected)
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = True
        kwargs['user'] = LOGGED_USER_USERNAME
        kwargs['named_args'] = OrderedDict([('key1', 'value foo bar'), ('key2', 'value "bar" foo'), ('key3', 'date ; whoami'), ('key4', '"date ; whoami"')])
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        expected = self._get_fixture('escaping_test_command_2.txt')
        self.assertEqual(command, expected)

    def test_various_ascii_parameters(self):
        if False:
            return 10
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = LOGGED_USER_USERNAME
        kwargs['named_args'] = {'foo1': 'bar1', 'foo2': 'bar2'}
        kwargs['positional_args'] = []
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        self.assertEqual(command, '/tmp/foo.sh foo1=bar1 foo2=bar2')

    def test_unicode_parameter_specifing(self):
        if False:
            return 10
        kwargs = copy.deepcopy(self._base_kwargs)
        kwargs['sudo'] = False
        kwargs['user'] = LOGGED_USER_USERNAME
        kwargs['named_args'] = {'ｆｏｏ': 'ｂａｒ'}
        kwargs['positional_args'] = []
        action = ShellScriptAction(**kwargs)
        command = action.get_full_command_string()
        self.assertEqual(command, "/tmp/foo.sh 'ｆｏｏ'='ｂａｒ'")

    def test_command_construction_correct_default_parameter_values_are_used(self):
        if False:
            print('Hello World!')
        runner_parameters = {}
        action_db_parameters = {'project': {'type': 'string', 'default': 'st2', 'position': 0}, 'version': {'type': 'string', 'position': 1, 'required': True}, 'fork': {'type': 'string', 'position': 2, 'default': 'StackStorm'}, 'branch': {'type': 'string', 'position': 3, 'default': 'master'}, 'update_changelog': {'type': 'boolean', 'position': 4, 'default': False}, 'local_repo': {'type': 'string', 'position': 5}}
        context = {}
        action_db = ActionDB(pack='dummy', name='action')
        runner = LocalShellScriptRunner('id')
        runner.runner_parameters = {}
        runner.action = action_db
        live_action_db_parameters = {'project': 'st2flow', 'version': '3.0.0', 'fork': 'StackStorm', 'local_repo': '/tmp/repo'}
        (runner_params, action_params) = param_utils.render_final_params(runner_parameters, action_db_parameters, live_action_db_parameters, context)
        self.assertDictEqual(action_params, {'project': 'st2flow', 'version': '3.0.0', 'fork': 'StackStorm', 'branch': 'master', 'update_changelog': False, 'local_repo': '/tmp/repo'})
        action_db.parameters = action_db_parameters
        (positional_args, named_args) = runner._get_script_args(action_params)
        named_args = runner._transform_named_args(named_args)
        shell_script_action = ShellScriptAction(name='dummy', action_exec_id='dummy', script_local_path_abs='/tmp/local.sh', named_args=named_args, positional_args=positional_args)
        command_string = shell_script_action.get_full_command_string()
        expected = '/tmp/local.sh st2flow 3.0.0 StackStorm master 0 /tmp/repo'
        self.assertEqual(command_string, expected)
        live_action_db_parameters = {'project': 'st2web', 'version': '3.1.0', 'fork': 'StackStorm1', 'update_changelog': True, 'local_repo': '/tmp/repob'}
        (runner_params, action_params) = param_utils.render_final_params(runner_parameters, action_db_parameters, live_action_db_parameters, context)
        self.assertDictEqual(action_params, {'project': 'st2web', 'version': '3.1.0', 'fork': 'StackStorm1', 'branch': 'master', 'update_changelog': True, 'local_repo': '/tmp/repob'})
        action_db.parameters = action_db_parameters
        (positional_args, named_args) = runner._get_script_args(action_params)
        named_args = runner._transform_named_args(named_args)
        shell_script_action = ShellScriptAction(name='dummy', action_exec_id='dummy', script_local_path_abs='/tmp/local.sh', named_args=named_args, positional_args=positional_args)
        command_string = shell_script_action.get_full_command_string()
        expected = '/tmp/local.sh st2web 3.1.0 StackStorm1 master 1 /tmp/repob'
        self.assertEqual(command_string, expected)
        live_action_db_parameters = {'project': 'st2rbac', 'version': '3.2.0', 'fork': 'StackStorm2', 'update_changelog': None, 'local_repo': '/tmp/repoc'}
        (runner_params, action_params) = param_utils.render_final_params(runner_parameters, action_db_parameters, live_action_db_parameters, context)
        self.assertDictEqual(action_params, {'project': 'st2rbac', 'version': '3.2.0', 'fork': 'StackStorm2', 'branch': 'master', 'update_changelog': False, 'local_repo': '/tmp/repoc'})
        action_db.parameters = action_db_parameters
        (positional_args, named_args) = runner._get_script_args(action_params)
        named_args = runner._transform_named_args(named_args)
        shell_script_action = ShellScriptAction(name='dummy', action_exec_id='dummy', script_local_path_abs='/tmp/local.sh', named_args=named_args, positional_args=positional_args)
        command_string = shell_script_action.get_full_command_string()
        expected = '/tmp/local.sh st2rbac 3.2.0 StackStorm2 master 0 /tmp/repoc'
        self.assertEqual(command_string, expected)