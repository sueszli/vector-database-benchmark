from __future__ import absolute_import
import bson
from mock import patch, Mock, MagicMock
import unittest2
import st2tests.config as tests_config
tests_config.parse_args()
from st2common.util import jsonify
from st2common.models.db.action import ActionDB
from st2common.runners.parallel_ssh import ParallelSSHClient
from st2common.exceptions.ssh import NoHostsConnectedToException
from st2common.models.system.paramiko_script_action import ParamikoRemoteScriptAction
from st2common.constants.action import LIVEACTION_STATUS_FAILED
from st2common.util import param as param_utils
from remote_runner.remote_script_runner import ParamikoRemoteScriptRunner
from st2tests.fixtures.generic.fixture import PACK_NAME as FIXTURES_PACK
from st2tests.fixturesloader import FixturesLoader
__all__ = ['ParamikoScriptRunnerTestCase']
TEST_MODELS = {'actions': ['a1.yaml']}
MODELS = FixturesLoader().load_models(fixtures_pack=FIXTURES_PACK, fixtures_dict=TEST_MODELS)
ACTION_1 = MODELS['actions']['a1.yaml']

class ParamikoScriptRunnerTestCase(unittest2.TestCase):

    @patch('st2common.runners.parallel_ssh.ParallelSSHClient', Mock)
    @patch.object(jsonify, 'json_loads', MagicMock(return_value={}))
    @patch.object(ParallelSSHClient, 'run', MagicMock(return_value={}))
    @patch.object(ParallelSSHClient, 'connect', MagicMock(return_value={}))
    def test_cwd_used_correctly(self):
        if False:
            i = 10
            return i + 15
        remote_action = ParamikoRemoteScriptAction('foo-script', bson.ObjectId(), script_local_path_abs='/home/stanley/shiz_storm.py', script_local_libs_path_abs=None, named_args={}, positional_args=['blank space'], env_vars={}, on_behalf_user='svetlana', user='stanley', private_key='---SOME RSA KEY---', remote_dir='/tmp', hosts=['127.0.0.1'], cwd='/test/cwd/')
        paramiko_runner = ParamikoRemoteScriptRunner('runner_1')
        paramiko_runner._parallel_ssh_client = ParallelSSHClient(['127.0.0.1'], 'stanley')
        paramiko_runner._run_script_on_remote_host(remote_action)
        exp_cmd = "cd /test/cwd/ && /tmp/shiz_storm.py 'blank space'"
        ParallelSSHClient.run.assert_called_with(exp_cmd, timeout=None)

    def test_username_invalid_private_key(self):
        if False:
            for i in range(10):
                print('nop')
        paramiko_runner = ParamikoRemoteScriptRunner('runner_1')
        paramiko_runner.runner_parameters = {'username': 'test_user', 'hosts': '127.0.0.1', 'private_key': 'invalid private key'}
        paramiko_runner.context = {}
        self.assertRaises(NoHostsConnectedToException, paramiko_runner.pre_run)

    @patch('st2common.runners.parallel_ssh.ParallelSSHClient', Mock)
    @patch.object(ParallelSSHClient, 'run', MagicMock(return_value={}))
    @patch.object(ParallelSSHClient, 'connect', MagicMock(return_value={}))
    def test_top_level_error_is_correctly_reported(self):
        if False:
            for i in range(10):
                print('nop')
        paramiko_runner = ParamikoRemoteScriptRunner('runner_1')
        paramiko_runner.runner_parameters = {'username': 'test_user', 'hosts': '127.0.0.1'}
        paramiko_runner.action = ACTION_1
        paramiko_runner.liveaction_id = 'foo'
        paramiko_runner.entry_point = 'foo'
        paramiko_runner.context = {}
        paramiko_runner._cwd = '/tmp'
        paramiko_runner._copy_artifacts = Mock(side_effect=Exception('fail!'))
        (status, result, _) = paramiko_runner.run(action_parameters={})
        self.assertEqual(status, LIVEACTION_STATUS_FAILED)
        self.assertEqual(result['failed'], True)
        self.assertEqual(result['succeeded'], False)
        self.assertIn('Failed copying content to remote boxes', result['error'])

    def test_command_construction_correct_default_parameter_values_are_used(self):
        if False:
            for i in range(10):
                print('nop')
        runner_parameters = {}
        action_db_parameters = {'project': {'type': 'string', 'default': 'st2', 'position': 0}, 'version': {'type': 'string', 'position': 1, 'required': True}, 'fork': {'type': 'string', 'position': 2, 'default': 'StackStorm'}, 'branch': {'type': 'string', 'position': 3, 'default': 'master'}, 'update_changelog': {'type': 'boolean', 'position': 4, 'default': False}, 'local_repo': {'type': 'string', 'position': 5}}
        context = {}
        action_db = ActionDB(pack='dummy', name='action')
        runner = ParamikoRemoteScriptRunner('id')
        runner.runner_parameters = {}
        runner.action = action_db
        live_action_db_parameters = {'project': 'st2flow', 'version': '3.0.0', 'fork': 'StackStorm', 'local_repo': '/tmp/repo'}
        (runner_params, action_params) = param_utils.render_final_params(runner_parameters, action_db_parameters, live_action_db_parameters, context)
        self.assertDictEqual(action_params, {'project': 'st2flow', 'version': '3.0.0', 'fork': 'StackStorm', 'branch': 'master', 'update_changelog': False, 'local_repo': '/tmp/repo'})
        action_db.parameters = action_db_parameters
        (positional_args, named_args) = runner._get_script_args(action_params)
        named_args = runner._transform_named_args(named_args)
        remote_action = ParamikoRemoteScriptAction('foo-script', 'id', script_local_path_abs='/tmp/script.sh', script_local_libs_path_abs=None, named_args=named_args, positional_args=positional_args, env_vars={}, on_behalf_user='svetlana', user='stanley', remote_dir='/tmp', hosts=['127.0.0.1'], cwd='/test/cwd/')
        command_string = remote_action.get_full_command_string()
        expected = 'cd /test/cwd/ && /tmp/script.sh st2flow 3.0.0 StackStorm master 0 /tmp/repo'
        self.assertEqual(command_string, expected)
        live_action_db_parameters = {'project': 'st2web', 'version': '3.1.0', 'fork': 'StackStorm1', 'update_changelog': True, 'local_repo': '/tmp/repob'}
        (runner_params, action_params) = param_utils.render_final_params(runner_parameters, action_db_parameters, live_action_db_parameters, context)
        self.assertDictEqual(action_params, {'project': 'st2web', 'version': '3.1.0', 'fork': 'StackStorm1', 'branch': 'master', 'update_changelog': True, 'local_repo': '/tmp/repob'})
        action_db.parameters = action_db_parameters
        (positional_args, named_args) = runner._get_script_args(action_params)
        named_args = runner._transform_named_args(named_args)
        remote_action = ParamikoRemoteScriptAction('foo-script', 'id', script_local_path_abs='/tmp/script.sh', script_local_libs_path_abs=None, named_args=named_args, positional_args=positional_args, env_vars={}, on_behalf_user='svetlana', user='stanley', remote_dir='/tmp', hosts=['127.0.0.1'], cwd='/test/cwd/')
        command_string = remote_action.get_full_command_string()
        expected = 'cd /test/cwd/ && /tmp/script.sh st2web 3.1.0 StackStorm1 master 1 /tmp/repob'
        self.assertEqual(command_string, expected)
        live_action_db_parameters = {'project': 'st2rbac', 'version': '3.2.0', 'fork': 'StackStorm2', 'update_changelog': None, 'local_repo': '/tmp/repoc'}
        (runner_params, action_params) = param_utils.render_final_params(runner_parameters, action_db_parameters, live_action_db_parameters, context)
        self.assertDictEqual(action_params, {'project': 'st2rbac', 'version': '3.2.0', 'fork': 'StackStorm2', 'branch': 'master', 'update_changelog': False, 'local_repo': '/tmp/repoc'})
        action_db.parameters = action_db_parameters
        (positional_args, named_args) = runner._get_script_args(action_params)
        named_args = runner._transform_named_args(named_args)
        remote_action = ParamikoRemoteScriptAction('foo-script', 'id', script_local_path_abs='/tmp/script.sh', script_local_libs_path_abs=None, named_args=named_args, positional_args=positional_args, env_vars={}, on_behalf_user='svetlana', user='stanley', remote_dir='/tmp', hosts=['127.0.0.1'], cwd='/test/cwd/')
        command_string = remote_action.get_full_command_string()
        expected = 'cd /test/cwd/ && /tmp/script.sh st2rbac 3.2.0 StackStorm2 master 0 /tmp/repoc'
        self.assertEqual(command_string, expected)