from __future__ import absolute_import
import os
import unittest2
import mock
from st2common.runners.paramiko_ssh_runner import BaseParallelSSHRunner
from st2common.runners.paramiko_ssh_runner import RUNNER_HOSTS
from st2common.runners.paramiko_ssh_runner import RUNNER_USERNAME
from st2common.runners.paramiko_ssh_runner import RUNNER_PASSWORD
from st2common.runners.paramiko_ssh_runner import RUNNER_PRIVATE_KEY
from st2common.runners.paramiko_ssh_runner import RUNNER_PASSPHRASE
from st2common.runners.paramiko_ssh_runner import RUNNER_SSH_PORT
import st2tests.config as tests_config
from st2tests.fixturesloader import get_resources_base_path
tests_config.parse_args()

class Runner(BaseParallelSSHRunner):

    def run(self):
        if False:
            i = 10
            return i + 15
        pass

class ParamikoSSHRunnerTestCase(unittest2.TestCase):

    @mock.patch('st2common.runners.paramiko_ssh_runner.ParallelSSHClient')
    def test_pre_run(self, mock_client):
        if False:
            for i in range(10):
                print('nop')
        private_key_path = os.path.join(get_resources_base_path(), 'ssh', 'dummy_rsa')
        with open(private_key_path, 'r') as fp:
            private_key = fp.read()
        runner = Runner('id')
        runner.context = {}
        runner_parameters = {RUNNER_HOSTS: 'localhost', RUNNER_USERNAME: 'someuser1', RUNNER_PASSWORD: 'somepassword'}
        runner.runner_parameters = runner_parameters
        runner.pre_run()
        expected_kwargs = {'hosts': ['localhost'], 'user': 'someuser1', 'password': 'somepassword', 'port': None, 'concurrency': 1, 'bastion_host': None, 'raise_on_any_error': False, 'connect': True, 'handle_stdout_line_func': mock.ANY, 'handle_stderr_line_func': mock.ANY}
        mock_client.assert_called_with(**expected_kwargs)
        runner = Runner('id')
        runner.context = {}
        runner_parameters = {RUNNER_HOSTS: 'localhost', RUNNER_USERNAME: 'someuser2', RUNNER_PRIVATE_KEY: private_key, RUNNER_SSH_PORT: 22}
        runner.runner_parameters = runner_parameters
        runner.pre_run()
        expected_kwargs = {'hosts': ['localhost'], 'user': 'someuser2', 'pkey_material': private_key, 'port': 22, 'concurrency': 1, 'bastion_host': None, 'raise_on_any_error': False, 'connect': True, 'handle_stdout_line_func': mock.ANY, 'handle_stderr_line_func': mock.ANY}
        mock_client.assert_called_with(**expected_kwargs)
        runner = Runner('id')
        runner.context = {}
        runner_parameters = {RUNNER_HOSTS: 'localhost21', RUNNER_USERNAME: 'someuser21', RUNNER_PRIVATE_KEY: private_key, RUNNER_PASSPHRASE: 'passphrase21', RUNNER_SSH_PORT: 22}
        runner.runner_parameters = runner_parameters
        runner.pre_run()
        expected_kwargs = {'hosts': ['localhost21'], 'user': 'someuser21', 'pkey_material': private_key, 'passphrase': 'passphrase21', 'port': 22, 'concurrency': 1, 'bastion_host': None, 'raise_on_any_error': False, 'connect': True, 'handle_stdout_line_func': mock.ANY, 'handle_stderr_line_func': mock.ANY}
        mock_client.assert_called_with(**expected_kwargs)
        runner = Runner('id')
        runner.context = {}
        runner_parameters = {RUNNER_HOSTS: 'localhost', RUNNER_USERNAME: 'someuser3', RUNNER_PRIVATE_KEY: private_key_path, RUNNER_SSH_PORT: 22}
        runner.runner_parameters = runner_parameters
        runner.pre_run()
        expected_kwargs = {'hosts': ['localhost'], 'user': 'someuser3', 'pkey_file': private_key_path, 'port': 22, 'concurrency': 1, 'bastion_host': None, 'raise_on_any_error': False, 'connect': True, 'handle_stdout_line_func': mock.ANY, 'handle_stderr_line_func': mock.ANY}
        mock_client.assert_called_with(**expected_kwargs)
        runner = Runner('id')
        runner.context = {}
        runner_parameters = {RUNNER_HOSTS: 'localhost31', RUNNER_USERNAME: 'someuser31', RUNNER_PRIVATE_KEY: private_key_path, RUNNER_PASSPHRASE: 'passphrase31', RUNNER_SSH_PORT: 22}
        runner.runner_parameters = runner_parameters
        runner.pre_run()
        expected_kwargs = {'hosts': ['localhost31'], 'user': 'someuser31', 'pkey_file': private_key_path, 'passphrase': 'passphrase31', 'port': 22, 'concurrency': 1, 'bastion_host': None, 'raise_on_any_error': False, 'connect': True, 'handle_stdout_line_func': mock.ANY, 'handle_stderr_line_func': mock.ANY}
        mock_client.assert_called_with(**expected_kwargs)
        runner = Runner('id')
        runner.context = {}
        runner_parameters = {RUNNER_HOSTS: 'localhost4', RUNNER_SSH_PORT: 22}
        runner.runner_parameters = runner_parameters
        runner.pre_run()
        expected_kwargs = {'hosts': ['localhost4'], 'user': None, 'pkey_file': None, 'port': 22, 'concurrency': 1, 'bastion_host': None, 'raise_on_any_error': False, 'connect': True, 'handle_stdout_line_func': mock.ANY, 'handle_stderr_line_func': mock.ANY}
        mock_client.assert_called_with(**expected_kwargs)

    @mock.patch('st2common.runners.paramiko_ssh_runner.ParallelSSHClient')
    def test_post_run(self, mock_client):
        if False:
            i = 10
            return i + 15
        runner = Runner('id')
        runner.context = {}
        runner_parameters = {RUNNER_HOSTS: 'localhost', RUNNER_USERNAME: 'someuser1', RUNNER_PASSWORD: 'somepassword'}
        runner.runner_parameters = runner_parameters
        runner.pre_run()
        expected_kwargs = {'hosts': ['localhost'], 'user': 'someuser1', 'password': 'somepassword', 'port': None, 'concurrency': 1, 'bastion_host': None, 'raise_on_any_error': False, 'connect': True, 'handle_stdout_line_func': mock.ANY, 'handle_stderr_line_func': mock.ANY}
        mock_client.assert_called_with(**expected_kwargs)
        self.assertEqual(runner._parallel_ssh_client.close.call_count, 0)
        runner.post_run(result=None, status=None)
        self.assertEqual(runner._parallel_ssh_client.close.call_count, 1)