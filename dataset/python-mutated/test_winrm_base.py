from __future__ import absolute_import
import collections
import time
import mock
from base64 import b64encode
from winrm import Response
from winrm.exceptions import WinRMOperationTimeoutError
from st2common.runners.base import ActionRunner
from st2tests.base import RunnerTestCase
from winrm_runner.winrm_base import WinRmBaseRunner, WinRmRunnerTimoutError
from winrm_runner.winrm_base import PS_ESCAPE_SEQUENCES
from winrm_runner.winrm_base import WINRM_MAX_CMD_LENGTH
from winrm_runner.winrm_base import WINRM_TIMEOUT_EXIT_CODE
from winrm_runner import winrm_ps_command_runner

class WinRmBaseTestCase(RunnerTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(WinRmBaseTestCase, self).setUpClass()
        self._runner = winrm_ps_command_runner.get_runner()

    def _init_runner(self):
        if False:
            return 10
        runner_parameters = {'host': 'host@domain.tld', 'username': 'user@domain.tld', 'password': 'xyz987'}
        self._runner.runner_parameters = runner_parameters
        self._runner.pre_run()

    def test_win_rm_runner_timout_error(self):
        if False:
            i = 10
            return i + 15
        error = WinRmRunnerTimoutError('test_response')
        self.assertIsInstance(error, Exception)
        self.assertEqual(error.response, 'test_response')
        with self.assertRaises(WinRmRunnerTimoutError):
            raise WinRmRunnerTimoutError('test raising')

    def test_init(self):
        if False:
            print('Hello World!')
        runner = winrm_ps_command_runner.WinRmPsCommandRunner('abcdef')
        self.assertIsInstance(runner, WinRmBaseRunner)
        self.assertIsInstance(runner, ActionRunner)
        self.assertEqual(runner.runner_id, 'abcdef')

    @mock.patch('winrm_runner.winrm_base.ActionRunner.pre_run')
    def test_pre_run(self, mock_pre_run):
        if False:
            while True:
                i = 10
        runner_parameters = {'host': 'host@domain.tld', 'username': 'user@domain.tld', 'password': 'abc123', 'timeout': 99, 'port': 1234, 'scheme': 'http', 'transport': 'ntlm', 'verify_ssl_cert': False, 'cwd': 'C:\\Test', 'env': {'TEST_VAR': 'TEST_VALUE'}, 'kwarg_op': '/'}
        self._runner.runner_parameters = runner_parameters
        self._runner.pre_run()
        mock_pre_run.assert_called_with()
        self.assertEqual(self._runner._session, None)
        self.assertEqual(self._runner._host, 'host@domain.tld')
        self.assertEqual(self._runner._username, 'user@domain.tld')
        self.assertEqual(self._runner._password, 'abc123')
        self.assertEqual(self._runner._timeout, 99)
        self.assertEqual(self._runner._read_timeout, 100)
        self.assertEqual(self._runner._port, 1234)
        self.assertEqual(self._runner._scheme, 'http')
        self.assertEqual(self._runner._transport, 'ntlm')
        self.assertEqual(self._runner._winrm_url, 'http://host@domain.tld:1234/wsman')
        self.assertEqual(self._runner._verify_ssl, False)
        self.assertEqual(self._runner._server_cert_validation, 'ignore')
        self.assertEqual(self._runner._cwd, 'C:\\Test')
        self.assertEqual(self._runner._env, {'TEST_VAR': 'TEST_VALUE'})
        self.assertEqual(self._runner._kwarg_op, '/')

    @mock.patch('winrm_runner.winrm_base.ActionRunner.pre_run')
    def test_pre_run_defaults(self, mock_pre_run):
        if False:
            for i in range(10):
                print('nop')
        runner_parameters = {'host': 'host@domain.tld', 'username': 'user@domain.tld', 'password': 'abc123'}
        self._runner.runner_parameters = runner_parameters
        self._runner.pre_run()
        mock_pre_run.assert_called_with()
        self.assertEqual(self._runner._host, 'host@domain.tld')
        self.assertEqual(self._runner._username, 'user@domain.tld')
        self.assertEqual(self._runner._password, 'abc123')
        self.assertEqual(self._runner._timeout, 60)
        self.assertEqual(self._runner._read_timeout, 61)
        self.assertEqual(self._runner._port, 5986)
        self.assertEqual(self._runner._scheme, 'https')
        self.assertEqual(self._runner._transport, 'ntlm')
        self.assertEqual(self._runner._winrm_url, 'https://host@domain.tld:5986/wsman')
        self.assertEqual(self._runner._verify_ssl, True)
        self.assertEqual(self._runner._server_cert_validation, 'validate')
        self.assertEqual(self._runner._cwd, None)
        self.assertEqual(self._runner._env, {})
        self.assertEqual(self._runner._kwarg_op, '-')

    @mock.patch('winrm_runner.winrm_base.ActionRunner.pre_run')
    def test_pre_run_5985_force_http(self, mock_pre_run):
        if False:
            while True:
                i = 10
        runner_parameters = {'host': 'host@domain.tld', 'username': 'user@domain.tld', 'password': 'abc123', 'port': 5985, 'scheme': 'https'}
        self._runner.runner_parameters = runner_parameters
        self._runner.pre_run()
        mock_pre_run.assert_called_with()
        self.assertEqual(self._runner._host, 'host@domain.tld')
        self.assertEqual(self._runner._username, 'user@domain.tld')
        self.assertEqual(self._runner._password, 'abc123')
        self.assertEqual(self._runner._timeout, 60)
        self.assertEqual(self._runner._read_timeout, 61)
        self.assertEqual(self._runner._port, 5985)
        self.assertEqual(self._runner._scheme, 'http')
        self.assertEqual(self._runner._transport, 'ntlm')
        self.assertEqual(self._runner._winrm_url, 'http://host@domain.tld:5985/wsman')
        self.assertEqual(self._runner._verify_ssl, True)
        self.assertEqual(self._runner._server_cert_validation, 'validate')
        self.assertEqual(self._runner._cwd, None)
        self.assertEqual(self._runner._env, {})
        self.assertEqual(self._runner._kwarg_op, '-')

    @mock.patch('winrm_runner.winrm_base.ActionRunner.pre_run')
    def test_pre_run_none_env(self, mock_pre_run):
        if False:
            i = 10
            return i + 15
        runner_parameters = {'host': 'host@domain.tld', 'username': 'user@domain.tld', 'password': 'abc123', 'env': None}
        self._runner.runner_parameters = runner_parameters
        self._runner.pre_run()
        mock_pre_run.assert_called_with()
        self.assertEqual(self._runner._env, {})

    @mock.patch('winrm_runner.winrm_base.ActionRunner.pre_run')
    def test_pre_run_ssl_verify_true(self, mock_pre_run):
        if False:
            while True:
                i = 10
        runner_parameters = {'host': 'host@domain.tld', 'username': 'user@domain.tld', 'password': 'abc123', 'verify_ssl_cert': True}
        self._runner.runner_parameters = runner_parameters
        self._runner.pre_run()
        mock_pre_run.assert_called_with()
        self.assertEqual(self._runner._verify_ssl, True)
        self.assertEqual(self._runner._server_cert_validation, 'validate')

    @mock.patch('winrm_runner.winrm_base.ActionRunner.pre_run')
    def test_pre_run_ssl_verify_false(self, mock_pre_run):
        if False:
            print('Hello World!')
        runner_parameters = {'host': 'host@domain.tld', 'username': 'user@domain.tld', 'password': 'abc123', 'verify_ssl_cert': False}
        self._runner.runner_parameters = runner_parameters
        self._runner.pre_run()
        mock_pre_run.assert_called_with()
        self.assertEqual(self._runner._verify_ssl, False)
        self.assertEqual(self._runner._server_cert_validation, 'ignore')

    @mock.patch('winrm_runner.winrm_base.Session')
    def test_get_session(self, mock_session):
        if False:
            return 10
        self._runner._session = None
        self._runner._winrm_url = 'https://host@domain.tld:5986/wsman'
        self._runner._username = 'user@domain.tld'
        self._runner._password = 'abc123'
        self._runner._transport = 'ntlm'
        self._runner._server_cert_validation = 'validate'
        self._runner._timeout = 60
        self._runner._read_timeout = 61
        mock_session.return_value = 'session'
        result = self._runner._get_session()
        self.assertEqual(result, 'session')
        self.assertEqual(result, self._runner._session)
        mock_session.assert_called_with('https://host@domain.tld:5986/wsman', auth=('user@domain.tld', 'abc123'), transport='ntlm', server_cert_validation='validate', operation_timeout_sec=60, read_timeout_sec=61)
        old_session = self._runner._session
        result = self._runner._get_session()
        self.assertEqual(result, old_session)

    def test_winrm_get_command_output(self):
        if False:
            for i in range(10):
                print('nop')
        self._runner._timeout = 0
        mock_protocol = mock.MagicMock()
        mock_protocol._raw_get_command_output.side_effect = [(b'output1', b'error1', 123, False), (b'output2', b'error2', 456, False), (b'output3', b'error3', 789, True)]
        result = self._runner._winrm_get_command_output(mock_protocol, 567, 890)
        self.assertEqual(result, (b'output1output2output3', b'error1error2error3', 789))
        mock_protocol._raw_get_command_output.assert_has_calls = [mock.call(567, 890), mock.call(567, 890), mock.call(567, 890)]

    def test_winrm_get_command_output_timeout(self):
        if False:
            print('Hello World!')
        self._runner._timeout = 0.1
        mock_protocol = mock.MagicMock()

        def sleep_for_timeout(*args, **kwargs):
            if False:
                while True:
                    i = 10
            time.sleep(0.2)
            return (b'output1', b'error1', 123, False)
        mock_protocol._raw_get_command_output.side_effect = sleep_for_timeout
        with self.assertRaises(WinRmRunnerTimoutError) as cm:
            self._runner._winrm_get_command_output(mock_protocol, 567, 890)
        timeout_exception = cm.exception
        self.assertEqual(timeout_exception.response.std_out, b'output1')
        self.assertEqual(timeout_exception.response.std_err, b'error1')
        self.assertEqual(timeout_exception.response.status_code, WINRM_TIMEOUT_EXIT_CODE)
        mock_protocol._raw_get_command_output.assert_called_with(567, 890)

    def test_winrm_get_command_output_operation_timeout(self):
        if False:
            print('Hello World!')
        self._runner._timeout = 0.1
        mock_protocol = mock.MagicMock()

        def sleep_for_timeout_then_raise(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            time.sleep(0.2)
            raise WinRMOperationTimeoutError()
        mock_protocol._raw_get_command_output.side_effect = sleep_for_timeout_then_raise
        with self.assertRaises(WinRmRunnerTimoutError) as cm:
            self._runner._winrm_get_command_output(mock_protocol, 567, 890)
        timeout_exception = cm.exception
        self.assertEqual(timeout_exception.response.std_out, b'')
        self.assertEqual(timeout_exception.response.std_err, b'')
        self.assertEqual(timeout_exception.response.status_code, WINRM_TIMEOUT_EXIT_CODE)
        mock_protocol._raw_get_command_output.assert_called_with(567, 890)

    def test_winrm_run_cmd(self):
        if False:
            return 10
        mock_protocol = mock.MagicMock()
        mock_protocol.open_shell.return_value = 123
        mock_protocol.run_command.return_value = 456
        mock_protocol._raw_get_command_output.return_value = (b'output', b'error', 9, True)
        mock_session = mock.MagicMock(protocol=mock_protocol)
        self._init_runner()
        result = self._runner._winrm_run_cmd(mock_session, 'fake-command', args=['arg1', 'arg2'], env={'PATH': 'C:\\st2\\bin'}, cwd='C:\\st2')
        expected_response = Response((b'output', b'error', 9))
        expected_response.timeout = False
        self.assertEqual(result.__dict__, expected_response.__dict__)
        mock_protocol.open_shell.assert_called_with(env_vars={'PATH': 'C:\\st2\\bin'}, working_directory='C:\\st2')
        mock_protocol.run_command.assert_called_with(123, 'fake-command', ['arg1', 'arg2'])
        mock_protocol._raw_get_command_output.assert_called_with(123, 456)
        mock_protocol.cleanup_command.assert_called_with(123, 456)
        mock_protocol.close_shell.assert_called_with(123)

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._winrm_get_command_output')
    def test_winrm_run_cmd_timeout(self, mock_get_command_output):
        if False:
            print('Hello World!')
        mock_protocol = mock.MagicMock()
        mock_protocol.open_shell.return_value = 123
        mock_protocol.run_command.return_value = 456
        mock_session = mock.MagicMock(protocol=mock_protocol)
        mock_get_command_output.side_effect = WinRmRunnerTimoutError(Response(('', '', 5)))
        self._init_runner()
        result = self._runner._winrm_run_cmd(mock_session, 'fake-command', args=['arg1', 'arg2'], env={'PATH': 'C:\\st2\\bin'}, cwd='C:\\st2')
        expected_response = Response(('', '', 5))
        expected_response.timeout = True
        self.assertEqual(result.__dict__, expected_response.__dict__)
        mock_protocol.open_shell.assert_called_with(env_vars={'PATH': 'C:\\st2\\bin'}, working_directory='C:\\st2')
        mock_protocol.run_command.assert_called_with(123, 'fake-command', ['arg1', 'arg2'])
        mock_protocol.cleanup_command.assert_called_with(123, 456)
        mock_protocol.close_shell.assert_called_with(123)

    def test_winrm_encode(self):
        if False:
            print('Hello World!')
        result = self._runner._winrm_encode('hello world')
        self.assertEqual(result, 'aABlAGwAbABvACAAdwBvAHIAbABkAA==')

    def test_winrm_ps_cmd(self):
        if False:
            return 10
        result = self._runner._winrm_ps_cmd('abc123==')
        self.assertEqual(result, 'powershell -encodedcommand abc123==')

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._winrm_run_cmd')
    def test_winrm_run_ps(self, mock_run_cmd):
        if False:
            print('Hello World!')
        mock_run_cmd.return_value = Response(('output', '', 3))
        script = 'Get-ADUser stanley'
        result = self._runner._winrm_run_ps('session', script, env={'PATH': 'C:\\st2\\bin'}, cwd='C:\\st2')
        self.assertEqual(result.__dict__, Response(('output', '', 3)).__dict__)
        expected_ps = 'powershell -encodedcommand ' + b64encode('Get-ADUser stanley'.encode('utf_16_le')).decode('ascii')
        mock_run_cmd.assert_called_with('session', expected_ps, env={'PATH': 'C:\\st2\\bin'}, cwd='C:\\st2')

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._winrm_run_cmd')
    def test_winrm_run_ps_clean_stderr(self, mock_run_cmd):
        if False:
            return 10
        mock_run_cmd.return_value = Response(('output', 'error', 3))
        mock_session = mock.MagicMock()
        mock_session._clean_error_msg.return_value = 'e'
        script = 'Get-ADUser stanley'
        result = self._runner._winrm_run_ps(mock_session, script, env={'PATH': 'C:\\st2\\bin'}, cwd='C:\\st2')
        self.assertEqual(result.__dict__, Response(('output', 'e', 3)).__dict__)
        expected_ps = 'powershell -encodedcommand ' + b64encode('Get-ADUser stanley'.encode('utf_16_le')).decode('ascii')
        mock_run_cmd.assert_called_with(mock_session, expected_ps, env={'PATH': 'C:\\st2\\bin'}, cwd='C:\\st2')
        mock_session._clean_error_msg.assert_called_with('error')

    def test_translate_response_success(self):
        if False:
            i = 10
            return i + 15
        response = Response(('output1', 'error1', 0))
        response.timeout = False
        result = self._runner._translate_response(response)
        self.assertEqual(result, ('succeeded', {'failed': False, 'succeeded': True, 'return_code': 0, 'stdout': 'output1', 'stderr': 'error1'}, None))

    def test_translate_response_failure(self):
        if False:
            return 10
        response = Response(('output1', 'error1', 123))
        response.timeout = False
        result = self._runner._translate_response(response)
        self.assertEqual(result, ('failed', {'failed': True, 'succeeded': False, 'return_code': 123, 'stdout': 'output1', 'stderr': 'error1'}, None))

    def test_translate_response_timeout(self):
        if False:
            print('Hello World!')
        response = Response(('output1', 'error1', 123))
        response.timeout = True
        result = self._runner._translate_response(response)
        self.assertEqual(result, ('timeout', {'failed': True, 'succeeded': False, 'return_code': -1, 'stdout': 'output1', 'stderr': 'error1'}, None))

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._run_ps_or_raise')
    def test_make_tmp_dir(self, mock_run_ps_or_raise):
        if False:
            return 10
        mock_run_ps_or_raise.return_value = {'stdout': ' expected \n'}
        result = self._runner._make_tmp_dir('C:\\Windows\\Temp')
        self.assertEqual(result, 'expected')
        mock_run_ps_or_raise.assert_called_with('$parent = C:\\Windows\\Temp\n$name = [System.IO.Path]::GetRandomFileName()\n$path = Join-Path $parent $name\nNew-Item -ItemType Directory -Path $path | Out-Null\n$path', 'Unable to make temporary directory for powershell script')

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._run_ps_or_raise')
    def test_rm_dir(self, mock_run_ps_or_raise):
        if False:
            while True:
                i = 10
        self._runner._rm_dir('C:\\Windows\\Temp\\testtmpdir')
        mock_run_ps_or_raise.assert_called_with('Remove-Item -Force -Recurse -Path "C:\\Windows\\Temp\\testtmpdir"', 'Unable to remove temporary directory for powershell script')

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._upload_chunk')
    @mock.patch('winrm_runner.winrm_base.open')
    @mock.patch('os.path.exists')
    def test_upload_chunk_file(self, mock_os_path_exists, mock_open, mock_upload_chunk):
        if False:
            i = 10
            return i + 15
        mock_os_path_exists.return_value = True
        mock_src_file = mock.MagicMock()
        mock_src_file.read.return_value = 'test data'
        mock_open.return_value.__enter__.return_value = mock_src_file
        self._runner._upload('/opt/data/test.ps1', 'C:\\Windows\\Temp\\test.ps1')
        mock_os_path_exists.assert_called_with('/opt/data/test.ps1')
        mock_open.assert_called_with('/opt/data/test.ps1', 'r')
        mock_src_file.read.assert_called_with()
        mock_upload_chunk.assert_has_calls([mock.call('C:\\Windows\\Temp\\test.ps1', 'test data')])

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._upload_chunk')
    @mock.patch('os.path.exists')
    def test_upload_chunk_data(self, mock_os_path_exists, mock_upload_chunk):
        if False:
            i = 10
            return i + 15
        mock_os_path_exists.return_value = False
        self._runner._upload('test data', 'C:\\Windows\\Temp\\test.ps1')
        mock_os_path_exists.assert_called_with('test data')
        mock_upload_chunk.assert_has_calls([mock.call('C:\\Windows\\Temp\\test.ps1', 'test data')])

    @mock.patch('winrm_runner.winrm_base.WINRM_UPLOAD_CHUNK_SIZE_BYTES', 2)
    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._upload_chunk')
    @mock.patch('os.path.exists')
    def test_upload_chunk_multiple_chunks(self, mock_os_path_exists, mock_upload_chunk):
        if False:
            print('Hello World!')
        mock_os_path_exists.return_value = False
        self._runner._upload('test data', 'C:\\Windows\\Temp\\test.ps1')
        mock_os_path_exists.assert_called_with('test data')
        mock_upload_chunk.assert_has_calls([mock.call('C:\\Windows\\Temp\\test.ps1', 'te'), mock.call('C:\\Windows\\Temp\\test.ps1', 'st'), mock.call('C:\\Windows\\Temp\\test.ps1', ' d'), mock.call('C:\\Windows\\Temp\\test.ps1', 'at'), mock.call('C:\\Windows\\Temp\\test.ps1', 'a')])

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._run_ps_or_raise')
    def test_upload_chunk(self, mock_run_ps_or_raise):
        if False:
            while True:
                i = 10
        self._runner._upload_chunk('C:\\Windows\\Temp\\testtmp.ps1', 'hello world')
        mock_run_ps_or_raise.assert_called_with('$filePath = "C:\\Windows\\Temp\\testtmp.ps1"\n$s = @"\naGVsbG8gd29ybGQ=\n"@\n$data = [System.Convert]::FromBase64String($s)\nAdd-Content -value $data -encoding byte -path $filePath\n', 'Failed to upload chunk of powershell script')

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._rm_dir')
    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._upload')
    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._make_tmp_dir')
    def test_tmp_script(self, mock_make_tmp_dir, mock_upload, mock_rm_dir):
        if False:
            print('Hello World!')
        mock_make_tmp_dir.return_value = 'C:\\Windows\\Temp\\abc123'
        with self._runner._tmp_script('C:\\Windows\\Temp', 'Get-ChildItem') as tmp:
            self.assertEqual(tmp, 'C:\\Windows\\Temp\\abc123\\script.ps1')
        mock_make_tmp_dir.assert_called_with('C:\\Windows\\Temp')
        mock_upload.assert_called_with('Get-ChildItem', 'C:\\Windows\\Temp\\abc123\\script.ps1')
        mock_rm_dir.assert_called_with('C:\\Windows\\Temp\\abc123')

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._rm_dir')
    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._upload')
    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._make_tmp_dir')
    def test_tmp_script_cleans_up_when_raises(self, mock_make_tmp_dir, mock_upload, mock_rm_dir):
        if False:
            print('Hello World!')
        mock_make_tmp_dir.return_value = 'C:\\Windows\\Temp\\abc123'
        mock_upload.side_effect = RuntimeError
        with self.assertRaises(RuntimeError):
            with self._runner._tmp_script('C:\\Windows\\Temp', 'Get-ChildItem') as tmp:
                self.assertEqual(tmp, 'can never get here')
        mock_make_tmp_dir.assert_called_with('C:\\Windows\\Temp')
        mock_upload.assert_called_with('Get-ChildItem', 'C:\\Windows\\Temp\\abc123\\script.ps1')
        mock_rm_dir.assert_called_with('C:\\Windows\\Temp\\abc123')

    @mock.patch('winrm.Protocol')
    def test_run_cmd(self, mock_protocol_init):
        if False:
            for i in range(10):
                print('nop')
        mock_protocol = mock.MagicMock()
        mock_protocol._raw_get_command_output.side_effect = [(b'output1', b'error1', 0, False), (b'output2', b'error2', 0, False), (b'output3', b'error3', 0, True)]
        mock_protocol_init.return_value = mock_protocol
        self._init_runner()
        result = self._runner.run_cmd('ipconfig /all')
        self.assertEqual(result, ('succeeded', {'failed': False, 'succeeded': True, 'return_code': 0, 'stdout': 'output1output2output3', 'stderr': 'error1error2error3'}, None))

    @mock.patch('winrm.Protocol')
    def test_run_cmd_failed(self, mock_protocol_init):
        if False:
            i = 10
            return i + 15
        mock_protocol = mock.MagicMock()
        mock_protocol._raw_get_command_output.side_effect = [(b'output1', b'error1', 0, False), (b'output2', b'error2', 0, False), (b'output3', b'error3', 1, True)]
        mock_protocol_init.return_value = mock_protocol
        self._init_runner()
        result = self._runner.run_cmd('ipconfig /all')
        self.assertEqual(result, ('failed', {'failed': True, 'succeeded': False, 'return_code': 1, 'stdout': 'output1output2output3', 'stderr': 'error1error2error3'}, None))

    @mock.patch('winrm.Protocol')
    def test_run_cmd_timeout(self, mock_protocol_init):
        if False:
            print('Hello World!')
        mock_protocol = mock.MagicMock()
        self._init_runner()
        self._runner._timeout = 0.1

        def sleep_for_timeout_then_raise(*args, **kwargs):
            if False:
                print('Hello World!')
            time.sleep(0.2)
            return (b'output1', b'error1', 123, False)
        mock_protocol._raw_get_command_output.side_effect = sleep_for_timeout_then_raise
        mock_protocol_init.return_value = mock_protocol
        result = self._runner.run_cmd('ipconfig /all')
        self.assertEqual(result, ('timeout', {'failed': True, 'succeeded': False, 'return_code': -1, 'stdout': 'output1', 'stderr': 'error1'}, None))

    @mock.patch('winrm.Protocol')
    def test_run_ps(self, mock_protocol_init):
        if False:
            for i in range(10):
                print('nop')
        mock_protocol = mock.MagicMock()
        mock_protocol._raw_get_command_output.side_effect = [(b'output1', b'error1', 0, False), (b'output2', b'error2', 0, False), (b'output3', b'error3', 0, True)]
        mock_protocol_init.return_value = mock_protocol
        self._init_runner()
        result = self._runner.run_ps('Get-Location')
        self.assertEqual(result, ('succeeded', {'failed': False, 'succeeded': True, 'return_code': 0, 'stdout': 'output1output2output3', 'stderr': 'error1error2error3'}, None))

    @mock.patch('winrm.Protocol')
    def test_run_ps_failed(self, mock_protocol_init):
        if False:
            while True:
                i = 10
        mock_protocol = mock.MagicMock()
        mock_protocol._raw_get_command_output.side_effect = [(b'output1', b'error1', 0, False), (b'output2', b'error2', 0, False), (b'output3', b'error3', 1, True)]
        mock_protocol_init.return_value = mock_protocol
        self._init_runner()
        result = self._runner.run_ps('Get-Location')
        self.assertEqual(result, ('failed', {'failed': True, 'succeeded': False, 'return_code': 1, 'stdout': 'output1output2output3', 'stderr': 'error1error2error3'}, None))

    @mock.patch('winrm.Protocol')
    def test_run_ps_timeout(self, mock_protocol_init):
        if False:
            while True:
                i = 10
        mock_protocol = mock.MagicMock()
        self._init_runner()
        self._runner._timeout = 0.1

        def sleep_for_timeout_then_raise(*args, **kwargs):
            if False:
                return 10
            time.sleep(0.2)
            return (b'output1', b'error1', 123, False)
        mock_protocol._raw_get_command_output.side_effect = sleep_for_timeout_then_raise
        mock_protocol_init.return_value = mock_protocol
        result = self._runner.run_ps('Get-Location')
        self.assertEqual(result, ('timeout', {'failed': True, 'succeeded': False, 'return_code': -1, 'stdout': 'output1', 'stderr': 'error1'}, None))

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._run_ps')
    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._winrm_encode')
    def test_run_ps_params(self, mock_winrm_encode, mock_run_ps):
        if False:
            print('Hello World!')
        mock_winrm_encode.return_value = 'xyz123=='
        mock_run_ps.return_value = 'expected'
        self._init_runner()
        result = self._runner.run_ps('Get-Location', '-param1 value1 arg1')
        self.assertEqual(result, 'expected')
        mock_winrm_encode.assert_called_with('& {Get-Location} -param1 value1 arg1')
        mock_run_ps.assert_called_with('xyz123==', is_b64=True)

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._winrm_ps_cmd')
    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._run_ps_script')
    def test_run_ps_large_command_convert_to_script(self, mock_run_ps_script, mock_winrm_ps_cmd):
        if False:
            for i in range(10):
                print('nop')
        mock_run_ps_script.return_value = 'expected'
        script = 'powershell -encodedcommand '
        script += '#' * (WINRM_MAX_CMD_LENGTH + 1 - len(script))
        mock_winrm_ps_cmd.return_value = script
        self._init_runner()
        result = self._runner.run_ps('$PSVersionTable')
        self.assertEqual(result, 'expected')
        mock_run_ps_script.assert_called_with('$PSVersionTable', None)

    @mock.patch('winrm.Protocol')
    def test__run_ps(self, mock_protocol_init):
        if False:
            while True:
                i = 10
        mock_protocol = mock.MagicMock()
        mock_protocol._raw_get_command_output.side_effect = [(b'output1', b'error1', 0, False), (b'output2', b'error2', 0, False), (b'output3', b'error3', 0, True)]
        mock_protocol_init.return_value = mock_protocol
        self._init_runner()
        result = self._runner._run_ps('Get-Location')
        self.assertEqual(result, ('succeeded', {'failed': False, 'succeeded': True, 'return_code': 0, 'stdout': 'output1output2output3', 'stderr': 'error1error2error3'}, None))

    @mock.patch('winrm.Protocol')
    def test__run_ps_failed(self, mock_protocol_init):
        if False:
            print('Hello World!')
        mock_protocol = mock.MagicMock()
        mock_protocol._raw_get_command_output.side_effect = [(b'output1', b'error1', 0, False), (b'output2', b'error2', 0, False), (b'output3', b'error3', 1, True)]
        mock_protocol_init.return_value = mock_protocol
        self._init_runner()
        result = self._runner._run_ps('Get-Location')
        self.assertEqual(result, ('failed', {'failed': True, 'succeeded': False, 'return_code': 1, 'stdout': 'output1output2output3', 'stderr': 'error1error2error3'}, None))

    @mock.patch('winrm.Protocol')
    def test__run_ps_timeout(self, mock_protocol_init):
        if False:
            for i in range(10):
                print('nop')
        mock_protocol = mock.MagicMock()
        self._init_runner()
        self._runner._timeout = 0.1

        def sleep_for_timeout_then_raise(*args, **kwargs):
            if False:
                while True:
                    i = 10
            time.sleep(0.2)
            return (b'output1', b'error1', 123, False)
        mock_protocol._raw_get_command_output.side_effect = sleep_for_timeout_then_raise
        mock_protocol_init.return_value = mock_protocol
        result = self._runner._run_ps('Get-Location')
        self.assertEqual(result, ('timeout', {'failed': True, 'succeeded': False, 'return_code': -1, 'stdout': 'output1', 'stderr': 'error1'}, None))

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._winrm_run_ps')
    def test__run_ps_b64_default(self, mock_winrm_run_ps):
        if False:
            while True:
                i = 10
        mock_winrm_run_ps.return_value = mock.MagicMock(status_code=0, timeout=False, std_out='output1', std_err='error1')
        self._init_runner()
        result = self._runner._run_ps('$PSVersionTable')
        self.assertEqual(result, ('succeeded', {'failed': False, 'succeeded': True, 'return_code': 0, 'stdout': 'output1', 'stderr': 'error1'}, None))
        mock_winrm_run_ps.assert_called_with(self._runner._session, '$PSVersionTable', env={}, cwd=None, is_b64=False)

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._winrm_run_ps')
    def test__run_ps_b64_true(self, mock_winrm_run_ps):
        if False:
            i = 10
            return i + 15
        mock_winrm_run_ps.return_value = mock.MagicMock(status_code=0, timeout=False, std_out='output1', std_err='error1')
        self._init_runner()
        result = self._runner._run_ps('xyz123', is_b64=True)
        self.assertEqual(result, ('succeeded', {'failed': False, 'succeeded': True, 'return_code': 0, 'stdout': 'output1', 'stderr': 'error1'}, None))
        mock_winrm_run_ps.assert_called_with(self._runner._session, 'xyz123', env={}, cwd=None, is_b64=True)

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._run_ps')
    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._tmp_script')
    def test__run_ps_script(self, mock_tmp_script, mock_run_ps):
        if False:
            print('Hello World!')
        mock_tmp_script.return_value.__enter__.return_value = 'C:\\tmpscript.ps1'
        mock_run_ps.return_value = 'expected'
        self._init_runner()
        result = self._runner._run_ps_script('$PSVersionTable')
        self.assertEqual(result, 'expected')
        mock_tmp_script.assert_called_with('[System.IO.Path]::GetTempPath()', '$PSVersionTable')
        mock_run_ps.assert_called_with('C:\\tmpscript.ps1')

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._run_ps')
    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._tmp_script')
    def test__run_ps_script_with_params(self, mock_tmp_script, mock_run_ps):
        if False:
            i = 10
            return i + 15
        mock_tmp_script.return_value.__enter__.return_value = 'C:\\tmpscript.ps1'
        mock_run_ps.return_value = 'expected'
        self._init_runner()
        result = self._runner._run_ps_script('Get-ChildItem', '-param1 value1 arg1')
        self.assertEqual(result, 'expected')
        mock_tmp_script.assert_called_with('[System.IO.Path]::GetTempPath()', 'Get-ChildItem')
        mock_run_ps.assert_called_with('C:\\tmpscript.ps1 -param1 value1 arg1')

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._run_ps')
    def test__run_ps_or_raise(self, mock_run_ps):
        if False:
            while True:
                i = 10
        mock_run_ps.return_value = ('success', {'failed': False, 'succeeded': True, 'return_code': 0, 'stdout': 'output', 'stderr': 'error'}, None)
        self._init_runner()
        result = self._runner._run_ps_or_raise('Get-ChildItem', 'my error message')
        self.assertEqual(result, {'failed': False, 'succeeded': True, 'return_code': 0, 'stdout': 'output', 'stderr': 'error'})

    @mock.patch('winrm_runner.winrm_base.WinRmBaseRunner._run_ps')
    def test__run_ps_or_raise_raises_on_failure(self, mock_run_ps):
        if False:
            while True:
                i = 10
        mock_run_ps.return_value = ('success', {'failed': True, 'succeeded': False, 'return_code': 1, 'stdout': 'output', 'stderr': 'error'}, None)
        self._init_runner()
        with self.assertRaises(RuntimeError):
            self._runner._run_ps_or_raise('Get-ChildItem', 'my error message')

    def test_multireplace(self):
        if False:
            while True:
                i = 10
        multireplace_map = {'a': 'x', 'c': 'y', 'aaa': 'z'}
        result = self._runner._multireplace('aaaccaa', multireplace_map)
        self.assertEqual(result, 'zyyxx')

    def test_multireplace_powershell(self):
        if False:
            for i in range(10):
                print('nop')
        param_str = '\n\r\t\x07\x08\x0c\x0b"\'`\x00$'
        result = self._runner._multireplace(param_str, PS_ESCAPE_SEQUENCES)
        self.assertEqual(result, '`n`r`t`a`b`f`v`"`\'```0`$')

    def test_param_to_ps_none(self):
        if False:
            return 10
        param = None
        result = self._runner._param_to_ps(param)
        self.assertEqual(result, '$null')

    def test_param_to_ps_string(self):
        if False:
            i = 10
            return i + 15
        param_str = 'StackStorm 1234'
        result = self._runner._param_to_ps(param_str)
        self.assertEqual(result, '"StackStorm 1234"')
        param_str = '\n\r\t'
        result = self._runner._param_to_ps(param_str)
        self.assertEqual(result, '"`n`r`t"')

    def test_param_to_ps_bool(self):
        if False:
            while True:
                i = 10
        result = self._runner._param_to_ps(True)
        self.assertEqual(result, '$true')
        result = self._runner._param_to_ps(False)
        self.assertEqual(result, '$false')

    def test_param_to_ps_integer(self):
        if False:
            for i in range(10):
                print('nop')
        result = self._runner._param_to_ps(9876)
        self.assertEqual(result, '9876')
        result = self._runner._param_to_ps(-765)
        self.assertEqual(result, '-765')

    def test_param_to_ps_float(self):
        if False:
            i = 10
            return i + 15
        result = self._runner._param_to_ps(98.76)
        self.assertEqual(result, '98.76')
        result = self._runner._param_to_ps(-76.5)
        self.assertEqual(result, '-76.5')

    def test_param_to_ps_list(self):
        if False:
            return 10
        input_list = ['StackStorm Test String', '`\x00$', True, 99]
        result = self._runner._param_to_ps(input_list)
        self.assertEqual(result, '@("StackStorm Test String", "```0`$", $true, 99)')

    def test_param_to_ps_list_nested(self):
        if False:
            for i in range(10):
                print('nop')
        input_list = [['a'], ['b'], [['c']]]
        result = self._runner._param_to_ps(input_list)
        self.assertEqual(result, '@(@("a"), @("b"), @(@("c")))')

    def test_param_to_ps_dict(self):
        if False:
            i = 10
            return i + 15
        input_list = collections.OrderedDict([('str key', 'Value String'), ('esc str\n', '\x08\x0c\x0b"'), (False, True), (11, 99), (18.3, 12.34)])
        result = self._runner._param_to_ps(input_list)
        expected_str = '@{"str key" = "Value String"; "esc str`n" = "`b`f`v`""; $false = $true; 11 = 99; 18.3 = 12.34}'
        self.assertEqual(result, expected_str)

    def test_param_to_ps_dict_nexted(self):
        if False:
            i = 10
            return i + 15
        input_list = collections.OrderedDict([('a', {'deep_a': 'value'}), ('b', {'deep_b': {'deep_deep_b': 'value'}})])
        result = self._runner._param_to_ps(input_list)
        expected_str = '@{"a" = @{"deep_a" = "value"}; "b" = @{"deep_b" = @{"deep_deep_b" = "value"}}}'
        self.assertEqual(result, expected_str)

    def test_param_to_ps_deep_nested_dict_outer(self):
        if False:
            for i in range(10):
                print('nop')
        input_dict = collections.OrderedDict([('a', [{'deep_a': 'value'}, {'deep_b': ['a', 'b', 'c']}])])
        result = self._runner._param_to_ps(input_dict)
        expected_str = '@{"a" = @(@{"deep_a" = "value"}, @{"deep_b" = @("a", "b", "c")})}'
        self.assertEqual(result, expected_str)

    def test_param_to_ps_deep_nested_list_outer(self):
        if False:
            for i in range(10):
                print('nop')
        input_list = [{'deep_a': 'value'}, {'deep_b': ['a', 'b', 'c']}, {'deep_c': [{'x': 'y'}]}]
        result = self._runner._param_to_ps(input_list)
        expected_str = '@(@{"deep_a" = "value"}, @{"deep_b" = @("a", "b", "c")}, @{"deep_c" = @(@{"x" = "y"})})'
        self.assertEqual(result, expected_str)

    def test_transform_params_to_ps(self):
        if False:
            return 10
        positional_args = [1, 'a', '\n']
        named_args = collections.OrderedDict([('a', 'value1'), ('b', True), ('c', ['x', 'y']), ('d', {'z': 'w'})])
        (result_pos, result_named) = self._runner._transform_params_to_ps(positional_args, named_args)
        self.assertEqual(result_pos, ['1', '"a"', '"`n"'])
        self.assertEqual(result_named, collections.OrderedDict([('a', '"value1"'), ('b', '$true'), ('c', '@("x", "y")'), ('d', '@{"z" = "w"}')]))

    def test_transform_params_to_ps_none(self):
        if False:
            return 10
        positional_args = None
        named_args = None
        (result_pos, result_named) = self._runner._transform_params_to_ps(positional_args, named_args)
        self.assertEqual(result_pos, None)
        self.assertEqual(result_named, None)

    def test_create_ps_params_string(self):
        if False:
            i = 10
            return i + 15
        positional_args = [1, 'a', '\n']
        named_args = collections.OrderedDict([('-a', 'value1'), ('-b', True), ('-c', ['x', 'y']), ('-d', {'z': 'w'})])
        result = self._runner.create_ps_params_string(positional_args, named_args)
        self.assertEqual(result, '-a "value1" -b $true -c @("x", "y") -d @{"z" = "w"} 1 "a" "`n"')

    def test_create_ps_params_string_none(self):
        if False:
            i = 10
            return i + 15
        positional_args = None
        named_args = None
        result = self._runner.create_ps_params_string(positional_args, named_args)
        self.assertEqual(result, '')