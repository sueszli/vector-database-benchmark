from __future__ import absolute_import
import base64
import os
import re
import six
import time
from base64 import b64encode
from contextlib import contextmanager
from st2common import log as logging
from st2common.constants import action as action_constants
from st2common.constants import exit_codes as exit_code_constants
from st2common.runners.base import ActionRunner
from st2common.util import jsonify
from winrm import Session, Response
from winrm.exceptions import WinRMOperationTimeoutError
__all__ = ['WinRmBaseRunner']
LOG = logging.getLogger(__name__)
RUNNER_CWD = 'cwd'
RUNNER_ENV = 'env'
RUNNER_HOST = 'host'
RUNNER_KWARG_OP = 'kwarg_op'
RUNNER_PASSWORD = 'password'
RUNNER_PORT = 'port'
RUNNER_SCHEME = 'scheme'
RUNNER_TIMEOUT = 'timeout'
RUNNER_TRANSPORT = 'transport'
RUNNER_USERNAME = 'username'
RUNNER_VERIFY_SSL = 'verify_ssl_cert'
WINRM_DEFAULT_TMP_DIR_PS = '[System.IO.Path]::GetTempPath()'
WINRM_MAX_CMD_LENGTH = 8191
WINRM_HTTPS_PORT = 5986
WINRM_HTTP_PORT = 5985
WINRM_TIMEOUT_EXIT_CODE = exit_code_constants.SUCCESS_EXIT_CODE - 1
WINRM_UPLOAD_CHUNK_SIZE_BYTES = 2048
DEFAULT_KWARG_OP = '-'
DEFAULT_PORT = WINRM_HTTPS_PORT
DEFAULT_SCHEME = 'https'
DEFAULT_TIMEOUT = 60
DEFAULT_TRANSPORT = 'ntlm'
DEFAULT_VERIFY_SSL = True
RESULT_KEYS_TO_TRANSFORM = ['stdout', 'stderr']
PS_ESCAPE_SEQUENCES = {'\n': '`n', '\r': '`r', '\t': '`t', '\x07': '`a', '\x08': '`b', '\x0c': '`f', '\x0b': '`v', '"': '`"', "'": "`'", '`': '``', '\x00': '`0', '$': '`$'}

class WinRmRunnerTimoutError(Exception):

    def __init__(self, response):
        if False:
            print('Hello World!')
        self.response = response

class WinRmBaseRunner(ActionRunner):

    def pre_run(self):
        if False:
            while True:
                i = 10
        super(WinRmBaseRunner, self).pre_run()
        self._session = None
        self._host = self.runner_parameters[RUNNER_HOST]
        self._username = self.runner_parameters[RUNNER_USERNAME]
        self._password = self.runner_parameters[RUNNER_PASSWORD]
        self._timeout = self.runner_parameters.get(RUNNER_TIMEOUT, DEFAULT_TIMEOUT)
        self._read_timeout = self._timeout + 1
        self._port = self.runner_parameters.get(RUNNER_PORT, DEFAULT_PORT)
        self._scheme = self.runner_parameters.get(RUNNER_SCHEME, DEFAULT_SCHEME)
        self._transport = self.runner_parameters.get(RUNNER_TRANSPORT, DEFAULT_TRANSPORT)
        if self._port == WINRM_HTTP_PORT:
            self._scheme = 'http'
        self._winrm_url = '{}://{}:{}/wsman'.format(self._scheme, self._host, self._port)
        self._verify_ssl = self.runner_parameters.get(RUNNER_VERIFY_SSL, DEFAULT_VERIFY_SSL)
        self._server_cert_validation = 'validate' if self._verify_ssl else 'ignore'
        self._cwd = self.runner_parameters.get(RUNNER_CWD, None)
        self._env = self.runner_parameters.get(RUNNER_ENV, {})
        self._env = self._env or {}
        self._kwarg_op = self.runner_parameters.get(RUNNER_KWARG_OP, DEFAULT_KWARG_OP)

    def _get_session(self):
        if False:
            i = 10
            return i + 15
        if not self._session:
            LOG.debug('Connecting via WinRM to url: {}'.format(self._winrm_url))
            self._session = Session(self._winrm_url, auth=(self._username, self._password), transport=self._transport, server_cert_validation=self._server_cert_validation, operation_timeout_sec=self._timeout, read_timeout_sec=self._read_timeout)
        return self._session

    def _winrm_get_command_output(self, protocol, shell_id, command_id):
        if False:
            while True:
                i = 10
        (stdout_buffer, stderr_buffer) = ([], [])
        return_code = 0
        command_done = False
        start_time = time.time()
        while not command_done:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if self._timeout and elapsed_time > self._timeout:
                raise WinRmRunnerTimoutError(Response((b''.join(stdout_buffer), b''.join(stderr_buffer), WINRM_TIMEOUT_EXIT_CODE)))
            try:
                (stdout, stderr, return_code, command_done) = protocol._raw_get_command_output(shell_id, command_id)
                stdout_buffer.append(stdout)
                stderr_buffer.append(stderr)
            except WinRMOperationTimeoutError:
                pass
        return (b''.join(stdout_buffer), b''.join(stderr_buffer), return_code)

    def _winrm_run_cmd(self, session, command, args=(), env=None, cwd=None):
        if False:
            for i in range(10):
                print('nop')
        shell_id = session.protocol.open_shell(env_vars=env, working_directory=cwd)
        command_id = session.protocol.run_command(shell_id, command, args)
        try:
            rs = Response(self._winrm_get_command_output(session.protocol, shell_id, command_id))
            rs.timeout = False
        except WinRmRunnerTimoutError as e:
            rs = e.response
            rs.timeout = True
        session.protocol.cleanup_command(shell_id, command_id)
        session.protocol.close_shell(shell_id)
        return rs

    def _winrm_encode(self, script):
        if False:
            return 10
        return b64encode(script.encode('utf_16_le')).decode('ascii')

    def _winrm_ps_cmd(self, encoded_ps):
        if False:
            i = 10
            return i + 15
        return 'powershell -encodedcommand {0}'.format(encoded_ps)

    def _winrm_run_ps(self, session, script, env=None, cwd=None, is_b64=False):
        if False:
            return 10
        LOG.debug('_winrm_run_ps() - script size = {}'.format(len(script)))
        encoded_ps = script if is_b64 else self._winrm_encode(script)
        ps_cmd = self._winrm_ps_cmd(encoded_ps)
        LOG.debug('_winrm_run_ps() - ps cmd size = {}'.format(len(ps_cmd)))
        rs = self._winrm_run_cmd(session, ps_cmd, env=env, cwd=cwd)
        if len(rs.std_err):
            rs.std_err = session._clean_error_msg(rs.std_err)
        return rs

    def _translate_response(self, response):
        if False:
            i = 10
            return i + 15
        succeeded = response.status_code == exit_code_constants.SUCCESS_EXIT_CODE
        status = action_constants.LIVEACTION_STATUS_SUCCEEDED
        status_code = response.status_code
        if response.timeout:
            status = action_constants.LIVEACTION_STATUS_TIMED_OUT
            status_code = WINRM_TIMEOUT_EXIT_CODE
        elif not succeeded:
            status = action_constants.LIVEACTION_STATUS_FAILED
        result = {'failed': not succeeded, 'succeeded': succeeded, 'return_code': status_code, 'stdout': response.std_out, 'stderr': response.std_err}
        if isinstance(result['stdout'], six.binary_type):
            result['stdout'] = result['stdout'].decode('utf-8')
        if isinstance(result['stderr'], six.binary_type):
            result['stderr'] = result['stderr'].decode('utf-8')
        return (status, jsonify.json_loads(result, RESULT_KEYS_TO_TRANSFORM), None)

    def _make_tmp_dir(self, parent):
        if False:
            print('Hello World!')
        LOG.debug('Creating temporary directory for WinRM script in parent: {}'.format(parent))
        ps = '$parent = {parent}\n$name = [System.IO.Path]::GetRandomFileName()\n$path = Join-Path $parent $name\nNew-Item -ItemType Directory -Path $path | Out-Null\n$path'.format(parent=parent)
        result = self._run_ps_or_raise(ps, 'Unable to make temporary directory for powershell script')
        return result['stdout'].strip()

    def _rm_dir(self, directory):
        if False:
            i = 10
            return i + 15
        ps = 'Remove-Item -Force -Recurse -Path "{}"'.format(directory)
        self._run_ps_or_raise(ps, 'Unable to remove temporary directory for powershell script')

    def _upload(self, src_path_or_data, dst_path):
        if False:
            return 10
        src_data = None
        if os.path.exists(src_path_or_data):
            LOG.debug('WinRM uploading local file: {}'.format(src_path_or_data))
            with open(src_path_or_data, 'r') as src_file:
                src_data = src_file.read()
        else:
            LOG.debug('WinRM uploading data from a string')
            src_data = src_path_or_data
        for i in range(0, len(src_data), WINRM_UPLOAD_CHUNK_SIZE_BYTES):
            LOG.debug('WinRM uploading data bytes: {}-{}'.format(i, i + WINRM_UPLOAD_CHUNK_SIZE_BYTES))
            self._upload_chunk(dst_path, src_data[i:i + WINRM_UPLOAD_CHUNK_SIZE_BYTES])

    def _upload_chunk(self, dst_path, src_data):
        if False:
            while True:
                i = 10
        if not isinstance(src_data, six.binary_type):
            src_data = src_data.encode('utf-8')
        ps = '$filePath = "{dst_path}"\n$s = @"\n{b64_data}\n"@\n$data = [System.Convert]::FromBase64String($s)\nAdd-Content -value $data -encoding byte -path $filePath\n'.format(dst_path=dst_path, b64_data=base64.b64encode(src_data).decode('utf-8'))
        LOG.debug('WinRM uploading chunk, size = {}'.format(len(ps)))
        self._run_ps_or_raise(ps, 'Failed to upload chunk of powershell script')

    @contextmanager
    def _tmp_script(self, parent, script):
        if False:
            while True:
                i = 10
        tmp_dir = None
        try:
            LOG.info('WinRM Script - Making temporary directory')
            tmp_dir = self._make_tmp_dir(parent)
            LOG.debug('WinRM Script - Tmp directory created: {}'.format(tmp_dir))
            LOG.info('WinRM Script = Upload starting')
            tmp_script = tmp_dir + '\\script.ps1'
            LOG.debug('WinRM Uploading script to: {}'.format(tmp_script))
            self._upload(script, tmp_script)
            LOG.info('WinRM Script - Upload complete')
            yield tmp_script
        finally:
            if tmp_dir:
                LOG.debug('WinRM Script - Removing script: {}'.format(tmp_dir))
                self._rm_dir(tmp_dir)

    def run_cmd(self, cmd):
        if False:
            while True:
                i = 10
        session = self._get_session()
        response = self._winrm_run_cmd(session, cmd, env=self._env, cwd=self._cwd)
        return self._translate_response(response)

    def run_ps(self, script, params=None):
        if False:
            while True:
                i = 10
        if params:
            powershell = '& {%s} %s' % (script, params)
        else:
            powershell = script
        encoded_ps = self._winrm_encode(powershell)
        ps_cmd = self._winrm_ps_cmd(encoded_ps)
        if len(ps_cmd) <= WINRM_MAX_CMD_LENGTH:
            LOG.info('WinRM powershell command size {} is > {}, the max size of a powershell command. Converting to a script execution.'.format(WINRM_MAX_CMD_LENGTH, len(ps_cmd)))
            return self._run_ps(encoded_ps, is_b64=True)
        else:
            return self._run_ps_script(script, params)

    def _run_ps(self, powershell, is_b64=False):
        if False:
            return 10
        "Executes a powershell command, no checks for length are done in this version.\n        The lack of checks here is intentional so that we don't run into an infinte loop\n        when converting a long command to a script"
        session = self._get_session()
        response = self._winrm_run_ps(session, powershell, env=self._env, cwd=self._cwd, is_b64=is_b64)
        return self._translate_response(response)

    def _run_ps_script(self, script, params=None):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = WINRM_DEFAULT_TMP_DIR_PS
        with self._tmp_script(tmp_dir, script) as tmp_script:
            ps = tmp_script
            if params:
                ps += ' ' + params
            return self._run_ps(ps)

    def _run_ps_or_raise(self, ps, error_msg):
        if False:
            while True:
                i = 10
        response = self._run_ps(ps)
        result = response[1]
        if result['failed']:
            raise RuntimeError('{}:\nstdout = {}\n\nstderr = {}'.format(error_msg, result['stdout'], result['stderr']))
        return result

    def _multireplace(self, string, replacements):
        if False:
            return 10
        '\n        Given a string and a replacement map, it returns the replaced string.\n        Source = https://gist.github.com/bgusach/a967e0587d6e01e889fd1d776c5f3729\n        Reference = https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string  # noqa\n        :param str string: string to execute replacements on\n        :param dict replacements: replacement dictionary {value to find: value to replace}\n        :rtype: str\n        '
        substrs = sorted(replacements, key=len, reverse=True)
        regexp = re.compile('|'.join([re.escape(s) for s in substrs]))
        return regexp.sub(lambda match: replacements[match.group(0)], string)

    def _param_to_ps(self, param):
        if False:
            print('Hello World!')
        ps_str = ''
        if param is None:
            ps_str = '$null'
        elif isinstance(param, six.string_types):
            ps_str = '"' + self._multireplace(param, PS_ESCAPE_SEQUENCES) + '"'
        elif isinstance(param, bool):
            ps_str = '$true' if param else '$false'
        elif isinstance(param, list):
            ps_str = '@('
            ps_str += ', '.join([self._param_to_ps(p) for p in param])
            ps_str += ')'
        elif isinstance(param, dict):
            ps_str = '@{'
            ps_str += '; '.join([self._param_to_ps(k) + ' = ' + self._param_to_ps(v) for (k, v) in six.iteritems(param)])
            ps_str += '}'
        else:
            ps_str = str(param)
        return ps_str

    def _transform_params_to_ps(self, positional_args, named_args):
        if False:
            for i in range(10):
                print('nop')
        if positional_args:
            for (i, arg) in enumerate(positional_args):
                positional_args[i] = self._param_to_ps(arg)
        if named_args:
            for (key, value) in six.iteritems(named_args):
                named_args[key] = self._param_to_ps(value)
        return (positional_args, named_args)

    def create_ps_params_string(self, positional_args, named_args):
        if False:
            return 10
        (positional_args, named_args) = self._transform_params_to_ps(positional_args, named_args)
        ps_params_str = ''
        if named_args:
            ps_params_str += ' '.join([k + ' ' + v for (k, v) in six.iteritems(named_args)])
            ps_params_str += ' '
        if positional_args:
            ps_params_str += ' '.join(positional_args)
        return ps_params_str