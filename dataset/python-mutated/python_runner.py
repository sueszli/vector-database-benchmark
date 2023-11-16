from __future__ import absolute_import
import os
import re
import sys
import uuid
import functools
from subprocess import list2cmdline
import six
from oslo_config import cfg
from six.moves import StringIO
from st2common import log as logging
from st2common.runners.base import GitWorktreeActionRunner
from st2common.runners.base import get_metadata as get_runner_metadata
from st2common.util import concurrency
from st2common.util.green.shell import run_command
from st2common.constants.action import ACTION_OUTPUT_RESULT_DELIMITER
from st2common.constants.action import LIVEACTION_STATUS_SUCCEEDED
from st2common.constants.action import LIVEACTION_STATUS_FAILED
from st2common.constants.action import LIVEACTION_STATUS_TIMED_OUT
from st2common.constants.action import MAX_PARAM_LENGTH
from st2common.constants.runners import PYTHON_RUNNER_INVALID_ACTION_STATUS_EXIT_CODE
from st2common.constants.error_messages import PACK_VIRTUALENV_DOESNT_EXIST
from st2common.constants.runners import PYTHON_RUNNER_DEFAULT_ACTION_TIMEOUT
from st2common.constants.runners import PYTHON_RUNNER_DEFAULT_LOG_LEVEL
from st2common.constants.system import API_URL_ENV_VARIABLE_NAME
from st2common.constants.system import AUTH_TOKEN_ENV_VARIABLE_NAME
from st2common.util.api import get_full_public_api_url
from st2common.util.pack import get_pack_common_libs_path_for_pack_ref
from st2common.content.utils import get_pack_base_path
from st2common.util.sandboxing import get_sandbox_path
from st2common.util.sandboxing import get_sandbox_python_path_for_python_action
from st2common.util.sandboxing import get_sandbox_python_binary_path
from st2common.util.sandboxing import get_sandbox_virtualenv_path
from st2common.util.shell import quote_unix
from st2common.services.action import store_execution_output_data
from st2common.runners.utils import make_read_and_store_stream_func
from st2common.util.jsonify import json_decode
from st2common.util.jsonify import json_encode
from python_runner import python_action_wrapper
__all__ = ['PythonRunner', 'get_runner', 'get_metadata']
LOG = logging.getLogger(__name__)
RUNNER_ENV = 'env'
RUNNER_TIMEOUT = 'timeout'
RUNNER_LOG_LEVEL = 'log_level'
BLACKLISTED_ENV_VARS = ['pythonpath']
BASE_DIR = os.path.dirname(os.path.abspath(python_action_wrapper.__file__))
WRAPPER_SCRIPT_NAME = 'python_action_wrapper.py'
WRAPPER_SCRIPT_PATH = os.path.join(BASE_DIR, WRAPPER_SCRIPT_NAME)

class PythonRunner(GitWorktreeActionRunner):

    def __init__(self, runner_id, config=None, timeout=PYTHON_RUNNER_DEFAULT_ACTION_TIMEOUT, log_level=None, sandbox=True, use_parent_args=True):
        if False:
            while True:
                i = 10
        '\n        :param timeout: Action execution timeout in seconds.\n        :type timeout: ``int``\n\n        :param log_level: Log level to use for the child actions.\n        :type log_level: ``str``\n\n        :param sandbox: True to use python binary from pack-specific virtual environment for the\n                        child action False to use a default system python binary from PATH.\n        :type sandbox: ``bool``\n\n        :param use_parent_args: True to use command line arguments from the parent process.\n        :type use_parent_args: ``bool``\n        '
        super(PythonRunner, self).__init__(runner_id=runner_id)
        self._config = config
        self._timeout = timeout
        self._enable_common_pack_libs = cfg.CONF.packs.enable_common_libs or False
        self._log_level = log_level or cfg.CONF.actionrunner.python_runner_log_level
        self._sandbox = sandbox
        self._use_parent_args = use_parent_args

    def pre_run(self):
        if False:
            while True:
                i = 10
        super(PythonRunner, self).pre_run()
        self._env = self.runner_parameters.get(RUNNER_ENV, {})
        self._timeout = self.runner_parameters.get(RUNNER_TIMEOUT, self._timeout)
        self._log_level = self.runner_parameters.get(RUNNER_LOG_LEVEL, self._log_level)
        if self._log_level == PYTHON_RUNNER_DEFAULT_LOG_LEVEL:
            self._log_level = cfg.CONF.actionrunner.python_runner_log_level

    def run(self, action_parameters):
        if False:
            return 10
        LOG.debug('Running pythonrunner.')
        LOG.debug('Getting pack name.')
        pack = self.get_pack_ref()
        LOG.debug('Getting user.')
        user = self.get_user()
        LOG.debug('Serializing parameters.')
        serialized_parameters = json_encode(action_parameters if action_parameters else {})
        LOG.debug('Getting virtualenv_path.')
        virtualenv_path = get_sandbox_virtualenv_path(pack=pack)
        LOG.debug('Getting python path.')
        if self._sandbox:
            python_path = get_sandbox_python_binary_path(pack=pack)
        else:
            python_path = sys.executable
        LOG.debug('Checking virtualenv path.')
        if virtualenv_path and (not os.path.isdir(virtualenv_path)):
            format_values = {'pack': pack, 'virtualenv_path': virtualenv_path}
            msg = PACK_VIRTUALENV_DOESNT_EXIST % format_values
            LOG.error('virtualenv_path set but not a directory: %s', msg)
            raise Exception(msg)
        LOG.debug('Checking entry_point.')
        if not self.entry_point:
            LOG.error('Action "%s" is missing entry_point attribute' % self.action.name)
            raise Exception('Action "%s" is missing entry_point attribute' % self.action.name)
        LOG.debug('Setting args.')
        if self._use_parent_args:
            parent_args = json_encode(sys.argv[1:])
        else:
            parent_args = json_encode([])
        args = [python_path, '-u', WRAPPER_SCRIPT_PATH, '--pack=%s' % pack, '--file-path=%s' % self.entry_point, '--user=%s' % user, '--parent-args=%s' % parent_args]
        subprocess = concurrency.get_subprocess_module()
        stdin = None
        stdin_params = None
        if len(serialized_parameters) >= MAX_PARAM_LENGTH:
            stdin = subprocess.PIPE
            LOG.debug('Parameters are too big...changing to stdin')
            stdin_params = '{"parameters": %s}\n' % serialized_parameters
            args.append('--stdin-parameters')
        else:
            LOG.debug('Parameters are just right...adding them to arguments')
            args.append('--parameters=%s' % serialized_parameters)
        if self._config:
            args.append('--config=%s' % json_encode(self._config))
        if self._log_level != PYTHON_RUNNER_DEFAULT_LOG_LEVEL:
            args.append('--log-level=%s' % self._log_level)
        LOG.debug('Setting env.')
        env = os.environ.copy()
        env['PATH'] = get_sandbox_path(virtualenv_path=virtualenv_path)
        sandbox_python_path = get_sandbox_python_path_for_python_action(pack=pack, inherit_from_parent=True, inherit_parent_virtualenv=True)
        if self._enable_common_pack_libs:
            try:
                pack_common_libs_path = self._get_pack_common_libs_path(pack_ref=pack)
            except Exception as e:
                LOG.debug('Failed to retrieve pack common lib path: %s' % six.text_type(e))
                pack_common_libs_path = None
        else:
            pack_common_libs_path = None
        if sandbox_python_path.startswith(':'):
            sandbox_python_path = sandbox_python_path[1:]
        if self._enable_common_pack_libs and pack_common_libs_path:
            sandbox_python_path = pack_common_libs_path + ':' + sandbox_python_path
        env['PYTHONPATH'] = sandbox_python_path
        user_env_vars = self._get_env_vars()
        env.update(user_env_vars)
        st2_env_vars = self._get_common_action_env_variables()
        env.update(st2_env_vars)
        datastore_env_vars = self._get_datastore_access_env_vars()
        env.update(datastore_env_vars)
        stdout = StringIO()
        stderr = StringIO()
        store_execution_stdout_line = functools.partial(store_execution_output_data, output_type='stdout')
        store_execution_stderr_line = functools.partial(store_execution_output_data, output_type='stderr')
        read_and_store_stdout = make_read_and_store_stream_func(execution_db=self.execution, action_db=self.action, store_data_func=store_execution_stdout_line)
        read_and_store_stderr = make_read_and_store_stream_func(execution_db=self.execution, action_db=self.action, store_data_func=store_execution_stderr_line)
        command_string = list2cmdline(args)
        if stdin_params:
            command_string = 'echo %s | %s' % (quote_unix(stdin_params), command_string)
        bufsize = cfg.CONF.actionrunner.stream_output_buffer_size
        LOG.debug('Running command (bufsize=%s): PATH=%s PYTHONPATH=%s %s' % (bufsize, env['PATH'], env['PYTHONPATH'], command_string))
        (exit_code, stdout, stderr, timed_out) = run_command(cmd=args, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, env=env, timeout=self._timeout, read_stdout_func=read_and_store_stdout, read_stderr_func=read_and_store_stderr, read_stdout_buffer=stdout, read_stderr_buffer=stderr, stdin_value=stdin_params, bufsize=bufsize)
        LOG.debug('Returning values: %s, %s, %s, %s', exit_code, stdout, stderr, timed_out)
        LOG.debug('Returning.')
        return self._get_output_values(exit_code, stdout, stderr, timed_out)

    def _get_pack_common_libs_path(self, pack_ref):
        if False:
            i = 10
            return i + 15
        '\n        Retrieve path to the pack common lib/ directory taking git work tree path into account\n        (if used).\n        '
        worktree_path = self.git_worktree_path
        pack_common_libs_path = get_pack_common_libs_path_for_pack_ref(pack_ref=pack_ref)
        if not worktree_path:
            return pack_common_libs_path
        pack_base_path = get_pack_base_path(pack_name=pack_ref)
        new_pack_common_libs_path = pack_common_libs_path.replace(pack_base_path, '')
        if new_pack_common_libs_path.startswith('/'):
            new_pack_common_libs_path = new_pack_common_libs_path[1:]
        new_pack_common_libs_path = os.path.join(worktree_path, new_pack_common_libs_path)
        common_prefix = os.path.commonprefix([worktree_path, new_pack_common_libs_path])
        if common_prefix != worktree_path:
            raise ValueError('pack libs path is not located inside the pack directory')
        return new_pack_common_libs_path

    def _get_output_values(self, exit_code, stdout, stderr, timed_out):
        if False:
            print('Hello World!')
        '\n        Return sanitized output values.\n\n        :return: Tuple with status, output and None\n\n        :rtype: ``tuple``\n        '
        if timed_out:
            error = 'Action failed to complete in %s seconds' % self._timeout
        else:
            error = None
        if exit_code == PYTHON_RUNNER_INVALID_ACTION_STATUS_EXIT_CODE:
            raise ValueError(stderr)
        if ACTION_OUTPUT_RESULT_DELIMITER in stdout:
            split = stdout.split(ACTION_OUTPUT_RESULT_DELIMITER)
            if len(split) != 3:
                raise ValueError(f'The result length should be 3, was {len(split)}.')
            action_result = split[1].strip()
            stdout = split[0] + split[2]
        else:
            action_result = None
        if action_result:
            try:
                action_result = json_decode(action_result)
            except Exception as e:
                LOG.warning('Failed to de-serialize result "%s": %s' % (str(action_result), six.text_type(e)))
        if action_result:
            if isinstance(action_result, dict):
                result = action_result.get('result', None)
                status = action_result.get('status', None)
            else:
                match = re.search("'result': (.*?)$", action_result or '')
                if match:
                    action_result = match.groups()[0]
                result = action_result
                status = None
        else:
            result = 'None'
            status = None
        output = {'stdout': stdout, 'stderr': stderr, 'exit_code': exit_code, 'result': result}
        if error:
            output['error'] = error
        status = self._get_final_status(action_status=status, timed_out=timed_out, exit_code=exit_code)
        return (status, output, None)

    def _get_final_status(self, action_status, timed_out, exit_code):
        if False:
            while True:
                i = 10
        "\n        Return final status based on action's status, time out value and\n        exit code. Example: succeeded, failed, timeout.\n\n        :return: status\n\n        :rtype: ``str``\n        "
        if action_status is not None:
            if exit_code == 0 and action_status is True:
                status = LIVEACTION_STATUS_SUCCEEDED
            elif exit_code == 0 and action_status is False:
                status = LIVEACTION_STATUS_FAILED
            else:
                status = LIVEACTION_STATUS_FAILED
        elif exit_code == 0:
            status = LIVEACTION_STATUS_SUCCEEDED
        else:
            status = LIVEACTION_STATUS_FAILED
        if timed_out:
            status = LIVEACTION_STATUS_TIMED_OUT
        return status

    def _get_env_vars(self):
        if False:
            i = 10
            return i + 15
        '\n        Return sanitized environment variables which will be used when launching\n        a subprocess.\n\n        :rtype: ``dict``\n        '
        env_vars = {}
        if self._env:
            env_vars.update(self._env)
        to_delete = []
        for (key, value) in env_vars.items():
            if key.lower() in BLACKLISTED_ENV_VARS:
                to_delete.append(key)
        for key in to_delete:
            LOG.debug('User specified environment variable "%s" which is being ignored...' % key)
            del env_vars[key]
        return env_vars

    def _get_datastore_access_env_vars(self):
        if False:
            while True:
                i = 10
        '\n        Return environment variables so datastore access using client (from st2client)\n        is possible with actions. This is done to be compatible with sensors.\n\n        :rtype: ``dict``\n        '
        env_vars = {}
        if self.auth_token:
            env_vars[AUTH_TOKEN_ENV_VARIABLE_NAME] = self.auth_token.token
        env_vars[API_URL_ENV_VARIABLE_NAME] = get_full_public_api_url()
        return env_vars

def get_runner(config=None):
    if False:
        for i in range(10):
            print('nop')
    return PythonRunner(runner_id=str(uuid.uuid4()), config=config)

def get_metadata():
    if False:
        while True:
            i = 10
    return get_runner_metadata('python_runner')[0]