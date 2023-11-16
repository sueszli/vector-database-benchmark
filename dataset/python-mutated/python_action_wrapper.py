from __future__ import absolute_import
import warnings
warnings.filterwarnings('ignore', message='Python 3.6 is no longer supported')
import os
import sys
import select
import traceback
import sysconfig
import orjson
RUNNERS_PATH_SUFFIX = 'st2common/runners'
if __name__ == '__main__':
    script_path = sys.path[0]
    if RUNNERS_PATH_SUFFIX in script_path:
        sys.path.pop(0)
    sys.path.insert(0, sysconfig.get_path('platlib'))
import sys
import argparse
import six
from st2common import log as logging
from st2common import config as st2common_config
from st2common.runners.base_action import Action
from st2common.runners.utils import get_logger_for_python_runner_action
from st2common.runners.utils import get_action_class_instance
from st2common.util import loader as action_loader
from st2common.constants.action import ACTION_OUTPUT_RESULT_DELIMITER
from st2common.constants.keyvalue import SYSTEM_SCOPE
from st2common.constants.runners import PYTHON_RUNNER_INVALID_ACTION_STATUS_EXIT_CODE
from st2common.constants.runners import PYTHON_RUNNER_DEFAULT_LOG_LEVEL
__all__ = ['PythonActionWrapper', 'ActionService']
LOG = logging.getLogger(__name__)
INVALID_STATUS_ERROR_MESSAGE = "\nIf this is an existing action which returns a tuple with two items, it needs to be updated to\neither:\n\n1. Return a list instead of a tuple\n2. Return a tuple where a first items is a status flag - (True, ('item1', 'item2'))\n\nFor more information, please see: https://docs.stackstorm.com/upgrade_notes.html#st2-v1-6\n".strip()
READ_STDIN_INPUT_TIMEOUT = 2

class ActionService(object):
    """
    Instance of this class is passed to the action instance and exposes "public" methods which can
    be called by the action.
    """

    def __init__(self, action_wrapper):
        if False:
            for i in range(10):
                print('nop')
        self._action_wrapper = action_wrapper
        self._datastore_service = None

    @property
    def datastore_service(self):
        if False:
            print('Hello World!')
        from st2common.services.datastore import ActionDatastoreService
        if not self._datastore_service:
            action_name = self._action_wrapper._class_name
            log_level = self._action_wrapper._log_level
            logger = get_logger_for_python_runner_action(action_name=action_name, log_level=log_level)
            pack_name = self._action_wrapper._pack
            class_name = self._action_wrapper._class_name
            auth_token = os.environ.get('ST2_ACTION_AUTH_TOKEN', None)
            self._datastore_service = ActionDatastoreService(logger=logger, pack_name=pack_name, class_name=class_name, auth_token=auth_token)
        return self._datastore_service

    def get_user_info(self):
        if False:
            while True:
                i = 10
        return self.datastore_service.get_user_info()

    def list_values(self, local=True, prefix=None):
        if False:
            for i in range(10):
                print('nop')
        return self.datastore_service.list_values(local=local, prefix=prefix)

    def get_value(self, name, local=True, scope=SYSTEM_SCOPE, decrypt=False):
        if False:
            i = 10
            return i + 15
        return self.datastore_service.get_value(name=name, local=local, scope=scope, decrypt=decrypt)

    def set_value(self, name, value, ttl=None, local=True, scope=SYSTEM_SCOPE, encrypt=False):
        if False:
            i = 10
            return i + 15
        return self.datastore_service.set_value(name=name, value=value, ttl=ttl, local=local, scope=scope, encrypt=encrypt)

    def delete_value(self, name, local=True, scope=SYSTEM_SCOPE):
        if False:
            return 10
        return self.datastore_service.delete_value(name=name, local=local, scope=scope)

class PythonActionWrapper(object):

    def __init__(self, pack, file_path, config=None, parameters=None, user=None, parent_args=None, log_level=PYTHON_RUNNER_DEFAULT_LOG_LEVEL):
        if False:
            print('Hello World!')
        '\n        :param pack: Name of the pack this action belongs to.\n        :type pack: ``str``\n\n        :param file_path: Path to the action module.\n        :type file_path: ``str``\n\n        :param config: Pack config.\n        :type config: ``dict``\n\n        :param parameters: action parameters.\n        :type parameters: ``dict`` or ``None``\n\n        :param user: Name of the user who triggered this action execution.\n        :type user: ``str``\n\n        :param parent_args: Command line arguments passed to the parent process.\n        :type parse_args: ``list``\n        '
        self._pack = pack
        self._file_path = file_path
        self._config = config or {}
        self._parameters = parameters or {}
        self._user = user
        self._parent_args = parent_args or []
        self._log_level = log_level
        self._class_name = None
        self._logger = logging.getLogger('PythonActionWrapper')
        try:
            st2common_config.parse_args(args=self._parent_args)
        except Exception as e:
            LOG.debug('Failed to parse config using parent args (parent_args=%s): %s' % (str(self._parent_args), six.text_type(e)))
        if not self._user:
            from oslo_config import cfg
            self._user = cfg.CONF.system_user.user

    def run(self):
        if False:
            i = 10
            return i + 15
        action = self._get_action_instance()
        output = action.run(**self._parameters)
        if isinstance(output, tuple) and len(output) == 2:
            action_status = output[0]
            action_result = output[1]
        else:
            action_status = None
            action_result = output
        action_output = {'result': action_result, 'status': None}
        if action_status is not None and (not isinstance(action_status, bool)):
            sys.stderr.write('Status returned from the action run() method must either be True or False, got: %s\n' % action_status)
            sys.stderr.write(INVALID_STATUS_ERROR_MESSAGE)
            sys.exit(PYTHON_RUNNER_INVALID_ACTION_STATUS_EXIT_CODE)
        if action_status is not None and isinstance(action_status, bool):
            action_output['status'] = action_status
            try:
                orjson.dumps(action_output['result'])
            except (TypeError, orjson.JSONDecodeError):
                action_output['result'] = str(action_output['result'])
        try:
            print_output = orjson.dumps(action_output)
        except Exception:
            print_output = str(action_output).encode('utf-8')
        sys.stdout.buffer.write(ACTION_OUTPUT_RESULT_DELIMITER.encode('utf-8'))
        sys.stdout.buffer.write(print_output + b'\n')
        sys.stdout.buffer.write(ACTION_OUTPUT_RESULT_DELIMITER.encode('utf-8'))
        sys.stdout.flush()

    def _get_action_instance(self):
        if False:
            i = 10
            return i + 15
        try:
            actions_cls = action_loader.register_plugin(Action, self._file_path)
        except Exception as e:
            tb_msg = traceback.format_exc()
            msg = 'Failed to load action class from file "%s" (action file most likely doesn\'t exist or contains invalid syntax): %s' % (self._file_path, six.text_type(e))
            msg += '\n\n' + tb_msg
            exc_cls = type(e)
            raise exc_cls(msg)
        action_cls = actions_cls[0] if actions_cls and len(actions_cls) > 0 else None
        if not action_cls:
            raise Exception('File "%s" has no action class or the file doesn\'t exist.' % self._file_path)
        self._class_name = action_cls.__name__
        action_service = ActionService(action_wrapper=self)
        action_instance = get_action_class_instance(action_cls=action_cls, config=self._config, action_service=action_service)
        return action_instance
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python action runner process wrapper')
    parser.add_argument('--pack', required=True, help='Name of the pack this action belongs to')
    parser.add_argument('--file-path', required=True, help='Path to the action module')
    parser.add_argument('--config', required=False, help='Pack config serialized as JSON')
    parser.add_argument('--parameters', required=False, help='Serialized action parameters')
    parser.add_argument('--stdin-parameters', required=False, action='store_true', help='Serialized action parameters via stdin')
    parser.add_argument('--user', required=False, help='User who triggered the action execution')
    parser.add_argument('--parent-args', required=False, help='Command line arguments passed to the parent process serialized as  JSON')
    parser.add_argument('--log-level', required=False, default=PYTHON_RUNNER_DEFAULT_LOG_LEVEL, help='Log level for actions')
    args = parser.parse_args()
    config = orjson.loads(args.config) if args.config else {}
    user = args.user
    parent_args = orjson.loads(args.parent_args) if args.parent_args else []
    log_level = args.log_level
    if not isinstance(config, dict):
        raise TypeError(f'Pack config needs to be a dictionary (was {type(config)}).')
    parameters = {}
    if args.parameters:
        LOG.debug('Getting parameters from argument')
        args_parameters = args.parameters
        args_parameters = orjson.loads(args_parameters) if args_parameters else {}
        parameters.update(args_parameters)
    if args.stdin_parameters:
        LOG.debug('Getting parameters from stdin')
        (i, _, _) = select.select([sys.stdin], [], [], READ_STDIN_INPUT_TIMEOUT)
        if not i:
            raise ValueError('No input received and timed out while waiting for parameters from stdin')
        stdin_data = sys.stdin.readline().strip()
        if not stdin_data:
            raise ValueError('Received no valid parameters data from sys.stdin')
        try:
            stdin_parameters = orjson.loads(stdin_data)
            stdin_parameters = stdin_parameters.get('parameters', {})
        except Exception as e:
            msg = 'Failed to parse parameters from stdin. Expected a JSON object with "parameters" attribute: %s' % six.text_type(e)
            raise ValueError(msg)
        parameters.update(stdin_parameters)
    LOG.debug('Received parameters: %s', parameters)
    if not isinstance(parent_args, list):
        raise TypeError(f'The parent_args is not a list (was {type(parent_args)}).')
    obj = PythonActionWrapper(pack=args.pack, file_path=args.file_path, config=config, parameters=parameters, user=user, parent_args=parent_args, log_level=log_level)
    obj.run()