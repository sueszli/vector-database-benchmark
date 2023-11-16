from __future__ import annotations
from datetime import datetime, timezone
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.action import ActionBase
from ansible.plugins.action.reboot import ActionModule as RebootActionModule
from ansible.utils.display import Display
display = Display()

class TimedOutException(Exception):
    pass

class ActionModule(RebootActionModule, ActionBase):
    TRANSFERS_FILES = False
    _VALID_ARGS = frozenset(('connect_timeout', 'connect_timeout_sec', 'msg', 'post_reboot_delay', 'post_reboot_delay_sec', 'pre_reboot_delay', 'pre_reboot_delay_sec', 'reboot_timeout', 'reboot_timeout_sec', 'shutdown_timeout', 'shutdown_timeout_sec', 'test_command'))
    DEFAULT_BOOT_TIME_COMMAND = '(Get-WmiObject -ClassName Win32_OperatingSystem).LastBootUpTime'
    DEFAULT_CONNECT_TIMEOUT = 5
    DEFAULT_PRE_REBOOT_DELAY = 2
    DEFAULT_SUDOABLE = False
    DEFAULT_SHUTDOWN_COMMAND_ARGS = '/r /t {delay_sec} /c "{message}"'
    DEPRECATED_ARGS = {'shutdown_timeout': '2.5', 'shutdown_timeout_sec': '2.5'}

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ActionModule, self).__init__(*args, **kwargs)

    def get_distribution(self, task_vars):
        if False:
            print('Hello World!')
        return {'name': 'windows', 'version': '', 'family': ''}

    def get_shutdown_command(self, task_vars, distribution):
        if False:
            i = 10
            return i + 15
        return self.DEFAULT_SHUTDOWN_COMMAND

    def run_test_command(self, distribution, **kwargs):
        if False:
            while True:
                i = 10
        test_command = self._task.args.get('test_command', self.DEFAULT_TEST_COMMAND)
        kwargs['test_command'] = self._connection._shell._encode_script(test_command)
        super(ActionModule, self).run_test_command(distribution, **kwargs)

    def perform_reboot(self, task_vars, distribution):
        if False:
            i = 10
            return i + 15
        shutdown_command = self.get_shutdown_command(task_vars, distribution)
        shutdown_command_args = self.get_shutdown_command_args(distribution)
        reboot_command = self._connection._shell._encode_script('{0} {1}'.format(shutdown_command, shutdown_command_args))
        display.vvv('{action}: rebooting server...'.format(action=self._task.action))
        display.debug('{action}: distribution: {dist}'.format(action=self._task.action, dist=distribution))
        display.debug("{action}: rebooting server with command '{command}'".format(action=self._task.action, command=reboot_command))
        result = {}
        reboot_result = self._low_level_execute_command(reboot_command, sudoable=self.DEFAULT_SUDOABLE)
        result['start'] = datetime.now(timezone.utc)
        stdout = reboot_result['stdout']
        stderr = reboot_result['stderr']
        if reboot_result['rc'] == 1190 or (reboot_result['rc'] != 0 and '(1190)' in reboot_result['stderr']):
            display.warning('A scheduled reboot was pre-empted by Ansible.')
            result1 = self._low_level_execute_command(self._connection._shell._encode_script('shutdown /a'), sudoable=self.DEFAULT_SUDOABLE)
            result2 = self._low_level_execute_command(reboot_command, sudoable=self.DEFAULT_SUDOABLE)
            reboot_result['rc'] = result2['rc']
            stdout += result1['stdout'] + result2['stdout']
            stderr += result1['stderr'] + result2['stderr']
        if reboot_result['rc'] != 0:
            result['failed'] = True
            result['rebooted'] = False
            result['msg'] = 'Reboot command failed, error was: {stdout} {stderr}'.format(stdout=to_native(stdout.strip()), stderr=to_native(stderr.strip()))
            return result
        result['failed'] = False
        return result