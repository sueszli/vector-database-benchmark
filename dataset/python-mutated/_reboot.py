"""Reboot action for Windows hosts

This contains the code to reboot a Windows host for use by other action plugins
in this collection. Right now it should only be used in this collection as the
interface is not final and count be subject to change.
"""
from __future__ import annotations
import datetime
import json
import random
import time
import traceback
import uuid
import typing as t
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
from ansible_collections.ansible.windows.plugins.plugin_utils._quote import quote_pwsh
try:
    from requests.exceptions import RequestException
except ImportError:
    RequestException = AnsibleConnectionFailure
_LOGON_UI_KEY = 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon\\AutoLogonChecked'
_DEFAULT_BOOT_TIME_COMMAND = '(Get-CimInstance -ClassName Win32_OperatingSystem -Property LastBootUpTime).LastBootUpTime.ToFileTime()'
T = t.TypeVar('T')
display = Display()

class _ReturnResultException(Exception):
    """Used to sneak results back to the return dict from an exception"""

    def __init__(self, msg, **result):
        if False:
            i = 10
            return i + 15
        super().__init__(msg)
        self.result = result

class _TestCommandFailure(Exception):
    """Differentiates between a connection failure and just a command assertion failure during the reboot loop"""

def reboot_host(task_action: str, connection: ConnectionBase, boot_time_command: str=_DEFAULT_BOOT_TIME_COMMAND, connect_timeout: int=5, msg: str='Reboot initiated by Ansible', post_reboot_delay: int=0, pre_reboot_delay: int=2, reboot_timeout: int=600, test_command: t.Optional[str]=None) -> t.Dict[str, t.Any]:
    if False:
        for i in range(10):
            print('nop')
    "Reboot a Windows Host.\n\n    Used by action plugins in ansible.windows to reboot a Windows host. It\n    takes in the connection plugin so it can run the commands on the targeted\n    host and monitor the reboot process. The return dict will have the\n    following keys set:\n\n        changed: Whether a change occurred (reboot was done)\n        elapsed: Seconds elapsed between the reboot and it coming back online\n        failed: Whether a failure occurred\n        unreachable: Whether it failed to connect to the host on the first cmd\n        rebooted: Whether the host was rebooted\n\n    When failed=True there may be more keys to give some information around\n    the failure like msg, exception. There are other keys that might be\n    returned as well but they are dependent on the failure that occurred.\n\n    Verbosity levels used:\n        2: Message when each reboot step is completed\n        4: Connection plugin operations and their results\n        5: Raw commands run and the results of those commands\n        Debug: Everything, very verbose\n\n    Args:\n        task_action: The name of the action plugin that is running for logging.\n        connection: The connection plugin to run the reboot commands on.\n        boot_time_command: The command to run when getting the boot timeout.\n        connect_timeout: Override the connection timeout of the connection\n            plugin when polling the rebooted host.\n        msg: The message to display to interactive users when rebooting the\n            host.\n        post_reboot_delay: Seconds to wait after sending the reboot command\n            before checking to see if it has returned.\n        pre_reboot_delay: Seconds to wait when sending the reboot command.\n        reboot_timeout: Seconds to wait while polling for the host to come\n            back online.\n        test_command: Command to run when the host is back online and\n            determines the machine is ready for management. When not defined\n            the default command should wait until the reboot is complete and\n            all pre-login configuration has completed.\n\n    Returns:\n        (Dict[str, Any]): The return result as a dictionary. Use the 'failed'\n            key to determine if there was a failure or not.\n    "
    result: t.Dict[str, t.Any] = {'changed': False, 'elapsed': 0, 'failed': False, 'unreachable': False, 'rebooted': False}
    host_context = {'do_close_on_reset': True}
    try:
        previous_boot_time = _do_until_success_or_retry_limit(task_action, connection, host_context, 'pre-reboot boot time check', 3, _get_system_boot_time, task_action, connection, boot_time_command)
    except Exception as e:
        if isinstance(e, _ReturnResultException):
            result.update(e.result)
        if isinstance(e, AnsibleConnectionFailure):
            result['unreachable'] = True
        else:
            result['failed'] = True
        result['msg'] = str(e)
        result['exception'] = traceback.format_exc()
        return result
    original_connection_timeout: t.Optional[float] = None
    try:
        original_connection_timeout = connection.get_option('connection_timeout')
        display.vvvv(f'{task_action}: saving original connection_timeout of {original_connection_timeout}')
    except KeyError:
        display.vvvv(f'{task_action}: connection_timeout connection option has not been set')
    reboot_command = '$ErrorActionPreference = \'Continue\'\n\nif ($%s) {\n    Remove-Item -LiteralPath \'%s\' -Force -ErrorAction SilentlyContinue\n}\n\n$stdout = $null\n$stderr = . { shutdown.exe /r /t %s /c %s | Set-Variable stdout } 2>&1 | ForEach-Object ToString\n\nConvertTo-Json -Compress -InputObject @{\n    stdout = (@($stdout) -join "`n")\n    stderr = (@($stderr) -join "`n")\n    rc = $LASTEXITCODE\n}\n' % (str(not test_command), _LOGON_UI_KEY, int(pre_reboot_delay), quote_pwsh(msg))
    expected_test_result = None
    if not test_command:
        expected_test_result = f'success-{uuid.uuid4()}'
        test_command = f"Get-Item -LiteralPath '{_LOGON_UI_KEY}' -ErrorAction Stop; '{expected_test_result}'"
    start = None
    try:
        _perform_reboot(task_action, connection, reboot_command)
        start = datetime.datetime.utcnow()
        result['changed'] = True
        result['rebooted'] = True
        if post_reboot_delay != 0:
            display.vv(f'{task_action}: waiting an additional {post_reboot_delay} seconds')
            time.sleep(post_reboot_delay)
        display.vv(f'{task_action} validating reboot')
        _do_until_success_or_timeout(task_action, connection, host_context, 'last boot time check', reboot_timeout, _check_boot_time, task_action, connection, host_context, previous_boot_time, boot_time_command, connect_timeout)
        if original_connection_timeout is not None:
            _set_connection_timeout(task_action, connection, host_context, original_connection_timeout)
        display.vv(f'{task_action} running post reboot test command')
        _do_until_success_or_timeout(task_action, connection, host_context, 'post-reboot test command', reboot_timeout, _run_test_command, task_action, connection, test_command, expected=expected_test_result)
        display.vv(f'{task_action}: system successfully rebooted')
    except Exception as e:
        if isinstance(e, _ReturnResultException):
            result.update(e.result)
        result['failed'] = True
        result['msg'] = str(e)
        result['exception'] = traceback.format_exc()
    if start:
        elapsed = datetime.datetime.utcnow() - start
        result['elapsed'] = elapsed.seconds
    return result

def _check_boot_time(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], previous_boot_time: int, boot_time_command: str, timeout: int):
    if False:
        i = 10
        return i + 15
    'Checks the system boot time has been changed or not'
    display.vvvv('%s: attempting to get system boot time' % task_action)
    if timeout:
        _set_connection_timeout(task_action, connection, host_context, timeout)
    current_boot_time = _get_system_boot_time(task_action, connection, boot_time_command)
    if current_boot_time == previous_boot_time:
        raise _TestCommandFailure('boot time has not changed')

def _do_until_success_or_retry_limit(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], action_desc: str, retries: int, func: t.Callable[..., T], *args: t.Any, **kwargs: t.Any) -> t.Optional[T]:
    if False:
        while True:
            i = 10
    'Runs the function multiple times ignoring errors until the retry limit is hit'

    def wait_condition(idx):
        if False:
            for i in range(10):
                print('nop')
        return idx < retries
    return _do_until_success_or_condition(task_action, connection, host_context, action_desc, wait_condition, func, *args, **kwargs)

def _do_until_success_or_timeout(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], action_desc: str, timeout: float, func: t.Callable[..., T], *args: t.Any, **kwargs: t.Any) -> t.Optional[T]:
    if False:
        while True:
            i = 10
    'Runs the function multiple times ignoring errors until a timeout occurs'
    max_end_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=timeout)

    def wait_condition(idx):
        if False:
            print('Hello World!')
        return datetime.datetime.utcnow() < max_end_time
    try:
        return _do_until_success_or_condition(task_action, connection, host_context, action_desc, wait_condition, func, *args, **kwargs)
    except Exception:
        raise Exception('Timed out waiting for %s (timeout=%s)' % (action_desc, timeout))

def _do_until_success_or_condition(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], action_desc: str, condition: t.Callable[[int], bool], func: t.Callable[..., T], *args: t.Any, **kwargs: t.Any) -> t.Optional[T]:
    if False:
        return 10
    'Runs the function multiple times ignoring errors until the condition is false'
    fail_count = 0
    max_fail_sleep = 12
    reset_required = False
    last_error = None
    while fail_count == 0 or condition(fail_count):
        try:
            if reset_required:
                _reset_connection(task_action, connection, host_context)
                reset_required = False
            else:
                res = func(*args, **kwargs)
                display.vvvvv('%s: %s success' % (task_action, action_desc))
                return res
        except Exception as e:
            last_error = e
            if not isinstance(e, _TestCommandFailure):
                reset_required = True
            random_int = random.randint(0, 1000) / 1000
            fail_sleep = 2 ** fail_count + random_int
            if fail_sleep > max_fail_sleep:
                fail_sleep = max_fail_sleep + random_int
            try:
                error = str(e).splitlines()[-1]
            except IndexError:
                error = str(e)
            display.vvvvv("{action}: {desc} fail {e_type} '{err}', retrying in {sleep:.4} seconds...\n{tcb}".format(action=task_action, desc=action_desc, e_type=type(e).__name__, err=error, sleep=fail_sleep, tcb=traceback.format_exc()))
            fail_count += 1
            time.sleep(fail_sleep)
    if last_error:
        raise last_error
    return None

def _execute_command(task_action: str, connection: ConnectionBase, command: str) -> t.Tuple[int, str, str]:
    if False:
        for i in range(10):
            print('nop')
    'Runs a command on the Windows host and returned the result'
    display.vvvvv(f'{task_action}: running command: {command}')
    command = connection._shell._encode_script(command)
    try:
        (rc, stdout, stderr) = connection.exec_command(command, in_data=None, sudoable=False)
    except RequestException as e:
        raise AnsibleConnectionFailure(f'Failed to connect to the host: {e}')
    rc = rc or 0
    stdout = to_text(stdout, errors='surrogate_or_strict').strip()
    stderr = to_text(stderr, errors='surrogate_or_strict').strip()
    display.vvvvv(f'{task_action}: command result - rc: {rc}, stdout: {stdout}, stderr: {stderr}')
    return (rc, stdout, stderr)

def _get_system_boot_time(task_action: str, connection: ConnectionBase, boot_time_command: str) -> str:
    if False:
        while True:
            i = 10
    'Gets a unique identifier to represent the boot time of the Windows host'
    display.vvvv(f'{task_action}: getting boot time')
    (rc, stdout, stderr) = _execute_command(task_action, connection, boot_time_command)
    if rc != 0:
        msg = f'{task_action}: failed to get host boot time info'
        raise _ReturnResultException(msg, rc=rc, stdout=stdout, stderr=stderr)
    display.vvvv(f'{task_action}: last boot time: {stdout}')
    return stdout

def _perform_reboot(task_action: str, connection: ConnectionBase, reboot_command: str, handle_abort: bool=True) -> None:
    if False:
        while True:
            i = 10
    'Runs the reboot command'
    display.vv(f'{task_action}: rebooting server...')
    stdout = stderr = None
    try:
        (rc, stdout, stderr) = _execute_command(task_action, connection, reboot_command)
    except AnsibleConnectionFailure as e:
        display.vvvv(f'{task_action}: AnsibleConnectionFailure caught and handled: {e}')
        rc = 0
    if stdout:
        try:
            reboot_result = json.loads(stdout)
        except getattr(json.decoder, 'JSONDecodeError', ValueError):
            pass
        else:
            stdout = reboot_result.get('stdout', stdout)
            stderr = reboot_result.get('stderr', stderr)
            rc = int(reboot_result.get('rc', rc))
    if handle_abort and (rc == 1190 or (rc != 0 and stderr and ('(1190)' in stderr))):
        display.warning('A scheduled reboot was pre-empted by Ansible.')
        (rc, stdout, stderr) = _execute_command(task_action, connection, 'shutdown.exe /a')
        display.vvvv(f'{task_action}: result from trying to abort existing shutdown - rc: {rc}, stdout: {stdout}, stderr: {stderr}')
        return _perform_reboot(task_action, connection, reboot_command, handle_abort=False)
    if rc != 0:
        msg = f'{task_action}: Reboot command failed'
        raise _ReturnResultException(msg, rc=rc, stdout=stdout, stderr=stderr)

def _reset_connection(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], ignore_errors: bool=False) -> None:
    if False:
        while True:
            i = 10
    'Resets the connection handling any errors'

    def _wrap_conn_err(func, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            func(*args, **kwargs)
        except (AnsibleError, RequestException) as e:
            if ignore_errors:
                return False
            raise AnsibleError(e)
        return True
    if host_context['do_close_on_reset']:
        display.vvvv(f'{task_action}: closing connection plugin')
        try:
            success = _wrap_conn_err(connection.close)
        except Exception:
            host_context['do_close_on_reset'] = False
            raise
        host_context['do_close_on_reset'] = success
    display.vvvv(f'{task_action}: resetting connection plugin')
    try:
        _wrap_conn_err(connection.reset)
    except AttributeError:
        pass

def _run_test_command(task_action: str, connection: ConnectionBase, command: str, expected: t.Optional[str]=None) -> None:
    if False:
        while True:
            i = 10
    'Runs the user specified test command until the host is able to run it properly'
    display.vvvv(f'{task_action}: attempting post-reboot test command')
    (rc, stdout, stderr) = _execute_command(task_action, connection, command)
    if rc != 0:
        msg = f'{task_action}: Test command failed - rc: {rc}, stdout: {stdout}, stderr: {stderr}'
        raise _TestCommandFailure(msg)
    if expected and expected not in stdout:
        msg = f"{task_action}: Test command failed - '{expected}' was not in stdout: {stdout}"
        raise _TestCommandFailure(msg)

def _set_connection_timeout(task_action: str, connection: ConnectionBase, host_context: t.Dict[str, t.Any], timeout: float) -> None:
    if False:
        i = 10
        return i + 15
    'Sets the connection plugin connection_timeout option and resets the connection'
    try:
        current_connection_timeout = connection.get_option('connection_timeout')
    except KeyError:
        return
    if timeout == current_connection_timeout:
        return
    display.vvvv(f'{task_action}: setting connect_timeout {timeout}')
    connection.set_option('connection_timeout', timeout)
    _reset_connection(task_action, connection, host_context, ignore_errors=True)