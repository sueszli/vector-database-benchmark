"""
Helper functions for use by mac modules
.. versionadded:: 2016.3.0
"""
import logging
import os
import plistlib
import subprocess
import time
import xml.parsers.expat
import salt.grains.extra
import salt.modules.cmdmod
import salt.utils.args
import salt.utils.files
import salt.utils.path
import salt.utils.platform
import salt.utils.stringutils
import salt.utils.timed_subprocess
from salt.exceptions import CommandExecutionError, SaltInvocationError, TimedProcTimeoutError
try:
    import pwd
except ImportError:
    pass
DEFAULT_SHELL = salt.grains.extra.shell()['shell']
log = logging.getLogger(__name__)
__virtualname__ = 'mac_utils'
__salt__ = {'cmd.run_all': salt.modules.cmdmod._run_all_quiet, 'cmd.run': salt.modules.cmdmod._run_quiet}

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Load only on Mac OS\n    '
    if not salt.utils.platform.is_darwin():
        return (False, 'The mac_utils utility could not be loaded: utility only works on MacOS systems.')
    return __virtualname__

def _run_all(cmd):
    if False:
        while True:
            i = 10
    '\n\n    Args:\n        cmd:\n\n    Returns:\n\n    '
    if not isinstance(cmd, list):
        cmd = salt.utils.args.shlex_split(cmd, posix=False)
    for (idx, item) in enumerate(cmd):
        if not isinstance(cmd[idx], str):
            cmd[idx] = str(cmd[idx])
    cmd = ' '.join(cmd)
    run_env = os.environ.copy()
    kwargs = {'cwd': None, 'shell': DEFAULT_SHELL, 'env': run_env, 'stdin': None, 'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE, 'with_communicate': True, 'timeout': None, 'bg': False}
    try:
        proc = salt.utils.timed_subprocess.TimedProc(cmd, **kwargs)
    except OSError as exc:
        raise CommandExecutionError("Unable to run command '{}' with the context '{}', reason: {}".format(cmd, kwargs, exc))
    ret = {}
    try:
        proc.run()
    except TimedProcTimeoutError as exc:
        ret['stdout'] = str(exc)
        ret['stderr'] = ''
        ret['retcode'] = 1
        ret['pid'] = proc.process.pid
        return ret
    (out, err) = (proc.stdout, proc.stderr)
    if out is not None:
        out = salt.utils.stringutils.to_str(out).rstrip()
    if err is not None:
        err = salt.utils.stringutils.to_str(err).rstrip()
    ret['pid'] = proc.process.pid
    ret['retcode'] = proc.process.returncode
    ret['stdout'] = out
    ret['stderr'] = err
    return ret

def _check_launchctl_stderr(ret):
    if False:
        print('Hello World!')
    '\n    helper class to check the launchctl stderr.\n    launchctl does not always return bad exit code\n    if there is a failure\n    '
    err = ret['stderr'].lower()
    if 'service is disabled' in err:
        return True
    return False

def execute_return_success(cmd):
    if False:
        while True:
            i = 10
    '\n    Executes the passed command. Returns True if successful\n\n    :param str cmd: The command to run\n\n    :return: True if successful, otherwise False\n    :rtype: bool\n\n    :raises: Error if command fails or is not supported\n    '
    ret = _run_all(cmd)
    log.debug('Execute return success %s: %r', cmd, ret)
    if ret['retcode'] != 0 or 'not supported' in ret['stdout'].lower():
        msg = 'Command Failed: {}\n'.format(cmd)
        msg += 'Return Code: {}\n'.format(ret['retcode'])
        msg += 'Output: {}\n'.format(ret['stdout'])
        msg += 'Error: {}\n'.format(ret['stderr'])
        raise CommandExecutionError(msg)
    return True

def execute_return_result(cmd):
    if False:
        return 10
    '\n    Executes the passed command. Returns the standard out if successful\n\n    :param str cmd: The command to run\n\n    :return: The standard out of the command if successful, otherwise returns\n    an error\n    :rtype: str\n\n    :raises: Error if command fails or is not supported\n    '
    ret = _run_all(cmd)
    if ret['retcode'] != 0 or 'not supported' in ret['stdout'].lower():
        msg = 'Command Failed: {}\n'.format(cmd)
        msg += 'Return Code: {}\n'.format(ret['retcode'])
        msg += 'Output: {}\n'.format(ret['stdout'])
        msg += 'Error: {}\n'.format(ret['stderr'])
        raise CommandExecutionError(msg)
    return ret['stdout']

def parse_return(data):
    if False:
        return 10
    '\n    Returns the data portion of a string that is colon separated.\n\n    :param str data: The string that contains the data to be parsed. Usually the\n    standard out from a command\n\n    For example:\n    ``Time Zone: America/Denver``\n    will return:\n    ``America/Denver``\n    '
    if ': ' in data:
        return data.split(': ')[1]
    if ':\n' in data:
        return data.split(':\n')[1]
    else:
        return data

def validate_enabled(enabled):
    if False:
        print('Hello World!')
    '\n    Helper function to validate the enabled parameter. Boolean values are\n    converted to "on" and "off". String values are checked to make sure they are\n    either "on" or "off"/"yes" or "no". Integer ``0`` will return "off". All\n    other integers will return "on"\n\n    :param enabled: Enabled can be boolean True or False, Integers, or string\n    values "on" and "off"/"yes" and "no".\n    :type: str, int, bool\n\n    :return: "on" or "off" or errors\n    :rtype: str\n    '
    if isinstance(enabled, str):
        if enabled.lower() not in ['on', 'off', 'yes', 'no']:
            msg = "\nMac Power: Invalid String Value for Enabled.\nString values must be 'on' or 'off'/'yes' or 'no'.\nPassed: {}".format(enabled)
            raise SaltInvocationError(msg)
        return 'on' if enabled.lower() in ['on', 'yes'] else 'off'
    return 'on' if bool(enabled) else 'off'

def confirm_updated(value, check_fun, normalize_ret=False, wait=5):
    if False:
        for i in range(10):
            print('nop')
    "\n    Wait up to ``wait`` seconds for a system parameter to be changed before\n    deciding it hasn't changed.\n\n    :param str value: The value indicating a successful change\n\n    :param function check_fun: The function whose return is compared with\n        ``value``\n\n    :param bool normalize_ret: Whether to normalize the return from\n        ``check_fun`` with ``validate_enabled``\n\n    :param int wait: The maximum amount of seconds to wait for a system\n        parameter to change\n    "
    for i in range(wait):
        state = validate_enabled(check_fun()) if normalize_ret else check_fun()
        log.debug('Confirm update try: %d func:%r state:%s value:%s', i, check_fun, state, value)
        if value in state:
            return True
        time.sleep(1)
    return False

def launchctl(sub_cmd, *args, **kwargs):
    if False:
        return 10
    "\n    Run a launchctl command and raise an error if it fails\n\n    Args: additional args are passed to launchctl\n        sub_cmd (str): Sub command supplied to launchctl\n\n    Kwargs: passed to ``cmd.run_all``\n        return_stdout (bool): A keyword argument. If true return the stdout of\n            the launchctl command\n\n    Returns:\n        bool: ``True`` if successful\n        str: The stdout of the launchctl command if requested\n\n    Raises:\n        CommandExecutionError: If command fails\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        import salt.utils.mac_service\n        salt.utils.mac_service.launchctl('debug', 'org.cups.cupsd')\n    "
    return_stdout = kwargs.pop('return_stdout', False)
    cmd = ['launchctl', sub_cmd]
    cmd.extend(args)
    if sub_cmd == 'bootout':
        kwargs['success_retcodes'] = [36]
    kwargs['python_shell'] = False
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    ret = __salt__['cmd.run_all'](cmd, **kwargs)
    error = _check_launchctl_stderr(ret)
    if ret['retcode'] or error:
        out = 'Failed to {} service:\n'.format(sub_cmd)
        out += 'stdout: {}\n'.format(ret['stdout'])
        out += 'stderr: {}\n'.format(ret['stderr'])
        out += 'retcode: {}'.format(ret['retcode'])
        raise CommandExecutionError(out)
    else:
        return ret['stdout'] if return_stdout else True

def _read_plist_file(root, file_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    :param root: The root path of the plist file\n    :param file_name: The name of the plist file\n    :return:  An empty dictionary if the plist file was invalid, otherwise, a dictionary with plist data\n    '
    file_path = os.path.join(root, file_name)
    log.debug('read_plist: Gathering service info for %s', file_path)
    if not file_path.lower().endswith('.plist'):
        log.debug('read_plist: Not a plist file: %s', file_path)
        return {}
    if not os.path.exists(os.path.realpath(file_path)):
        log.warning('read_plist: Ignoring broken symlink: %s', file_path)
        return {}
    try:
        with salt.utils.files.fopen(file_path, 'rb') as handle:
            plist = plistlib.load(handle)
    except plistlib.InvalidFileException:
        log.warning('read_plist: Unable to parse "%s" as it is invalid XML: InvalidFileException.', file_path)
        return {}
    except ValueError as err:
        log.debug("Caught ValueError: '%s', while trying to parse '%s'.", err, file_path)
        return {}
    except xml.parsers.expat.ExpatError:
        log.warning('read_plist: Unable to parse "%s" as it is invalid XML: xml.parsers.expat.ExpatError.', file_path)
        return {}
    if 'Label' not in plist:
        log.debug('read_plist: Service does not contain a Label key. Skipping %s.', file_path)
        return {}
    return {'file_name': file_name, 'file_path': file_path, 'plist': plist}

def _available_services(refresh=False):
    if False:
        print('Hello World!')
    '\n    This is a helper function for getting the available macOS services.\n\n    The strategy is to look through the known system locations for\n    launchd plist files, parse them, and use their information for\n    populating the list of services. Services can run without a plist\n    file present, but normally services which have an automated startup\n    will have a plist file, so this is a minor compromise.\n    '
    if 'available_services' in __context__ and (not refresh):
        log.debug('Found context for available services.')
        __context__['using_cached_services'] = True
        return __context__['available_services']
    launchd_paths = {'/Library/LaunchAgents', '/Library/LaunchDaemons', '/System/Library/LaunchAgents', '/System/Library/LaunchDaemons'}
    agent_path = '/Users/{}/Library/LaunchAgents'
    launchd_paths.update({agent_path.format(user) for user in os.listdir('/Users/') if os.path.isdir(agent_path.format(user))})
    result = {}
    for launch_dir in launchd_paths:
        for (root, dirs, files) in salt.utils.path.os_walk(launch_dir):
            for file_name in files:
                data = _read_plist_file(root, file_name)
                if data:
                    result[data['plist']['Label'].lower()] = data
    __context__['available_services'] = result
    __context__['using_cached_services'] = False
    return result

def available_services(refresh=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a dictionary of all available services on the system\n\n    :param bool refresh: If you wish to refresh the available services\n    as this data is cached on the first run.\n\n    Returns:\n        dict: All available services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        import salt.utils.mac_service\n        salt.utils.mac_service.available_services()\n    '
    log.debug('Loading available services')
    return _available_services(refresh)

def console_user(username=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Gets the UID or Username of the current console user.\n\n    :return: The uid or username of the console user.\n\n    :param bool username: Whether to return the username of the console\n    user instead of the UID. Defaults to False\n\n    :rtype: Interger of the UID, or a string of the username.\n\n    Raises:\n        CommandExecutionError: If we fail to get the UID.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        import salt.utils.mac_service\n        salt.utils.mac_service.console_user()\n    '
    try:
        uid = os.stat('/dev/console')[4]
    except (OSError, IndexError):
        raise CommandExecutionError('Failed to get a UID for the console user.')
    if username:
        return pwd.getpwuid(uid)[0]
    return uid

def git_is_stub():
    if False:
        return 10
    '\n    Return whether macOS git is the standard OS stub or a real binary.\n    '
    try:
        cmd = ['/usr/bin/xcode-select', '-p']
        _ = subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
        log.debug('Xcode command line tools present')
        return False
    except subprocess.CalledProcessError:
        log.debug('Xcode command line tools not present')
        return True