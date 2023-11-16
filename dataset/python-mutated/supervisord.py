"""
Provide the service module for system supervisord or supervisord in a
virtualenv
"""
import configparser
import os
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError, CommandNotFoundError

def __virtual__():
    if False:
        i = 10
        return i + 15
    return True

def _get_supervisorctl_bin(bin_env):
    if False:
        i = 10
        return i + 15
    '\n    Return supervisorctl command to call, either from a virtualenv, an argument\n    passed in, or from the global modules options\n    '
    cmd = 'supervisorctl'
    if not bin_env:
        which_result = __salt__['cmd.which_bin']([cmd])
        if which_result is None:
            raise CommandNotFoundError('Could not find a `{}` binary'.format(cmd))
        return which_result
    if os.path.isdir(bin_env):
        cmd_bin = os.path.join(bin_env, 'bin', cmd)
        if os.path.isfile(cmd_bin):
            return cmd_bin
        raise CommandNotFoundError('Could not find a `{}` binary'.format(cmd))
    return bin_env

def _ctl_cmd(cmd, name, conf_file, bin_env):
    if False:
        return 10
    '\n    Return the command list to use\n    '
    ret = [_get_supervisorctl_bin(bin_env)]
    if conf_file is not None:
        ret += ['-c', conf_file]
    ret.append(cmd)
    if name:
        ret.append(name)
    return ret

def _get_return(ret):
    if False:
        while True:
            i = 10
    retmsg = ret['stdout']
    if ret['retcode'] != 0:
        if 'ERROR' not in retmsg:
            retmsg = 'ERROR: {}'.format(retmsg)
    return retmsg

def start(name='all', user=None, conf_file=None, bin_env=None):
    if False:
        print('Hello World!')
    "\n    Start the named service.\n    Process group names should not include a trailing asterisk.\n\n    user\n        user to run supervisorctl as\n    conf_file\n        path to supervisord config file\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.start <service>\n        salt '*' supervisord.start <group>:\n    "
    if name.endswith(':*'):
        name = name[:-1]
    ret = __salt__['cmd.run_all'](_ctl_cmd('start', name, conf_file, bin_env), runas=user, python_shell=False)
    return _get_return(ret)

def restart(name='all', user=None, conf_file=None, bin_env=None):
    if False:
        return 10
    "\n    Restart the named service.\n    Process group names should not include a trailing asterisk.\n\n    user\n        user to run supervisorctl as\n    conf_file\n        path to supervisord config file\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.restart <service>\n        salt '*' supervisord.restart <group>:\n    "
    if name.endswith(':*'):
        name = name[:-1]
    ret = __salt__['cmd.run_all'](_ctl_cmd('restart', name, conf_file, bin_env), runas=user, python_shell=False)
    return _get_return(ret)

def stop(name='all', user=None, conf_file=None, bin_env=None):
    if False:
        i = 10
        return i + 15
    "\n    Stop the named service.\n    Process group names should not include a trailing asterisk.\n\n    user\n        user to run supervisorctl as\n    conf_file\n        path to supervisord config file\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.stop <service>\n        salt '*' supervisord.stop <group>:\n    "
    if name.endswith(':*'):
        name = name[:-1]
    ret = __salt__['cmd.run_all'](_ctl_cmd('stop', name, conf_file, bin_env), runas=user, python_shell=False)
    return _get_return(ret)

def add(name, user=None, conf_file=None, bin_env=None):
    if False:
        return 10
    "\n    Activates any updates in config for process/group.\n\n    user\n        user to run supervisorctl as\n    conf_file\n        path to supervisord config file\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.add <name>\n    "
    if name.endswith(':'):
        name = name[:-1]
    elif name.endswith(':*'):
        name = name[:-2]
    ret = __salt__['cmd.run_all'](_ctl_cmd('add', name, conf_file, bin_env), runas=user, python_shell=False)
    return _get_return(ret)

def remove(name, user=None, conf_file=None, bin_env=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Removes process/group from active config\n\n    user\n        user to run supervisorctl as\n    conf_file\n        path to supervisord config file\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.remove <name>\n    "
    if name.endswith(':'):
        name = name[:-1]
    elif name.endswith(':*'):
        name = name[:-2]
    ret = __salt__['cmd.run_all'](_ctl_cmd('remove', name, conf_file, bin_env), runas=user, python_shell=False)
    return _get_return(ret)

def reread(user=None, conf_file=None, bin_env=None):
    if False:
        while True:
            i = 10
    "\n    Reload the daemon's configuration files\n\n    user\n        user to run supervisorctl as\n    conf_file\n        path to supervisord config file\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.reread\n    "
    ret = __salt__['cmd.run_all'](_ctl_cmd('reread', None, conf_file, bin_env), runas=user, python_shell=False)
    return _get_return(ret)

def update(user=None, conf_file=None, bin_env=None, name=None):
    if False:
        print('Hello World!')
    "\n    Reload config and add/remove/update as necessary\n\n    user\n        user to run supervisorctl as\n    conf_file\n        path to supervisord config file\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n    name\n        name of the process group to update. if none then update any\n        process group that has changes\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.update\n    "
    if isinstance(name, str):
        if name.endswith(':'):
            name = name[:-1]
        elif name.endswith(':*'):
            name = name[:-2]
    ret = __salt__['cmd.run_all'](_ctl_cmd('update', name, conf_file, bin_env), runas=user, python_shell=False)
    return _get_return(ret)

def status(name=None, user=None, conf_file=None, bin_env=None):
    if False:
        while True:
            i = 10
    "\n    List programs and its state\n\n    user\n        user to run supervisorctl as\n    conf_file\n        path to supervisord config file\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.status\n    "
    all_process = {}
    for line in status_raw(name, user, conf_file, bin_env).splitlines():
        if len(line.split()) > 2:
            (process, state, reason) = line.split(None, 2)
        else:
            (process, state, reason) = line.split() + ['']
        all_process[process] = {'state': state, 'reason': reason}
    return all_process

def status_raw(name=None, user=None, conf_file=None, bin_env=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Display the raw output of status\n\n    user\n        user to run supervisorctl as\n    conf_file\n        path to supervisord config file\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.status_raw\n    "
    ret = __salt__['cmd.run_all'](_ctl_cmd('status', name, conf_file, bin_env), runas=user, python_shell=False)
    return _get_return(ret)

def custom(command, user=None, conf_file=None, bin_env=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run any custom supervisord command\n\n    user\n        user to run supervisorctl as\n    conf_file\n        path to supervisord config file\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' supervisord.custom "mstop \'*gunicorn*\'"\n    '
    ret = __salt__['cmd.run_all'](_ctl_cmd(command, None, conf_file, bin_env), runas=user, python_shell=False)
    return _get_return(ret)

def _read_config(conf_file=None):
    if False:
        i = 10
        return i + 15
    '\n    Reads the config file using configparser\n    '
    if conf_file is None:
        paths = ('/etc/supervisor/supervisord.conf', '/etc/supervisord.conf')
        for path in paths:
            if os.path.exists(path):
                conf_file = path
                break
    if conf_file is None:
        raise CommandExecutionError('No suitable config file found')
    config = configparser.ConfigParser()
    try:
        config.read(conf_file)
    except OSError as exc:
        raise CommandExecutionError('Unable to read from {}: {}'.format(conf_file, exc))
    return config

def options(name, conf_file=None):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2014.1.0\n\n    Read the config file and return the config options for a given process\n\n    name\n        Name of the configured process\n    conf_file\n        path to supervisord config file\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.options foo\n    "
    config = _read_config(conf_file)
    section_name = 'program:{}'.format(name)
    if section_name not in config.sections():
        raise CommandExecutionError("Process '{}' not found".format(name))
    ret = {}
    for (key, val) in config.items(section_name):
        val = salt.utils.stringutils.to_num(val.split(';')[0].strip())
        if isinstance(val, str):
            if val.lower() == 'true':
                val = True
            elif val.lower() == 'false':
                val = False
        ret[key] = val
    return ret

def status_bool(name, expected_state=None, user=None, conf_file=None, bin_env=None):
    if False:
        return 10
    "\n    Check for status of a specific supervisord process and return boolean result.\n\n    name\n        name of the process to check\n\n    expected_state\n        search for a specific process state. If set to ``None`` - any process state will match.\n\n    user\n        user to run supervisorctl as\n\n    conf_file\n        path to supervisord config file\n\n    bin_env\n        path to supervisorctl bin or path to virtualenv with supervisor\n        installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' supervisord.status_bool nginx expected_state='RUNNING'\n    "
    cmd = 'status {}'.format(name)
    for line in custom(cmd, user, conf_file, bin_env).splitlines():
        if len(line.split()) > 2:
            (process, state, reason) = line.split(None, 2)
        else:
            (process, state, reason) = line.split() + ['']
    if reason == '(no such process)' or process != name:
        return False
    if expected_state is None or state == expected_state:
        return True
    else:
        return False