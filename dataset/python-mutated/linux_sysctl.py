"""
Module for viewing and modifying sysctl parameters
"""
import logging
import os
import re
import salt.utils.data
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
import salt.utils.systemd
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'sysctl'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only run on Linux systems\n    '
    if __grains__['kernel'] != 'Linux':
        return (False, 'The linux_sysctl execution module cannot be loaded: only available on Linux systems.')
    return __virtualname__

def _which(cmd):
    if False:
        print('Hello World!')
    '\n    Utility function wrapper to error out early if a command is not found\n    '
    _cmd = salt.utils.path.which(cmd)
    if not _cmd:
        raise CommandExecutionError("Command '{}' cannot be found".format(cmd))
    return _cmd

def default_config():
    if False:
        print('Hello World!')
    "\n    Linux hosts using systemd 207 or later ignore ``/etc/sysctl.conf`` and only\n    load from ``/etc/sysctl.d/*.conf``. This function will do the proper checks\n    and return a default config file which will be valid for the Minion. Hosts\n    running systemd >= 207 will use ``/etc/sysctl.d/99-salt.conf``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt -G 'kernel:Linux' sysctl.default_config\n    "
    if salt.utils.systemd.booted(__context__) and salt.utils.systemd.version(__context__) >= 207:
        return '/etc/sysctl.d/99-salt.conf'
    return '/etc/sysctl.conf'

def show(config_file=False):
    if False:
        while True:
            i = 10
    "\n    Return a list of sysctl parameters for this minion\n\n    config: Pull the data from the system configuration file\n        instead of the live data.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.show\n    "
    ret = {}
    if config_file:
        if not os.path.exists(config_file):
            return []
        try:
            with salt.utils.files.fopen(config_file) as fp_:
                for line in fp_:
                    line = salt.utils.stringutils.to_str(line).strip()
                    if not line.startswith('#') and '=' in line:
                        (key, value) = line.split('=', 1)
                        ret[key.rstrip()] = value.lstrip()
        except OSError:
            log.error('Could not open sysctl file')
            return None
    else:
        _sysctl = '{}'.format(_which('sysctl'))
        cmd = [_sysctl, '-a']
        out = __salt__['cmd.run_stdout'](cmd, output_loglevel='trace')
        for line in out.splitlines():
            if not line or ' = ' not in line:
                continue
            comps = line.split(' = ', 1)
            ret[comps[0]] = comps[1]
    return ret

def get(name):
    if False:
        print('Hello World!')
    "\n    Return a single sysctl parameter for this minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.get net.ipv4.ip_forward\n    "
    _sysctl = '{}'.format(_which('sysctl'))
    cmd = [_sysctl, '-n', name]
    out = __salt__['cmd.run'](cmd, python_shell=False)
    return out

def assign(name, value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Assign a single sysctl parameter for this minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.assign net.ipv4.ip_forward 1\n    "
    value = str(value)
    tran_tab = name.translate(''.maketrans('./', '/.'))
    sysctl_file = '/proc/sys/{}'.format(tran_tab)
    if not os.path.exists(sysctl_file):
        raise CommandExecutionError('sysctl {} does not exist'.format(name))
    ret = {}
    _sysctl = '{}'.format(_which('sysctl'))
    cmd = [_sysctl, '-w', '{}={}'.format(name, value)]
    data = __salt__['cmd.run_all'](cmd, python_shell=False)
    out = data['stdout']
    err = data['stderr']
    regex = re.compile('^{}\\s+=\\s+{}$'.format(re.escape(name), re.escape(value)))
    if not regex.match(out) or 'Invalid argument' in str(err):
        if data['retcode'] != 0 and err:
            error = err
        else:
            error = out
        raise CommandExecutionError('sysctl -w failed: {}'.format(error))
    (new_name, new_value) = out.split(' = ', 1)
    ret[new_name] = new_value
    return ret

def _sanitize_sysctl_value(value):
    if False:
        while True:
            i = 10
    'Replace separating whitespaces by exactly one tab.\n\n    On Linux procfs, files such as /proc/sys/net/ipv4/tcp_rmem or many\n    other sysctl with whitespace in it consistently use one tab. When\n    setting the value, spaces or tabs can be used and will be converted\n    to tabs by the kernel (when reading them again).\n    '
    return re.sub('\\s+', '\t', str(value))

def persist(name, value, config=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Assign and persist a simple sysctl parameter for this minion. If ``config``\n    is not specified, a sensible default will be chosen using\n    :mod:`sysctl.default_config <salt.modules.linux_sysctl.default_config>`.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.persist net.ipv4.ip_forward 1\n    "
    if config is None:
        config = default_config()
    edited = False
    if not os.path.isfile(config):
        sysctl_dir = os.path.dirname(config)
        if not os.path.exists(sysctl_dir):
            os.makedirs(sysctl_dir)
        try:
            with salt.utils.files.fopen(config, 'w+') as _fh:
                _fh.write('#\n# Kernel sysctl configuration\n#\n')
        except OSError:
            msg = 'Could not write to file: {0}'
            raise CommandExecutionError(msg.format(config))
    nlines = []
    try:
        with salt.utils.files.fopen(config, 'r') as _fh:
            config_data = salt.utils.data.decode(_fh.readlines())
    except OSError:
        msg = 'Could not read from file: {0}'
        raise CommandExecutionError(msg.format(config))
    for line in config_data:
        if '=' not in line:
            nlines.append(line)
            continue
        comps = [i.strip() for i in line.split('=', 1)]
        if comps[0].startswith('#'):
            nlines.append(line)
            continue
        if name == comps[0]:
            sanitized_value = _sanitize_sysctl_value(value)
            if _sanitize_sysctl_value(comps[1]) == sanitized_value:
                if _sanitize_sysctl_value(get(name)) != sanitized_value:
                    assign(name, value)
                    return 'Updated'
                else:
                    return 'Already set'
            nlines.append('{} = {}\n'.format(name, value))
            edited = True
            continue
        else:
            nlines.append(line)
    if not edited:
        nlines.append('{} = {}\n'.format(name, value))
    try:
        with salt.utils.files.fopen(config, 'wb') as _fh:
            _fh.writelines(salt.utils.data.encode(nlines))
    except OSError:
        msg = 'Could not write to file: {0}'
        raise CommandExecutionError(msg.format(config))
    assign(name, value)
    return 'Updated'