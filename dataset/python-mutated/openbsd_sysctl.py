"""
Module for viewing and modifying OpenBSD sysctl parameters
"""
import os
import re
import salt.utils.data
import salt.utils.files
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError
__virtualname__ = 'sysctl'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only run on OpenBSD systems\n    '
    if __grains__['os'] == 'OpenBSD':
        return __virtualname__
    return (False, 'The openbsd_sysctl execution module cannot be loaded: only available on OpenBSD systems.')

def show(config_file=False):
    if False:
        print('Hello World!')
    "\n    Return a list of sysctl parameters for this minion\n\n    config: Pull the data from the system configuration file\n        instead of the live data.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.show\n    "
    cmd = 'sysctl'
    ret = {}
    out = __salt__['cmd.run_stdout'](cmd, output_loglevel='trace')
    for line in out.splitlines():
        if not line or '=' not in line:
            continue
        comps = line.split('=', 1)
        ret[comps[0]] = comps[1]
    return ret

def get(name):
    if False:
        i = 10
        return i + 15
    "\n    Return a single sysctl parameter for this minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.get hw.physmem\n    "
    cmd = 'sysctl -n {}'.format(name)
    out = __salt__['cmd.run'](cmd)
    return out

def assign(name, value):
    if False:
        print('Hello World!')
    "\n    Assign a single sysctl parameter for this minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.assign net.inet.ip.forwarding 1\n    "
    ret = {}
    cmd = 'sysctl {}="{}"'.format(name, value)
    data = __salt__['cmd.run_all'](cmd)
    if re.match('^sysctl:.*: Operation not permitted$', data['stderr']) or data['retcode'] != 0:
        raise CommandExecutionError('sysctl failed: {}'.format(data['stderr']))
    (new_name, new_value) = data['stdout'].split(':', 1)
    ret[new_name] = new_value.split(' -> ')[-1]
    return ret

def persist(name, value, config='/etc/sysctl.conf'):
    if False:
        while True:
            i = 10
    "\n    Assign and persist a simple sysctl parameter for this minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.persist net.inet.ip.forwarding 1\n    "
    nlines = []
    edited = False
    value = str(value)
    if not os.path.isfile(config):
        try:
            with salt.utils.files.fopen(config, 'w+'):
                pass
        except OSError:
            msg = 'Could not create {0}'
            raise CommandExecutionError(msg.format(config))
    with salt.utils.files.fopen(config, 'r') as ifile:
        for line in ifile:
            line = salt.utils.stringutils.to_unicode(line)
            if not line.startswith('{}='.format(name)):
                nlines.append(line)
                continue
            else:
                (key, rest) = line.split('=', 1)
                if rest.startswith('"'):
                    (_, rest_v, rest) = rest.split('"', 2)
                elif rest.startswith("'"):
                    (_, rest_v, rest) = rest.split("'", 2)
                else:
                    rest_v = rest.split()[0]
                    rest = rest[len(rest_v):]
                if rest_v == value:
                    return 'Already set'
                new_line = '{}={}{}'.format(key, value, rest)
                nlines.append(new_line)
                edited = True
    if not edited:
        nlines.append('{}={}\n'.format(name, value))
    with salt.utils.files.fopen(config, 'wb') as ofile:
        ofile.writelines(salt.utils.data.encode(nlines))
    assign(name, value)
    return 'Updated'