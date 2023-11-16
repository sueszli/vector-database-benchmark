"""
Module for viewing and modifying sysctl parameters
"""
import os
import salt.utils.files
from salt.exceptions import CommandExecutionError
__virtualname__ = 'sysctl'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only run on Darwin (macOS) systems\n    '
    if __grains__['os'] == 'MacOS':
        return __virtualname__
    return (False, 'The darwin_sysctl execution module cannot be loaded: Only available on macOS systems.')

def show(config_file=False):
    if False:
        return 10
    "\n    Return a list of sysctl parameters for this minion\n\n    config: Pull the data from the system configuration file\n        instead of the live data.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.show\n    "
    roots = ('audit', 'debug', 'hw', 'hw', 'kern', 'machdep', 'net', 'net', 'security', 'user', 'vfs', 'vm')
    cmd = 'sysctl -a'
    ret = {}
    out = __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False)
    comps = ['']
    for line in out.splitlines():
        if any([line.startswith('{}.'.format(root)) for root in roots]):
            comps = line.split(': ' if ': ' in line else ' = ', 1)
            if len(comps) == 2:
                ret[comps[0]] = comps[1]
            else:
                ret[comps[0]] = ''
        elif comps[0]:
            ret[comps[0]] += '{}\n'.format(line)
        else:
            continue
    return ret

def get(name):
    if False:
        print('Hello World!')
    "\n    Return a single sysctl parameter for this minion\n\n    name\n        The name of the sysctl value to display.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.get hw.physmem\n    "
    cmd = 'sysctl -n {}'.format(name)
    out = __salt__['cmd.run'](cmd, python_shell=False)
    return out

def assign(name, value):
    if False:
        return 10
    "\n    Assign a single sysctl parameter for this minion\n\n    name\n        The name of the sysctl value to edit.\n\n    value\n        The sysctl value to apply.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.assign net.inet.icmp.icmplim 50\n    "
    ret = {}
    cmd = 'sysctl -w {}="{}"'.format(name, value)
    data = __salt__['cmd.run_all'](cmd, python_shell=False)
    if data['retcode'] != 0:
        raise CommandExecutionError('sysctl failed: {}'.format(data['stderr']))
    (new_name, new_value) = data['stdout'].split(':', 1)
    ret[new_name] = new_value.split(' -> ')[-1]
    return ret

def persist(name, value, config='/etc/sysctl.conf', apply_change=False):
    if False:
        i = 10
        return i + 15
    "\n    Assign and persist a simple sysctl parameter for this minion\n\n    name\n        The name of the sysctl value to edit.\n\n    value\n        The sysctl value to apply.\n\n    config\n        The location of the sysctl configuration file.\n\n    apply_change\n        Default is False; Default behavior only creates or edits\n        the sysctl.conf file. If apply is set to True, the changes are\n        applied to the system.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' sysctl.persist net.inet.icmp.icmplim 50\n        salt '*' sysctl.persist coretemp_load NO config=/etc/sysctl.conf\n    "
    nlines = []
    edited = False
    value = str(value)
    if not os.path.isfile(config):
        try:
            with salt.utils.files.fopen(config, 'w+') as _fh:
                _fh.write('#\n# Kernel sysctl configuration\n#\n')
        except OSError:
            msg = 'Could not write to file: {0}'
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
                nlines.append('{}={}\n'.format(name, value))
                edited = True
    if not edited:
        nlines.append('{}={}\n'.format(name, value))
    nlines = [salt.utils.stringutils.to_str(_l) for _l in nlines]
    with salt.utils.files.fopen(config, 'w+') as ofile:
        ofile.writelines(nlines)
    if apply_change is True:
        assign(name, value)
        return 'Updated and applied'
    return 'Updated'