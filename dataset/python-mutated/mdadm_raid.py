"""
Salt module to manage RAID arrays with mdadm
"""
import logging
import os
import re
import salt.utils.path
from salt.exceptions import CommandExecutionError, SaltInvocationError
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}
__virtualname__ = 'raid'
_VOL_REGEX_PATTERN_MATCH = '^ARRAY\\s+{0}\\s+.*$'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    mdadm provides raid functions for Linux\n    '
    if __grains__['kernel'] != 'Linux':
        return (False, 'The mdadm execution module cannot be loaded: only available on Linux.')
    if not salt.utils.path.which('mdadm'):
        return (False, 'The mdadm execution module cannot be loaded: the mdadm binary is not in the path.')
    return __virtualname__

def list_():
    if False:
        while True:
            i = 10
    "\n    List the RAID devices.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' raid.list\n    "
    ret = {}
    for line in __salt__['cmd.run_stdout'](['mdadm', '--detail', '--scan'], python_shell=False).splitlines():
        if ' ' not in line:
            continue
        comps = line.split()
        device = comps[1]
        ret[device] = {'device': device}
        for comp in comps[2:]:
            key = comp.split('=')[0].lower()
            value = comp.split('=')[1]
            ret[device][key] = value
    return ret

def detail(device='/dev/md0'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Show detail for a specified RAID device\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' raid.detail '/dev/md0'\n    "
    ret = {}
    ret['members'] = {}
    if not os.path.exists(device):
        msg = "Device {0} doesn't exist!"
        raise CommandExecutionError(msg.format(device))
    cmd = ['mdadm', '--detail', device]
    for line in __salt__['cmd.run_stdout'](cmd, python_shell=False).splitlines():
        if line.startswith(device):
            continue
        if ' ' not in line:
            continue
        if ':' not in line:
            if '/dev/' in line:
                comps = line.split()
                state = comps[4:-1]
                ret['members'][comps[0]] = {'device': comps[-1], 'major': comps[1], 'minor': comps[2], 'number': comps[0], 'raiddevice': comps[3], 'state': ' '.join(state)}
            continue
        comps = line.split(' : ')
        comps[0] = comps[0].lower()
        comps[0] = comps[0].strip()
        comps[0] = comps[0].replace(' ', '_')
        ret[comps[0]] = comps[1].strip()
    return ret

def destroy(device):
    if False:
        print('Hello World!')
    "\n    Destroy a RAID device.\n\n    WARNING This will zero the superblock of all members of the RAID array..\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' raid.destroy /dev/md0\n    "
    try:
        details = detail(device)
    except CommandExecutionError:
        return False
    stop_cmd = ['mdadm', '--stop', device]
    zero_cmd = ['mdadm', '--zero-superblock']
    if __salt__['cmd.retcode'](stop_cmd, python_shell=False) == 0:
        for number in details['members']:
            zero_cmd.append(details['members'][number]['device'])
        __salt__['cmd.retcode'](zero_cmd, python_shell=False)
    if __grains__.get('os_family') == 'Debian':
        cfg_file = '/etc/mdadm/mdadm.conf'
    else:
        cfg_file = '/etc/mdadm.conf'
    try:
        __salt__['file.replace'](cfg_file, 'ARRAY {} .*'.format(device), '')
    except SaltInvocationError:
        pass
    if __salt__['raid.list']().get(device) is None:
        return True
    else:
        return False

def stop():
    if False:
        for i in range(10):
            print('nop')
    "\n    Shut down all arrays that can be shut down (i.e. are not currently in use).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' raid.stop\n    "
    cmd = 'mdadm --stop --scan'
    if __salt__['cmd.retcode'](cmd):
        return True
    return False

def create(name, level, devices, metadata='default', test_mode=False, **kwargs):
    if False:
        return 10
    '\n    Create a RAID device.\n\n    .. versionchanged:: 2014.7.0\n\n    .. warning::\n        Use with CAUTION, as this function can be very destructive if not used\n        properly!\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' raid.create /dev/md0 level=1 chunk=256 devices="[\'/dev/xvdd\', \'/dev/xvde\']" test_mode=True\n\n    .. note::\n\n        Adding ``test_mode=True`` as an argument will print out the mdadm\n        command that would have been run.\n\n    name\n        The name of the array to create.\n\n    level\n        The RAID level to use when creating the raid.\n\n    devices\n        A list of devices used to build the array.\n\n    metadata\n        Version of metadata to use when creating the array.\n\n    kwargs\n        Optional arguments to be passed to mdadm.\n\n    returns\n        test_mode=True:\n            Prints out the full command.\n        test_mode=False (Default):\n            Executes command on remote the host(s) and\n            Prints out the mdadm output.\n\n    .. note::\n\n        It takes time to create a RAID array. You can check the progress in\n        "resync_status:" field of the results from the following command:\n\n        .. code-block:: bash\n\n            salt \'*\' raid.detail /dev/md0\n\n    For more info, read the ``mdadm(8)`` manpage\n    '
    opts = []
    raid_devices = len(devices)
    for key in kwargs:
        if not key.startswith('__'):
            opts.append('--{}'.format(key))
            if kwargs[key] is not True:
                opts.append(str(kwargs[key]))
        if key == 'spare-devices':
            raid_devices -= int(kwargs[key])
    cmd = ['mdadm', '-C', name, '-R', '-v', '-l', str(level)] + opts + ['-e', str(metadata), '-n', str(raid_devices)] + devices
    cmd_str = ' '.join(cmd)
    if test_mode is True:
        return cmd_str
    elif test_mode is False:
        return __salt__['cmd.run'](cmd, python_shell=False)

def save_config():
    if False:
        return 10
    "\n    Save RAID configuration to config file.\n\n    Same as:\n    mdadm --detail --scan >> /etc/mdadm/mdadm.conf\n\n    Fixes this issue with Ubuntu\n    REF: http://askubuntu.com/questions/209702/why-is-my-raid-dev-md1-showing-up-as-dev-md126-is-mdadm-conf-being-ignored\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' raid.save_config\n\n    "
    scan = __salt__['cmd.run']('mdadm --detail --scan', python_shell=False).splitlines()
    if __grains__['os'] == 'Ubuntu':
        buggy_ubuntu_tags = ['name', 'metadata']
        for (i, elem) in enumerate(scan):
            for bad_tag in buggy_ubuntu_tags:
                pattern = '\\s{}=\\S+'.format(re.escape(bad_tag))
                pattern = re.compile(pattern, flags=re.I)
                scan[i] = re.sub(pattern, '', scan[i])
    if __grains__.get('os_family') == 'Debian':
        cfg_file = '/etc/mdadm/mdadm.conf'
    else:
        cfg_file = '/etc/mdadm.conf'
    try:
        vol_d = {line.split()[1]: line for line in scan}
        for vol in vol_d:
            pattern = _VOL_REGEX_PATTERN_MATCH.format(re.escape(vol))
            __salt__['file.replace'](cfg_file, pattern, vol_d[vol], append_if_not_found=True)
    except SaltInvocationError:
        __salt__['file.write'](cfg_file, args=scan)
    if __grains__.get('os_family') == 'Debian':
        return __salt__['cmd.run']('update-initramfs -u')
    elif __grains__.get('os_family') == 'RedHat':
        return __salt__['cmd.run']('dracut --force')

def assemble(name, devices, test_mode=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Assemble a RAID device.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' raid.assemble /dev/md0 ['/dev/xvdd', '/dev/xvde']\n\n    .. note::\n\n        Adding ``test_mode=True`` as an argument will print out the mdadm\n        command that would have been run.\n\n    name\n        The name of the array to assemble.\n\n    devices\n        The list of devices comprising the array to assemble.\n\n    kwargs\n        Optional arguments to be passed to mdadm.\n\n    returns\n        test_mode=True:\n            Prints out the full command.\n        test_mode=False (Default):\n            Executes command on the host(s) and prints out the mdadm output.\n\n    For more info, read the ``mdadm`` manpage.\n    "
    opts = []
    for key in kwargs:
        if not key.startswith('__'):
            opts.append('--{}'.format(key))
            if kwargs[key] is not True:
                opts.append(kwargs[key])
    if isinstance(devices, str):
        devices = devices.split(',')
    cmd = ['mdadm', '-A', name, '-v'] + opts + devices
    if test_mode is True:
        return cmd
    elif test_mode is False:
        return __salt__['cmd.run'](cmd, python_shell=False)

def examine(device, quiet=False):
    if False:
        i = 10
        return i + 15
    "\n    Show detail for a specified RAID component device\n\n    device\n        Device to examine, that is part of the RAID\n\n    quiet\n        If the device is not part of the RAID, do not show any error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' raid.examine '/dev/sda1'\n    "
    res = __salt__['cmd.run_stdout']('mdadm -Y -E {}'.format(device), python_shell=False, ignore_retcode=quiet)
    ret = {}
    for line in res.splitlines():
        (name, var) = line.partition('=')[::2]
        ret[name] = var
    return ret

def add(name, device):
    if False:
        i = 10
        return i + 15
    "\n    Add new device to RAID array.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' raid.add /dev/md0 /dev/sda1\n\n    "
    cmd = 'mdadm --manage {} --add {}'.format(name, device)
    if __salt__['cmd.retcode'](cmd) == 0:
        return True
    return False