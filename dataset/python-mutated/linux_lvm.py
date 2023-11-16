"""
Support for Linux LVM2
"""
import logging
import os.path
import salt.utils.path
log = logging.getLogger(__name__)
__virtualname__ = 'lvm'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load the module if lvm is installed\n    '
    if salt.utils.path.which('lvm'):
        return __virtualname__
    return (False, 'The linux_lvm execution module cannot be loaded: the lvm binary is not in the path.')

def version():
    if False:
        while True:
            i = 10
    "\n    Return LVM version from lvm version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvm.version\n    "
    cmd = 'lvm version'
    out = __salt__['cmd.run'](cmd).splitlines()
    ret = out[0].split(': ')
    return ret[1].strip()

def fullversion():
    if False:
        print('Hello World!')
    "\n    Return all version info from lvm version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvm.fullversion\n    "
    ret = {}
    cmd = 'lvm version'
    out = __salt__['cmd.run'](cmd).splitlines()
    for line in out:
        comps = line.split(':')
        ret[comps[0].strip()] = comps[1].strip()
    return ret

def pvdisplay(pvname='', real=False, quiet=False):
    if False:
        print('Hello World!')
    "\n    Return information about the physical volume(s)\n\n    pvname\n        physical device name\n\n    real\n        dereference any symlinks and report the real device\n\n        .. versionadded:: 2015.8.7\n\n    quiet\n        if the physical volume is not present, do not show any error\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lvm.pvdisplay\n        salt '*' lvm.pvdisplay /dev/md0\n    "
    ret = {}
    cmd = ['pvdisplay', '-c']
    if pvname:
        cmd.append(pvname)
    cmd_ret = __salt__['cmd.run_all'](cmd, python_shell=False, ignore_retcode=quiet)
    if cmd_ret['retcode'] != 0:
        return {}
    out = cmd_ret['stdout'].splitlines()
    for line in out:
        if 'is a new physical volume' not in line:
            comps = line.strip().split(':')
            if real:
                device = os.path.realpath(comps[0])
            else:
                device = comps[0]
            ret[device] = {'Physical Volume Device': comps[0], 'Volume Group Name': comps[1], 'Physical Volume Size (kB)': comps[2], 'Internal Physical Volume Number': comps[3], 'Physical Volume Status': comps[4], 'Physical Volume (not) Allocatable': comps[5], 'Current Logical Volumes Here': comps[6], 'Physical Extent Size (kB)': comps[7], 'Total Physical Extents': comps[8], 'Free Physical Extents': comps[9], 'Allocated Physical Extents': comps[10]}
            if real:
                ret[device]['Real Physical Volume Device'] = device
    return ret

def vgdisplay(vgname='', quiet=False):
    if False:
        i = 10
        return i + 15
    "\n    Return information about the volume group(s)\n\n    vgname\n        volume group name\n\n    quiet\n        if the volume group is not present, do not show any error\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lvm.vgdisplay\n        salt '*' lvm.vgdisplay nova-volumes\n    "
    ret = {}
    cmd = ['vgdisplay', '-c']
    if vgname:
        cmd.append(vgname)
    cmd_ret = __salt__['cmd.run_all'](cmd, python_shell=False, ignore_retcode=quiet)
    if cmd_ret['retcode'] != 0:
        return {}
    out = cmd_ret['stdout'].splitlines()
    for line in out:
        comps = line.strip().split(':')
        ret[comps[0]] = {'Volume Group Name': comps[0], 'Volume Group Access': comps[1], 'Volume Group Status': comps[2], 'Internal Volume Group Number': comps[3], 'Maximum Logical Volumes': comps[4], 'Current Logical Volumes': comps[5], 'Open Logical Volumes': comps[6], 'Maximum Logical Volume Size': comps[7], 'Maximum Physical Volumes': comps[8], 'Current Physical Volumes': comps[9], 'Actual Physical Volumes': comps[10], 'Volume Group Size (kB)': comps[11], 'Physical Extent Size (kB)': comps[12], 'Total Physical Extents': comps[13], 'Allocated Physical Extents': comps[14], 'Free Physical Extents': comps[15], 'UUID': comps[16]}
    return ret

def lvdisplay(lvname='', quiet=False):
    if False:
        return 10
    "\n    Return information about the logical volume(s)\n\n    lvname\n        logical device name\n\n    quiet\n        if the logical volume is not present, do not show any error\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lvm.lvdisplay\n        salt '*' lvm.lvdisplay /dev/vg_myserver/root\n    "
    ret = {}
    cmd = ['lvdisplay', '-c']
    if lvname:
        cmd.append(lvname)
    cmd_ret = __salt__['cmd.run_all'](cmd, python_shell=False, ignore_retcode=quiet)
    if cmd_ret['retcode'] != 0:
        return {}
    out = cmd_ret['stdout'].splitlines()
    for line in out:
        comps = line.strip().split(':')
        ret[comps[0]] = {'Logical Volume Name': comps[0], 'Volume Group Name': comps[1], 'Logical Volume Access': comps[2], 'Logical Volume Status': comps[3], 'Internal Logical Volume Number': comps[4], 'Open Logical Volumes': comps[5], 'Logical Volume Size': comps[6], 'Current Logical Extents Associated': comps[7], 'Allocated Logical Extents': comps[8], 'Allocation Policy': comps[9], 'Read Ahead Sectors': comps[10], 'Major Device Number': comps[11], 'Minor Device Number': comps[12]}
    return ret

def pvcreate(devices, override=True, force=True, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Set a physical device to be used as an LVM physical volume\n\n    override\n        Skip devices, if they are already LVM physical volumes\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt mymachine lvm.pvcreate /dev/sdb1,/dev/sdb2\n        salt mymachine lvm.pvcreate /dev/sdb1 dataalignmentoffset=7s\n    '
    if not devices:
        return 'Error: at least one device is required'
    if isinstance(devices, str):
        devices = devices.split(',')
    cmd = ['pvcreate']
    if force:
        cmd.append('--yes')
    else:
        cmd.append('-qq')
    for device in devices:
        if not os.path.exists(device):
            return '{} does not exist'.format(device)
        if not pvdisplay(device, quiet=True):
            cmd.append(device)
        elif not override:
            return 'Device "{}" is already an LVM physical volume.'.format(device)
    if not cmd[2:]:
        return True
    valid = ('metadatasize', 'dataalignment', 'dataalignmentoffset', 'pvmetadatacopies', 'metadatacopies', 'metadataignore', 'restorefile', 'norestorefile', 'labelsector', 'setphysicalvolumesize')
    no_parameter = 'norestorefile'
    for var in kwargs:
        if kwargs[var] and var in valid:
            cmd.extend(['--{}'.format(var), kwargs[var]])
        elif kwargs[var] and var in no_parameter:
            cmd.append('--{}'.format(var))
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out.get('retcode'):
        return out.get('stderr')
    for device in devices:
        if not pvdisplay(device):
            return 'Device "{}" was not affected.'.format(device)
    return True

def pvremove(devices, override=True, force=True):
    if False:
        print('Hello World!')
    '\n    Remove a physical device being used as an LVM physical volume\n\n    override\n        Skip devices, if they are already not used as LVM physical volumes\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt mymachine lvm.pvremove /dev/sdb1,/dev/sdb2\n    '
    if isinstance(devices, str):
        devices = devices.split(',')
    cmd = ['pvremove']
    if force:
        cmd.append('--yes')
    else:
        cmd.append('-qq')
    for device in devices:
        if pvdisplay(device):
            cmd.append(device)
        elif not override:
            return '{} is not a physical volume'.format(device)
    if not cmd[2:]:
        return True
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out.get('retcode'):
        return out.get('stderr')
    for device in devices:
        if pvdisplay(device, quiet=True):
            return 'Device "{}" was not affected.'.format(device)
    return True

def vgcreate(vgname, devices, force=False, **kwargs):
    if False:
        return 10
    '\n    Create an LVM volume group\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt mymachine lvm.vgcreate my_vg /dev/sdb1,/dev/sdb2\n        salt mymachine lvm.vgcreate my_vg /dev/sdb1 clustered=y\n    '
    if not vgname or not devices:
        return 'Error: vgname and device(s) are both required'
    if isinstance(devices, str):
        devices = devices.split(',')
    cmd = ['vgcreate', vgname]
    for device in devices:
        cmd.append(device)
    if force:
        cmd.append('--yes')
    else:
        cmd.append('-qq')
    valid = ('addtag', 'alloc', 'autobackup', 'clustered', 'maxlogicalvolumes', 'maxphysicalvolumes', 'metadatatype', 'vgmetadatacopies', 'metadatacopies', 'physicalextentsize', 'zero')
    for var in kwargs:
        if kwargs[var] and var in valid:
            cmd.append('--{}'.format(var))
            cmd.append(kwargs[var])
    cmd_ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if cmd_ret.get('retcode'):
        out = cmd_ret.get('stderr').strip()
    else:
        out = 'Volume group "{}" successfully created'.format(vgname)
    vgdata = vgdisplay(vgname)
    vgdata['Output from vgcreate'] = out
    return vgdata

def vgextend(vgname, devices, force=False):
    if False:
        return 10
    '\n    Add physical volumes to an LVM volume group\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt mymachine lvm.vgextend my_vg /dev/sdb1,/dev/sdb2\n        salt mymachine lvm.vgextend my_vg /dev/sdb1\n    '
    if not vgname or not devices:
        return 'Error: vgname and device(s) are both required'
    if isinstance(devices, str):
        devices = devices.split(',')
    cmd = ['vgextend', vgname]
    if force:
        cmd.append('--yes')
    else:
        cmd.append('-qq')
    for device in devices:
        cmd.append(device)
    cmd_ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if cmd_ret.get('retcode'):
        out = cmd_ret.get('stderr').strip()
    else:
        out = 'Volume group "{}" successfully extended'.format(vgname)
    vgdata = {'Output from vgextend': out}
    return vgdata

def lvcreate(lvname, vgname, size=None, extents=None, snapshot=None, pv=None, thinvolume=False, thinpool=False, force=False, **kwargs):
    if False:
        return 10
    "\n    Create a new logical volume, with option for which physical volume to be used\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lvm.lvcreate new_volume_name     vg_name size=10G\n        salt '*' lvm.lvcreate new_volume_name     vg_name extents=100 pv=/dev/sdb\n        salt '*' lvm.lvcreate new_snapshot        vg_name snapshot=volume_name size=3G\n\n    .. versionadded:: 0.12.0\n\n    Support for thin pools and thin volumes\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lvm.lvcreate new_thinpool_name   vg_name               size=20G thinpool=True\n        salt '*' lvm.lvcreate new_thinvolume_name vg_name/thinpool_name size=10G thinvolume=True\n\n    "
    if size and extents:
        return 'Error: Please specify only one of size or extents'
    if thinvolume and thinpool:
        return 'Error: Please set only one of thinvolume or thinpool to True'
    valid = ('activate', 'chunksize', 'contiguous', 'discards', 'stripes', 'stripesize', 'minor', 'persistent', 'mirrors', 'nosync', 'noudevsync', 'monitor', 'ignoremonitoring', 'permission', 'poolmetadatasize', 'readahead', 'regionsize', 'type', 'virtualsize', 'zero')
    no_parameter = ('nosync', 'noudevsync', 'ignoremonitoring', 'thin')
    extra_arguments = []
    if kwargs:
        for (k, v) in kwargs.items():
            if k in no_parameter:
                extra_arguments.append('--{}'.format(k))
            elif k in valid:
                extra_arguments.extend(['--{}'.format(k), '{}'.format(v)])
    cmd = [salt.utils.path.which('lvcreate')]
    if thinvolume:
        cmd.extend(['--thin', '-n', lvname])
    elif thinpool:
        cmd.extend(['--thinpool', lvname])
    else:
        cmd.extend(['-n', lvname])
    if snapshot:
        cmd.extend(['-s', '{}/{}'.format(vgname, snapshot)])
    else:
        cmd.append(vgname)
    if size and thinvolume:
        cmd.extend(['-V', '{}'.format(size)])
    elif extents and thinvolume:
        return 'Error: Thin volume size cannot be specified as extents'
    elif size:
        cmd.extend(['-L', '{}'.format(size)])
    elif extents:
        cmd.extend(['-l', '{}'.format(extents)])
    else:
        return 'Error: Either size or extents must be specified'
    if pv:
        cmd.append(pv)
    if extra_arguments:
        cmd.extend(extra_arguments)
    if force:
        cmd.append('--yes')
    else:
        cmd.append('-qq')
    cmd_ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if cmd_ret.get('retcode'):
        out = cmd_ret.get('stderr').strip()
    else:
        out = 'Logical volume "{}" created.'.format(lvname)
    lvdev = '/dev/{}/{}'.format(vgname, lvname)
    lvdata = lvdisplay(lvdev)
    lvdata['Output from lvcreate'] = out
    return lvdata

def vgremove(vgname, force=True):
    if False:
        print('Hello World!')
    '\n    Remove an LVM volume group\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt mymachine lvm.vgremove vgname\n        salt mymachine lvm.vgremove vgname force=True\n    '
    cmd = ['vgremove', vgname]
    if force:
        cmd.append('--yes')
    else:
        cmd.append('-qq')
    cmd_ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if cmd_ret.get('retcode'):
        out = cmd_ret.get('stderr').strip()
    else:
        out = 'Volume group "{}" successfully removed'.format(vgname)
    return out

def lvremove(lvname, vgname, force=True):
    if False:
        i = 10
        return i + 15
    "\n    Remove a given existing logical volume from a named existing volume group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lvm.lvremove lvname vgname force=True\n    "
    cmd = ['lvremove', '{}/{}'.format(vgname, lvname)]
    if force:
        cmd.append('--yes')
    else:
        cmd.append('-qq')
    cmd_ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if cmd_ret.get('retcode'):
        out = cmd_ret.get('stderr').strip()
    else:
        out = 'Logical volume "{}" successfully removed'.format(lvname)
    return out

def lvresize(size=None, lvpath=None, extents=None, force=False, resizefs=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Resize a logical volume to specific size.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n\n        salt '*' lvm.lvresize +12M /dev/mapper/vg1-test\n        salt '*' lvm.lvresize lvpath=/dev/mapper/vg1-test extents=+100%FREE\n\n    "
    if size and extents:
        log.error('Error: Please specify only one of size or extents')
        return {}
    cmd = ['lvresize']
    if force:
        cmd.append('--force')
    else:
        cmd.append('-qq')
    if resizefs:
        cmd.append('--resizefs')
    if size:
        cmd.extend(['-L', '{}'.format(size)])
    elif extents:
        cmd.extend(['-l', '{}'.format(extents)])
    else:
        log.error('Error: Either size or extents must be specified')
        return {}
    cmd.append(lvpath)
    cmd_ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if cmd_ret.get('retcode'):
        out = cmd_ret.get('stderr').strip()
    else:
        out = 'Logical volume "{}" successfully resized.'.format(lvpath)
    return {'Output from lvresize': out}

def lvextend(size=None, lvpath=None, extents=None, force=False, resizefs=False):
    if False:
        print('Hello World!')
    "\n    Increase a logical volume to specific size.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n\n        salt '*' lvm.lvextend +12M /dev/mapper/vg1-test\n        salt '*' lvm.lvextend lvpath=/dev/mapper/vg1-test extents=+100%FREE\n\n    "
    if size and extents:
        log.error('Error: Please specify only one of size or extents')
        return {}
    cmd = ['lvextend']
    if force:
        cmd.append('--yes')
    else:
        cmd.append('-qq')
    if resizefs:
        cmd.append('--resizefs')
    if size:
        cmd.extend(['-L', '{}'.format(size)])
    elif extents:
        cmd.extend(['-l', '{}'.format(extents)])
    else:
        log.error('Error: Either size or extents must be specified')
        return {}
    cmd.append(lvpath)
    cmd_ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if cmd_ret.get('retcode'):
        out = cmd_ret.get('stderr').strip()
    else:
        out = 'Logical volume "{}" successfully extended.'.format(lvpath)
    return {'Output from lvextend': out}

def pvresize(devices, override=True, force=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Resize a LVM physical volume to the physical device size\n\n    override\n        Skip devices, if they are already not used as LVM physical volumes\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt mymachine lvm.pvresize /dev/sdb1,/dev/sdb2\n    '
    if isinstance(devices, str):
        devices = devices.split(',')
    cmd = ['pvresize']
    if force:
        cmd.append('--yes')
    else:
        cmd.append('-qq')
    for device in devices:
        if pvdisplay(device):
            cmd.append(device)
        elif not override:
            return '{} is not a physical volume'.format(device)
    if not cmd[2:]:
        return True
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out.get('retcode'):
        return out.get('stderr')
    return True