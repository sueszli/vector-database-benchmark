"""
Module for managing BTRFS file systems.
"""
import itertools
import os
import re
import subprocess
import uuid
import salt.utils.fsutils
import salt.utils.platform
from salt.exceptions import CommandExecutionError

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only work on POSIX-like systems\n    '
    return not salt.utils.platform.is_windows() and __grains__.get('kernel') == 'Linux'

def version():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return BTRFS version.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.version\n    "
    out = __salt__['cmd.run_all']('btrfs --version')
    if out.get('stderr'):
        raise CommandExecutionError(out['stderr'])
    return {'version': out['stdout'].split(' ', 1)[-1]}

def _parse_btrfs_info(data):
    if False:
        return 10
    '\n    Parse BTRFS device info data.\n    '
    ret = {}
    for line in [line for line in data.split('\n') if line][:-1]:
        if line.startswith('Label:'):
            line = re.sub('Label:\\s+', '', line)
            (label, uuid_) = (tkn.strip() for tkn in line.split('uuid:'))
            ret['label'] = label != 'none' and label or None
            ret['uuid'] = uuid_
            continue
        if line.startswith('\tdevid'):
            dev_data = re.split('\\s+', line.strip())
            dev_id = dev_data[-1]
            ret[dev_id] = {'device_id': dev_data[1], 'size': dev_data[3], 'used': dev_data[5]}
    return ret

def info(device):
    if False:
        while True:
            i = 10
    "\n    Get BTRFS filesystem information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.info /dev/sda1\n    "
    out = __salt__['cmd.run_all']('btrfs filesystem show {}'.format(device))
    salt.utils.fsutils._verify_run(out)
    return _parse_btrfs_info(out['stdout'])

def devices():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get known BTRFS formatted devices on the system.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.devices\n    "
    out = __salt__['cmd.run_all']('blkid -o export')
    salt.utils.fsutils._verify_run(out)
    return salt.utils.fsutils._blkid_output(out['stdout'], fs_type='btrfs')

def _defragment_mountpoint(mountpoint):
    if False:
        print('Hello World!')
    '\n    Defragment only one BTRFS mountpoint.\n    '
    out = __salt__['cmd.run_all']('btrfs filesystem defragment -f {}'.format(mountpoint))
    return {'mount_point': mountpoint, 'passed': not out['stderr'], 'log': out['stderr'] or False, 'range': False}

def defragment(path):
    if False:
        while True:
            i = 10
    "\n    Defragment mounted BTRFS filesystem.\n    In order to defragment a filesystem, device should be properly mounted and writable.\n\n    If passed a device name, then defragmented whole filesystem, mounted on in.\n    If passed a moun tpoint of the filesystem, then only this mount point is defragmented.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.defragment /dev/sda1\n        salt '*' btrfs.defragment /path/on/filesystem\n    "
    is_device = salt.utils.fsutils._is_device(path)
    mounts = salt.utils.fsutils._get_mounts('btrfs')
    if is_device and (not mounts.get(path)):
        raise CommandExecutionError('Device "{}" is not mounted'.format(path))
    result = []
    if is_device:
        for mount_point in mounts[path]:
            result.append(_defragment_mountpoint(mount_point['mount_point']))
    else:
        is_mountpoint = False
        for mountpoints in mounts.values():
            for mpnt in mountpoints:
                if path == mpnt['mount_point']:
                    is_mountpoint = True
                    break
        d_res = _defragment_mountpoint(path)
        if not is_mountpoint and (not d_res['passed']) and ('range ioctl not supported' in d_res['log']):
            d_res['log'] = 'Range ioctl defragmentation is not supported in this kernel.'
        if not is_mountpoint:
            d_res['mount_point'] = False
            d_res['range'] = os.path.exists(path) and path or False
        result.append(d_res)
    return result

def features():
    if False:
        for i in range(10):
            print('nop')
    "\n    List currently available BTRFS features.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.mkfs_features\n    "
    out = __salt__['cmd.run_all']('mkfs.btrfs -O list-all')
    salt.utils.fsutils._verify_run(out)
    ret = {}
    for line in [re.sub('\\s+', ' ', line) for line in out['stderr'].split('\n') if ' - ' in line]:
        (option, description) = line.split(' - ', 1)
        ret[option] = description
    return ret

def _usage_overall(raw):
    if False:
        return 10
    '\n    Parse usage/overall.\n    '
    data = {}
    for line in raw.split('\n')[1:]:
        keyset = [item.strip() for item in re.sub('\\s+', ' ', line).split(':', 1) if item.strip()]
        if len(keyset) == 2:
            key = re.sub('[()]', '', keyset[0]).replace(' ', '_').lower()
            if key in ['free_estimated', 'global_reserve']:
                subk = keyset[1].split('(')
                data[key] = subk[0].strip()
                subk = subk[1].replace(')', '').split(': ')
                data['{}_{}'.format(key, subk[0])] = subk[1]
            else:
                data[key] = keyset[1]
    return data

def _usage_specific(raw):
    if False:
        print('Hello World!')
    '\n    Parse usage/specific.\n    '
    get_key = lambda val: dict([tuple(val.split(':'))])
    raw = raw.split('\n')
    (section, size, used) = raw[0].split(' ')
    section = section.replace(',', '_').replace(':', '').lower()
    data = {}
    data[section] = {}
    for val in [size, used]:
        data[section].update(get_key(val.replace(',', '')))
    for devices in raw[1:]:
        data[section].update(get_key(re.sub('\\s+', ':', devices.strip())))
    return data

def _usage_unallocated(raw):
    if False:
        while True:
            i = 10
    '\n    Parse usage/unallocated.\n    '
    ret = {}
    for line in raw.split('\n')[1:]:
        keyset = re.sub('\\s+', ' ', line.strip()).split(' ')
        if len(keyset) == 2:
            ret[keyset[0]] = keyset[1]
    return ret

def usage(path):
    if False:
        return 10
    "\n    Show in which disk the chunks are allocated.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.usage /your/mountpoint\n    "
    out = __salt__['cmd.run_all']('btrfs filesystem usage {}'.format(path))
    salt.utils.fsutils._verify_run(out)
    ret = {}
    for section in out['stdout'].split('\n\n'):
        if section.startswith('Overall:\n'):
            ret['overall'] = _usage_overall(section)
        elif section.startswith('Unallocated:\n'):
            ret['unallocated'] = _usage_unallocated(section)
        else:
            ret.update(_usage_specific(section))
    return ret

def mkfs(*devices, **kwargs):
    if False:
        print('Hello World!')
    "\n    Create a file system on the specified device. By default wipes out with force.\n\n    General options:\n\n    * **allocsize**: Specify the BTRFS offset from the start of the device.\n    * **bytecount**: Specify the size of the resultant filesystem.\n    * **nodesize**: Node size.\n    * **leafsize**: Specify the nodesize, the tree block size in which btrfs stores data.\n    * **noforce**: Prevent force overwrite when an existing filesystem is detected on the device.\n    * **sectorsize**: Specify the sectorsize, the minimum data block allocation unit.\n    * **nodiscard**: Do not perform whole device TRIM operation by default.\n    * **uuid**: Pass UUID or pass True to generate one.\n\n\n    Options:\n\n    * **dto**: (raid0|raid1|raid5|raid6|raid10|single|dup)\n               Specify how the data must be spanned across the devices specified.\n    * **mto**: (raid0|raid1|raid5|raid6|raid10|single|dup)\n               Specify how metadata must be spanned across the devices specified.\n    * **fts**: Features (call ``salt <host> btrfs.features`` for full list of available features)\n\n    See the ``mkfs.btrfs(8)`` manpage for a more complete description of corresponding options description.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.mkfs /dev/sda1\n        salt '*' btrfs.mkfs /dev/sda1 noforce=True\n    "
    if not devices:
        raise CommandExecutionError('No devices specified')
    mounts = salt.utils.fsutils._get_mounts('btrfs')
    for device in devices:
        if mounts.get(device):
            raise CommandExecutionError('Device "{}" should not be mounted'.format(device))
    cmd = ['mkfs.btrfs']
    dto = kwargs.get('dto')
    mto = kwargs.get('mto')
    if len(devices) == 1:
        if dto:
            cmd.append('-d single')
        if mto:
            cmd.append('-m single')
    else:
        if dto:
            cmd.append('-d {}'.format(dto))
        if mto:
            cmd.append('-m {}'.format(mto))
    for (key, option) in [('-l', 'leafsize'), ('-L', 'label'), ('-O', 'fts'), ('-A', 'allocsize'), ('-b', 'bytecount'), ('-n', 'nodesize'), ('-s', 'sectorsize')]:
        if option == 'label' and option in kwargs:
            kwargs['label'] = "'{}'".format(kwargs['label'])
        if kwargs.get(option):
            cmd.append('{} {}'.format(key, kwargs.get(option)))
    if kwargs.get('uuid'):
        cmd.append('-U {}'.format(kwargs.get('uuid') is True and uuid.uuid1() or kwargs.get('uuid')))
    if kwargs.get('nodiscard'):
        cmd.append('-K')
    if not kwargs.get('noforce'):
        cmd.append('-f')
    cmd.extend(devices)
    out = __salt__['cmd.run_all'](' '.join(cmd))
    salt.utils.fsutils._verify_run(out)
    ret = {'log': out['stdout']}
    ret.update(__salt__['btrfs.info'](devices[0]))
    return ret

def resize(mountpoint, size):
    if False:
        for i in range(10):
            print('nop')
    "\n    Resize filesystem.\n\n    General options:\n\n    * **mountpoint**: Specify the BTRFS mountpoint to resize.\n    * **size**: ([+/-]<newsize>[kKmMgGtTpPeE]|max) Specify the new size of the target.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.resize /mountpoint size=+1g\n        salt '*' btrfs.resize /dev/sda1 size=max\n    "
    if size == 'max':
        if not salt.utils.fsutils._is_device(mountpoint):
            raise CommandExecutionError('Mountpoint "{}" should be a valid device'.format(mountpoint))
        if not salt.utils.fsutils._get_mounts('btrfs').get(mountpoint):
            raise CommandExecutionError('Device "{}" should be mounted'.format(mountpoint))
    elif len(size) < 3 or size[0] not in '-+' or size[-1] not in 'kKmMgGtTpPeE' or re.sub('\\d', '', size[1:][:-1]):
        raise CommandExecutionError('Unknown size: "{}". Expected: [+/-]<newsize>[kKmMgGtTpPeE]|max'.format(size))
    out = __salt__['cmd.run_all']('btrfs filesystem resize {} {}'.format(size, mountpoint))
    salt.utils.fsutils._verify_run(out)
    ret = {'log': out['stdout']}
    ret.update(__salt__['btrfs.info'](mountpoint))
    return ret

def _fsck_ext(device):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check an ext2/ext3/ext4 file system.\n\n    This is forced check to determine a filesystem is clean or not.\n    NOTE: Maybe this function needs to be moved as a standard method in extfs module in a future.\n    '
    msgs = {0: 'No errors', 1: 'Filesystem errors corrected', 2: 'System should be rebooted', 4: 'Filesystem errors left uncorrected', 8: 'Operational error', 16: 'Usage or syntax error', 32: 'Fsck canceled by user request', 128: 'Shared-library error'}
    return msgs.get(__salt__['cmd.run_all']('fsck -f -n {}'.format(device))['retcode'], 'Unknown error')

def convert(device, permanent=False, keeplf=False):
    if False:
        while True:
            i = 10
    "\n    Convert ext2/3/4 to BTRFS. Device should be mounted.\n\n    Filesystem can be converted temporarily so the further processing and rollback is possible,\n    or permanently, where previous extended filesystem image gets deleted. Please note, permanent\n    conversion takes a while as BTRFS filesystem needs to be properly rebalanced afterwards.\n\n    General options:\n\n    * **permanent**: Specify if the migration should be permanent (false by default)\n    * **keeplf**: Keep ``lost+found`` of the partition (removed by default,\n                  but still in the image, if not permanent migration)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.convert /dev/sda1\n        salt '*' btrfs.convert /dev/sda1 permanent=True\n    "
    out = __salt__['cmd.run_all']('blkid -o export')
    salt.utils.fsutils._verify_run(out)
    devices = salt.utils.fsutils._blkid_output(out['stdout'])
    if not devices.get(device):
        raise CommandExecutionError('The device "{}" was is not found.'.format(device))
    if not devices[device]['type'] in ['ext2', 'ext3', 'ext4']:
        raise CommandExecutionError('The device "{}" is a "{}" file system.'.format(device, devices[device]['type']))
    mountpoint = salt.utils.fsutils._get_mounts(devices[device]['type']).get(device, [{'mount_point': None}])[0].get('mount_point')
    if mountpoint == '/':
        raise CommandExecutionError('One does not simply converts a root filesystem!\n\nConverting an extended root filesystem to BTRFS is a careful\nand lengthy process, among other steps including the following\nrequirements:\n\n  1. Proper verified backup.\n  2. System outage.\n  3. Offline system access.\n\nFor further details, please refer to your OS vendor\ndocumentation regarding this topic.\n')
    salt.utils.fsutils._verify_run(__salt__['cmd.run_all']('umount {}'.format(device)))
    ret = {'before': {'fsck_status': _fsck_ext(device), 'mount_point': mountpoint, 'type': devices[device]['type']}}
    salt.utils.fsutils._verify_run(__salt__['cmd.run_all']('btrfs-convert {}'.format(device)))
    salt.utils.fsutils._verify_run(__salt__['cmd.run_all']('mount {} {}'.format(device, mountpoint)))
    out = __salt__['cmd.run_all']('blkid -o export')
    salt.utils.fsutils._verify_run(out)
    devices = salt.utils.fsutils._blkid_output(out['stdout'])
    ret['after'] = {'fsck_status': 'N/A', 'mount_point': mountpoint, 'type': devices[device]['type']}
    image_path = '{}/ext2_saved'.format(mountpoint)
    orig_fstype = ret['before']['type']
    if not os.path.exists(image_path):
        raise CommandExecutionError('BTRFS migration went wrong: the image "{}" not found!'.format(image_path))
    if not permanent:
        ret['after']['{}_image'.format(orig_fstype)] = image_path
        image_info_proc = subprocess.run(['file', '{}/image'.format(image_path)], check=True, stdout=subprocess.PIPE)
        ret['after']['{}_image_info'.format(orig_fstype)] = image_info_proc.stdout.strip()
    else:
        ret['after']['{}_image'.format(orig_fstype)] = 'removed'
        ret['after']['{}_image_info'.format(orig_fstype)] = 'N/A'
        salt.utils.fsutils._verify_run(__salt__['cmd.run_all']('btrfs subvolume delete {}'.format(image_path)))
        out = __salt__['cmd.run_all']('btrfs filesystem balance {}'.format(mountpoint))
        salt.utils.fsutils._verify_run(out)
        ret['after']['balance_log'] = out['stdout']
    lost_found = '{}/lost+found'.format(mountpoint)
    if os.path.exists(lost_found) and (not keeplf):
        salt.utils.fsutils._verify_run(__salt__['cmd.run_all']('rm -rf {}'.format(lost_found)))
    return ret

def _restripe(mountpoint, direction, *devices, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Restripe BTRFS: add or remove devices from the particular mounted filesystem.\n    '
    fs_log = []
    if salt.utils.fsutils._is_device(mountpoint):
        raise CommandExecutionError('Mountpount expected, while device "{}" specified'.format(mountpoint))
    mounted = False
    for (device, mntpoints) in salt.utils.fsutils._get_mounts('btrfs').items():
        for mntdata in mntpoints:
            if mntdata['mount_point'] == mountpoint:
                mounted = True
                break
    if not mounted:
        raise CommandExecutionError('No BTRFS device mounted on "{}" mountpoint'.format(mountpoint))
    if not devices:
        raise CommandExecutionError('No devices specified.')
    available_devices = __salt__['btrfs.devices']()
    for device in devices:
        if device not in available_devices.keys():
            raise CommandExecutionError('Device "{}" is not recognized'.format(device))
    cmd = ['btrfs device {}'.format(direction)]
    for device in devices:
        cmd.append(device)
    if direction == 'add':
        if kwargs.get('nodiscard'):
            cmd.append('-K')
        if kwargs.get('force'):
            cmd.append('-f')
    cmd.append(mountpoint)
    out = __salt__['cmd.run_all'](' '.join(cmd))
    salt.utils.fsutils._verify_run(out)
    if out['stdout']:
        fs_log.append(out['stdout'])
    if direction == 'add':
        out = None
        data_conversion = kwargs.get('dc')
        meta_conversion = kwargs.get('mc')
        if data_conversion and meta_conversion:
            out = __salt__['cmd.run_all']('btrfs balance start -dconvert={} -mconvert={} {}'.format(data_conversion, meta_conversion, mountpoint))
        else:
            out = __salt__['cmd.run_all']('btrfs filesystem balance {}'.format(mountpoint))
        salt.utils.fsutils._verify_run(out)
        if out['stdout']:
            fs_log.append(out['stdout'])
    ret = {}
    if fs_log:
        ret.update({'log': '\n'.join(fs_log)})
    ret.update(__salt__['btrfs.info'](mountpoint))
    return ret

def add(mountpoint, *devices, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Add a devices to a BTRFS filesystem.\n\n    General options:\n\n    * **nodiscard**: Do not perform whole device TRIM\n    * **force**: Force overwrite existing filesystem on the disk\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.add /mountpoint /dev/sda1 /dev/sda2\n    "
    return _restripe(mountpoint, 'add', *devices, **kwargs)

def delete(mountpoint, *devices, **kwargs):
    if False:
        print('Hello World!')
    "\n    Remove devices from a BTRFS filesystem.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.delete /mountpoint /dev/sda1 /dev/sda2\n    "
    return _restripe(mountpoint, 'delete', *devices, **kwargs)

def _parse_proplist(data):
    if False:
        print('Hello World!')
    '\n    Parse properties list.\n    '
    out = {}
    for line in data.split('\n'):
        line = re.split('\\s+', line, 1)
        if len(line) == 2:
            out[line[0]] = line[1]
    return out

def properties(obj, type=None, set=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    List properties for given btrfs object. The object can be path of BTRFS device,\n    mount point, or any directories/files inside the BTRFS filesystem.\n\n    General options:\n\n    * **type**: Possible types are s[ubvol], f[ilesystem], i[node] and d[evice].\n    * **force**: Force overwrite existing filesystem on the disk\n    * **set**: <key=value,key1=value1...> Options for a filesystem properties.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' btrfs.properties /mountpoint\n        salt \'*\' btrfs.properties /dev/sda1 type=subvol set=\'ro=false,label="My Storage"\'\n    '
    if type and type not in ['s', 'subvol', 'f', 'filesystem', 'i', 'inode', 'd', 'device']:
        raise CommandExecutionError('Unknown property type: "{}" specified'.format(type))
    cmd = ['btrfs']
    cmd.append('property')
    cmd.append(set and 'set' or 'list')
    if type:
        cmd.append('-t{}'.format(type))
    cmd.append(obj)
    if set:
        try:
            for (key, value) in [[item.strip() for item in keyset.split('=')] for keyset in set.split(',')]:
                cmd.append(key)
                cmd.append(value)
        except Exception as ex:
            raise CommandExecutionError(ex)
    out = __salt__['cmd.run_all'](' '.join(cmd))
    salt.utils.fsutils._verify_run(out)
    if not set:
        ret = {}
        for (prop, descr) in _parse_proplist(out['stdout']).items():
            ret[prop] = {'description': descr}
            value = __salt__['cmd.run_all']('btrfs property get {} {}'.format(obj, prop))['stdout']
            ret[prop]['value'] = value and value.split('=')[-1] or 'N/A'
        return ret

def subvolume_exists(path):
    if False:
        i = 10
        return i + 15
    "\n    Check if a subvolume is present in the filesystem.\n\n    path\n        Mount point for the subvolume (full path)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.subvolume_exists /mnt/var\n\n    "
    cmd = ['btrfs', 'subvolume', 'show', path]
    return __salt__['cmd.retcode'](cmd, ignore_retcode=True) == 0

def subvolume_create(name, dest=None, qgroupids=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create subvolume `name` in `dest`.\n\n    Return True if the subvolume is created, False is the subvolume is\n    already there.\n\n    name\n         Name of the new subvolume\n\n    dest\n         If not given, the subvolume will be created in the current\n         directory, if given will be in /dest/name\n\n    qgroupids\n         Add the newly created subcolume to a qgroup. This parameter\n         is a list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.subvolume_create var\n        salt '*' btrfs.subvolume_create var dest=/mnt\n        salt '*' btrfs.subvolume_create var qgroupids='[200]'\n\n    "
    if qgroupids and type(qgroupids) is not list:
        raise CommandExecutionError('Qgroupids parameter must be a list')
    if dest:
        name = os.path.join(dest, name)
    if subvolume_exists(name):
        return False
    cmd = ['btrfs', 'subvolume', 'create']
    if type(qgroupids) is list:
        cmd.append('-i')
        cmd.extend(qgroupids)
    cmd.append(name)
    res = __salt__['cmd.run_all'](cmd)
    salt.utils.fsutils._verify_run(res)
    return True

def subvolume_delete(name=None, names=None, commit=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete the subvolume(s) from the filesystem\n\n    The user can remove one single subvolume (name) or multiple of\n    then at the same time (names). One of the two parameters needs to\n    specified.\n\n    Please, refer to the documentation to understand the implication\n    on the transactions, and when the subvolume is really deleted.\n\n    Return True if the subvolume is deleted, False is the subvolume\n    was already missing.\n\n    name\n        Name of the subvolume to remove\n\n    names\n        List of names of subvolumes to remove\n\n    commit\n        * 'after': Wait for transaction commit at the end\n        * 'each': Wait for transaction commit after each delete\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.subvolume_delete /var/volumes/tmp\n        salt '*' btrfs.subvolume_delete /var/volumes/tmp commit=after\n\n    "
    if not name and (not (names and type(names) is list)):
        raise CommandExecutionError('Provide a value for the name parameter')
    if commit and commit not in ('after', 'each'):
        raise CommandExecutionError('Value for commit not recognized')
    names = [n for n in itertools.chain([name], names or []) if n and subvolume_exists(n)]
    if not names:
        return False
    cmd = ['btrfs', 'subvolume', 'delete']
    if commit == 'after':
        cmd.append('--commit-after')
    elif commit == 'each':
        cmd.append('--commit-each')
    cmd.extend(names)
    res = __salt__['cmd.run_all'](cmd)
    salt.utils.fsutils._verify_run(res)
    return True

def subvolume_find_new(name, last_gen):
    if False:
        print('Hello World!')
    "\n    List the recently modified files in a subvolume\n\n    name\n        Name of the subvolume\n\n    last_gen\n        Last transid marker from where to compare\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.subvolume_find_new /var/volumes/tmp 1024\n\n    "
    cmd = ['btrfs', 'subvolume', 'find-new', name, last_gen]
    res = __salt__['cmd.run_all'](cmd)
    salt.utils.fsutils._verify_run(res)
    lines = res['stdout'].splitlines()
    files = [l.split()[-1] for l in lines if l.startswith('inode')]
    transid = lines[-1].split()[-1]
    return {'files': files, 'transid': transid}

def subvolume_get_default(path):
    if False:
        print('Hello World!')
    "\n    Get the default subvolume of the filesystem path\n\n    path\n        Mount point for the subvolume\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.subvolume_get_default /var/volumes/tmp\n\n    "
    cmd = ['btrfs', 'subvolume', 'get-default', path]
    res = __salt__['cmd.run_all'](cmd)
    salt.utils.fsutils._verify_run(res)
    line = res['stdout'].strip()
    id_ = line.split()[1]
    name = line.split()[-1]
    return {'id': id_, 'name': name}

def _pop(line, key, use_rest):
    if False:
        print('Hello World!')
    '\n    Helper for the line parser.\n\n    If key is a prefix of line, will remove ir from the line and will\n    extract the value (space separation), and the rest of the line.\n\n    If use_rest is True, the value will be the rest of the line.\n\n    Return a tuple with the value and the rest of the line.\n    '
    value = None
    if line.startswith(key):
        line = line[len(key):].strip()
        if use_rest:
            value = line
            line = ''
        else:
            (value, line) = line.split(' ', 1)
    return (value, line.strip())

def subvolume_list(path, parent_id=False, absolute=False, ogeneration=False, generation=False, subvolumes=False, uuid=False, parent_uuid=False, sent_subvolume_uuid=False, snapshots=False, readonly=False, deleted=False, generation_cmp=None, ogeneration_cmp=None, sort=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    List the subvolumes present in the filesystem.\n\n    path\n        Mount point for the subvolume\n\n    parent_id\n        Print parent ID\n\n    absolute\n        Print all the subvolumes in the filesystem and distinguish\n        between absolute and relative path with respect to the given\n        <path>\n\n    ogeneration\n        Print the ogeneration of the subvolume\n\n    generation\n        Print the generation of the subvolume\n\n    subvolumes\n        Print only subvolumes below specified <path>\n\n    uuid\n        Print the UUID of the subvolume\n\n    parent_uuid\n        Print the parent uuid of subvolumes (and snapshots)\n\n    sent_subvolume_uuid\n        Print the UUID of the sent subvolume, where the subvolume is\n        the result of a receive operation\n\n    snapshots\n        Only snapshot subvolumes in the filesystem will be listed\n\n    readonly\n        Only readonly subvolumes in the filesystem will be listed\n\n    deleted\n        Only deleted subvolumens that are ye not cleaned\n\n    generation_cmp\n        List subvolumes in the filesystem that its generation is >=,\n        <= or = value. '+' means >= value, '-' means <= value, If\n        there is neither '+' nor '-', it means = value\n\n    ogeneration_cmp\n        List subvolumes in the filesystem that its ogeneration is >=,\n        <= or = value\n\n    sort\n        List subvolumes in order by specified items. Possible values:\n        * rootid\n        * gen\n        * ogen\n        * path\n        You can add '+' or '-' in front of each items, '+' means\n        ascending, '-' means descending. The default is ascending. You\n        can combite it in a list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.subvolume_list /var/volumes/tmp\n        salt '*' btrfs.subvolume_list /var/volumes/tmp path=True\n        salt '*' btrfs.subvolume_list /var/volumes/tmp sort='[-rootid]'\n\n    "
    if sort and type(sort) is not list:
        raise CommandExecutionError('Sort parameter must be a list')
    valid_sorts = [''.join((order, attrib)) for (order, attrib) in itertools.product(('-', '', '+'), ('rootid', 'gen', 'ogen', 'path'))]
    if sort and (not all((s in valid_sorts for s in sort))):
        raise CommandExecutionError('Value for sort not recognized')
    cmd = ['btrfs', 'subvolume', 'list']
    params = ((parent_id, '-p'), (absolute, '-a'), (ogeneration, '-c'), (generation, '-g'), (subvolumes, '-o'), (uuid, '-u'), (parent_uuid, '-q'), (sent_subvolume_uuid, '-R'), (snapshots, '-s'), (readonly, '-r'), (deleted, '-d'))
    cmd.extend((p[1] for p in params if p[0]))
    if generation_cmp:
        cmd.extend(['-G', generation_cmp])
    if ogeneration_cmp:
        cmd.extend(['-C', ogeneration_cmp])
    if sort:
        cmd.append('--sort={}'.format(','.join(sort)))
    cmd.append(path)
    res = __salt__['cmd.run_all'](cmd)
    salt.utils.fsutils._verify_run(res)
    columns = ('ID', 'gen', 'cgen', 'parent', 'top level', 'otime', 'parent_uuid', 'received_uuid', 'uuid', 'path')
    result = []
    for line in res['stdout'].splitlines():
        table = {}
        for key in columns:
            (value, line) = _pop(line, key, key == 'path')
            if value:
                table[key.lower()] = value
        if not line:
            result.append(table)
    return result

def subvolume_set_default(subvolid, path):
    if False:
        return 10
    "\n    Set the subvolume as default\n\n    subvolid\n        ID of the new default subvolume\n\n    path\n        Mount point for the filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.subvolume_set_default 257 /var/volumes/tmp\n\n    "
    cmd = ['btrfs', 'subvolume', 'set-default', subvolid, path]
    res = __salt__['cmd.run_all'](cmd)
    salt.utils.fsutils._verify_run(res)
    return True

def subvolume_show(path):
    if False:
        i = 10
        return i + 15
    "\n    Show information of a given subvolume\n\n    path\n        Mount point for the filesystem\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.subvolume_show /var/volumes/tmp\n\n    "
    cmd = ['btrfs', 'subvolume', 'show', path]
    res = __salt__['cmd.run_all'](cmd)
    salt.utils.fsutils._verify_run(res)
    result = {}
    table = {}
    stdout = res['stdout'].splitlines()
    key = stdout.pop(0)
    result[key.strip()] = table
    for line in stdout:
        (key, value) = line.split(':', 1)
        table[key.lower().strip()] = value.strip()
    return result

def subvolume_snapshot(source, dest=None, name=None, read_only=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a snapshot of a source subvolume\n\n    source\n        Source subvolume from where to create the snapshot\n\n    dest\n        If only dest is given, the subvolume will be named as the\n        basename of the source\n\n    name\n       Name of the snapshot\n\n    read_only\n        Create a read only snapshot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.subvolume_snapshot /var/volumes/tmp dest=/.snapshots\n        salt '*' btrfs.subvolume_snapshot /var/volumes/tmp name=backup\n\n    "
    if not dest and (not name):
        raise CommandExecutionError('Provide parameter dest, name, or both')
    cmd = ['btrfs', 'subvolume', 'snapshot']
    if read_only:
        cmd.append('-r')
    cmd.append(source)
    if dest and (not name):
        cmd.append(dest)
    if dest and name:
        name = os.path.join(dest, name)
    if name:
        cmd.append(name)
    res = __salt__['cmd.run_all'](cmd)
    salt.utils.fsutils._verify_run(res)
    return True

def subvolume_sync(path, subvolids=None, sleep=None):
    if False:
        i = 10
        return i + 15
    "\n    Wait until given subvolume are completely removed from the\n    filesystem after deletion.\n\n    path\n        Mount point for the filesystem\n\n    subvolids\n        List of IDs of subvolumes to wait for\n\n    sleep\n        Sleep N seconds betwenn checks (default: 1)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' btrfs.subvolume_sync /var/volumes/tmp\n        salt '*' btrfs.subvolume_sync /var/volumes/tmp subvolids='[257]'\n\n    "
    if subvolids and type(subvolids) is not list:
        raise CommandExecutionError('Subvolids parameter must be a list')
    cmd = ['btrfs', 'subvolume', 'sync']
    if sleep:
        cmd.extend(['-s', sleep])
    cmd.append(path)
    if subvolids:
        cmd.extend(subvolids)
    res = __salt__['cmd.run_all'](cmd)
    salt.utils.fsutils._verify_run(res)
    return True