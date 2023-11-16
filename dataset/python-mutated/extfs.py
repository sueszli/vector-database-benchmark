"""
Module for managing ext2/3/4 file systems
"""
import logging
import salt.utils.platform
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only work on POSIX-like systems\n    '
    if salt.utils.platform.is_windows():
        return (False, 'The extfs execution module cannot be loaded: only available on non-Windows systems.')
    return True

def mkfs(device, fs_type, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Create a file system on the specified device\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' extfs.mkfs /dev/sda1 fs_type=ext4 opts=\'acl,noexec\'\n\n    Valid options are:\n\n    * **block_size**: 1024, 2048 or 4096\n    * **check**: check for bad blocks\n    * **direct**: use direct IO\n    * **ext_opts**: extended file system options (comma-separated)\n    * **fragment_size**: size of fragments\n    * **force**: setting force to True will cause mke2fs to specify the -F\n      option twice (it is already set once); this is truly dangerous\n    * **blocks_per_group**: number of blocks in a block group\n    * **number_of_groups**: ext4 option for a virtual block group\n    * **bytes_per_inode**: set the bytes/inode ratio\n    * **inode_size**: size of the inode\n    * **journal**: set to True to create a journal (default on ext3/4)\n    * **journal_opts**: options for the fs journal (comma separated)\n    * **blocks_file**: read bad blocks from file\n    * **label**: label to apply to the file system\n    * **reserved**: percentage of blocks reserved for super-user\n    * **last_dir**: last mounted directory\n    * **test**: set to True to not actually create the file system (mke2fs -n)\n    * **number_of_inodes**: override default number of inodes\n    * **creator_os**: override "creator operating system" field\n    * **opts**: mount options (comma separated)\n    * **revision**: set the filesystem revision (default 1)\n    * **super**: write superblock and group descriptors only\n    * **fs_type**: set the filesystem type (REQUIRED)\n    * **usage_type**: how the filesystem is going to be used\n    * **uuid**: set the UUID for the file system\n\n    See the ``mke2fs(8)`` manpage for a more complete description of these\n    options.\n    '
    kwarg_map = {'block_size': 'b', 'check': 'c', 'direct': 'D', 'ext_opts': 'E', 'fragment_size': 'f', 'force': 'F', 'blocks_per_group': 'g', 'number_of_groups': 'G', 'bytes_per_inode': 'i', 'inode_size': 'I', 'journal': 'j', 'journal_opts': 'J', 'blocks_file': 'l', 'label': 'L', 'reserved': 'm', 'last_dir': 'M', 'test': 'n', 'number_of_inodes': 'N', 'creator_os': 'o', 'opts': 'O', 'revision': 'r', 'super': 'S', 'usage_type': 'T', 'uuid': 'U'}
    opts = ''
    for key in kwargs:
        if key in kwarg_map:
            opt = kwarg_map[key]
            if kwargs[key] == 'True':
                opts += '-{} '.format(opt)
            else:
                opts += '-{} {} '.format(opt, kwargs[key])
    cmd = 'mke2fs -F -t {} {}{}'.format(fs_type, opts, device)
    out = __salt__['cmd.run'](cmd, python_shell=False).splitlines()
    ret = []
    for line in out:
        if not line:
            continue
        elif line.startswith('mke2fs'):
            continue
        elif line.startswith('Discarding device blocks'):
            continue
        elif line.startswith('Allocating group tables'):
            continue
        elif line.startswith('Writing inode tables'):
            continue
        elif line.startswith('Creating journal'):
            continue
        elif line.startswith('Writing superblocks'):
            continue
        ret.append(line)
    return ret

def tune(device, **kwargs):
    if False:
        print('Hello World!')
    "\n    Set attributes for the specified device (using tune2fs)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' extfs.tune /dev/sda1 force=True label=wildstallyns opts='acl,noexec'\n\n    Valid options are:\n\n    * **max**: max mount count\n    * **count**: mount count\n    * **error**: error behavior\n    * **extended_opts**: extended options (comma separated)\n    * **force**: force, even if there are errors (set to True)\n    * **group**: group name or gid that can use the reserved blocks\n    * **interval**: interval between checks\n    * **journal**: set to True to create a journal (default on ext3/4)\n    * **journal_opts**: options for the fs journal (comma separated)\n    * **label**: label to apply to the file system\n    * **reserved**: percentage of blocks reserved for super-user\n    * **last_dir**: last mounted directory\n    * **opts**: mount options (comma separated)\n    * **feature**: set or clear a feature (comma separated)\n    * **mmp_check**: mmp check interval\n    * **reserved**: reserved blocks count\n    * **quota_opts**: quota options (comma separated)\n    * **time**: time last checked\n    * **user**: user or uid who can use the reserved blocks\n    * **uuid**: set the UUID for the file system\n\n    See the ``mke2fs(8)`` manpage for a more complete description of these\n    options.\n    "
    kwarg_map = {'max': 'c', 'count': 'C', 'error': 'e', 'extended_opts': 'E', 'force': 'f', 'group': 'g', 'interval': 'i', 'journal': 'j', 'journal_opts': 'J', 'label': 'L', 'last_dir': 'M', 'opts': 'o', 'feature': 'O', 'mmp_check': 'p', 'reserved': 'r', 'quota_opts': 'Q', 'time': 'T', 'user': 'u', 'uuid': 'U'}
    opts = ''
    for key in kwargs:
        if key in kwarg_map:
            opt = kwarg_map[key]
            if kwargs[key] == 'True':
                opts += '-{} '.format(opt)
            else:
                opts += '-{} {} '.format(opt, kwargs[key])
    cmd = 'tune2fs {}{}'.format(opts, device)
    out = __salt__['cmd.run'](cmd, python_shell=False).splitlines()
    return out

def attributes(device, args=None):
    if False:
        return 10
    "\n    Return attributes from dumpe2fs for a specified device\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' extfs.attributes /dev/sda1\n    "
    fsdump = dump(device, args)
    return fsdump['attributes']

def blocks(device, args=None):
    if False:
        while True:
            i = 10
    "\n    Return block and inode info from dumpe2fs for a specified device\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' extfs.blocks /dev/sda1\n    "
    fsdump = dump(device, args)
    return fsdump['blocks']

def dump(device, args=None):
    if False:
        return 10
    "\n    Return all contents of dumpe2fs for a specified device\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' extfs.dump /dev/sda1\n    "
    cmd = 'dumpe2fs {}'.format(device)
    if args:
        cmd = cmd + ' -' + args
    ret = {'attributes': {}, 'blocks': {}}
    out = __salt__['cmd.run'](cmd, python_shell=False).splitlines()
    mode = 'opts'
    group = None
    for line in out:
        if not line:
            continue
        if line.startswith('dumpe2fs'):
            continue
        if mode == 'opts':
            line = line.replace('\t', ' ')
            comps = line.split(': ')
            if line.startswith('Filesystem features'):
                ret['attributes'][comps[0]] = comps[1].split()
            elif line.startswith('Group') and (not line.startswith('Group descriptor size')):
                mode = 'blocks'
            else:
                if len(comps) < 2:
                    continue
                ret['attributes'][comps[0]] = comps[1].strip()
        if mode == 'blocks':
            if line.startswith('Group'):
                line = line.replace(':', '')
                line = line.replace('(', '')
                line = line.replace(')', '')
                line = line.replace('[', '')
                line = line.replace(']', '')
                comps = line.split()
                blkgrp = comps[1]
                group = 'Group {}'.format(blkgrp)
                ret['blocks'][group] = {}
                ret['blocks'][group]['group'] = blkgrp
                ret['blocks'][group]['range'] = comps[3]
                ret['blocks'][group]['extra'] = []
            elif 'Free blocks:' in line:
                comps = line.split(': ')
                free_blocks = comps[1].split(', ')
                ret['blocks'][group]['free blocks'] = free_blocks
            elif 'Free inodes:' in line:
                comps = line.split(': ')
                inodes = comps[1].split(', ')
                ret['blocks'][group]['free inodes'] = inodes
            else:
                line = line.strip()
                ret['blocks'][group]['extra'].append(line)
    return ret