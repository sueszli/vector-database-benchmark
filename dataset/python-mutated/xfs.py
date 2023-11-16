"""
Module for managing XFS file systems.
"""
import logging
import os
import re
import time
import salt.utils.data
import salt.utils.files
import salt.utils.path
import salt.utils.platform
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        return 10
    '\n    Only work on POSIX-like systems\n    '
    return not salt.utils.platform.is_windows() and __grains__.get('kernel') == 'Linux'

def _verify_run(out, cmd=None):
    if False:
        i = 10
        return i + 15
    '\n    Crash to the log if command execution was not successful.\n    '
    if out.get('retcode', 0) and out['stderr']:
        if cmd:
            log.debug('Command: "%s"', cmd)
        log.debug('Return code: %s', out.get('retcode'))
        log.debug('Error output:\n%s', out.get('stderr', 'N/A'))
        raise CommandExecutionError(out['stderr'])

def _xfs_info_get_kv(serialized):
    if False:
        print('Hello World!')
    '\n    Parse one line of the XFS info output.\n    '
    if serialized.startswith('='):
        serialized = serialized[1:].strip()
    serialized = serialized.replace(' = ', '=*** ').replace(' =', '=')
    opt = []
    for tkn in serialized.split(' '):
        if not opt or '=' in tkn:
            opt.append(tkn)
        else:
            opt[len(opt) - 1] = opt[len(opt) - 1] + ' ' + tkn
    return [tuple(items.split('=')) for items in opt]

def _parse_xfs_info(data):
    if False:
        while True:
            i = 10
    '\n    Parse output from "xfs_info" or "xfs_growfs -n".\n    '
    ret = {}
    spr = re.compile('\\s+')
    entry = None
    for line in [spr.sub(' ', l).strip().replace(', ', ' ') for l in data.split('\n')]:
        if not line or '=' not in line:
            continue
        nfo = _xfs_info_get_kv(line)
        if not line.startswith('='):
            entry = nfo.pop(0)
            ret[entry[0]] = {'section': entry[entry[1] != '***' and 1 or 0]}
        ret[entry[0]].update(dict(nfo))
    return ret

def info(device):
    if False:
        print('Hello World!')
    "\n    Get filesystem geometry information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xfs.info /dev/sda1\n    "
    out = __salt__['cmd.run_all']('xfs_info {}'.format(device))
    if out.get('stderr'):
        raise CommandExecutionError(out['stderr'].replace('xfs_info:', '').strip())
    return _parse_xfs_info(out['stdout'])

def _xfsdump_output(data):
    if False:
        i = 10
        return i + 15
    '\n    Parse CLI output of the xfsdump utility.\n    '
    out = {}
    summary = []
    summary_block = False
    for line in [l.strip() for l in data.split('\n') if l.strip()]:
        line = re.sub('^xfsdump: ', '', line)
        if line.startswith('session id:'):
            out['Session ID'] = line.split(' ')[-1]
        elif line.startswith('session label:'):
            out['Session label'] = re.sub('^session label: ', '', line)
        elif line.startswith('media file size'):
            out['Media size'] = re.sub('^media file size\\s+', '', line)
        elif line.startswith('dump complete:'):
            out['Dump complete'] = re.sub('^dump complete:\\s+', '', line)
        elif line.startswith('Dump Status:'):
            out['Status'] = re.sub('^Dump Status:\\s+', '', line)
        elif line.startswith('Dump Summary:'):
            summary_block = True
            continue
        if line.startswith(' ') and summary_block:
            summary.append(line.strip())
        elif not line.startswith(' ') and summary_block:
            summary_block = False
    if summary:
        out['Summary'] = ' '.join(summary)
    return out

def dump(device, destination, level=0, label=None, noerase=None):
    if False:
        i = 10
        return i + 15
    "\n    Dump filesystem device to the media (file, tape etc).\n\n    Required parameters:\n\n    * **device**: XFS device, content of which to be dumped.\n    * **destination**: Specifies a dump destination.\n\n    Valid options are:\n\n    * **label**: Label of the dump. Otherwise automatically generated label is used.\n    * **level**: Specifies a dump level of 0 to 9.\n    * **noerase**: Pre-erase media.\n\n    Other options are not used in order to let ``xfsdump`` use its default\n    values, as they are most optimal. See the ``xfsdump(8)`` manpage for\n    a more complete description of these options.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xfs.dump /dev/sda1 /detination/on/the/client\n        salt '*' xfs.dump /dev/sda1 /detination/on/the/client label='Company accountancy'\n        salt '*' xfs.dump /dev/sda1 /detination/on/the/client noerase=True\n    "
    if not salt.utils.path.which('xfsdump'):
        raise CommandExecutionError('Utility "xfsdump" has to be installed or missing.')
    label = label and label or time.strftime('XFS dump for "{}" of %Y.%m.%d, %H:%M'.format(device), time.localtime()).replace("'", '"')
    cmd = ['xfsdump']
    cmd.append('-F')
    if not noerase:
        cmd.append('-E')
    cmd.append("-L '{}'".format(label))
    cmd.append('-l {}'.format(level))
    cmd.append('-f {}'.format(destination))
    cmd.append(device)
    cmd = ' '.join(cmd)
    out = __salt__['cmd.run_all'](cmd)
    _verify_run(out, cmd=cmd)
    return _xfsdump_output(out['stdout'])

def _xr_to_keyset(line):
    if False:
        print('Hello World!')
    '\n    Parse xfsrestore output keyset elements.\n    '
    tkns = [elm for elm in line.strip().split(':', 1) if elm]
    if len(tkns) == 1:
        return "'{}': ".format(tkns[0])
    else:
        (key, val) = tkns
        return "'{}': '{}',".format(key.strip(), val.strip())

def _xfs_inventory_output(out):
    if False:
        i = 10
        return i + 15
    '\n    Transform xfsrestore inventory data output to a Python dict source and evaluate it.\n    '
    data = []
    out = [line for line in out.split('\n') if line.strip()]
    if len(out) == 1 and 'restore status' in out[0].lower():
        return {'restore_status': out[0]}
    ident = 0
    data.append('{')
    for line in out[:-1]:
        if len([elm for elm in line.strip().split(':') if elm]) == 1:
            n_ident = len(re.sub('[^\t]', '', line))
            if ident > n_ident:
                for step in range(ident):
                    data.append('},')
            ident = n_ident
            data.append(_xr_to_keyset(line))
            data.append('{')
        else:
            data.append(_xr_to_keyset(line))
    for step in range(ident + 1):
        data.append('},')
    data.append('},')
    data = eval('\n'.join(data))[0]
    data['restore_status'] = out[-1]
    return data

def inventory():
    if False:
        i = 10
        return i + 15
    "\n    Display XFS dump inventory without restoration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xfs.inventory\n    "
    out = __salt__['cmd.run_all']('xfsrestore -I')
    _verify_run(out)
    return _xfs_inventory_output(out['stdout'])

def _xfs_prune_output(out, uuid):
    if False:
        print('Hello World!')
    '\n    Parse prune output.\n    '
    data = {}
    cnt = []
    cutpoint = False
    for line in [l.strip() for l in out.split('\n') if l]:
        if line.startswith('-'):
            if cutpoint:
                break
            else:
                cutpoint = True
                continue
        if cutpoint:
            cnt.append(line)
    for kset in [e for e in cnt[1:] if ':' in e]:
        (key, val) = (t.strip() for t in kset.split(':', 1))
        data[key.lower().replace(' ', '_')] = val
    return data.get('uuid') == uuid and data or {}

def prune_dump(sessionid):
    if False:
        i = 10
        return i + 15
    "\n    Prunes the dump session identified by the given session id.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xfs.prune_dump b74a3586-e52e-4a4a-8775-c3334fa8ea2c\n\n    "
    out = __salt__['cmd.run_all']('xfsinvutil -s {} -F'.format(sessionid))
    _verify_run(out)
    data = _xfs_prune_output(out['stdout'], sessionid)
    if data:
        return data
    raise CommandExecutionError('Session UUID "{}" was not found.'.format(sessionid))

def _blkid_output(out):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse blkid output.\n    '
    flt = lambda data: [el for el in data if el.strip()]
    data = {}
    for dev_meta in flt(out.split('\n\n')):
        dev = {}
        for items in flt(dev_meta.strip().split('\n')):
            (key, val) = items.split('=', 1)
            dev[key.lower()] = val
        if dev.pop('type', None) == 'xfs':
            dev['label'] = dev.get('label')
            data[dev.pop('devname')] = dev
    mounts = _get_mounts()
    for device in mounts:
        if data.get(device):
            data[device].update(mounts[device])
    return data

def devices():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get known XFS formatted devices on the system.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xfs.devices\n    "
    out = __salt__['cmd.run_all']('blkid -o export')
    _verify_run(out)
    return _blkid_output(out['stdout'])

def _xfs_estimate_output(out):
    if False:
        i = 10
        return i + 15
    '\n    Parse xfs_estimate output.\n    '
    spc = re.compile('\\s+')
    data = {}
    for line in [l for l in out.split('\n') if l.strip()][1:]:
        (directory, bsize, blocks, megabytes, logsize) = spc.sub(' ', line).split(' ')
        data[directory] = {'block _size': bsize, 'blocks': blocks, 'megabytes': megabytes, 'logsize': logsize}
    return data

def estimate(path):
    if False:
        print('Hello World!')
    "\n    Estimate the space that an XFS filesystem will take.\n    For each directory estimate the space that directory would take\n    if it were copied to an XFS filesystem.\n    Estimation does not cross mount points.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xfs.estimate /path/to/file\n        salt '*' xfs.estimate /path/to/dir/*\n    "
    if not os.path.exists(path):
        raise CommandExecutionError('Path "{}" was not found.'.format(path))
    out = __salt__['cmd.run_all']('xfs_estimate -v {}'.format(path))
    _verify_run(out)
    return _xfs_estimate_output(out['stdout'])

def mkfs(device, label=None, ssize=None, noforce=None, bso=None, gmo=None, ino=None, lso=None, rso=None, nmo=None, dso=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a file system on the specified device. By default wipes out with force.\n\n    General options:\n\n    * **label**: Specify volume label.\n    * **ssize**: Specify the fundamental sector size of the filesystem.\n    * **noforce**: Do not force create filesystem, if disk is already formatted.\n\n    Filesystem geometry options:\n\n    * **bso**: Block size options.\n    * **gmo**: Global metadata options.\n    * **dso**: Data section options. These options specify the location, size,\n               and other parameters of the data section of the filesystem.\n    * **ino**: Inode options to specify the inode size of the filesystem, and other inode allocation parameters.\n    * **lso**: Log section options.\n    * **nmo**: Naming options.\n    * **rso**: Realtime section options.\n\n    See the ``mkfs.xfs(8)`` manpage for a more complete description of corresponding options description.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xfs.mkfs /dev/sda1\n        salt '*' xfs.mkfs /dev/sda1 dso='su=32k,sw=6' noforce=True\n        salt '*' xfs.mkfs /dev/sda1 dso='su=32k,sw=6' lso='logdev=/dev/sda2,size=10000b'\n    "
    getopts = lambda args: dict((args and '=' in args and args or None) and [kw.split('=') for kw in args.split(',')] or [])
    cmd = ['mkfs.xfs']
    if label:
        cmd.append('-L')
        cmd.append("'{}'".format(label))
    if ssize:
        cmd.append('-s')
        cmd.append(ssize)
    for (switch, opts) in [('-b', bso), ('-m', gmo), ('-n', nmo), ('-i', ino), ('-d', dso), ('-l', lso), ('-r', rso)]:
        try:
            if getopts(opts):
                cmd.append(switch)
                cmd.append(opts)
        except Exception:
            raise CommandExecutionError('Wrong parameters "{}" for option "{}"'.format(opts, switch))
    if not noforce:
        cmd.append('-f')
    cmd.append(device)
    cmd = ' '.join(cmd)
    out = __salt__['cmd.run_all'](cmd)
    _verify_run(out, cmd=cmd)
    return _parse_xfs_info(out['stdout'])

def modify(device, label=None, lazy_counting=None, uuid=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Modify parameters of an XFS filesystem.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xfs.modify /dev/sda1 label='My backup' lazy_counting=False\n        salt '*' xfs.modify /dev/sda1 uuid=False\n        salt '*' xfs.modify /dev/sda1 uuid=True\n    "
    if not label and lazy_counting is None and (uuid is None):
        raise CommandExecutionError('Nothing specified for modification for "{}" device'.format(device))
    cmd = ['xfs_admin']
    if label:
        cmd.append('-L')
        cmd.append("'{}'".format(label))
    if lazy_counting is False:
        cmd.append('-c')
        cmd.append('0')
    elif lazy_counting:
        cmd.append('-c')
        cmd.append('1')
    if uuid is False:
        cmd.append('-U')
        cmd.append('nil')
    elif uuid:
        cmd.append('-U')
        cmd.append('generate')
    cmd.append(device)
    cmd = ' '.join(cmd)
    _verify_run(__salt__['cmd.run_all'](cmd), cmd=cmd)
    out = __salt__['cmd.run_all']('blkid -o export {}'.format(device))
    _verify_run(out)
    return _blkid_output(out['stdout'])

def _get_mounts():
    if False:
        while True:
            i = 10
    '\n    List mounted filesystems.\n    '
    mounts = {}
    with salt.utils.files.fopen('/proc/mounts') as fhr:
        for line in salt.utils.data.decode(fhr.readlines()):
            (device, mntpnt, fstype, options, fs_freq, fs_passno) = line.strip().split(' ')
            if fstype != 'xfs':
                continue
            mounts[device] = {'mount_point': mntpnt, 'options': options.split(',')}
    return mounts

def defragment(device):
    if False:
        for i in range(10):
            print('nop')
    "\n    Defragment mounted XFS filesystem.\n    In order to mount a filesystem, device should be properly mounted and writable.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xfs.defragment /dev/sda1\n    "
    if device == '/':
        raise CommandExecutionError('Root is not a device.')
    if not _get_mounts().get(device):
        raise CommandExecutionError('Device "{}" is not mounted'.format(device))
    out = __salt__['cmd.run_all']('xfs_fsr {}'.format(device))
    _verify_run(out)
    return {'log': out['stdout']}