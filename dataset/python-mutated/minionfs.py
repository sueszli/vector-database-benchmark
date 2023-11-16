"""
Fileserver backend which serves files pushed to the Master

The :mod:`cp.push <salt.modules.cp.push>` function allows Minions to push files
up to the Master. Using this backend, these pushed files are exposed to other
Minions via the Salt fileserver.

To enable minionfs, :conf_master:`file_recv` needs to be set to ``True`` in the
master config file (otherwise :mod:`cp.push <salt.modules.cp.push>` will not be
allowed to push files to the Master), and ``minionfs`` must be added to the
:conf_master:`fileserver_backends` list.

.. code-block:: yaml

    fileserver_backend:
      - minionfs

.. note::
    ``minion`` also works here. Prior to the 2018.3.0 release, *only*
    ``minion`` would work.

Other minionfs settings include: :conf_master:`minionfs_whitelist`,
:conf_master:`minionfs_blacklist`, :conf_master:`minionfs_mountpoint`, and
:conf_master:`minionfs_env`.

.. seealso:: :ref:`tutorial-minionfs`

"""
import logging
import os
import salt.fileserver
import salt.utils.files
import salt.utils.gzip_util
import salt.utils.hashutils
import salt.utils.path
import salt.utils.stringutils
import salt.utils.url
import salt.utils.versions
log = logging.getLogger(__name__)
__virtualname__ = 'minionfs'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if file_recv is enabled\n    '
    if __virtualname__ not in __opts__['fileserver_backend']:
        return False
    return __virtualname__ if __opts__['file_recv'] else False

def _is_exposed(minion):
    if False:
        i = 10
        return i + 15
    '\n    Check if the minion is exposed, based on the whitelist and blacklist\n    '
    return salt.utils.stringutils.check_whitelist_blacklist(minion, whitelist=__opts__['minionfs_whitelist'], blacklist=__opts__['minionfs_blacklist'])

def find_file(path, tgt_env='base', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Search the environment for the relative path\n    '
    fnd = {'path': '', 'rel': ''}
    if os.path.isabs(path):
        return fnd
    if tgt_env not in envs():
        return fnd
    if os.path.basename(path) == 'top.sls':
        log.debug('minionfs will NOT serve top.sls for security reasons (path requested: %s)', path)
        return fnd
    mountpoint = salt.utils.url.strip_proto(__opts__['minionfs_mountpoint'])
    path = path[len(mountpoint):].lstrip(os.path.sep)
    try:
        (minion, pushed_file) = path.split(os.sep, 1)
    except ValueError:
        return fnd
    if not _is_exposed(minion):
        return fnd
    full = os.path.join(__opts__['cachedir'], 'minions', minion, 'files', pushed_file)
    if os.path.isfile(full) and (not salt.fileserver.is_file_ignored(__opts__, full)):
        fnd['path'] = full
        fnd['rel'] = path
        fnd['stat'] = list(os.stat(full))
        return fnd
    return fnd

def envs():
    if False:
        i = 10
        return i + 15
    '\n    Returns the one environment specified for minionfs in the master\n    configuration.\n    '
    return [__opts__['minionfs_env']]

def serve_file(load, fnd):
    if False:
        while True:
            i = 10
    "\n    Return a chunk from a file based on the data received\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Push the file to the master\n        $ salt 'source-minion' cp.push /path/to/the/file\n        $ salt 'destination-minion' cp.get_file salt://source-minion/path/to/the/file /destination/file\n    "
    ret = {'data': '', 'dest': ''}
    if not fnd['path']:
        return ret
    ret['dest'] = fnd['rel']
    gzip = load.get('gzip', None)
    fpath = os.path.normpath(fnd['path'])
    with salt.utils.files.fopen(fpath, 'rb') as fp_:
        fp_.seek(load['loc'])
        data = fp_.read(__opts__['file_buffer_size'])
        if data and (not salt.utils.files.is_binary(fpath)):
            data = data.decode(__salt_system_encoding__)
        if gzip and data:
            data = salt.utils.gzip_util.compress(data, gzip)
            ret['gzip'] = gzip
        ret['data'] = data
    return ret

def update():
    if False:
        return 10
    '\n    When we are asked to update (regular interval) lets reap the cache\n    '
    try:
        salt.fileserver.reap_fileserver_cache_dir(os.path.join(__opts__['cachedir'], 'minionfs/hash'), find_file)
    except os.error:
        pass

def file_hash(load, fnd):
    if False:
        return 10
    '\n    Return a file hash, the hash type is set in the master config file\n    '
    path = fnd['path']
    ret = {}
    if 'env' in load:
        load.pop('env')
    if load['saltenv'] not in envs():
        return {}
    if not path or not os.path.isfile(path):
        return ret
    ret['hash_type'] = __opts__['hash_type']
    cache_path = os.path.join(__opts__['cachedir'], 'minionfs', 'hash', load['saltenv'], '{}.hash.{}'.format(fnd['rel'], __opts__['hash_type']))
    if os.path.exists(cache_path):
        try:
            with salt.utils.files.fopen(cache_path, 'rb') as fp_:
                try:
                    (hsum, mtime) = salt.utils.stringutils.to_unicode(fp_.read()).split(':')
                except ValueError:
                    log.debug('Fileserver attempted to read incomplete cache file. Retrying.')
                    file_hash(load, fnd)
                    return ret
                if os.path.getmtime(path) == mtime:
                    ret['hsum'] = hsum
                    return ret
        except os.error:
            log.debug('Fileserver encountered lock when reading cache file. Retrying.')
            file_hash(load, fnd)
            return ret
    ret['hsum'] = salt.utils.hashutils.get_hash(path, __opts__['hash_type'])
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_object = '{}:{}'.format(ret['hsum'], os.path.getmtime(path))
    with salt.utils.files.flopen(cache_path, 'w') as fp_:
        fp_.write(cache_object)
    return ret

def file_list(load):
    if False:
        while True:
            i = 10
    '\n    Return a list of all files on the file server in a specified environment\n    '
    if 'env' in load:
        load.pop('env')
    if load['saltenv'] not in envs():
        return []
    mountpoint = salt.utils.url.strip_proto(__opts__['minionfs_mountpoint'])
    prefix = load.get('prefix', '').strip('/')
    if mountpoint and prefix.startswith(mountpoint + os.path.sep):
        prefix = prefix[len(mountpoint + os.path.sep):]
    minions_cache_dir = os.path.join(__opts__['cachedir'], 'minions')
    minion_dirs = os.listdir(minions_cache_dir)
    if prefix:
        (tgt_minion, _, prefix) = prefix.partition('/')
        if not prefix:
            return []
        if tgt_minion not in minion_dirs:
            log.warning("No files found in minionfs cache for minion ID '%s'", tgt_minion)
            return []
        minion_dirs = [tgt_minion]
    ret = []
    for minion in minion_dirs:
        if not _is_exposed(minion):
            continue
        minion_files_dir = os.path.join(minions_cache_dir, minion, 'files')
        if not os.path.isdir(minion_files_dir):
            log.debug('minionfs: could not find files directory under %s!', os.path.join(minions_cache_dir, minion))
            continue
        walk_dir = os.path.join(minion_files_dir, prefix)
        for (root, _, files) in salt.utils.path.os_walk(walk_dir, followlinks=False):
            for fname in files:
                if os.path.islink(os.path.join(root, fname)):
                    continue
                relpath = os.path.relpath(os.path.join(root, fname), minion_files_dir)
                if relpath.startswith('../'):
                    continue
                rel_fn = os.path.join(mountpoint, minion, relpath)
                if not salt.fileserver.is_file_ignored(__opts__, rel_fn):
                    ret.append(rel_fn)
    return ret

def dir_list(load):
    if False:
        print('Hello World!')
    "\n    Return a list of all directories on the master\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        $ salt 'source-minion' cp.push /absolute/path/file  # Push the file to the master\n        $ salt 'destination-minion' cp.list_master_dirs\n        destination-minion:\n            - source-minion/absolute\n            - source-minion/absolute/path\n    "
    if 'env' in load:
        load.pop('env')
    if load['saltenv'] not in envs():
        return []
    mountpoint = salt.utils.url.strip_proto(__opts__['minionfs_mountpoint'])
    prefix = load.get('prefix', '').strip('/')
    if mountpoint and prefix.startswith(mountpoint + os.path.sep):
        prefix = prefix[len(mountpoint + os.path.sep):]
    minions_cache_dir = os.path.join(__opts__['cachedir'], 'minions')
    minion_dirs = os.listdir(minions_cache_dir)
    if prefix:
        (tgt_minion, _, prefix) = prefix.partition('/')
        if not prefix:
            return []
        if tgt_minion not in minion_dirs:
            log.warning("No files found in minionfs cache for minion ID '%s'", tgt_minion)
            return []
        minion_dirs = [tgt_minion]
    ret = []
    for minion in os.listdir(minions_cache_dir):
        if not _is_exposed(minion):
            continue
        minion_files_dir = os.path.join(minions_cache_dir, minion, 'files')
        if not os.path.isdir(minion_files_dir):
            log.warning('minionfs: could not find files directory under %s!', os.path.join(minions_cache_dir, minion))
            continue
        walk_dir = os.path.join(minion_files_dir, prefix)
        for (root, _, _) in salt.utils.path.os_walk(walk_dir, followlinks=False):
            relpath = os.path.relpath(root, minion_files_dir)
            if relpath in ('.', '..') or relpath.startswith('../'):
                continue
            ret.append(os.path.join(mountpoint, minion, relpath))
    return ret