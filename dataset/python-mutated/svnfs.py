"""
Subversion Fileserver Backend

After enabling this backend, branches and tags in a remote subversion
repository are exposed to salt as different environments. To enable this
backend, add ``svnfs`` to the :conf_master:`fileserver_backend` option in the
Master config file.

.. code-block:: yaml

    fileserver_backend:
      - svnfs

.. note::
    ``svn`` also works here. Prior to the 2018.3.0 release, *only* ``svn``
    would work.

This backend assumes a standard svn layout with directories for ``branches``,
``tags``, and ``trunk``, at the repository root.

:depends:   - subversion
            - pysvn

.. versionchanged:: 2014.7.0
    The paths to the trunk, branches, and tags have been made configurable, via
    the config options :conf_master:`svnfs_trunk`,
    :conf_master:`svnfs_branches`, and :conf_master:`svnfs_tags`.
    :conf_master:`svnfs_mountpoint` was also added. Finally, support for
    per-remote configuration parameters was added. See the
    :conf_master:`documentation <svnfs_remotes>` for more information.
"""
import copy
import errno
import fnmatch
import hashlib
import logging
import os
import shutil
from datetime import datetime
import salt.fileserver
import salt.utils.data
import salt.utils.files
import salt.utils.gzip_util
import salt.utils.hashutils
import salt.utils.path
import salt.utils.stringutils
import salt.utils.url
import salt.utils.versions
from salt.exceptions import FileserverConfigError
from salt.utils.event import tagify
PER_REMOTE_OVERRIDES = ('mountpoint', 'root', 'trunk', 'branches', 'tags')
HAS_SVN = False
try:
    import pysvn
    HAS_SVN = True
    CLIENT = pysvn.Client()
except ImportError:
    pass
log = logging.getLogger(__name__)
__virtualname__ = 'svnfs'
__virtual_aliases__ = ('svn',)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if subversion is available\n    '
    if __virtualname__ not in __opts__['fileserver_backend']:
        return False
    if not HAS_SVN:
        log.error('Subversion fileserver backend is enabled in configuration but could not be loaded, is pysvn installed?')
        return False
    errors = []
    for param in ('svnfs_trunk', 'svnfs_branches', 'svnfs_tags'):
        if os.path.isabs(__opts__[param]):
            errors.append("Master configuration parameter '{}' (value: {}) cannot be an absolute path".format(param, __opts__[param]))
    if errors:
        for error in errors:
            log.error(error)
        log.error('Subversion fileserver backed will be disabled')
        return False
    return __virtualname__

def _rev(repo):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns revision ID of repo\n    '
    try:
        repo_info = dict(CLIENT.info(repo['repo']).items())
    except (pysvn._pysvn.ClientError, TypeError, KeyError, AttributeError) as exc:
        log.error('Error retrieving revision ID for svnfs remote %s (cachedir: %s): %s', repo['url'], repo['repo'], exc)
    else:
        return repo_info['revision'].number
    return None

def _failhard():
    if False:
        return 10
    '\n    Fatal fileserver configuration issue, raise an exception\n    '
    raise FileserverConfigError('Failed to load svn fileserver backend')

def init():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the list of svn remotes and their configuration information\n    '
    bp_ = os.path.join(__opts__['cachedir'], 'svnfs')
    new_remote = False
    repos = []
    per_remote_defaults = {}
    for param in PER_REMOTE_OVERRIDES:
        per_remote_defaults[param] = str(__opts__['svnfs_{}'.format(param)])
    for remote in __opts__['svnfs_remotes']:
        repo_conf = copy.deepcopy(per_remote_defaults)
        if isinstance(remote, dict):
            repo_url = next(iter(remote))
            per_remote_conf = {key: str(val) for (key, val) in salt.utils.data.repack_dictlist(remote[repo_url]).items()}
            if not per_remote_conf:
                log.error('Invalid per-remote configuration for remote %s. If no per-remote parameters are being specified, there may be a trailing colon after the URL, which should be removed. Check the master configuration file.', repo_url)
                _failhard()
            per_remote_errors = False
            for param in (x for x in per_remote_conf if x not in PER_REMOTE_OVERRIDES):
                log.error("Invalid configuration parameter '%s' for remote %s. Valid parameters are: %s. See the documentation for further information.", param, repo_url, ', '.join(PER_REMOTE_OVERRIDES))
                per_remote_errors = True
            if per_remote_errors:
                _failhard()
            repo_conf.update(per_remote_conf)
        else:
            repo_url = remote
        if not isinstance(repo_url, str):
            log.error('Invalid svnfs remote %s. Remotes must be strings, you may need to enclose the URL in quotes', repo_url)
            _failhard()
        try:
            repo_conf['mountpoint'] = salt.utils.url.strip_proto(repo_conf['mountpoint'])
        except TypeError:
            pass
        hash_type = getattr(hashlib, __opts__.get('hash_type', 'md5'))
        repo_hash = hash_type(repo_url).hexdigest()
        rp_ = os.path.join(bp_, repo_hash)
        if not os.path.isdir(rp_):
            os.makedirs(rp_)
        if not os.listdir(rp_):
            try:
                CLIENT.checkout(repo_url, rp_)
                repos.append(rp_)
                new_remote = True
            except pysvn._pysvn.ClientError as exc:
                log.error("Failed to initialize svnfs remote '%s': %s", repo_url, exc)
                _failhard()
        else:
            try:
                CLIENT.status(rp_)
            except pysvn._pysvn.ClientError as exc:
                log.error('Cache path %s (corresponding remote: %s) exists but is not a valid subversion checkout. You will need to manually delete this directory on the master to continue to use this svnfs remote.', rp_, repo_url)
                _failhard()
        repo_conf.update({'repo': rp_, 'url': repo_url, 'hash': repo_hash, 'cachedir': rp_, 'lockfile': os.path.join(rp_, 'update.lk')})
        repos.append(repo_conf)
    if new_remote:
        remote_map = os.path.join(__opts__['cachedir'], 'svnfs/remote_map.txt')
        try:
            with salt.utils.files.fopen(remote_map, 'w+') as fp_:
                timestamp = datetime.now().strftime('%d %b %Y %H:%M:%S.%f')
                fp_.write('# svnfs_remote map as of {}\n'.format(timestamp))
                for repo_conf in repos:
                    fp_.write(salt.utils.stringutils.to_str('{} = {}\n'.format(repo_conf['hash'], repo_conf['url'])))
        except OSError:
            pass
        else:
            log.info('Wrote new svnfs_remote map to %s', remote_map)
    return repos

def _clear_old_remotes():
    if False:
        i = 10
        return i + 15
    '\n    Remove cache directories for remotes no longer configured\n    '
    bp_ = os.path.join(__opts__['cachedir'], 'svnfs')
    try:
        cachedir_ls = os.listdir(bp_)
    except OSError:
        cachedir_ls = []
    repos = init()
    for repo in repos:
        try:
            cachedir_ls.remove(repo['hash'])
        except ValueError:
            pass
    to_remove = []
    for item in cachedir_ls:
        if item in ('hash', 'refs'):
            continue
        path = os.path.join(bp_, item)
        if os.path.isdir(path):
            to_remove.append(path)
    failed = []
    if to_remove:
        for rdir in to_remove:
            try:
                shutil.rmtree(rdir)
            except OSError as exc:
                log.error('Unable to remove old svnfs remote cachedir %s: %s', rdir, exc)
                failed.append(rdir)
            else:
                log.debug('svnfs removed old cachedir %s', rdir)
    for fdir in failed:
        to_remove.remove(fdir)
    return (bool(to_remove), repos)

def clear_cache():
    if False:
        i = 10
        return i + 15
    '\n    Completely clear svnfs cache\n    '
    fsb_cachedir = os.path.join(__opts__['cachedir'], 'svnfs')
    list_cachedir = os.path.join(__opts__['cachedir'], 'file_lists/svnfs')
    errors = []
    for rdir in (fsb_cachedir, list_cachedir):
        if os.path.exists(rdir):
            try:
                shutil.rmtree(rdir)
            except OSError as exc:
                errors.append('Unable to delete {}: {}'.format(rdir, exc))
    return errors

def clear_lock(remote=None):
    if False:
        print('Hello World!')
    '\n    Clear update.lk\n\n    ``remote`` can either be a dictionary containing repo configuration\n    information, or a pattern. If the latter, then remotes for which the URL\n    matches the pattern will be locked.\n    '

    def _do_clear_lock(repo):
        if False:
            i = 10
            return i + 15

        def _add_error(errlist, repo, exc):
            if False:
                print('Hello World!')
            msg = 'Unable to remove update lock for {} ({}): {} '.format(repo['url'], repo['lockfile'], exc)
            log.debug(msg)
            errlist.append(msg)
        success = []
        failed = []
        if os.path.exists(repo['lockfile']):
            try:
                os.remove(repo['lockfile'])
            except OSError as exc:
                if exc.errno == errno.EISDIR:
                    try:
                        shutil.rmtree(repo['lockfile'])
                    except OSError as exc:
                        _add_error(failed, repo, exc)
                else:
                    _add_error(failed, repo, exc)
            else:
                msg = 'Removed lock for {}'.format(repo['url'])
                log.debug(msg)
                success.append(msg)
        return (success, failed)
    if isinstance(remote, dict):
        return _do_clear_lock(remote)
    cleared = []
    errors = []
    for repo in init():
        if remote:
            try:
                if remote not in repo['url']:
                    continue
            except TypeError:
                if str(remote) not in repo['url']:
                    continue
        (success, failed) = _do_clear_lock(repo)
        cleared.extend(success)
        errors.extend(failed)
    return (cleared, errors)

def lock(remote=None):
    if False:
        print('Hello World!')
    '\n    Place an update.lk\n\n    ``remote`` can either be a dictionary containing repo configuration\n    information, or a pattern. If the latter, then remotes for which the URL\n    matches the pattern will be locked.\n    '

    def _do_lock(repo):
        if False:
            print('Hello World!')
        success = []
        failed = []
        if not os.path.exists(repo['lockfile']):
            try:
                with salt.utils.files.fopen(repo['lockfile'], 'w+') as fp_:
                    fp_.write('')
            except OSError as exc:
                msg = 'Unable to set update lock for {} ({}): {} '.format(repo['url'], repo['lockfile'], exc)
                log.debug(msg)
                failed.append(msg)
            else:
                msg = 'Set lock for {}'.format(repo['url'])
                log.debug(msg)
                success.append(msg)
        return (success, failed)
    if isinstance(remote, dict):
        return _do_lock(remote)
    locked = []
    errors = []
    for repo in init():
        if remote:
            try:
                if not fnmatch.fnmatch(repo['url'], remote):
                    continue
            except TypeError:
                if not fnmatch.fnmatch(repo['url'], str(remote)):
                    continue
        (success, failed) = _do_lock(repo)
        locked.extend(success)
        errors.extend(failed)
    return (locked, errors)

def update():
    if False:
        print('Hello World!')
    '\n    Execute an svn update on all of the repos\n    '
    data = {'changed': False, 'backend': 'svnfs'}
    (data['changed'], repos) = _clear_old_remotes()
    for repo in repos:
        if os.path.exists(repo['lockfile']):
            log.warning("Update lockfile is present for svnfs remote %s, skipping. If this warning persists, it is possible that the update process was interrupted. Removing %s or running 'salt-run fileserver.clear_lock svnfs' will allow updates to continue for this remote.", repo['url'], repo['lockfile'])
            continue
        (_, errors) = lock(repo)
        if errors:
            log.error('Unable to set update lock for svnfs remote %s, skipping.', repo['url'])
            continue
        log.debug('svnfs is fetching from %s', repo['url'])
        old_rev = _rev(repo)
        try:
            CLIENT.update(repo['repo'])
        except pysvn._pysvn.ClientError as exc:
            log.error('Error updating svnfs remote %s (cachedir: %s): %s', repo['url'], repo['cachedir'], exc)
        new_rev = _rev(repo)
        if any((x is None for x in (old_rev, new_rev))):
            continue
        if new_rev != old_rev:
            data['changed'] = True
        clear_lock(repo)
    env_cache = os.path.join(__opts__['cachedir'], 'svnfs/envs.p')
    if data.get('changed', False) is True or not os.path.isfile(env_cache):
        env_cachedir = os.path.dirname(env_cache)
        if not os.path.exists(env_cachedir):
            os.makedirs(env_cachedir)
        new_envs = envs(ignore_cache=True)
        with salt.utils.files.fopen(env_cache, 'wb+') as fp_:
            fp_.write(salt.payload.dumps(new_envs))
            log.trace('Wrote env cache data to %s', env_cache)
    if __opts__.get('fileserver_events', False):
        with salt.utils.event.get_event('master', __opts__['sock_dir'], opts=__opts__, listen=False) as event:
            event.fire_event(data, tagify(['svnfs', 'update'], prefix='fileserver'))
    try:
        salt.fileserver.reap_fileserver_cache_dir(os.path.join(__opts__['cachedir'], 'svnfs/hash'), find_file)
    except OSError:
        pass

def _env_is_exposed(env):
    if False:
        return 10
    '\n    Check if an environment is exposed by comparing it against a whitelist and\n    blacklist.\n    '
    return salt.utils.stringutils.check_whitelist_blacklist(env, whitelist=__opts__['svnfs_saltenv_whitelist'], blacklist=__opts__['svnfs_saltenv_blacklist'])

def envs(ignore_cache=False):
    if False:
        return 10
    '\n    Return a list of refs that can be used as environments\n    '
    if not ignore_cache:
        env_cache = os.path.join(__opts__['cachedir'], 'svnfs/envs.p')
        cache_match = salt.fileserver.check_env_cache(__opts__, env_cache)
        if cache_match is not None:
            return cache_match
    ret = set()
    for repo in init():
        trunk = os.path.join(repo['repo'], repo['trunk'])
        if os.path.isdir(trunk):
            ret.add('base')
        else:
            log.error("svnfs trunk path '%s' does not exist in repo %s, no base environment will be provided by this remote", repo['trunk'], repo['url'])
        branches = os.path.join(repo['repo'], repo['branches'])
        if os.path.isdir(branches):
            ret.update(os.listdir(branches))
        else:
            log.error("svnfs branches path '%s' does not exist in repo %s", repo['branches'], repo['url'])
        tags = os.path.join(repo['repo'], repo['tags'])
        if os.path.isdir(tags):
            ret.update(os.listdir(tags))
        else:
            log.error("svnfs tags path '%s' does not exist in repo %s", repo['tags'], repo['url'])
    return [x for x in sorted(ret) if _env_is_exposed(x)]

def _env_root(repo, saltenv):
    if False:
        while True:
            i = 10
    '\n    Return the root of the directory corresponding to the desired environment,\n    or None if the environment was not found.\n    '
    if saltenv == 'base':
        trunk = os.path.join(repo['repo'], repo['trunk'])
        if os.path.isdir(trunk):
            return trunk
        else:
            return None
    branches = os.path.join(repo['repo'], repo['branches'])
    if os.path.isdir(branches) and saltenv in os.listdir(branches):
        return os.path.join(branches, saltenv)
    tags = os.path.join(repo['repo'], repo['tags'])
    if os.path.isdir(tags) and saltenv in os.listdir(tags):
        return os.path.join(tags, saltenv)
    return None

def find_file(path, tgt_env='base', **kwargs):
    if False:
        return 10
    '\n    Find the first file to match the path and ref. This operates similarly to\n    the roots file sever but with assumptions of the directory structure\n    based on svn standard practices.\n    '
    fnd = {'path': '', 'rel': ''}
    if os.path.isabs(path) or tgt_env not in envs():
        return fnd
    for repo in init():
        env_root = _env_root(repo, tgt_env)
        if env_root is None:
            continue
        if repo['mountpoint'] and (not path.startswith(repo['mountpoint'] + os.path.sep)):
            continue
        repo_path = path[len(repo['mountpoint']):].lstrip(os.path.sep)
        if repo['root']:
            repo_path = os.path.join(repo['root'], repo_path)
        full = os.path.join(env_root, repo_path)
        if os.path.isfile(full):
            fnd['rel'] = path
            fnd['path'] = full
            try:
                fnd['stat'] = list(os.stat(full))
            except Exception:
                pass
            return fnd
    return fnd

def serve_file(load, fnd):
    if False:
        print('Hello World!')
    '\n    Return a chunk from a file based on the data received\n    '
    if 'env' in load:
        load.pop('env')
    ret = {'data': '', 'dest': ''}
    if not all((x in load for x in ('path', 'loc', 'saltenv'))):
        return ret
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

def file_hash(load, fnd):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a file hash, the hash type is set in the master config file\n    '
    if 'env' in load:
        load.pop('env')
    if not all((x in load for x in ('path', 'saltenv'))):
        return ''
    saltenv = load['saltenv']
    if saltenv == 'base':
        saltenv = 'trunk'
    ret = {}
    relpath = fnd['rel']
    path = fnd['path']
    if not path or not os.path.isfile(path):
        return ret
    ret['hash_type'] = __opts__['hash_type']
    cache_path = os.path.join(__opts__['cachedir'], 'svnfs', 'hash', saltenv, '{}.hash.{}'.format(relpath, __opts__['hash_type']))
    if os.path.exists(cache_path):
        with salt.utils.files.fopen(cache_path, 'rb') as fp_:
            (hsum, mtime) = fp_.read().split(':')
            if os.path.getmtime(path) == mtime:
                ret['hsum'] = hsum
                return ret
    ret['hsum'] = salt.utils.hashutils.get_hash(path, __opts__['hash_type'])
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with salt.utils.files.fopen(cache_path, 'w') as fp_:
        fp_.write('{}:{}'.format(ret['hsum'], os.path.getmtime(path)))
    return ret

def _file_lists(load, form):
    if False:
        print('Hello World!')
    '\n    Return a dict containing the file lists for files, dirs, emptydirs and symlinks\n    '
    if 'env' in load:
        load.pop('env')
    if 'saltenv' not in load or load['saltenv'] not in envs():
        return []
    list_cachedir = os.path.join(__opts__['cachedir'], 'file_lists/svnfs')
    if not os.path.isdir(list_cachedir):
        try:
            os.makedirs(list_cachedir)
        except os.error:
            log.critical('Unable to make cachedir %s', list_cachedir)
            return []
    list_cache = os.path.join(list_cachedir, '{}.p'.format(load['saltenv']))
    w_lock = os.path.join(list_cachedir, '.{}.w'.format(load['saltenv']))
    (cache_match, refresh_cache, save_cache) = salt.fileserver.check_file_list_cache(__opts__, form, list_cache, w_lock)
    if cache_match is not None:
        return cache_match
    if refresh_cache:
        ret = {'files': set(), 'dirs': set(), 'empty_dirs': set()}
        for repo in init():
            env_root = _env_root(repo, load['saltenv'])
            if env_root is None:
                continue
            if repo['root']:
                env_root = os.path.join(env_root, repo['root']).rstrip(os.path.sep)
                if not os.path.isdir(env_root):
                    continue
            for (root, dirs, files) in salt.utils.path.os_walk(env_root):
                relpath = os.path.relpath(root, env_root)
                dir_rel_fn = os.path.join(repo['mountpoint'], relpath)
                if relpath != '.':
                    ret['dirs'].add(dir_rel_fn)
                    if not dirs and (not files):
                        ret['empty_dirs'].add(dir_rel_fn)
                for fname in files:
                    rel_fn = os.path.relpath(os.path.join(root, fname), env_root)
                    ret['files'].add(os.path.join(repo['mountpoint'], rel_fn))
        if repo['mountpoint']:
            ret['dirs'].add(repo['mountpoint'])
        for key in ret:
            ret[key] = sorted(ret[key])
        if save_cache:
            salt.fileserver.write_file_list_cache(__opts__, ret, list_cache, w_lock)
        return ret.get(form, [])
    return []

def file_list(load):
    if False:
        while True:
            i = 10
    '\n    Return a list of all files on the file server in a specified\n    environment\n    '
    return _file_lists(load, 'files')

def file_list_emptydirs(load):
    if False:
        return 10
    '\n    Return a list of all empty directories on the master\n    '
    return _file_lists(load, 'empty_dirs')

def dir_list(load):
    if False:
        while True:
            i = 10
    '\n    Return a list of all directories on the master\n    '
    return _file_lists(load, 'dirs')