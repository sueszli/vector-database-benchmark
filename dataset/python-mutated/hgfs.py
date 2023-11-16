"""
Mercurial Fileserver Backend

To enable, add ``hgfs`` to the :conf_master:`fileserver_backend` option in the
Master config file.

.. code-block:: yaml

    fileserver_backend:
      - hgfs

.. note::
    ``hg`` also works here. Prior to the 2018.3.0 release, *only* ``hg`` would
    work.

After enabling this backend, branches, bookmarks, and tags in a remote
mercurial repository are exposed to salt as different environments. This
feature is managed by the :conf_master:`fileserver_backend` option in the salt
master config file.

This fileserver has an additional option :conf_master:`hgfs_branch_method` that
will set the desired branch method. Possible values are: ``branches``,
``bookmarks``, or ``mixed``. If using ``branches`` or ``mixed``, the
``default`` branch will be mapped to ``base``.


.. versionchanged:: 2014.1.0
    The :conf_master:`hgfs_base` master config parameter was added, allowing
    for a branch other than ``default`` to be used for the ``base``
    environment, and allowing for a ``base`` environment to be specified when
    using an :conf_master:`hgfs_branch_method` of ``bookmarks``.


:depends:   - mercurial
            - python bindings for mercurial (``python-hglib``)
"""
import copy
import errno
import fnmatch
import glob
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
import salt.utils.stringutils
import salt.utils.url
import salt.utils.versions
from salt.exceptions import FileserverConfigError
from salt.utils.event import tagify
VALID_BRANCH_METHODS = ('branches', 'bookmarks', 'mixed')
PER_REMOTE_OVERRIDES = ('base', 'branch_method', 'mountpoint', 'root')
try:
    import hglib
    HAS_HG = True
except ImportError:
    HAS_HG = False
log = logging.getLogger(__name__)
__virtualname__ = 'hgfs'
__virtual_aliases__ = ('hg',)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if mercurial is available\n    '
    if __virtualname__ not in __opts__['fileserver_backend']:
        return False
    if not HAS_HG:
        log.error('Mercurial fileserver backend is enabled in configuration but could not be loaded, is hglib installed?')
        return False
    if __opts__['hgfs_branch_method'] not in VALID_BRANCH_METHODS:
        log.error("Invalid hgfs_branch_method '%s'. Valid methods are: %s", __opts__['hgfs_branch_method'], VALID_BRANCH_METHODS)
        return False
    if salt.utils.path.which('hg') is None:
        log.error('hgfs requested but hg executable is not available.')
        return False
    return __virtualname__

def _all_branches(repo):
    if False:
        return 10
    '\n    Returns all branches for the specified repo\n    '
    branches = [(salt.utils.stringutils.to_str(x[0]), x[1], salt.utils.stringutils.to_str(x[2])) for x in repo.branches()]
    return branches

def _get_branch(repo, name):
    if False:
        i = 10
        return i + 15
    '\n    Find the requested branch in the specified repo\n    '
    try:
        return [x for x in _all_branches(repo) if x[0] == name][0]
    except IndexError:
        return False

def _all_bookmarks(repo):
    if False:
        while True:
            i = 10
    '\n    Returns all bookmarks for the specified repo\n    '
    bookmarks = [(salt.utils.stringutils.to_str(x[0]), x[1], salt.utils.stringutils.to_str(x[2])) for x in repo.bookmarks()[0]]
    return bookmarks

def _get_bookmark(repo, name):
    if False:
        return 10
    '\n    Find the requested bookmark in the specified repo\n    '
    try:
        return [x for x in _all_bookmarks(repo) if x[0] == name][0]
    except IndexError:
        return False

def _all_tags(repo):
    if False:
        print('Hello World!')
    '\n    Returns all tags for the specified repo\n    '
    return [(salt.utils.stringutils.to_str(x[0]), x[1], salt.utils.stringutils.to_str(x[2]), x[3]) for x in repo.tags() if salt.utils.stringutils.to_str(x[0]) != 'tip']

def _get_tag(repo, name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find the requested tag in the specified repo\n    '
    try:
        return [x for x in _all_tags(repo) if x[0] == name][0]
    except IndexError:
        return False

def _get_ref(repo, name):
    if False:
        while True:
            i = 10
    '\n    Return ref tuple if ref is in the repo.\n    '
    if name == 'base':
        name = repo['base']
    if name == repo['base'] or name in envs():
        if repo['branch_method'] == 'branches':
            return _get_branch(repo['repo'], name) or _get_tag(repo['repo'], name)
        elif repo['branch_method'] == 'bookmarks':
            return _get_bookmark(repo['repo'], name) or _get_tag(repo['repo'], name)
        elif repo['branch_method'] == 'mixed':
            return _get_branch(repo['repo'], name) or _get_bookmark(repo['repo'], name) or _get_tag(repo['repo'], name)
    return False

def _get_manifest(repo, ref):
    if False:
        return 10
    '\n    Get manifest for ref\n    '
    manifest = [(salt.utils.stringutils.to_str(x[0]), salt.utils.stringutils.to_str(x[1]), x[2], x[3], salt.utils.stringutils.to_str(x[4])) for x in repo.manifest(rev=ref[1])]
    return manifest

def _failhard():
    if False:
        print('Hello World!')
    '\n    Fatal fileserver configuration issue, raise an exception\n    '
    raise FileserverConfigError('Failed to load hg fileserver backend')

def init():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of hglib objects for the various hgfs remotes\n    '
    bp_ = os.path.join(__opts__['cachedir'], 'hgfs')
    new_remote = False
    repos = []
    per_remote_defaults = {}
    for param in PER_REMOTE_OVERRIDES:
        per_remote_defaults[param] = str(__opts__['hgfs_{}'.format(param)])
    for remote in __opts__['hgfs_remotes']:
        repo_conf = copy.deepcopy(per_remote_defaults)
        if isinstance(remote, dict):
            repo_url = next(iter(remote))
            per_remote_conf = {key: str(val) for (key, val) in salt.utils.data.repack_dictlist(remote[repo_url]).items()}
            if not per_remote_conf:
                log.error('Invalid per-remote configuration for hgfs remote %s. If no per-remote parameters are being specified, there may be a trailing colon after the URL, which should be removed. Check the master configuration file.', repo_url)
                _failhard()
            branch_method = per_remote_conf.get('branch_method', per_remote_defaults['branch_method'])
            if branch_method not in VALID_BRANCH_METHODS:
                log.error("Invalid branch_method '%s' for remote %s. Valid branch methods are: %s. This remote will be ignored.", branch_method, repo_url, ', '.join(VALID_BRANCH_METHODS))
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
            log.error('Invalid hgfs remote %s. Remotes must be strings, you may need to enclose the URL in quotes', repo_url)
            _failhard()
        try:
            repo_conf['mountpoint'] = salt.utils.url.strip_proto(repo_conf['mountpoint'])
        except TypeError:
            pass
        hash_type = getattr(hashlib, __opts__.get('hash_type', 'md5'))
        repo_hash = hash_type(repo_url.encode('utf-8')).hexdigest()
        rp_ = os.path.join(bp_, repo_hash)
        if not os.path.isdir(rp_):
            os.makedirs(rp_)
        if not os.listdir(rp_):
            client = hglib.init(rp_)
            client.close()
            new_remote = True
        repo = None
        try:
            try:
                repo = hglib.open(rp_)
            except hglib.error.ServerError:
                log.error('Cache path %s (corresponding remote: %s) exists but is not a valid mercurial repository. You will need to manually delete this directory on the master to continue to use this hgfs remote.', rp_, repo_url)
                _failhard()
            except Exception as exc:
                log.error("Exception '%s' encountered while initializing hgfs remote %s", exc, repo_url)
                _failhard()
            try:
                refs = repo.config(names=b'paths')
            except hglib.error.CommandError:
                refs = None
            if not refs:
                hgconfpath = os.path.join(rp_, '.hg', 'hgrc')
                with salt.utils.files.fopen(hgconfpath, 'w+') as hgconfig:
                    hgconfig.write('[paths]\n')
                    hgconfig.write(salt.utils.stringutils.to_str('default = {}\n'.format(repo_url)))
            repo_conf.update({'repo': repo, 'url': repo_url, 'hash': repo_hash, 'cachedir': rp_, 'lockfile': os.path.join(__opts__['cachedir'], 'hgfs', '{}.update.lk'.format(repo_hash))})
            repos.append(repo_conf)
        finally:
            if repo:
                repo.close()
    if new_remote:
        remote_map = os.path.join(__opts__['cachedir'], 'hgfs/remote_map.txt')
        try:
            with salt.utils.files.fopen(remote_map, 'w+') as fp_:
                timestamp = datetime.now().strftime('%d %b %Y %H:%M:%S.%f')
                fp_.write('# hgfs_remote map as of {}\n'.format(timestamp))
                for repo in repos:
                    fp_.write(salt.utils.stringutils.to_str('{} = {}\n'.format(repo['hash'], repo['url'])))
        except OSError:
            pass
        else:
            log.info('Wrote new hgfs_remote map to %s', remote_map)
    return repos

def _clear_old_remotes():
    if False:
        while True:
            i = 10
    '\n    Remove cache directories for remotes no longer configured\n    '
    bp_ = os.path.join(__opts__['cachedir'], 'hgfs')
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
                log.error('Unable to remove old hgfs remote cachedir %s: %s', rdir, exc)
                failed.append(rdir)
            else:
                log.debug('hgfs removed old cachedir %s', rdir)
    for fdir in failed:
        to_remove.remove(fdir)
    return (bool(to_remove), repos)

def clear_cache():
    if False:
        return 10
    '\n    Completely clear hgfs cache\n    '
    fsb_cachedir = os.path.join(__opts__['cachedir'], 'hgfs')
    list_cachedir = os.path.join(__opts__['cachedir'], 'file_lists/hgfs')
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
            for i in range(10):
                print('nop')

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
        try:
            if remote:
                try:
                    if not fnmatch.fnmatch(repo['url'], remote):
                        continue
                except TypeError:
                    if not fnmatch.fnmatch(repo['url'], str(remote)):
                        continue
            (success, failed) = _do_clear_lock(repo)
            cleared.extend(success)
            errors.extend(failed)
        finally:
            repo['repo'].close()
    return (cleared, errors)

def lock(remote=None):
    if False:
        print('Hello World!')
    '\n    Place an update.lk\n\n    ``remote`` can either be a dictionary containing repo configuration\n    information, or a pattern. If the latter, then remotes for which the URL\n    matches the pattern will be locked.\n    '

    def _do_lock(repo):
        if False:
            i = 10
            return i + 15
        success = []
        failed = []
        if not os.path.exists(repo['lockfile']):
            try:
                with salt.utils.files.fopen(repo['lockfile'], 'w'):
                    pass
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
        try:
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
        finally:
            repo['repo'].close()
    return (locked, errors)

def update():
    if False:
        while True:
            i = 10
    '\n    Execute an hg pull on all of the repos\n    '
    data = {'changed': False, 'backend': 'hgfs'}
    (data['changed'], repos) = _clear_old_remotes()
    for repo in repos:
        try:
            if os.path.exists(repo['lockfile']):
                log.warning("Update lockfile is present for hgfs remote %s, skipping. If this warning persists, it is possible that the update process was interrupted. Removing %s or running 'salt-run fileserver.clear_lock hgfs' will allow updates to continue for this remote.", repo['url'], repo['lockfile'])
                continue
            (_, errors) = lock(repo)
            if errors:
                log.error('Unable to set update lock for hgfs remote %s, skipping.', repo['url'])
                continue
            log.debug('hgfs is fetching from %s', repo['url'])
            repo['repo'].open()
            curtip = repo['repo'].tip()
            try:
                repo['repo'].pull()
            except Exception as exc:
                log.error('Exception %s caught while updating hgfs remote %s', exc, repo['url'], exc_info_on_loglevel=logging.DEBUG)
            else:
                newtip = repo['repo'].tip()
                if curtip[1] != newtip[1]:
                    data['changed'] = True
        finally:
            repo['repo'].close()
        clear_lock(repo)
    env_cache = os.path.join(__opts__['cachedir'], 'hgfs/envs.p')
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
            event.fire_event(data, tagify(['hgfs', 'update'], prefix='fileserver'))
    try:
        salt.fileserver.reap_fileserver_cache_dir(os.path.join(__opts__['cachedir'], 'hgfs/hash'), find_file)
    except OSError:
        pass

def _env_is_exposed(env):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if an environment is exposed by comparing it against a whitelist and\n    blacklist.\n    '
    return salt.utils.stringutils.check_whitelist_blacklist(env, whitelist=__opts__['hgfs_saltenv_whitelist'], blacklist=__opts__['hgfs_saltenv_blacklist'])

def envs(ignore_cache=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of refs that can be used as environments\n    '
    if not ignore_cache:
        env_cache = os.path.join(__opts__['cachedir'], 'hgfs/envs.p')
        cache_match = salt.fileserver.check_env_cache(__opts__, env_cache)
        if cache_match is not None:
            return cache_match
    ret = set()
    for repo in init():
        try:
            repo['repo'].open()
            if repo['branch_method'] in ('branches', 'mixed'):
                for branch in _all_branches(repo['repo']):
                    branch_name = branch[0]
                    if branch_name == repo['base']:
                        branch_name = 'base'
                    ret.add(branch_name)
            if repo['branch_method'] in ('bookmarks', 'mixed'):
                for bookmark in _all_bookmarks(repo['repo']):
                    bookmark_name = bookmark[0]
                    if bookmark_name == repo['base']:
                        bookmark_name = 'base'
                    ret.add(bookmark_name)
            ret.update([x[0] for x in _all_tags(repo['repo'])])
        finally:
            repo['repo'].close()
    return [x for x in sorted(ret) if _env_is_exposed(x)]

def find_file(path, tgt_env='base', **kwargs):
    if False:
        print('Hello World!')
    '\n    Find the first file to match the path and ref, read the file out of hg\n    and send the path to the newly cached file\n    '
    fnd = {'path': '', 'rel': ''}
    if os.path.isabs(path) or tgt_env not in envs():
        return fnd
    dest = os.path.join(__opts__['cachedir'], 'hgfs/refs', tgt_env, path)
    hashes_glob = os.path.join(__opts__['cachedir'], 'hgfs/hash', tgt_env, '{}.hash.*'.format(path))
    blobshadest = os.path.join(__opts__['cachedir'], 'hgfs/hash', tgt_env, '{}.hash.blob_sha1'.format(path))
    lk_fn = os.path.join(__opts__['cachedir'], 'hgfs/hash', tgt_env, '{}.lk'.format(path))
    destdir = os.path.dirname(dest)
    hashdir = os.path.dirname(blobshadest)
    if not os.path.isdir(destdir):
        try:
            os.makedirs(destdir)
        except OSError:
            os.remove(destdir)
            os.makedirs(destdir)
    if not os.path.isdir(hashdir):
        try:
            os.makedirs(hashdir)
        except OSError:
            os.remove(hashdir)
            os.makedirs(hashdir)
    for repo in init():
        try:
            if repo['mountpoint'] and (not path.startswith(repo['mountpoint'] + os.path.sep)):
                continue
            repo_path = path[len(repo['mountpoint']):].lstrip(os.path.sep)
            if repo['root']:
                repo_path = os.path.join(repo['root'], repo_path)
            repo['repo'].open()
            ref = _get_ref(repo, tgt_env)
            if not ref:
                repo['repo'].close()
                continue
            salt.fileserver.wait_lock(lk_fn, dest)
            if os.path.isfile(blobshadest) and os.path.isfile(dest):
                with salt.utils.files.fopen(blobshadest, 'r') as fp_:
                    sha = fp_.read()
                    if sha == ref[2]:
                        fnd['rel'] = path
                        fnd['path'] = dest
                        repo['repo'].close()
                        return fnd
            try:
                repo['repo'].cat([salt.utils.stringutils.to_bytes('path:{}'.format(repo_path))], rev=ref[2], output=dest)
            except hglib.error.CommandError:
                repo['repo'].close()
                continue
            with salt.utils.files.fopen(lk_fn, 'w'):
                pass
            for filename in glob.glob(hashes_glob):
                try:
                    os.remove(filename)
                except Exception:
                    pass
            with salt.utils.files.fopen(blobshadest, 'w+') as fp_:
                fp_.write(salt.utils.stringutils.to_str(ref[2]))
            try:
                os.remove(lk_fn)
            except OSError:
                pass
            fnd['rel'] = path
            fnd['path'] = dest
            try:
                fnd['stat'] = list(os.stat(dest))
            except Exception:
                pass
        finally:
            repo['repo'].close()
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
    ret = {'hash_type': __opts__['hash_type']}
    relpath = fnd['rel']
    path = fnd['path']
    hashdest = os.path.join(__opts__['cachedir'], 'hgfs/hash', load['saltenv'], '{}.hash.{}'.format(relpath, __opts__['hash_type']))
    if not os.path.isfile(hashdest):
        ret['hsum'] = salt.utils.hashutils.get_hash(path, __opts__['hash_type'])
        with salt.utils.files.fopen(hashdest, 'w+') as fp_:
            fp_.write(ret['hsum'])
        return ret
    else:
        with salt.utils.files.fopen(hashdest, 'rb') as fp_:
            ret['hsum'] = salt.utils.stringutils.to_unicode(fp_.read())
        return ret

def _file_lists(load, form):
    if False:
        while True:
            i = 10
    '\n    Return a dict containing the file lists for files and dirs\n    '
    if 'env' in load:
        load.pop('env')
    list_cachedir = os.path.join(__opts__['cachedir'], 'file_lists/hgfs')
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
        ret = {}
        ret['files'] = _get_file_list(load)
        ret['dirs'] = _get_dir_list(load)
        if save_cache:
            salt.fileserver.write_file_list_cache(__opts__, ret, list_cache, w_lock)
        return ret.get(form, [])
    return []

def file_list(load):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of all files on the file server in a specified environment\n    '
    return _file_lists(load, 'files')

def _get_file_list(load):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a list of all files on the file server in a specified environment\n    '
    if 'env' in load:
        load.pop('env')
    if 'saltenv' not in load or load['saltenv'] not in envs():
        return []
    ret = set()
    for repo in init():
        try:
            repo['repo'].open()
            ref = _get_ref(repo, load['saltenv'])
            if ref:
                manifest = _get_manifest(repo['repo'], ref=ref)
                for tup in manifest:
                    relpath = os.path.relpath(tup[4], repo['root'])
                    if not relpath.startswith('../'):
                        ret.add(os.path.join(repo['mountpoint'], relpath))
        finally:
            repo['repo'].close()
    return sorted(ret)

def file_list_emptydirs(load):
    if False:
        print('Hello World!')
    '\n    Return a list of all empty directories on the master\n    '
    return []

def dir_list(load):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a list of all directories on the master\n    '
    return _file_lists(load, 'dirs')

def _get_dir_list(load):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a list of all directories on the master\n    '
    if 'env' in load:
        load.pop('env')
    if 'saltenv' not in load or load['saltenv'] not in envs():
        return []
    ret = set()
    for repo in init():
        try:
            repo['repo'].open()
            ref = _get_ref(repo, load['saltenv'])
            if ref:
                manifest = _get_manifest(repo['repo'], ref=ref)
                for tup in manifest:
                    filepath = tup[4]
                    split = filepath.rsplit('/', 1)
                    while len(split) > 1:
                        relpath = os.path.relpath(split[0], repo['root'])
                        if relpath != '.':
                            if not relpath.startswith('../'):
                                ret.add(os.path.join(repo['mountpoint'], relpath))
                        split = split[0].rsplit('/', 1)
        finally:
            repo['repo'].close()
    if repo['mountpoint']:
        ret.add(repo['mountpoint'])
    return sorted(ret)