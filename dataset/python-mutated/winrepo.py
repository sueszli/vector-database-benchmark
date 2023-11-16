"""
Runner to manage Windows software repo
"""
import logging
import os
import salt.loader
import salt.minion
import salt.template
import salt.utils.files
import salt.utils.gitfs
import salt.utils.msgpack
import salt.utils.path
from salt.exceptions import CommandExecutionError, SaltRenderError
log = logging.getLogger(__name__)
PER_REMOTE_OVERRIDES = ('ssl_verify', 'refspecs', 'fallback')
PER_REMOTE_ONLY = salt.utils.gitfs.PER_REMOTE_ONLY
GLOBAL_ONLY = ('branch',)

def _legacy_git():
    if False:
        for i in range(10):
            print('nop')
    return not any((salt.utils.gitfs.GITPYTHON_VERSION, salt.utils.gitfs.PYGIT2_VERSION))

def genrepo(opts=None, fire_event=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate winrepo_cachefile based on sls files in the winrepo_dir\n\n    opts\n        Specify an alternate opts dict. Should not be used unless this function\n        is imported into an execution module.\n\n    fire_event : True\n        Fire an event on failure. Only supported on the master.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run winrepo.genrepo\n    '
    if opts is None:
        opts = __opts__
    winrepo_dir = opts['winrepo_dir']
    winrepo_cachefile = opts['winrepo_cachefile']
    ret = {}
    if not os.path.exists(winrepo_dir):
        os.makedirs(winrepo_dir)
    renderers = salt.loader.render(opts, __salt__)
    for (root, _, files) in salt.utils.path.os_walk(winrepo_dir):
        for name in files:
            if name.endswith('.sls'):
                try:
                    config = salt.template.compile_template(os.path.join(root, name), renderers, opts['renderer'], opts['renderer_blacklist'], opts['renderer_whitelist'])
                except SaltRenderError as exc:
                    log.debug('Failed to render %s.', os.path.join(root, name))
                    log.debug('Error: %s.', exc)
                    continue
                if config:
                    revmap = {}
                    for (pkgname, versions) in config.items():
                        log.debug("Compiling winrepo data for package '%s'", pkgname)
                        for (version, repodata) in versions.items():
                            log.debug('Compiling winrepo data for %s version %s', pkgname, version)
                            if not isinstance(version, str):
                                config[pkgname][str(version)] = config[pkgname].pop(version)
                            if not isinstance(repodata, dict):
                                msg = 'Failed to compile {}.'.format(os.path.join(root, name))
                                log.debug(msg)
                                if fire_event:
                                    try:
                                        __jid_event__.fire_event({'error': msg}, 'progress')
                                    except NameError:
                                        log.error('Attempted to fire the an event with the following error, but event firing is not supported: %s', msg)
                                continue
                            revmap[repodata['full_name']] = pkgname
                    ret.setdefault('repo', {}).update(config)
                    ret.setdefault('name_map', {}).update(revmap)
    with salt.utils.files.fopen(os.path.join(winrepo_dir, winrepo_cachefile), 'w+b') as repo:
        repo.write(salt.utils.msgpack.dumps(ret))
    return ret

def update_git_repos(opts=None, clean=False, masterless=False):
    if False:
        i = 10
        return i + 15
    '\n    Checkout git repos containing Windows Software Package Definitions\n\n    opts\n        Specify an alternate opts dict. Should not be used unless this function\n        is imported into an execution module.\n\n    clean : False\n        Clean repo cachedirs which are not configured under\n        :conf_master:`winrepo_remotes`.\n\n        .. warning::\n            This argument should not be set to ``True`` if a mix of git and\n            non-git repo definitions are being used, as it will result in the\n            non-git repo definitions being removed.\n\n        .. versionadded:: 2015.8.0\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt-run winrepo.update_git_repos\n        salt-run winrepo.update_git_repos clean=True\n    '
    if opts is None:
        opts = __opts__
    winrepo_dir = opts['winrepo_dir']
    winrepo_remotes = opts['winrepo_remotes']
    winrepo_cfg = [(winrepo_remotes, winrepo_dir), (opts['winrepo_remotes_ng'], opts['winrepo_dir_ng'])]
    ret = {}
    for (remotes, base_dir) in winrepo_cfg:
        if _legacy_git():
            winrepo_result = {}
            for remote_info in remotes:
                if '/' in remote_info:
                    targetname = remote_info.split('/')[-1]
                else:
                    targetname = remote_info
                rev = 'HEAD'
                try:
                    (rev, remote_url) = remote_info.strip().split()
                except ValueError:
                    remote_url = remote_info
                gittarget = os.path.join(base_dir, targetname).replace('.', '_')
                if masterless:
                    result = __salt__['state.single']('git.latest', name=remote_url, rev=rev, branch='winrepo', target=gittarget, force_checkout=True, force_reset=True)
                    if isinstance(result, list):
                        raise CommandExecutionError('Failed up update winrepo remotes: {}'.format('\n'.join(result)))
                    if 'name' not in result:
                        key = next(iter(result))
                        result = result[key]
                else:
                    mminion = salt.minion.MasterMinion(opts)
                    result = mminion.functions['state.single']('git.latest', name=remote_url, rev=rev, branch='winrepo', target=gittarget, force_checkout=True, force_reset=True)
                    if isinstance(result, list):
                        raise CommandExecutionError('Failed to update winrepo remotes: {}'.format('\n'.join(result)))
                    if 'name' not in result:
                        key = next(iter(result))
                        result = result[key]
                winrepo_result[result['name']] = result['result']
            ret.update(winrepo_result)
        else:
            try:
                winrepo = salt.utils.gitfs.WinRepo(opts, remotes, per_remote_overrides=PER_REMOTE_OVERRIDES, per_remote_only=PER_REMOTE_ONLY, global_only=GLOBAL_ONLY, cache_root=base_dir)
                winrepo.fetch_remotes()
                if clean:
                    winrepo.clear_old_remotes()
                winrepo.checkout()
            except Exception as exc:
                msg = 'Failed to update winrepo_remotes: {}'.format(exc)
                log.error(msg, exc_info_on_loglevel=logging.DEBUG)
                return msg
            ret.update(winrepo.winrepo_dirs)
    return ret