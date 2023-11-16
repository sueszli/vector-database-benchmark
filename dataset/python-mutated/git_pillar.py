"""
Runner module to directly manage the git external pillar
"""
import logging
import salt.pillar.git_pillar
import salt.utils.gitfs
from salt.exceptions import SaltRunnerError
log = logging.getLogger(__name__)

def update(branch=None, repo=None):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2014.1.0\n\n    .. versionchanged:: 2015.8.4\n        This runner function now supports the :ref:`git_pillar\n        configuration schema <git-pillar-configuration>` introduced in\n        2015.8.0. Additionally, the branch and repo can now be omitted to\n        update all git_pillar remotes. The return data has also changed to\n        a dictionary. The values will be ``True`` only if new commits were\n        fetched, and ``False`` if there were errors or no new commits were\n        fetched.\n\n    .. versionchanged:: 2018.3.0\n        The return for a given git_pillar remote will now be ``None`` when no\n        changes were fetched. ``False`` now is reserved only for instances in\n        which there were errors.\n\n    .. versionchanged:: 3001\n        The repo parameter also matches against the repo name.\n\n    Fetch one or all configured git_pillar remotes.\n\n    .. note::\n        This will *not* fast-forward the git_pillar cachedir on the master. All\n        it does is perform a ``git fetch``. If this runner is executed with\n        ``-l debug``, you may see a log message that says that the repo is\n        up-to-date. Keep in mind that Salt automatically fetches git_pillar\n        repos roughly every 60 seconds (or whatever\n        :conf_master:`loop_interval` is set to). So, it is possible that the\n        repo was fetched automatically in the time between when changes were\n        pushed to the repo, and when this runner was executed. When in doubt,\n        simply refresh pillar data using :py:func:`saltutil.refresh_pillar\n        <salt.modules.saltutil.refresh_pillar>` and then use\n        :py:func:`pillar.item <salt.modules.pillar.item>` to check if the\n        pillar data has changed as expected.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Update specific branch and repo\n        salt-run git_pillar.update branch='branch' repo='https://foo.com/bar.git'\n        # Update specific repo, by name\n        salt-run git_pillar.update repo=myrepo\n        # Update all repos\n        salt-run git_pillar.update\n        # Run with debug logging\n        salt-run git_pillar.update -l debug\n    "
    ret = {}
    for ext_pillar in __opts__.get('ext_pillar', []):
        pillar_type = next(iter(ext_pillar))
        if pillar_type != 'git':
            continue
        pillar_conf = ext_pillar[pillar_type]
        pillar = salt.utils.gitfs.GitPillar(__opts__, pillar_conf, per_remote_overrides=salt.pillar.git_pillar.PER_REMOTE_OVERRIDES, per_remote_only=salt.pillar.git_pillar.PER_REMOTE_ONLY, global_only=salt.pillar.git_pillar.GLOBAL_ONLY)
        for remote in pillar.remotes:
            if branch is not None:
                if branch != remote.branch:
                    continue
            if repo is not None:
                if repo != remote.url and repo != getattr(remote, 'name', None):
                    continue
            try:
                result = remote.fetch()
            except Exception as exc:
                log.error("Exception '%s' caught while fetching git_pillar remote '%s'", exc, remote.id, exc_info_on_loglevel=logging.DEBUG)
                result = False
            finally:
                remote.clear_lock()
            ret[remote.id] = result
    if not ret:
        if branch is not None or repo is not None:
            raise SaltRunnerError('Specified git branch/repo not found in ext_pillar config')
        else:
            raise SaltRunnerError('No git_pillar remotes are configured')
    return ret