"""
Support for the Mercurial SCM
"""
import logging
import salt.utils.data
import salt.utils.path
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if hg is installed\n    '
    if salt.utils.path.which('hg') is None:
        return (False, 'The hg execution module cannot be loaded: hg unavailable.')
    else:
        return True

def _ssh_flag(identity_path):
    if False:
        i = 10
        return i + 15
    return ['--ssh', 'ssh -i {}'.format(identity_path)]

def revision(cwd, rev='tip', short=False, user=None):
    if False:
        while True:
            i = 10
    "\n    Returns the long hash of a given identifier (hash, branch, tag, HEAD, etc)\n\n    cwd\n        The path to the Mercurial repository\n\n    rev: tip\n        The revision\n\n    short: False\n        Return an abbreviated commit hash\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hg.revision /path/to/repo mybranch\n    "
    cmd = ['hg', 'id', '-i', '--debug' if not short else '', '-r', '{}'.format(rev)]
    result = __salt__['cmd.run_all'](cmd, cwd=cwd, runas=user, python_shell=False)
    if result['retcode'] == 0:
        return result['stdout']
    else:
        return ''

def describe(cwd, rev='tip', user=None):
    if False:
        return 10
    "\n    Mimic git describe and return an identifier for the given revision\n\n    cwd\n        The path to the Mercurial repository\n\n    rev: tip\n        The path to the archive tarball\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hg.describe /path/to/repo\n    "
    cmd = ['hg', 'log', '-r', '{}'.format(rev), '--template', "'{{latesttag}}-{{latesttagdistance}}-{{node|short}}'"]
    desc = __salt__['cmd.run_stdout'](cmd, cwd=cwd, runas=user, python_shell=False)
    return desc or revision(cwd, rev, short=True)

def archive(cwd, output, rev='tip', fmt=None, prefix=None, user=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Export a tarball from the repository\n\n    cwd\n        The path to the Mercurial repository\n\n    output\n        The path to the archive tarball\n\n    rev: tip\n        The revision to create an archive from\n\n    fmt: None\n        Format of the resulting archive. Mercurial supports: tar,\n        tbz2, tgz, zip, uzip, and files formats.\n\n    prefix : None\n        Prepend <prefix>/ to every filename in the archive\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    If ``prefix`` is not specified it defaults to the basename of the repo\n    directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hg.archive /path/to/repo output=/tmp/archive.tgz fmt=tgz\n    "
    cmd = ['hg', 'archive', '{}'.format(output), '--rev', '{}'.format(rev)]
    if fmt:
        cmd.append('--type')
        cmd.append('{}'.format(fmt))
    if prefix:
        cmd.append('--prefix')
        cmd.append('"{}"'.format(prefix))
    return __salt__['cmd.run'](cmd, cwd=cwd, runas=user, python_shell=False)

def pull(cwd, opts=None, user=None, identity=None, repository=None):
    if False:
        while True:
            i = 10
    "\n    Perform a pull on the given repository\n\n    cwd\n        The path to the Mercurial repository\n\n    repository : None\n        Perform pull from the repository different from .hg/hgrc:[paths]:default\n\n    opts : None\n        Any additional options to add to the command line\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    identity : None\n        Private SSH key on the minion server for authentication (ssh://)\n\n        .. versionadded:: 2015.5.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hg.pull /path/to/repo opts=-u\n    "
    cmd = ['hg', 'pull']
    if identity:
        cmd.extend(_ssh_flag(identity))
    if opts:
        for opt in opts.split():
            cmd.append(opt)
    if repository is not None:
        cmd.append(repository)
    ret = __salt__['cmd.run_all'](cmd, cwd=cwd, runas=user, python_shell=False)
    if ret['retcode'] != 0:
        raise CommandExecutionError('Hg command failed: {}'.format(ret.get('stderr', ret['stdout'])))
    return ret['stdout']

def update(cwd, rev, force=False, user=None):
    if False:
        while True:
            i = 10
    '\n    Update to a given revision\n\n    cwd\n        The path to the Mercurial repository\n\n    rev\n        The revision to update to\n\n    force : False\n        Force an update\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt devserver1 hg.update /path/to/repo somebranch\n    '
    cmd = ['hg', 'update', '{}'.format(rev)]
    if force:
        cmd.append('-C')
    ret = __salt__['cmd.run_all'](cmd, cwd=cwd, runas=user, python_shell=False)
    if ret['retcode'] != 0:
        raise CommandExecutionError('Hg command failed: {}'.format(ret.get('stderr', ret['stdout'])))
    return ret['stdout']

def clone(cwd, repository, opts=None, user=None, identity=None):
    if False:
        while True:
            i = 10
    "\n    Clone a new repository\n\n    cwd\n        The path to the Mercurial repository\n\n    repository\n        The hg URI of the repository\n\n    opts : None\n        Any additional options to add to the command line\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    identity : None\n        Private SSH key on the minion server for authentication (ssh://)\n\n        .. versionadded:: 2015.5.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hg.clone /path/to/repo https://bitbucket.org/birkenfeld/sphinx\n    "
    cmd = ['hg', 'clone', '{}'.format(repository), '{}'.format(cwd)]
    if opts:
        for opt in opts.split():
            cmd.append('{}'.format(opt))
    if identity:
        cmd.extend(_ssh_flag(identity))
    ret = __salt__['cmd.run_all'](cmd, runas=user, python_shell=False)
    if ret['retcode'] != 0:
        raise CommandExecutionError('Hg command failed: {}'.format(ret.get('stderr', ret['stdout'])))
    return ret['stdout']

def status(cwd, opts=None, user=None):
    if False:
        while True:
            i = 10
    "\n    Show changed files of the given repository\n\n    cwd\n        The path to the Mercurial repository\n\n    opts : None\n        Any additional options to add to the command line\n\n    user : None\n        Run hg as a user other than what the minion runs as\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' hg.status /path/to/repo\n    "

    def _status(cwd):
        if False:
            print('Hello World!')
        cmd = ['hg', 'status']
        if opts:
            for opt in opts.split():
                cmd.append('{}'.format(opt))
        out = __salt__['cmd.run_stdout'](cmd, cwd=cwd, runas=user, python_shell=False)
        types = {'M': 'modified', 'A': 'added', 'R': 'removed', 'C': 'clean', '!': 'missing', '?': 'not tracked', 'I': 'ignored', ' ': 'origin of the previous file'}
        ret = {}
        for line in out.splitlines():
            (t, f) = (types[line[0]], line[2:])
            if t not in ret:
                ret[t] = []
            ret[t].append(f)
        return ret
    if salt.utils.data.is_iter(cwd):
        return {cwd: _status(cwd) for cwd in cwd}
    else:
        return _status(cwd)