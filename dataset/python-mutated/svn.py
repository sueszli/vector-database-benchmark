"""
Subversion SCM
"""
import re
import salt.utils.args
import salt.utils.path
from salt.exceptions import CommandExecutionError
_INI_RE = re.compile('^([^:]+):\\s+(\\S.*)$', re.M)

def __virtual__():
    if False:
        return 10
    '\n    Only load if svn is installed\n    '
    if salt.utils.path.which('svn') is None:
        return (False, 'The svn execution module cannot be loaded: svn unavailable.')
    else:
        return True

def _run_svn(cmd, cwd, user, username, password, opts, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Execute svn\n    return the output of the command\n\n    cmd\n        The command to run.\n\n    cwd\n        The path to the Subversion repository\n\n    user\n        Run svn as a user other than what the minion runs as\n\n    username\n        Connect to the Subversion server as another user\n\n    password\n        Connect to the Subversion server with this password\n\n        .. versionadded:: 0.17.0\n\n    opts\n        Any additional options to add to the command line\n\n    kwargs\n        Additional options to pass to the run-cmd\n    '
    cmd = ['svn', '--non-interactive', cmd]
    options = list(opts)
    if username:
        options.extend(['--username', username])
    if password:
        options.extend(['--password', password])
    cmd.extend(options)
    result = __salt__['cmd.run_all'](cmd, python_shell=False, cwd=cwd, runas=user, **kwargs)
    retcode = result['retcode']
    if retcode == 0:
        return result['stdout']
    raise CommandExecutionError(result['stderr'] + '\n\n' + ' '.join(cmd))

def info(cwd, targets=None, user=None, username=None, password=None, fmt='str'):
    if False:
        print('Hello World!')
    "\n    Display the Subversion information from the checkout.\n\n    cwd\n        The path to the Subversion repository\n\n    targets : None\n        files, directories, and URLs to pass to the command as arguments\n        svn uses '.' by default\n\n    user : None\n        Run svn as a user other than what the minion runs as\n\n    username : None\n        Connect to the Subversion server as another user\n\n    password : None\n        Connect to the Subversion server with this password\n\n        .. versionadded:: 0.17.0\n\n    fmt : str\n        How to fmt the output from info.\n        (str, xml, list, dict)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' svn.info /path/to/svn/repo\n    "
    opts = list()
    if fmt == 'xml':
        opts.append('--xml')
    if targets:
        opts += salt.utils.args.shlex_split(targets)
    infos = _run_svn('info', cwd, user, username, password, opts)
    if fmt in ('str', 'xml'):
        return infos
    info_list = []
    for infosplit in infos.split('\n\n'):
        info_list.append(_INI_RE.findall(infosplit))
    if fmt == 'list':
        return info_list
    if fmt == 'dict':
        return [dict(tmp) for tmp in info_list]

def checkout(cwd, remote, target=None, user=None, username=None, password=None, *opts):
    if False:
        print('Hello World!')
    "\n    Download a working copy of the remote Subversion repository\n    directory or file\n\n    cwd\n        The path to the Subversion repository\n\n    remote : None\n        URL to checkout\n\n    target : None\n        The name to give the file or directory working copy\n        Default: svn uses the remote basename\n\n    user : None\n        Run svn as a user other than what the minion runs as\n\n    username : None\n        Connect to the Subversion server as another user\n\n    password : None\n        Connect to the Subversion server with this password\n\n        .. versionadded:: 0.17.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' svn.checkout /path/to/repo svn://remote/repo\n    "
    opts += (remote,)
    if target:
        opts += (target,)
    return _run_svn('checkout', cwd, user, username, password, opts)

def switch(cwd, remote, target=None, user=None, username=None, password=None, *opts):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2014.1.0\n\n    Switch a working copy of a remote Subversion repository\n    directory\n\n    cwd\n        The path to the Subversion repository\n\n    remote : None\n        URL to switch\n\n    target : None\n        The name to give the file or directory working copy\n        Default: svn uses the remote basename\n\n    user : None\n        Run svn as a user other than what the minion runs as\n\n    username : None\n        Connect to the Subversion server as another user\n\n    password : None\n        Connect to the Subversion server with this password\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' svn.switch /path/to/repo svn://remote/repo\n    "
    opts += (remote,)
    if target:
        opts += (target,)
    return _run_svn('switch', cwd, user, username, password, opts)

def update(cwd, targets=None, user=None, username=None, password=None, *opts):
    if False:
        return 10
    "\n    Update the current directory, files, or directories from\n    the remote Subversion repository\n\n    cwd\n        The path to the Subversion repository\n\n    targets : None\n        files and directories to pass to the command as arguments\n        Default: svn uses '.'\n\n    user : None\n        Run svn as a user other than what the minion runs as\n\n    password : None\n        Connect to the Subversion server with this password\n\n        .. versionadded:: 0.17.0\n\n    username : None\n        Connect to the Subversion server as another user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' svn.update /path/to/repo\n    "
    if targets:
        opts += tuple(salt.utils.args.shlex_split(targets))
    return _run_svn('update', cwd, user, username, password, opts)

def diff(cwd, targets=None, user=None, username=None, password=None, *opts):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the diff of the current directory, files, or directories from\n    the remote Subversion repository\n\n    cwd\n        The path to the Subversion repository\n\n    targets : None\n        files and directories to pass to the command as arguments\n        Default: svn uses '.'\n\n    user : None\n        Run svn as a user other than what the minion runs as\n\n    username : None\n        Connect to the Subversion server as another user\n\n    password : None\n        Connect to the Subversion server with this password\n\n        .. versionadded:: 0.17.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' svn.diff /path/to/repo\n    "
    if targets:
        opts += tuple(salt.utils.args.shlex_split(targets))
    return _run_svn('diff', cwd, user, username, password, opts)

def commit(cwd, targets=None, msg=None, user=None, username=None, password=None, *opts):
    if False:
        i = 10
        return i + 15
    "\n    Commit the current directory, files, or directories to\n    the remote Subversion repository\n\n    cwd\n        The path to the Subversion repository\n\n    targets : None\n        files and directories to pass to the command as arguments\n        Default: svn uses '.'\n\n    msg : None\n        Message to attach to the commit log\n\n    user : None\n        Run svn as a user other than what the minion runs as\n\n    username : None\n        Connect to the Subversion server as another user\n\n    password : None\n        Connect to the Subversion server with this password\n\n        .. versionadded:: 0.17.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' svn.commit /path/to/repo\n    "
    if msg:
        opts += ('-m', msg)
    if targets:
        opts += tuple(salt.utils.args.shlex_split(targets))
    return _run_svn('commit', cwd, user, username, password, opts)

def add(cwd, targets, user=None, username=None, password=None, *opts):
    if False:
        return 10
    "\n    Add files to be tracked by the Subversion working-copy checkout\n\n    cwd\n        The path to the Subversion repository\n\n    targets : None\n        files and directories to pass to the command as arguments\n\n    user : None\n        Run svn as a user other than what the minion runs as\n\n    username : None\n        Connect to the Subversion server as another user\n\n    password : None\n        Connect to the Subversion server with this password\n\n        .. versionadded:: 0.17.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' svn.add /path/to/repo /path/to/new/file\n    "
    if targets:
        opts += tuple(salt.utils.args.shlex_split(targets))
    return _run_svn('add', cwd, user, username, password, opts)

def remove(cwd, targets, msg=None, user=None, username=None, password=None, *opts):
    if False:
        print('Hello World!')
    "\n    Remove files and directories from the Subversion repository\n\n    cwd\n        The path to the Subversion repository\n\n    targets : None\n        files, directories, and URLs to pass to the command as arguments\n\n    msg : None\n        Message to attach to the commit log\n\n    user : None\n        Run svn as a user other than what the minion runs as\n\n    username : None\n        Connect to the Subversion server as another user\n\n    password : None\n        Connect to the Subversion server with this password\n\n        .. versionadded:: 0.17.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' svn.remove /path/to/repo /path/to/repo/remove\n    "
    if msg:
        opts += ('-m', msg)
    if targets:
        opts += tuple(salt.utils.args.shlex_split(targets))
    return _run_svn('remove', cwd, user, username, password, opts)

def status(cwd, targets=None, user=None, username=None, password=None, *opts):
    if False:
        return 10
    "\n    Display the status of the current directory, files, or\n    directories in the Subversion repository\n\n    cwd\n        The path to the Subversion repository\n\n    targets : None\n        files, directories, and URLs to pass to the command as arguments\n        Default: svn uses '.'\n\n    user : None\n        Run svn as a user other than what the minion runs as\n\n    username : None\n        Connect to the Subversion server as another user\n\n    password : None\n        Connect to the Subversion server with this password\n\n        .. versionadded:: 0.17.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' svn.status /path/to/repo\n    "
    if targets:
        opts += tuple(salt.utils.args.shlex_split(targets))
    return _run_svn('status', cwd, user, username, password, opts)

def export(cwd, remote, target=None, user=None, username=None, password=None, revision='HEAD', *opts):
    if False:
        return 10
    "\n    Create an unversioned copy of a tree.\n\n    cwd\n        The path to the Subversion repository\n\n    remote : None\n        URL and path to file or directory checkout\n\n    target : None\n        The name to give the file or directory working copy\n        Default: svn uses the remote basename\n\n    user : None\n        Run svn as a user other than what the minion runs as\n\n    username : None\n        Connect to the Subversion server as another user\n\n    password : None\n        Connect to the Subversion server with this password\n\n        .. versionadded:: 0.17.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' svn.export /path/to/repo svn://remote/repo\n    "
    opts += (remote,)
    if target:
        opts += (target,)
    revision_args = '-r'
    opts += (revision_args, str(revision))
    return _run_svn('export', cwd, user, username, password, opts)