"""
Module for using the locate utilities
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
        return (False, 'The locate execution module cannot be loaded: only available on non-Windows systems.')
    return True

def version():
    if False:
        i = 10
        return i + 15
    "\n    Returns the version of locate\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' locate.version\n    "
    cmd = 'locate -V'
    out = __salt__['cmd.run'](cmd).splitlines()
    return out

def stats():
    if False:
        i = 10
        return i + 15
    "\n    Returns statistics about the locate database\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' locate.stats\n    "
    ret = {}
    cmd = 'locate -S'
    out = __salt__['cmd.run'](cmd).splitlines()
    for line in out:
        comps = line.strip().split()
        if line.startswith('Database'):
            ret['database'] = comps[1].replace(':', '')
            continue
        ret[' '.join(comps[1:])] = comps[0]
    return ret

def updatedb():
    if False:
        for i in range(10):
            print('nop')
    "\n    Updates the locate database\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' locate.updatedb\n    "
    cmd = 'updatedb'
    out = __salt__['cmd.run'](cmd).splitlines()
    return out

def locate(pattern, database='', limit=0, **kwargs):
    if False:
        return 10
    "\n    Performs a file lookup. Valid options (and their defaults) are::\n\n        basename=False\n        count=False\n        existing=False\n        follow=True\n        ignore=False\n        nofollow=False\n        wholename=True\n        regex=False\n        database=<locate's default database>\n        limit=<integer, not set by default>\n\n    See the manpage for ``locate(1)`` for further explanation of these options.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' locate.locate\n    "
    options = ''
    toggles = {'basename': 'b', 'count': 'c', 'existing': 'e', 'follow': 'L', 'ignore': 'i', 'nofollow': 'P', 'wholename': 'w'}
    for option in kwargs:
        if bool(kwargs[option]) is True and option in toggles:
            options += toggles[option]
    if options:
        options = '-{}'.format(options)
    if database:
        options += ' -d {}'.format(database)
    if limit > 0:
        options += ' -l {}'.format(limit)
    if 'regex' in kwargs and bool(kwargs['regex']) is True:
        options += ' --regex'
    cmd = 'locate {} {}'.format(options, pattern)
    out = __salt__['cmd.run'](cmd, python_shell=False).splitlines()
    return out