"""
Manage PHP pecl extensions.
"""
import logging
import re
import shlex
import salt.utils.data
import salt.utils.path
__func_alias__ = {'list_': 'list'}
log = logging.getLogger(__name__)
__virtualname__ = 'pecl'

def __virtual__():
    if False:
        print('Hello World!')
    if salt.utils.path.which('pecl'):
        return __virtualname__
    return (False, 'The pecl execution module not loaded: pecl binary is not in the path.')

def _pecl(command, defaults=False):
    if False:
        print('Hello World!')
    '\n    Execute the command passed with pecl\n    '
    cmdline = 'pecl {}'.format(command)
    if salt.utils.data.is_true(defaults):
        cmdline = "yes ''" + ' | ' + cmdline
    ret = __salt__['cmd.run_all'](cmdline, python_shell=True)
    if ret['retcode'] == 0:
        return ret['stdout']
    else:
        log.error('Problem running pecl. Is php-pear installed?')
        return ''

def install(pecls, defaults=False, force=False, preferred_state='stable'):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 0.17.0\n\n    Installs one or several pecl extensions.\n\n    pecls\n        The pecl extensions to install.\n\n    defaults\n        Use default answers for extensions such as pecl_http which ask\n        questions before installation. Without this option, the pecl.installed\n        state will hang indefinitely when trying to install these extensions.\n\n    force\n        Whether to force the installed version or not\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pecl.install fuse\n    "
    if isinstance(pecls, str):
        pecls = [pecls]
    preferred_state = '-d preferred_state={}'.format(shlex.quote(preferred_state))
    if force:
        return _pecl('{} install -f {}'.format(preferred_state, shlex.quote(' '.join(pecls))), defaults=defaults)
    else:
        _pecl('{} install {}'.format(preferred_state, shlex.quote(' '.join(pecls))), defaults=defaults)
        if not isinstance(pecls, list):
            pecls = [pecls]
        for pecl in pecls:
            found = False
            if '/' in pecl:
                (channel, pecl) = pecl.split('/')
            else:
                channel = None
            installed_pecls = list_(channel)
            for pecl in installed_pecls:
                installed_pecl_with_version = '{}-{}'.format(pecl, installed_pecls.get(pecl)[0])
                if pecl in installed_pecl_with_version:
                    found = True
            if not found:
                return False
        return True

def uninstall(pecls):
    if False:
        return 10
    "\n    Uninstall one or several pecl extensions.\n\n    pecls\n        The pecl extensions to uninstall.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pecl.uninstall fuse\n    "
    if isinstance(pecls, str):
        pecls = [pecls]
    return _pecl('uninstall {}'.format(shlex.quote(' '.join(pecls))))

def update(pecls):
    if False:
        return 10
    "\n    Update one or several pecl extensions.\n\n    pecls\n        The pecl extensions to update.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pecl.update fuse\n    "
    if isinstance(pecls, str):
        pecls = [pecls]
    return _pecl('install -U {}'.format(shlex.quote(' '.join(pecls))))

def list_(channel=None):
    if False:
        print('Hello World!')
    "\n    List installed pecl extensions.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pecl.list\n    "
    pecl_channel_pat = re.compile('^([^ ]+)[ ]+([^ ]+)[ ]+([^ ]+)')
    pecls = {}
    command = 'list'
    if channel:
        command = '{} -c {}'.format(command, shlex.quote(channel))
    lines = _pecl(command).splitlines()
    lines = (l for l in lines if pecl_channel_pat.match(l))
    for line in lines:
        match = pecl_channel_pat.match(line)
        if match:
            pecls[match.group(1)] = [match.group(2), match.group(3)]
    return pecls