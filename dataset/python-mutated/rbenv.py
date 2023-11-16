"""
Manage ruby installations with rbenv. rbenv is supported on Linux and macOS.
rbenv doesn't work on Windows (and isn't really necessary on Windows as there is
no system Ruby on Windows). On Windows, the RubyInstaller and/or Pik are both
good alternatives to work with multiple versions of Ruby on the same box.

http://misheska.com/blog/2013/06/15/using-rbenv-to-manage-multiple-versions-of-ruby/

.. versionadded:: 0.16.0
"""
import logging
import os
import re
import salt.utils.args
import salt.utils.data
import salt.utils.path
import salt.utils.platform
from salt.exceptions import SaltInvocationError
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}
__opts__ = {'rbenv.root': None, 'rbenv.build_env': None}

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work on POSIX-like systems\n    '
    if salt.utils.platform.is_windows():
        return (False, 'The rbenv execution module failed to load: only available on non-Windows systems.')
    return True

def _shlex_split(s):
    if False:
        return 10
    if s is None:
        ret = salt.utils.args.shlex_split('')
    else:
        ret = salt.utils.args.shlex_split(s)
    return ret

def _parse_env(env):
    if False:
        while True:
            i = 10
    if not env:
        env = {}
    if isinstance(env, list):
        env = salt.utils.data.repack_dictlist(env)
    if not isinstance(env, dict):
        env = {}
    for bad_env_key in (x for (x, y) in env.items() if y is None):
        log.error("Environment variable '%s' passed without a value. Setting value to an empty string", bad_env_key)
        env[bad_env_key] = ''
    return env

def _rbenv_bin(runas=None):
    if False:
        print('Hello World!')
    path = _rbenv_path(runas)
    return '{}/bin/rbenv'.format(path)

def _rbenv_path(runas=None):
    if False:
        i = 10
        return i + 15
    path = None
    if runas in (None, 'root'):
        path = __salt__['config.option']('rbenv.root') or '/usr/local/rbenv'
    else:
        path = __salt__['config.option']('rbenv.root') or '~{}/.rbenv'.format(runas)
    return os.path.expanduser(path)

def _rbenv_exec(command, env=None, runas=None, ret=None):
    if False:
        return 10
    if not is_installed(runas):
        return False
    binary = _rbenv_bin(runas)
    path = _rbenv_path(runas)
    environ = _parse_env(env)
    environ['RBENV_ROOT'] = path
    result = __salt__['cmd.run_all']([binary] + command, runas=runas, env=environ)
    if isinstance(ret, dict):
        ret.update(result)
        return ret
    if result['retcode'] == 0:
        return result['stdout']
    else:
        return False

def _install_rbenv(path, runas=None):
    if False:
        print('Hello World!')
    if os.path.isdir(path):
        return True
    cmd = ['git', 'clone', 'https://github.com/rbenv/rbenv.git', path]
    return __salt__['cmd.retcode'](cmd, runas=runas, python_shell=False) == 0

def _install_ruby_build(path, runas=None):
    if False:
        while True:
            i = 10
    path = '{}/plugins/ruby-build'.format(path)
    if os.path.isdir(path):
        return True
    cmd = ['git', 'clone', 'https://github.com/rbenv/ruby-build.git', path]
    return __salt__['cmd.retcode'](cmd, runas=runas, python_shell=False) == 0

def _update_rbenv(path, runas=None):
    if False:
        i = 10
        return i + 15
    if not os.path.isdir(path):
        return False
    return __salt__['cmd.retcode'](['git', 'pull'], runas=runas, cwd=path, python_shell=False) == 0

def _update_ruby_build(path, runas=None):
    if False:
        for i in range(10):
            print('nop')
    path = '{}/plugins/ruby-build'.format(path)
    if not os.path.isdir(path):
        return False
    return __salt__['cmd.retcode'](['git', 'pull'], runas=runas, cwd=path, python_shell=False) == 0

def install(runas=None, path=None):
    if False:
        i = 10
        return i + 15
    "\n    Install rbenv systemwide\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbenv.install\n    "
    path = path or _rbenv_path(runas)
    path = os.path.expanduser(path)
    return _install_rbenv(path, runas) and _install_ruby_build(path, runas)

def update(runas=None, path=None):
    if False:
        print('Hello World!')
    "\n    Updates the current versions of rbenv and ruby-build\n\n    runas\n        The user under which to run rbenv. If not specified, then rbenv will be\n        run as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbenv.update\n    "
    path = path or _rbenv_path(runas)
    path = os.path.expanduser(path)
    return _update_rbenv(path, runas) and _update_ruby_build(path, runas)

def is_installed(runas=None):
    if False:
        while True:
            i = 10
    "\n    Check if rbenv is installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbenv.is_installed\n    "
    return __salt__['cmd.has_exec'](_rbenv_bin(runas))

def install_ruby(ruby, runas=None):
    if False:
        print('Hello World!')
    '\n    Install a ruby implementation.\n\n    ruby\n        The version of Ruby to install, should match one of the\n        versions listed by :py:func:`rbenv.list <salt.modules.rbenv.list>`\n\n    runas\n        The user under which to run rbenv. If not specified, then rbenv will be\n        run as the user under which Salt is running.\n\n    Additional environment variables can be configured in pillar /\n    grains / master:\n\n    .. code-block:: yaml\n\n        rbenv:\n          build_env: \'CONFIGURE_OPTS="--no-tcmalloc" CFLAGS="-fno-tree-dce"\'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' rbenv.install_ruby 2.0.0-p0\n    '
    ruby = re.sub('^ruby-', '', ruby)
    env = None
    env_list = []
    if __grains__['os'] in ('FreeBSD', 'NetBSD', 'OpenBSD'):
        env_list.append('MAKE=gmake')
    if __salt__['config.get']('rbenv:build_env'):
        env_list.append(__salt__['config.get']('rbenv:build_env'))
    elif __salt__['config.option']('rbenv.build_env'):
        env_list.append(__salt__['config.option']('rbenv.build_env'))
    if env_list:
        env = ' '.join(env_list)
    ret = {}
    ret = _rbenv_exec(['install', ruby], env=env, runas=runas, ret=ret)
    if ret is not False and ret['retcode'] == 0:
        rehash(runas=runas)
        return ret['stderr']
    else:
        uninstall_ruby(ruby, runas=runas)
        return False

def uninstall_ruby(ruby, runas=None):
    if False:
        return 10
    "\n    Uninstall a ruby implementation.\n\n    ruby\n        The version of ruby to uninstall. Should match one of the versions\n        listed by :py:func:`rbenv.versions <salt.modules.rbenv.versions>`.\n\n    runas\n        The user under which to run rbenv. If not specified, then rbenv will be\n        run as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbenv.uninstall_ruby 2.0.0-p0\n    "
    ruby = re.sub('^ruby-', '', ruby)
    _rbenv_exec(['uninstall', '--force', ruby], runas=runas)
    return True

def versions(runas=None):
    if False:
        print('Hello World!')
    "\n    List the installed versions of ruby\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbenv.versions\n    "
    ret = _rbenv_exec(['versions', '--bare'], runas=runas)
    return [] if ret is False else ret.splitlines()

def default(ruby=None, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns or sets the currently defined default ruby\n\n    ruby\n        The version to set as the default. Should match one of the versions\n        listed by :py:func:`rbenv.versions <salt.modules.rbenv.versions>`.\n        Leave blank to return the current default.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbenv.default\n        salt '*' rbenv.default 2.0.0-p0\n    "
    if ruby:
        _rbenv_exec(['global', ruby], runas=runas)
        return True
    else:
        ret = _rbenv_exec(['global'], runas=runas)
        return '' if ret is False else ret.strip()

def list_(runas=None):
    if False:
        print('Hello World!')
    "\n    List the installable versions of ruby\n\n    runas\n        The user under which to run rbenv. If not specified, then rbenv will be\n        run as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbenv.list\n    "
    ret = []
    output = _rbenv_exec(['install', '--list'], runas=runas)
    if output:
        for line in output.splitlines():
            if line == 'Available versions:':
                continue
            ret.append(line.strip())
    return ret

def rehash(runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Run ``rbenv rehash`` to update the installed shims\n\n    runas\n        The user under which to run rbenv. If not specified, then rbenv will be\n        run as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbenv.rehash\n    "
    _rbenv_exec(['rehash'], runas=runas)
    return True

def do(cmdline, runas=None, env=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Execute a ruby command with rbenv's shims from the user or the system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbenv.do 'gem list bundler'\n        salt '*' rbenv.do 'gem list bundler' deploy\n    "
    if not cmdline:
        raise SaltInvocationError('Command must be specified')
    path = _rbenv_path(runas)
    if not env:
        env = {}
    env['PATH'] = salt.utils.stringutils.to_str(os.pathsep.join((salt.utils.path.join(path, 'shims'), salt.utils.stringutils.to_unicode(os.environ['PATH']))))
    try:
        cmdline = salt.utils.args.shlex_split(cmdline)
    except AttributeError:
        cmdauth = salt.utils.args.shlex_split(str(cmdline))
    result = __salt__['cmd.run_all'](cmdline, runas=runas, env=env, python_shell=False)
    if result['retcode'] == 0:
        rehash(runas=runas)
        return result['stdout']
    else:
        return False

def do_with_ruby(ruby, cmdline, runas=None):
    if False:
        i = 10
        return i + 15
    "\n    Execute a ruby command with rbenv's shims using a specific ruby version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rbenv.do_with_ruby 2.0.0-p0 'gem list bundler'\n        salt '*' rbenv.do_with_ruby 2.0.0-p0 'gem list bundler' runas=deploy\n    "
    if not cmdline:
        raise SaltInvocationError('Command must be specified')
    try:
        cmdline = salt.utils.args.shlex_split(cmdline)
    except AttributeError:
        cmdline = salt.utils.args.shlex_split(str(cmdline))
    env = {}
    if ruby:
        env['RBENV_VERSION'] = ruby
        cmd = cmdline
    else:
        cmd = cmdline
    return do(cmd, runas=runas, env=env)