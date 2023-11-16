"""
Manage ruby installations and gemsets with RVM, the Ruby Version Manager.
"""
import logging
import os
import re
import salt.utils.args
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}
__opts__ = {'rvm.runas': None}

def _get_rvm_location(runas=None):
    if False:
        for i in range(10):
            print('nop')
    if runas:
        runas_home = os.path.expanduser('~{}'.format(runas))
        rvmpath = '{}/.rvm/bin/rvm'.format(runas_home)
        if os.path.exists(rvmpath):
            return [rvmpath]
    return ['/usr/local/rvm/bin/rvm']

def _rvm(command, runas=None, cwd=None, env=None):
    if False:
        i = 10
        return i + 15
    if runas is None:
        runas = __salt__['config.option']('rvm.runas')
    if not is_installed(runas):
        return False
    cmd = _get_rvm_location(runas) + command
    ret = __salt__['cmd.run_all'](cmd, runas=runas, cwd=cwd, python_shell=False, env=env)
    if ret['retcode'] == 0:
        return ret['stdout']
    return False

def _rvm_do(ruby, command, runas=None, cwd=None, env=None):
    if False:
        while True:
            i = 10
    return _rvm([ruby or 'default', 'do'] + command, runas=runas, cwd=cwd, env=env)

def is_installed(runas=None):
    if False:
        return 10
    "\n    Check if RVM is installed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.is_installed\n    "
    try:
        return __salt__['cmd.has_exec'](_get_rvm_location(runas)[0])
    except IndexError:
        return False

def install(runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Install RVM system-wide\n\n    runas\n        The user under which to run the rvm installer script. If not specified,\n        then it be run as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.install\n    "
    installer = 'https://raw.githubusercontent.com/rvm/rvm/master/binscripts/rvm-installer'
    ret = __salt__['cmd.run_all']('curl -Ls {installer} | bash -s stable'.format(installer=installer), runas=runas, python_shell=True)
    if ret['retcode'] > 0:
        msg = 'Error encountered while downloading the RVM installer'
        if ret['stderr']:
            msg += '. stderr follows:\n\n' + ret['stderr']
        raise CommandExecutionError(msg)
    return True

def install_ruby(ruby, runas=None, opts=None, env=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Install a ruby implementation.\n\n    ruby\n        The version of ruby to install\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    env\n        Environment to set for the install command. Useful for exporting compilation\n        flags such as RUBY_CONFIGURE_OPTS\n\n    opts\n        List of options to pass to the RVM installer (ie -C, --patch, etc)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.install_ruby 1.9.3-p385\n    "
    if opts is None:
        opts = []
    if runas and runas != 'root':
        _rvm(['autolibs', 'disable', ruby] + opts, runas=runas)
        opts.append('--disable-binary')
    return _rvm(['install', ruby] + opts, runas=runas, env=env)

def reinstall_ruby(ruby, runas=None, env=None):
    if False:
        while True:
            i = 10
    "\n    Reinstall a ruby implementation\n\n    ruby\n        The version of ruby to reinstall\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.reinstall_ruby 1.9.3-p385\n    "
    return _rvm(['reinstall', ruby], runas=runas, env=env)

def list_(runas=None):
    if False:
        print('Hello World!')
    "\n    List all rvm-installed rubies\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.list\n    "
    rubies = []
    output = _rvm(['list'], runas=runas)
    if output:
        regex = re.compile('^[= ]([*> ]) ([^- ]+)-([^ ]+) \\[ (.*) \\]')
        for line in output.splitlines():
            match = regex.match(line)
            if match:
                rubies.append([match.group(2), match.group(3), match.group(1) == '*'])
    return rubies

def set_default(ruby, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the default ruby\n\n    ruby\n        The version of ruby to make the default\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.set_default 2.0.0\n    "
    return _rvm(['alias', 'create', 'default', ruby], runas=runas)

def get(version='stable', runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Update RVM\n\n    version : stable\n        Which version of RVM to install, (e.g. stable or head)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.get\n    "
    return _rvm(['get', version], runas=runas)

def wrapper(ruby_string, wrapper_prefix, runas=None, *binaries):
    if False:
        return 10
    "\n    Install RVM wrapper scripts\n\n    ruby_string\n        Ruby/gemset to install wrappers for\n\n    wrapper_prefix\n        What to prepend to the name of the generated wrapper binaries\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    binaries : None\n        The names of the binaries to create wrappers for. When nothing is\n        given, wrappers for ruby, gem, rake, irb, rdoc, ri and testrb are\n        generated.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.wrapper <ruby_string> <wrapper_prefix>\n    "
    cmd = ['wrapper', ruby_string, wrapper_prefix]
    cmd.extend(binaries)
    return _rvm(cmd, runas=runas)

def rubygems(ruby, version, runas=None):
    if False:
        i = 10
        return i + 15
    "\n    Installs a specific rubygems version in the given ruby\n\n    ruby\n        The ruby for which to install rubygems\n\n    version\n        The version of rubygems to install, or 'remove' to use the version that\n        ships with 1.9\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.rubygems 2.0.0 1.8.24\n    "
    return _rvm_do(ruby, ['rubygems', version], runas=runas)

def gemset_create(ruby, gemset, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Creates a gemset.\n\n    ruby\n        The ruby version for which to create the gemset\n\n    gemset\n        The name of the gemset to create\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.gemset_create 2.0.0 foobar\n    "
    return _rvm_do(ruby, ['rvm', 'gemset', 'create', gemset], runas=runas)

def gemset_list(ruby='default', runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all gemsets for the given ruby.\n\n    ruby : default\n        The ruby version for which to list the gemsets\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.gemset_list\n    "
    gemsets = []
    output = _rvm_do(ruby, ['rvm', 'gemset', 'list'], runas=runas)
    if output:
        regex = re.compile('^   ([^ ]+)')
        for line in output.splitlines():
            match = regex.match(line)
            if match:
                gemsets.append(match.group(1))
    return gemsets

def gemset_delete(ruby, gemset, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete a gemset\n\n    ruby\n        The ruby version to which the gemset belongs\n\n    gemset\n        The gemset to delete\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.gemset_delete 2.0.0 foobar\n    "
    return _rvm_do(ruby, ['rvm', '--force', 'gemset', 'delete', gemset], runas=runas)

def gemset_empty(ruby, gemset, runas=None):
    if False:
        print('Hello World!')
    "\n    Remove all gems from a gemset.\n\n    ruby\n        The ruby version to which the gemset belongs\n\n    gemset\n        The gemset to empty\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.gemset_empty 2.0.0 foobar\n    "
    return _rvm_do(ruby, ['rvm', '--force', 'gemset', 'empty', gemset], runas=runas)

def gemset_copy(source, destination, runas=None):
    if False:
        return 10
    "\n    Copy all gems from one gemset to another.\n\n    source\n        The name of the gemset to copy, complete with ruby version\n\n    destination\n        The destination gemset\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.gemset_copy foobar bazquo\n    "
    return _rvm(['gemset', 'copy', source, destination], runas=runas)

def gemset_list_all(runas=None):
    if False:
        i = 10
        return i + 15
    "\n    List all gemsets for all installed rubies.\n\n    Note that you must have set a default ruby before this can work.\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.gemset_list_all\n    "
    gemsets = {}
    current_ruby = None
    output = _rvm_do('default', ['rvm', 'gemset', 'list_all'], runas=runas)
    if output:
        gems_regex = re.compile('^   ([^ ]+)')
        gemset_regex = re.compile('^gemsets for ([^ ]+)')
        for line in output.splitlines():
            match = gemset_regex.match(line)
            if match:
                current_ruby = match.group(1)
                gemsets[current_ruby] = []
            match = gems_regex.match(line)
            if match:
                gemsets[current_ruby].append(match.group(1))
    return gemsets

def do(ruby, command, runas=None, cwd=None, env=None):
    if False:
        print('Hello World!')
    "\n    Execute a command in an RVM controlled environment.\n\n    ruby\n        Which ruby to use\n\n    command\n        The rvm command to execute\n\n    runas\n        The user under which to run rvm. If not specified, then rvm will be run\n        as the user under which Salt is running.\n\n    cwd\n        The directory from which to run the rvm command. Defaults to the user's\n        home directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' rvm.do 2.0.0 <command>\n    "
    try:
        command = salt.utils.args.shlex_split(command)
    except AttributeError:
        command = salt.utils.args.shlex_split(str(command))
    return _rvm_do(ruby, command, runas=runas, cwd=cwd, env=env)