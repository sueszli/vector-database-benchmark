"""
Manage and query Bower packages
===============================

This module manages the installed packages using Bower.
Note that npm, git and bower must be installed for this module to be
available.

"""
import logging
import shlex
import salt.utils.json
import salt.utils.path
from salt.exceptions import CommandExecutionError
from salt.utils.versions import Version
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}

def __virtual__():
    if False:
        return 10
    '\n    Only work when Bower is installed\n    '
    if salt.utils.path.which('bower') is None:
        return (False, 'The bower module could not be loaded: bower command not found')
    return True

def _check_valid_version():
    if False:
        for i in range(10):
            print('nop')
    '\n    Check the version of Bower to ensure this module will work. Currently\n    bower must be at least version 1.3.\n    '
    bower_version = Version(__salt__['cmd.run']('{} --version'.format(salt.utils.path.which('bower'))))
    valid_version = Version('1.3')
    if bower_version < valid_version:
        raise CommandExecutionError("'bower' is not recent enough({} < {}). Please Upgrade.".format(bower_version, valid_version))

def _construct_bower_command(bower_command):
    if False:
        return 10
    '\n    Create bower command line string\n    '
    if not bower_command:
        raise CommandExecutionError('bower_command, e.g. install, must be specified')
    cmd = [salt.utils.path.which('bower')] + shlex.split(bower_command)
    cmd.extend(['--config.analytics', 'false', '--config.interactive', 'false', '--allow-root', '--json'])
    return cmd

def install(pkg, dir, pkgs=None, runas=None, env=None):
    if False:
        return 10
    "\n    Install a Bower package.\n\n    If no package is specified, the dependencies (from bower.json) of the\n    package in the given directory will be installed.\n\n    pkg\n        A package name in any format accepted by Bower, including a version\n        identifier\n\n    dir\n        The target directory in which to install the package\n\n    pkgs\n        A list of package names in the same format as the ``pkg`` parameter\n\n    runas\n        The user to run Bower with\n\n    env\n        Environment variables to set when invoking Bower. Uses the same ``env``\n        format as the :py:func:`cmd.run <salt.modules.cmdmod.run>` execution\n        function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bower.install underscore /path/to/project\n\n        salt '*' bower.install jquery#2.0 /path/to/project\n\n    "
    _check_valid_version()
    cmd = _construct_bower_command('install')
    if pkg:
        cmd.append(pkg)
    elif pkgs:
        cmd.extend(pkgs)
    result = __salt__['cmd.run_all'](cmd, cwd=dir, runas=runas, env=env, python_shell=False)
    if result['retcode'] != 0:
        raise CommandExecutionError(result['stderr'])
    stdout = salt.utils.json.loads(result['stdout'])
    return stdout != {}

def uninstall(pkg, dir, runas=None, env=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Uninstall a Bower package.\n\n    pkg\n        A package name in any format accepted by Bower\n\n    dir\n        The target directory from which to uninstall the package\n\n    runas\n        The user to run Bower with\n\n    env\n        Environment variables to set when invoking Bower. Uses the same ``env``\n        format as the :py:func:`cmd.run <salt.modules.cmdmod.run>` execution\n        function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bower.uninstall underscore /path/to/project\n\n    "
    _check_valid_version()
    cmd = _construct_bower_command('uninstall')
    cmd.append(pkg)
    result = __salt__['cmd.run_all'](cmd, cwd=dir, runas=runas, env=env, python_shell=False)
    if result['retcode'] != 0:
        raise CommandExecutionError(result['stderr'])
    stdout = salt.utils.json.loads(result['stdout'])
    return stdout != {}

def list_(dir, runas=None, env=None):
    if False:
        i = 10
        return i + 15
    "\n    List installed Bower packages.\n\n    dir\n        The directory whose packages will be listed\n\n    runas\n        The user to run Bower with\n\n    env\n        Environment variables to set when invoking Bower. Uses the same ``env``\n        format as the :py:func:`cmd.run <salt.modules.cmdmod.run>` execution\n        function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bower.list /path/to/project\n\n    "
    _check_valid_version()
    cmd = _construct_bower_command('list')
    cmd.append('--offline')
    result = __salt__['cmd.run_all'](cmd, cwd=dir, runas=runas, env=env, python_shell=False)
    if result['retcode'] != 0:
        raise CommandExecutionError(result['stderr'])
    return salt.utils.json.loads(result['stdout'])['dependencies']

def prune(dir, runas=None, env=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2017.7.0\n\n    Remove extraneous local Bower packages, i.e. those not referenced in bower.json\n\n    dir\n        The directory whose packages will be pruned\n\n    runas\n        The user to run Bower with\n\n    env\n        Environment variables to set when invoking Bower. Uses the same ``env``\n        format as the :py:func:`cmd.run <salt.modules.cmdmod.run>` execution\n        function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bower.prune /path/to/project\n\n    "
    _check_valid_version()
    cmd = _construct_bower_command('prune')
    result = __salt__['cmd.run_all'](cmd, cwd=dir, runas=runas, env=env, python_shell=False)
    if result['retcode'] != 0:
        raise CommandExecutionError(result['stderr'])
    return salt.utils.json.loads(result['stdout'])