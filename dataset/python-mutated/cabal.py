"""
Manage and query Cabal packages
===============================

.. versionadded:: 2015.8.0

"""
import logging
import salt.utils.path
from salt.exceptions import CommandExecutionError
logger = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only work when cabal-install is installed.\n    '
    return salt.utils.path.which('cabal') is not None and salt.utils.path.which('ghc-pkg') is not None

def update(user=None, env=None):
    if False:
        print('Hello World!')
    "\n    Updates list of known packages.\n\n    user\n        The user to run cabal update with\n\n    env\n        Environment variables to set when invoking cabal. Uses the\n        same ``env`` format as the :py:func:`cmd.run\n        <salt.modules.cmdmod.run>` execution function.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cabal.update\n\n    "
    return __salt__['cmd.run_all']('cabal update', runas=user, env=env)

def install(pkg=None, pkgs=None, user=None, install_global=False, env=None):
    if False:
        print('Hello World!')
    "\n    Install a cabal package.\n\n    pkg\n        A package name in format accepted by cabal-install. See:\n        https://wiki.haskell.org/Cabal-Install\n\n    pkgs\n        A list of packages names in same format as ``pkg``\n\n    user\n        The user to run cabal install with\n\n    install_global\n        Install package globally instead of locally\n\n    env\n        Environment variables to set when invoking cabal. Uses the\n        same ``env`` format as the :py:func:`cmd.run\n        <salt.modules.cmdmod.run>` execution function\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cabal.install shellcheck\n        salt '*' cabal.install shellcheck-0.3.5\n    "
    cmd = ['cabal install']
    if install_global:
        cmd.append('--global')
    if pkg:
        cmd.append('"{}"'.format(pkg))
    elif pkgs:
        cmd.append('"{}"'.format('" "'.join(pkgs)))
    result = __salt__['cmd.run_all'](' '.join(cmd), runas=user, env=env)
    if result['retcode'] != 0:
        raise CommandExecutionError(result['stderr'])
    return result

def list_(pkg=None, user=None, installed=False, env=None):
    if False:
        return 10
    "\n    List packages matching a search string.\n\n    pkg\n        Search string for matching package names\n    user\n        The user to run cabal list with\n    installed\n        If True, only return installed packages.\n    env\n        Environment variables to set when invoking cabal. Uses the\n        same ``env`` format as the :py:func:`cmd.run\n        <salt.modules.cmdmod.run>` execution function\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cabal.list\n        salt '*' cabal.list ShellCheck\n    "
    cmd = ['cabal list --simple-output']
    if installed:
        cmd.append('--installed')
    if pkg:
        cmd.append('"{}"'.format(pkg))
    result = __salt__['cmd.run_all'](' '.join(cmd), runas=user, env=env)
    packages = {}
    for line in result['stdout'].splitlines():
        data = line.split()
        package_name = data[0]
        package_version = data[1]
        packages[package_name] = package_version
    return packages

def uninstall(pkg, user=None, env=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Uninstall a cabal package.\n\n    pkg\n        The package to uninstall\n    user\n        The user to run ghc-pkg unregister with\n    env\n        Environment variables to set when invoking cabal. Uses the\n        same ``env`` format as the :py:func:`cmd.run\n        <salt.modules.cmdmod.run>` execution function\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cabal.uninstall ShellCheck\n\n    "
    cmd = ['ghc-pkg unregister']
    cmd.append('"{}"'.format(pkg))
    result = __salt__['cmd.run_all'](' '.join(cmd), runas=user, env=env)
    if result['retcode'] != 0:
        raise CommandExecutionError(result['stderr'])
    return result