"""
Work with Nix packages
======================

.. versionadded:: 2017.7.0

Does not require the machine to be Nixos, just have Nix installed and available
to use for the user running this command. Their profile must be located in
their home, under ``$HOME/.nix-profile/``, and the nix store, unless specially
set up, should be in ``/nix``. To easily use this with multiple users or a root
user, set up the `nix-daemon`_.

This module exposes most of the common nix operations. Currently not meant to be run as a ``pkg`` module, but explicitly as ``nix.*``.

For more information on nix, see the `nix documentation`_.

.. _`nix documentation`: https://nixos.org/nix/manual/
.. _`nix-daemon`: https://nixos.org/nix/manual/#ssec-multi-user
"""
import itertools
import logging
import os
import salt.utils.itertools
import salt.utils.path
logger = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    This only works if we have access to nix-env\n    '
    nixhome = os.path.join(os.path.expanduser('~{}'.format(__opts__['user'])), '.nix-profile/bin/')
    if salt.utils.path.which(os.path.join(nixhome, 'nix-env')) and salt.utils.path.which(os.path.join(nixhome, 'nix-collect-garbage')):
        return True
    else:
        return (False, 'The `nix` binaries required cannot be found or are not installed. (`nix-store` and `nix-env`)')

def _run(cmd):
    if False:
        for i in range(10):
            print('nop')
    "\n    Just a convenience function for ``__salt__['cmd.run_all'](cmd)``\n    "
    return __salt__['cmd.run_all'](cmd, env={'HOME': os.path.expanduser('~{}'.format(__opts__['user']))})

def _nix_env():
    if False:
        for i in range(10):
            print('nop')
    '\n    nix-env with quiet option. By default, nix is extremely verbose and prints the build log of every package to stderr. This tells nix to\n    only show changes.\n    '
    nixhome = os.path.join(os.path.expanduser('~{}'.format(__opts__['user'])), '.nix-profile/bin/')
    return [os.path.join(nixhome, 'nix-env')]

def _nix_collect_garbage():
    if False:
        while True:
            i = 10
    '\n    Make sure we get the right nix-store, too.\n    '
    nixhome = os.path.join(os.path.expanduser('~{}'.format(__opts__['user'])), '.nix-profile/bin/')
    return [os.path.join(nixhome, 'nix-collect-garbage')]

def _quietnix():
    if False:
        for i in range(10):
            print('nop')
    '\n    nix-env with quiet option. By default, nix is extremely verbose and prints the build log of every package to stderr. This tells nix to\n    only show changes.\n    '
    p = _nix_env()
    p.append('--no-build-output')
    return p

def _zip_flatten(x, ys):
    if False:
        i = 10
        return i + 15
    '\n    intersperse x into ys, with an extra element at the beginning.\n    '
    return itertools.chain.from_iterable(zip(itertools.repeat(x), ys))

def _output_format(out, operation):
    if False:
        i = 10
        return i + 15
    '\n    gets a list of all the packages that were affected by ``operation``, splits it up (there can be multiple packages on a line), and then\n    flattens that list. We make it to a list for easier parsing.\n    '
    return [s.split()[1:] for s in out if s.startswith(operation)]

def _format_upgrade(s):
    if False:
        while True:
            i = 10
    "\n    split the ``upgrade`` responses on ``' to '``\n    "
    return s.split(' to ')

def _strip_quotes(s):
    if False:
        i = 10
        return i + 15
    '\n    nix likes to quote itself in a backtick and a single quote. This just strips those.\n    '
    return s.strip("'`")

def upgrade(*pkgs):
    if False:
        print('Hello World!')
    "\n    Runs an update operation on the specified packages, or all packages if none is specified.\n\n    :type pkgs: list(str)\n    :param pkgs:\n        List of packages to update\n\n    :return: The upgraded packages. Example element: ``['libxslt-1.1.0', 'libxslt-1.1.10']``\n    :rtype: list(tuple(str, str))\n\n    .. code-block:: bash\n\n        salt '*' nix.update\n        salt '*' nix.update pkgs=one,two\n    "
    cmd = _quietnix()
    cmd.append('--upgrade')
    cmd.extend(pkgs)
    out = _run(cmd)
    upgrades = [_format_upgrade(s.split(maxsplit=1)[1]) for s in out['stderr'].splitlines() if s.startswith('upgrading')]
    return [[_strip_quotes(s_) for s_ in s] for s in upgrades]

def install(*pkgs, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Installs a single or multiple packages via nix\n\n    :type pkgs: list(str)\n    :param pkgs:\n        packages to update\n    :param bool attributes:\n        Pass the list of packages or single package as attribues, not package names.\n        default: False\n\n    :return: Installed packages. Example element: ``gcc-3.3.2``\n    :rtype: list(str)\n\n    .. code-block:: bash\n\n        salt '*' nix.install package [package2 ...]\n        salt '*' nix.install attributes=True attr.name [attr.name2 ...]\n    "
    attributes = kwargs.get('attributes', False)
    if not pkgs:
        return 'Plese specify a package or packages to upgrade'
    cmd = _quietnix()
    cmd.append('--install')
    if kwargs.get('attributes', False):
        cmd.extend(_zip_flatten('--attr', pkgs))
    else:
        cmd.extend(pkgs)
    out = _run(cmd)
    installs = list(itertools.chain.from_iterable([s.split()[1:] for s in out['stderr'].splitlines() if s.startswith('installing')]))
    return [_strip_quotes(s) for s in installs]

def list_pkgs(installed=True, attributes=True):
    if False:
        while True:
            i = 10
    "\n    Lists installed packages. Due to how nix works, it defaults to just doing a ``nix-env -q``.\n\n    :param bool installed:\n        list only installed packages. This can be a very long list (12,000+ elements), so caution is advised.\n        Default: True\n\n    :param bool attributes:\n        show the attributes of the packages when listing all packages.\n        Default: True\n\n    :return: Packages installed or available, along with their attributes.\n    :rtype: list(list(str))\n\n    .. code-block:: bash\n\n        salt '*' nix.list_pkgs\n        salt '*' nix.list_pkgs installed=False\n    "
    cmd = _nix_env()
    cmd.append('--query')
    if installed:
        cmd.append('--installed')
    if not installed:
        cmd.append('--available')
        if attributes:
            cmd.append('--attr-path')
    out = _run(cmd)
    return [s.split() for s in salt.utils.itertools.split(out['stdout'], '\n')]

def uninstall(*pkgs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Erases a package from the current nix profile. Nix uninstalls work differently than other package managers, and the symlinks in the\n    profile are removed, while the actual package remains. There is also a ``nix.purge`` function, to clear the package cache of unused\n    packages.\n\n    :type pkgs: list(str)\n    :param pkgs:\n        List, single package to uninstall\n\n    :return: Packages that have been uninstalled\n    :rtype: list(str)\n\n    .. code-block:: bash\n\n        salt '*' nix.uninstall pkg1 [pkg2 ...]\n    "
    cmd = _quietnix()
    cmd.append('--uninstall')
    cmd.extend(pkgs)
    out = _run(cmd)
    fmtout = (out['stderr'].splitlines(), 'uninstalling')
    return [_strip_quotes(s.split()[1]) for s in out['stderr'].splitlines() if s.startswith('uninstalling')]

def collect_garbage():
    if False:
        return 10
    "\n    Completely removed all currently 'uninstalled' packages in the nix store.\n\n    Tells the user how many store paths were removed and how much space was freed.\n\n    :return: How much space was freed and how many derivations were removed\n    :rtype: str\n\n    .. warning::\n       This is a destructive action on the nix store.\n\n    .. code-block:: bash\n\n        salt '*' nix.collect_garbage\n    "
    cmd = _nix_collect_garbage()
    cmd.append('--delete-old')
    out = _run(cmd)
    return out['stdout'].splitlines()