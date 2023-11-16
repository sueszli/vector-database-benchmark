"""Locations where we look for configs, install stuff, etc"""
try:
    __import__('_distutils_hack').remove_shim()
except (ImportError, AttributeError):
    pass
import logging
import os
import sys
from distutils.cmd import Command as DistutilsCommand
from distutils.command.install import SCHEME_KEYS
from distutils.command.install import install as distutils_install_command
from distutils.sysconfig import get_python_lib
from typing import Dict, List, Optional, Union, cast
from pip._internal.models.scheme import Scheme
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import get_major_minor_version
logger = logging.getLogger(__name__)

def distutils_scheme(dist_name: str, user: bool=False, home: Optional[str]=None, root: Optional[str]=None, isolated: bool=False, prefix: Optional[str]=None, *, ignore_config_files: bool=False) -> Dict[str, str]:
    if False:
        return 10
    '\n    Return a distutils install scheme\n    '
    from distutils.dist import Distribution
    dist_args: Dict[str, Union[str, List[str]]] = {'name': dist_name}
    if isolated:
        dist_args['script_args'] = ['--no-user-cfg']
    d = Distribution(dist_args)
    if not ignore_config_files:
        try:
            d.parse_config_files()
        except UnicodeDecodeError:
            paths = d.find_config_files()
            logger.warning('Ignore distutils configs in %s due to encoding errors.', ', '.join((os.path.basename(p) for p in paths)))
    obj: Optional[DistutilsCommand] = None
    obj = d.get_command_obj('install', create=True)
    assert obj is not None
    i = cast(distutils_install_command, obj)
    assert not (user and prefix), f'user={user} prefix={prefix}'
    assert not (home and prefix), f'home={home} prefix={prefix}'
    i.user = user or i.user
    if user or home:
        i.prefix = ''
    i.prefix = prefix or i.prefix
    i.home = home or i.home
    i.root = root or i.root
    i.finalize_options()
    scheme = {}
    for key in SCHEME_KEYS:
        scheme[key] = getattr(i, 'install_' + key)
    if 'install_lib' in d.get_option_dict('install'):
        scheme.update({'purelib': i.install_lib, 'platlib': i.install_lib})
    if running_under_virtualenv():
        if home:
            prefix = home
        elif user:
            prefix = i.install_userbase
        else:
            prefix = i.prefix
        scheme['headers'] = os.path.join(prefix, 'include', 'site', f'python{get_major_minor_version()}', dist_name)
        if root is not None:
            path_no_drive = os.path.splitdrive(os.path.abspath(scheme['headers']))[1]
            scheme['headers'] = os.path.join(root, path_no_drive[1:])
    return scheme

def get_scheme(dist_name: str, user: bool=False, home: Optional[str]=None, root: Optional[str]=None, isolated: bool=False, prefix: Optional[str]=None) -> Scheme:
    if False:
        while True:
            i = 10
    '\n    Get the "scheme" corresponding to the input parameters. The distutils\n    documentation provides the context for the available schemes:\n    https://docs.python.org/3/install/index.html#alternate-installation\n\n    :param dist_name: the name of the package to retrieve the scheme for, used\n        in the headers scheme path\n    :param user: indicates to use the "user" scheme\n    :param home: indicates to use the "home" scheme and provides the base\n        directory for the same\n    :param root: root under which other directories are re-based\n    :param isolated: equivalent to --no-user-cfg, i.e. do not consider\n        ~/.pydistutils.cfg (posix) or ~/pydistutils.cfg (non-posix) for\n        scheme paths\n    :param prefix: indicates to use the "prefix" scheme and provides the\n        base directory for the same\n    '
    scheme = distutils_scheme(dist_name, user, home, root, isolated, prefix)
    return Scheme(platlib=scheme['platlib'], purelib=scheme['purelib'], headers=scheme['headers'], scripts=scheme['scripts'], data=scheme['data'])

def get_bin_prefix() -> str:
    if False:
        return 10
    prefix = os.path.normpath(sys.prefix)
    if WINDOWS:
        bin_py = os.path.join(prefix, 'Scripts')
        if not os.path.exists(bin_py):
            bin_py = os.path.join(prefix, 'bin')
        return bin_py
    if sys.platform[:6] == 'darwin' and prefix[:16] == '/System/Library/':
        return '/usr/local/bin'
    return os.path.join(prefix, 'bin')

def get_purelib() -> str:
    if False:
        i = 10
        return i + 15
    return get_python_lib(plat_specific=False)

def get_platlib() -> str:
    if False:
        while True:
            i = 10
    return get_python_lib(plat_specific=True)