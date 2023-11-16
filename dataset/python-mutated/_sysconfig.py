import logging
import os
import sys
import sysconfig
import typing
from pip._internal.exceptions import InvalidSchemeCombination, UserInstallationInvalid
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import change_root, get_major_minor_version, is_osx_framework
logger = logging.getLogger(__name__)
_AVAILABLE_SCHEMES = set(sysconfig.get_scheme_names())
_PREFERRED_SCHEME_API = getattr(sysconfig, 'get_preferred_scheme', None)

def _should_use_osx_framework_prefix() -> bool:
    if False:
        return 10
    "Check for Apple's ``osx_framework_library`` scheme.\n\n    Python distributed by Apple's Command Line Tools has this special scheme\n    that's used when:\n\n    * This is a framework build.\n    * We are installing into the system prefix.\n\n    This does not account for ``pip install --prefix`` (also means we're not\n    installing to the system prefix), which should use ``posix_prefix``, but\n    logic here means ``_infer_prefix()`` outputs ``osx_framework_library``. But\n    since ``prefix`` is not available for ``sysconfig.get_default_scheme()``,\n    which is the stdlib replacement for ``_infer_prefix()``, presumably Apple\n    wouldn't be able to magically switch between ``osx_framework_library`` and\n    ``posix_prefix``. ``_infer_prefix()`` returning ``osx_framework_library``\n    means its behavior is consistent whether we use the stdlib implementation\n    or our own, and we deal with this special case in ``get_scheme()`` instead.\n    "
    return 'osx_framework_library' in _AVAILABLE_SCHEMES and (not running_under_virtualenv()) and is_osx_framework()

def _infer_prefix() -> str:
    if False:
        return 10
    'Try to find a prefix scheme for the current platform.\n\n    This tries:\n\n    * A special ``osx_framework_library`` for Python distributed by Apple\'s\n      Command Line Tools, when not running in a virtual environment.\n    * Implementation + OS, used by PyPy on Windows (``pypy_nt``).\n    * Implementation without OS, used by PyPy on POSIX (``pypy``).\n    * OS + "prefix", used by CPython on POSIX (``posix_prefix``).\n    * Just the OS name, used by CPython on Windows (``nt``).\n\n    If none of the above works, fall back to ``posix_prefix``.\n    '
    if _PREFERRED_SCHEME_API:
        return _PREFERRED_SCHEME_API('prefix')
    if _should_use_osx_framework_prefix():
        return 'osx_framework_library'
    implementation_suffixed = f'{sys.implementation.name}_{os.name}'
    if implementation_suffixed in _AVAILABLE_SCHEMES:
        return implementation_suffixed
    if sys.implementation.name in _AVAILABLE_SCHEMES:
        return sys.implementation.name
    suffixed = f'{os.name}_prefix'
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    if os.name in _AVAILABLE_SCHEMES:
        return os.name
    return 'posix_prefix'

def _infer_user() -> str:
    if False:
        return 10
    'Try to find a user scheme for the current platform.'
    if _PREFERRED_SCHEME_API:
        return _PREFERRED_SCHEME_API('user')
    if is_osx_framework() and (not running_under_virtualenv()):
        suffixed = 'osx_framework_user'
    else:
        suffixed = f'{os.name}_user'
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    if 'posix_user' not in _AVAILABLE_SCHEMES:
        raise UserInstallationInvalid()
    return 'posix_user'

def _infer_home() -> str:
    if False:
        print('Hello World!')
    'Try to find a home for the current platform.'
    if _PREFERRED_SCHEME_API:
        return _PREFERRED_SCHEME_API('home')
    suffixed = f'{os.name}_home'
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    return 'posix_home'
_HOME_KEYS = ['installed_base', 'base', 'installed_platbase', 'platbase', 'prefix', 'exec_prefix']
if sysconfig.get_config_var('userbase') is not None:
    _HOME_KEYS.append('userbase')

def get_scheme(dist_name: str, user: bool=False, home: typing.Optional[str]=None, root: typing.Optional[str]=None, isolated: bool=False, prefix: typing.Optional[str]=None) -> Scheme:
    if False:
        i = 10
        return i + 15
    '\n    Get the "scheme" corresponding to the input parameters.\n\n    :param dist_name: the name of the package to retrieve the scheme for, used\n        in the headers scheme path\n    :param user: indicates to use the "user" scheme\n    :param home: indicates to use the "home" scheme\n    :param root: root under which other directories are re-based\n    :param isolated: ignored, but kept for distutils compatibility (where\n        this controls whether the user-site pydistutils.cfg is honored)\n    :param prefix: indicates to use the "prefix" scheme and provides the\n        base directory for the same\n    '
    if user and prefix:
        raise InvalidSchemeCombination('--user', '--prefix')
    if home and prefix:
        raise InvalidSchemeCombination('--home', '--prefix')
    if home is not None:
        scheme_name = _infer_home()
    elif user:
        scheme_name = _infer_user()
    else:
        scheme_name = _infer_prefix()
    if prefix is not None and scheme_name == 'osx_framework_library':
        scheme_name = 'posix_prefix'
    if home is not None:
        variables = {k: home for k in _HOME_KEYS}
    elif prefix is not None:
        variables = {k: prefix for k in _HOME_KEYS}
    else:
        variables = {}
    paths = sysconfig.get_paths(scheme=scheme_name, vars=variables)
    if running_under_virtualenv():
        if user:
            base = variables.get('userbase', sys.prefix)
        else:
            base = variables.get('base', sys.prefix)
        python_xy = f'python{get_major_minor_version()}'
        paths['include'] = os.path.join(base, 'include', 'site', python_xy)
    elif not dist_name:
        dist_name = 'UNKNOWN'
    scheme = Scheme(platlib=paths['platlib'], purelib=paths['purelib'], headers=os.path.join(paths['include'], dist_name), scripts=paths['scripts'], data=paths['data'])
    if root is not None:
        for key in SCHEME_KEYS:
            value = change_root(root, getattr(scheme, key))
            setattr(scheme, key, value)
    return scheme

def get_bin_prefix() -> str:
    if False:
        while True:
            i = 10
    if sys.platform[:6] == 'darwin' and sys.prefix[:16] == '/System/Library/':
        return '/usr/local/bin'
    return sysconfig.get_paths()['scripts']

def get_purelib() -> str:
    if False:
        while True:
            i = 10
    return sysconfig.get_paths()['purelib']

def get_platlib() -> str:
    if False:
        while True:
            i = 10
    return sysconfig.get_paths()['platlib']