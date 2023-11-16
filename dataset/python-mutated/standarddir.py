"""Utilities to get and initialize data/config paths."""
import os
import os.path
import sys
import contextlib
import enum
import argparse
from typing import Iterator, Optional
from qutebrowser.qt.core import QStandardPaths
from qutebrowser.qt.widgets import QApplication
from qutebrowser.utils import log, debug, utils, version, qtutils
_locations = {}

class _Location(enum.Enum):
    """A key for _locations."""
    config = enum.auto()
    auto_config = enum.auto()
    data = enum.auto()
    system_data = enum.auto()
    cache = enum.auto()
    download = enum.auto()
    runtime = enum.auto()
    config_py = enum.auto()
APPNAME = 'qutebrowser'

class EmptyValueError(Exception):
    """Error raised when QStandardPaths returns an empty value."""

@contextlib.contextmanager
def _unset_organization() -> Iterator[None]:
    if False:
        return 10
    'Temporarily unset QApplication.organizationName().\n\n    This is primarily needed in config.py.\n    '
    qapp = QApplication.instance()
    if qapp is not None:
        orgname = qapp.organizationName()
        qapp.setOrganizationName(qtutils.QT_NONE)
    try:
        yield
    finally:
        if qapp is not None:
            qapp.setOrganizationName(orgname)

def _init_config(args: Optional[argparse.Namespace]) -> None:
    if False:
        while True:
            i = 10
    'Initialize the location for configs.'
    typ = QStandardPaths.StandardLocation.ConfigLocation
    path = _from_args(typ, args)
    if path is None:
        if utils.is_windows:
            app_data_path = _writable_location(QStandardPaths.StandardLocation.AppDataLocation)
            path = os.path.join(app_data_path, 'config')
        else:
            path = _writable_location(typ)
    _create(path)
    _locations[_Location.config] = path
    _locations[_Location.auto_config] = path
    if utils.is_mac:
        path = _from_args(typ, args)
        if path is None:
            path = os.path.expanduser('~/.' + APPNAME)
            _create(path)
            _locations[_Location.config] = path
    config_py_file = os.path.join(_locations[_Location.config], 'config.py')
    if getattr(args, 'config_py', None) is not None:
        assert args is not None
        config_py_file = os.path.abspath(args.config_py)
    _locations[_Location.config_py] = config_py_file

def config(auto: bool=False) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get the location for the config directory.\n\n    If auto=True is given, get the location for the autoconfig.yml directory,\n    which is different on macOS.\n    '
    if auto:
        return _locations[_Location.auto_config]
    return _locations[_Location.config]

def config_py() -> str:
    if False:
        print('Hello World!')
    'Get the location for config.py.\n\n    Usually, config.py is in standarddir.config(), but this can be overridden\n    with the --config-py argument.\n    '
    return _locations[_Location.config_py]

def _init_data(args: Optional[argparse.Namespace]) -> None:
    if False:
        i = 10
        return i + 15
    'Initialize the location for data.'
    typ = QStandardPaths.StandardLocation.AppDataLocation
    path = _from_args(typ, args)
    if path is None:
        if utils.is_windows:
            app_data_path = _writable_location(typ)
            path = os.path.join(app_data_path, 'data')
        elif sys.platform.startswith('haiku'):
            config_path = _writable_location(QStandardPaths.StandardLocation.ConfigLocation)
            path = os.path.join(config_path, 'data')
        else:
            path = _writable_location(typ)
    _create(path)
    _locations[_Location.data] = path
    _locations.pop(_Location.system_data, None)
    if utils.is_linux:
        prefix = '/app' if version.is_flatpak() else '/usr'
        path = f'{prefix}/share/{APPNAME}'
        if os.path.exists(path):
            _locations[_Location.system_data] = path

def data(system: bool=False) -> str:
    if False:
        i = 10
        return i + 15
    'Get the data directory.\n\n    If system=True is given, gets the system-wide (probably non-writable) data\n    directory.\n    '
    if system:
        try:
            return _locations[_Location.system_data]
        except KeyError:
            pass
    return _locations[_Location.data]

def _init_cache(args: Optional[argparse.Namespace]) -> None:
    if False:
        i = 10
        return i + 15
    'Initialize the location for the cache.'
    typ = QStandardPaths.StandardLocation.CacheLocation
    path = _from_args(typ, args)
    if path is None:
        if utils.is_windows:
            data_path = _writable_location(QStandardPaths.StandardLocation.AppLocalDataLocation)
            path = os.path.join(data_path, 'cache')
        else:
            path = _writable_location(typ)
    _create(path)
    _locations[_Location.cache] = path

def cache() -> str:
    if False:
        print('Hello World!')
    return _locations[_Location.cache]

def _init_download(args: Optional[argparse.Namespace]) -> None:
    if False:
        i = 10
        return i + 15
    "Initialize the location for downloads.\n\n    Note this is only the default directory as found by Qt.\n    Therefore, we also don't create it.\n    "
    typ = QStandardPaths.StandardLocation.DownloadLocation
    path = _from_args(typ, args)
    if path is None:
        path = _writable_location(typ)
    _locations[_Location.download] = path

def download() -> str:
    if False:
        print('Hello World!')
    return _locations[_Location.download]

def _init_runtime(args: Optional[argparse.Namespace]) -> None:
    if False:
        print('Hello World!')
    'Initialize location for runtime data.'
    if utils.is_mac or utils.is_windows:
        typ = QStandardPaths.StandardLocation.TempLocation
    else:
        typ = QStandardPaths.StandardLocation.RuntimeLocation
    path = _from_args(typ, args)
    if path is None:
        try:
            path = _writable_location(typ)
        except EmptyValueError:
            if typ == QStandardPaths.StandardLocation.TempLocation:
                raise
            path = _writable_location(QStandardPaths.StandardLocation.TempLocation)
        if version.is_flatpak():
            (*parts, app_name) = os.path.split(path)
            assert app_name == APPNAME, app_name
            flatpak_id = version.flatpak_id()
            assert flatpak_id is not None
            path = os.path.join(*parts, 'app', flatpak_id)
    _create(path)
    _locations[_Location.runtime] = path

def runtime() -> str:
    if False:
        return 10
    return _locations[_Location.runtime]

def _writable_location(typ: QStandardPaths.StandardLocation) -> str:
    if False:
        while True:
            i = 10
    'Wrapper around QStandardPaths.writableLocation.\n\n    Arguments:\n        typ: A QStandardPaths::StandardLocation member.\n    '
    typ_str = debug.qenum_key(QStandardPaths, typ)
    assert typ in [QStandardPaths.StandardLocation.ConfigLocation, QStandardPaths.StandardLocation.AppLocalDataLocation, QStandardPaths.StandardLocation.CacheLocation, QStandardPaths.StandardLocation.DownloadLocation, QStandardPaths.StandardLocation.RuntimeLocation, QStandardPaths.StandardLocation.TempLocation, QStandardPaths.StandardLocation.AppDataLocation], typ_str
    with _unset_organization():
        path = QStandardPaths.writableLocation(typ)
    log.misc.debug('writable location for {}: {}'.format(typ_str, path))
    if not path:
        raise EmptyValueError('QStandardPaths returned an empty value!')
    path = path.replace('/', os.sep)
    if typ != QStandardPaths.StandardLocation.DownloadLocation and path.split(os.sep)[-1] != APPNAME:
        path = os.path.join(path, APPNAME)
    return path

def _from_args(typ: QStandardPaths.StandardLocation, args: Optional[argparse.Namespace]) -> Optional[str]:
    if False:
        while True:
            i = 10
    'Get the standard directory from an argparse namespace.\n\n    Return:\n        The overridden path, or None if there is no override.\n    '
    basedir_suffix = {QStandardPaths.StandardLocation.ConfigLocation: 'config', QStandardPaths.StandardLocation.AppDataLocation: 'data', QStandardPaths.StandardLocation.AppLocalDataLocation: 'data', QStandardPaths.StandardLocation.CacheLocation: 'cache', QStandardPaths.StandardLocation.DownloadLocation: 'download', QStandardPaths.StandardLocation.RuntimeLocation: 'runtime'}
    if getattr(args, 'basedir', None) is None:
        return None
    assert args is not None
    try:
        suffix = basedir_suffix[typ]
    except KeyError:
        return None
    return os.path.abspath(os.path.join(args.basedir, suffix))

def _create(path: str) -> None:
    if False:
        print('Hello World!')
    'Create the `path` directory.\n\n    From the XDG basedir spec:\n        If, when attempting to write a file, the destination directory is\n        non-existent an attempt should be made to create it with permission\n        0700. If the destination directory exists already the permissions\n        should not be changed.\n    '
    if APPNAME == 'qute_test' and path.startswith('/home'):
        for (k, v) in os.environ.items():
            if k == 'HOME' or k.startswith('XDG_'):
                log.init.debug(f'{k} = {v}')
        raise AssertionError('Trying to create directory inside /home during tests, this should not happen.')
    os.makedirs(path, 448, exist_ok=True)

def _init_dirs(args: argparse.Namespace=None) -> None:
    if False:
        i = 10
        return i + 15
    'Create and cache standard directory locations.\n\n    Mainly in a separate function because we need to call it in tests.\n    '
    _init_config(args)
    _init_data(args)
    _init_cache(args)
    _init_download(args)
    _init_runtime(args)

def init(args: Optional[argparse.Namespace]) -> None:
    if False:
        return 10
    'Initialize all standard dirs.'
    if args is not None:
        log.init.debug('Base directory: {}'.format(args.basedir))
    _init_dirs(args)
    _init_cachedir_tag()

def _init_cachedir_tag() -> None:
    if False:
        print('Hello World!')
    "Create CACHEDIR.TAG if it doesn't exist.\n\n    See https://bford.info/cachedir/\n    "
    cachedir_tag = os.path.join(cache(), 'CACHEDIR.TAG')
    if not os.path.exists(cachedir_tag):
        try:
            with open(cachedir_tag, 'w', encoding='utf-8') as f:
                f.write('Signature: 8a477f597d28d172789f06886806bc55\n')
                f.write('# This file is a cache directory tag created by qutebrowser.\n')
                f.write('# For information about cache directory tags, see:\n')
                f.write('#  https://bford.info/cachedir/\n')
        except OSError:
            log.init.exception('Failed to create CACHEDIR.TAG')