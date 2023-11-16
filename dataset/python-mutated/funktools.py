"""
A small collection of useful functional tools for working with iterables.
"""
import errno
import locale
import os
import stat
import subprocess
import time
import warnings
from functools import partial
from itertools import count, islice
from typing import Any, Iterable
DIRECTORY_CLEANUP_TIMEOUT = 1.0

def _is_iterable(elem: Any) -> bool:
    if False:
        return 10
    if getattr(elem, '__iter__', False) or isinstance(elem, Iterable):
        return True
    return False

def take(n: int, iterable: Iterable) -> Iterable:
    if False:
        return 10
    'Take n elements from the supplied iterable without consuming it.\n\n    :param int n: Number of unique groups\n    :param iter iterable: An iterable to split up\n    '
    return list(islice(iterable, n))

def chunked(n: int, iterable: Iterable) -> Iterable:
    if False:
        print('Hello World!')
    'Split an iterable into lists of length *n*.\n\n    :param int n: Number of unique groups\n    :param iter iterable: An iterable to split up\n\n    '
    return iter(partial(take, n, iter(iterable)), [])

def unnest(elem: Iterable) -> Any:
    if False:
        return 10
    'Flatten an arbitrarily nested iterable.\n\n    :param elem: An iterable to flatten\n    :type elem: :class:`~collections.Iterable`\n    >>> nested_iterable = (\n            1234, (3456, 4398345, (234234)), (\n                2396, (\n                    928379, 29384, (\n                        293759, 2347, (\n                            2098, 7987, 27599\n                        )\n                    )\n                )\n            )\n        )\n    >>> list(unnest(nested_iterable))\n    [1234, 3456, 4398345, 234234, 2396, 928379, 29384, 293759,\n     2347, 2098, 7987, 27599]\n    '
    if isinstance(elem, Iterable) and (not isinstance(elem, str)):
        for el in elem:
            if isinstance(el, Iterable) and (not isinstance(el, str)):
                yield from unnest(el)
            else:
                yield el
    else:
        yield elem

def dedup(iterable: Iterable) -> Iterable:
    if False:
        for i in range(10):
            print('nop')
    'Deduplicate an iterable object like iter(set(iterable)) but order-\n    preserved.'
    return iter(dict.fromkeys(iterable))

def is_readonly_path(fn: os.PathLike) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'check if a provided path exists and is readonly.\n\n    permissions check is `bool(path.stat & stat.s_iread)` or `not\n    os.access(path, os.w_ok)`\n    '
    if os.path.exists(fn):
        file_stat = os.stat(fn).st_mode
        return not bool(file_stat & stat.s_iwrite) or not os.access(fn, os.w_ok)
    return False

def _wait_for_files(path):
    if False:
        while True:
            i = 10
    'Retry with backoff up to 1 second to delete files from a directory.\n\n    :param str path: The path to crawl to delete files from\n    :return: A list of remaining paths or None\n    :rtype: Optional[List[str]]\n    '
    timeout = 0.001
    remaining = []
    while timeout < DIRECTORY_CLEANUP_TIMEOUT:
        remaining = []
        if os.path.isdir(path):
            L = os.listdir(path)
            for target in L:
                _remaining = _wait_for_files(target)
                if _remaining:
                    remaining.extend(_remaining)
            continue
        try:
            os.unlink(path)
        except FileNotFoundError as e:
            if e.errno == errno.ENOENT:
                return
        except (OSError, PermissionError):
            time.sleep(timeout)
            timeout *= 2
            remaining.append(path)
        else:
            return
    return remaining

def _walk_for_powershell(directory):
    if False:
        i = 10
        return i + 15
    for (_, dirs, files) in os.walk(directory):
        powershell = next(iter((fn for fn in files if fn.lower() == 'powershell.exe')), None)
        if powershell is not None:
            return os.path.join(directory, powershell)
        for subdir in dirs:
            powershell = _walk_for_powershell(os.path.join(directory, subdir))
            if powershell:
                return powershell
    return None

def _get_powershell_path():
    if False:
        for i in range(10):
            print('nop')
    paths = [os.path.expandvars('%windir%\\{0}\\WindowsPowerShell').format(subdir) for subdir in ('SysWOW64', 'system32')]
    powershell_path = next(iter((_walk_for_powershell(pth) for pth in paths)), None)
    if not powershell_path:
        powershell_path = subprocess.run(['where', 'powershell'], check=False)
    if powershell_path.stdout:
        return powershell_path.stdout.strip()

def _get_sid_with_powershell():
    if False:
        i = 10
        return i + 15
    powershell_path = _get_powershell_path()
    if not powershell_path:
        return None
    args = [powershell_path, '-ExecutionPolicy', 'Bypass', '-Command', "Invoke-Expression '[System.Security.Principal.WindowsIdentity]::GetCurrent().user | Write-Host'"]
    sid = subprocess.run(args, capture_output=True, check=False)
    return sid.stdout.strip()

def get_value_from_tuple(value, value_type):
    if False:
        print('Hello World!')
    try:
        import winreg
    except ImportError:
        import _winreg as winreg
    if value_type in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
        if '\x00' in value:
            return value[:value.index('\x00')]
        return value
    return None

def query_registry_value(root, key_name, value):
    if False:
        for i in range(10):
            print('nop')
    try:
        import winreg
    except ImportError:
        import _winreg as winreg
    try:
        with winreg.OpenKeyEx(root, key_name, 0, winreg.KEY_READ) as key:
            return get_value_from_tuple(*winreg.QueryValueEx(key, value))
    except OSError:
        return None

def _get_sid_from_registry():
    if False:
        i = 10
        return i + 15
    try:
        import winreg
    except ImportError:
        import _winreg as winreg
    var_names = ('%USERPROFILE%', '%HOME%')
    current_user_home = next(iter((os.path.expandvars(v) for v in var_names if v)), None)
    (root, subkey) = (winreg.HKEY_LOCAL_MACHINE, 'Software\\Microsoft\\Windows NT\\CurrentVersion\\ProfileList')
    subkey_names = []
    value = None
    matching_key = None
    try:
        with winreg.OpenKeyEx(root, subkey, 0, winreg.KEY_READ) as key:
            for i in count():
                key_name = winreg.EnumKey(key, i)
                subkey_names.append(key_name)
                value = query_registry_value(root, f'{subkey}\\{key_name}', 'ProfileImagePath')
                if value and value.lower() == current_user_home.lower():
                    matching_key = key_name
                    break
    except OSError:
        pass
    if matching_key is not None:
        return matching_key

def _get_current_user():
    if False:
        i = 10
        return i + 15
    fns = (_get_sid_from_registry, _get_sid_with_powershell)
    for fn in fns:
        result = fn()
        if result:
            return result
    return None

def _find_icacls_exe():
    if False:
        i = 10
        return i + 15
    if os.name == 'nt':
        paths = [os.path.expandvars('%windir%\\{0}').format(subdir) for subdir in ('system32', 'SysWOW64')]
        for path in paths:
            icacls_path = next(iter((fn for fn in os.listdir(path) if fn.lower() == 'icacls.exe')), None)
            if icacls_path is not None:
                icacls_path = os.path.join(path, icacls_path)
                return icacls_path
    return None

def set_write_bit(fn: str) -> None:
    if False:
        while True:
            i = 10
    "Set read-write permissions for the current user on the target path. Fail\n    silently if the path doesn't exist.\n\n    :param str fn: The target filename or path\n    :return: None\n    "
    if not os.path.exists(fn):
        return
    file_stat = os.stat(fn).st_mode
    os.chmod(fn, file_stat | stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    if os.name == 'nt':
        user_sid = _get_current_user()
        icacls_exe = _find_icacls_exe() or 'icacls'
        if user_sid:
            c = subprocess.run([icacls_exe, f"''{fn}''", '/grant', f'{user_sid}:WD', '/T', '/C', '/Q'], capture_output=True, encoding=locale.getpreferredencoding(), check=False)
            if not c.err and c.returncode == 0:
                return
    if not os.path.isdir(fn):
        for path in [fn, os.path.dirname(fn)]:
            try:
                os.chflags(path, 0)
            except AttributeError:
                pass
        return None
    for (root, dirs, files) in os.walk(fn, topdown=False):
        for dir_ in [os.path.join(root, d) for d in dirs]:
            set_write_bit(dir_)
        for file_ in [os.path.join(root, f) for f in files]:
            set_write_bit(file_)

def handle_remove_readonly(func, path, exc):
    if False:
        print('Hello World!')
    'Error handler for shutil.rmtree.\n\n    Windows source repo folders are read-only by default, so this error handler\n    attempts to set them as writeable and then proceed with deletion.\n\n    :param function func: The caller function\n    :param str path: The target path for removal\n    :param Exception exc: The raised exception\n\n    This function will call check :func:`is_readonly_path` before attempting to call\n    :func:`set_write_bit` on the target path and try again.\n    '
    PERM_ERRORS = (errno.EACCES, errno.EPERM, errno.ENOENT)
    default_warning_message = 'Unable to remove file due to permissions restriction: {!r}'
    (exc_type, exc_exception, exc_tb) = exc
    if is_readonly_path(path):
        set_write_bit(path)
        try:
            func(path)
        except (OSError, FileNotFoundError, PermissionError) as e:
            if e.errno in PERM_ERRORS:
                if e.errno == errno.ENOENT:
                    return
                remaining = None
                if os.path.isdir(path):
                    remaining = _wait_for_files(path)
                if remaining:
                    warnings.warn(default_warning_message.format(path), ResourceWarning, stacklevel=2)
                else:
                    func(path, ignore_errors=True)
                return
    if exc_exception.errno in PERM_ERRORS:
        set_write_bit(path)
        remaining = _wait_for_files(path)
        try:
            func(path)
        except (OSError, FileNotFoundError, PermissionError) as e:
            if e.errno in PERM_ERRORS and e.errno != errno.ENOENT:
                warnings.warn(default_warning_message.format(path), ResourceWarning, stacklevel=2)
            return
    else:
        raise exc_exception