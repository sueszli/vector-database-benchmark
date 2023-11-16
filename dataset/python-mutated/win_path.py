"""
Manage the Windows System PATH

Note that not all Windows applications will rehash the PATH environment variable,
Only the ones that listen to the WM_SETTINGCHANGE message.
"""
import logging
import os
import salt.utils.args
import salt.utils.data
import salt.utils.platform
import salt.utils.stringutils
import salt.utils.win_functions
try:
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
log = logging.getLogger(__name__)
HIVE = 'HKEY_LOCAL_MACHINE'
KEY = 'SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment'
VNAME = 'PATH'
VTYPE = 'REG_EXPAND_SZ'
PATHSEP = str(os.pathsep)

def __virtual__():
    if False:
        return 10
    '\n    Load only on Windows\n    '
    if salt.utils.platform.is_windows() and HAS_WIN32:
        return 'win_path'
    return (False, 'Module win_path: module only works on Windows systems')

def _normalize_dir(string_):
    if False:
        return 10
    '\n    Normalize the directory to make comparison possible\n    '
    return os.path.normpath(salt.utils.stringutils.to_unicode(string_))

def rehash():
    if False:
        i = 10
        return i + 15
    "\n    Send a WM_SETTINGCHANGE Broadcast to Windows to refresh the Environment\n    variables for new processes.\n\n    .. note::\n        This will only affect new processes that aren't launched by services. To\n        apply changes to the path to services, the host must be restarted. The\n        ``salt-minion``, if running as a service, will not see changes to the\n        environment until the system is restarted. See\n        `MSDN Documentation <https://support.microsoft.com/en-us/help/821761/changes-that-you-make-to-environment-variables-do-not-affect-services>`_\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_path.rehash\n    "
    return salt.utils.win_functions.broadcast_setting_change('Environment')

def get_path():
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a list of items in the SYSTEM path\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_path.get_path\n    "
    ret = salt.utils.stringutils.to_unicode(__utils__['reg.read_value']('HKEY_LOCAL_MACHINE', 'SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment', 'PATH')['vdata']).split(';')
    ret = ret[:-1] if ret[-1] == '' else ret
    return list(map(_normalize_dir, ret))

def exists(path):
    if False:
        return 10
    "\n    Check if the directory is configured in the SYSTEM path\n    Case-insensitive and ignores trailing backslash\n\n    Returns:\n        boolean True if path exists, False if not\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_path.exists 'c:\\python27'\n        salt '*' win_path.exists 'c:\\python27\\'\n        salt '*' win_path.exists 'C:\\pyThon27'\n    "
    path = _normalize_dir(path)
    sysPath = get_path()
    return path.lower() in (x.lower() for x in sysPath)

def _update_local_path(local_path):
    if False:
        print('Hello World!')
    os.environ['PATH'] = PATHSEP.join(local_path)

def add(path, index=None, **kwargs):
    if False:
        return 10
    "\n    Add the directory to the SYSTEM path in the index location. Returns\n    ``True`` if successful, otherwise ``False``.\n\n    path\n        Directory to add to path\n\n    index\n        Optionally specify an index at which to insert the directory\n\n    rehash : True\n        If the registry was updated, and this value is set to ``True``, sends a\n        WM_SETTINGCHANGE broadcast to refresh the environment variables. Set\n        this to ``False`` to skip this broadcast.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        # Will add to the beginning of the path\n        salt '*' win_path.add 'c:\\python27' 0\n\n        # Will add to the end of the path\n        salt '*' win_path.add 'c:\\python27' index='-1'\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    rehash_ = kwargs.pop('rehash', True)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    path = _normalize_dir(path)
    if path == '.':
        return False
    path_str = salt.utils.stringutils.to_str(path)
    system_path = get_path()
    local_path = [salt.utils.stringutils.to_str(x) for x in os.environ['PATH'].split(PATHSEP)]
    if index is not None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = None

    def _check_path(dirs, path, index):
        if False:
            i = 10
            return i + 15
        '\n        Check the dir list for the specified path, at the specified index, and\n        make changes to the list if needed. Return True if changes were made to\n        the list, otherwise return False.\n        '
        dirs_lc = [x.lower() for x in dirs]
        try:
            cur_index = dirs_lc.index(path.lower())
        except ValueError:
            cur_index = None
        num_dirs = len(dirs)
        pos = index
        if index is not None:
            if index >= num_dirs or index == -1:
                pos = 'END'
            elif index <= -num_dirs:
                index = pos = 0
            elif index < 0:
                pos += 1
        if pos == 'END':
            if cur_index is not None:
                if cur_index == num_dirs - 1:
                    return False
                else:
                    dirs.pop(cur_index)
                    dirs.append(path)
                    return True
            else:
                dirs.append(path)
                return True
        elif index is None:
            if cur_index is not None:
                return False
            else:
                dirs.append(path)
                return True
        elif cur_index is not None:
            if index < 0 and cur_index != num_dirs + index or (index >= 0 and cur_index != index):
                dirs.pop(cur_index)
                dirs.insert(pos, path)
                return True
            else:
                return False
        else:
            dirs.insert(pos, path)
            return True
    if _check_path(local_path, path_str, index):
        _update_local_path(local_path)
    if not _check_path(system_path, path, index):
        return True
    result = __utils__['reg.set_value'](HIVE, KEY, VNAME, ';'.join(salt.utils.data.decode(system_path)), VTYPE)
    if result and rehash_:
        return rehash()
    else:
        return result

def remove(path, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Remove the directory from the SYSTEM path\n\n    Returns:\n        boolean True if successful, False if unsuccessful\n\n    rehash : True\n        If the registry was updated, and this value is set to ``True``, sends a\n        WM_SETTINGCHANGE broadcast to refresh the environment variables. Set\n        this to ``False`` to skip this broadcast.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Will remove C:\\Python27 from the path\n        salt '*' win_path.remove 'c:\\\\python27'\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    rehash_ = kwargs.pop('rehash', True)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    path = _normalize_dir(path)
    path_str = salt.utils.stringutils.to_str(path)
    system_path = get_path()
    local_path = [salt.utils.stringutils.to_str(x) for x in os.environ['PATH'].split(PATHSEP)]

    def _check_path(dirs, path):
        if False:
            return 10
        '\n        Check the dir list for the specified path, and make changes to the list\n        if needed. Return True if changes were made to the list, otherwise\n        return False.\n        '
        dirs_lc = [x.lower() for x in dirs]
        path_lc = path.lower()
        new_dirs = []
        for (index, dirname) in enumerate(dirs_lc):
            if path_lc != dirname:
                new_dirs.append(dirs[index])
        if len(new_dirs) != len(dirs):
            dirs[:] = new_dirs[:]
            return True
        else:
            return False
    if _check_path(local_path, path_str):
        _update_local_path(local_path)
    if not _check_path(system_path, path):
        return True
    result = __utils__['reg.set_value'](HIVE, KEY, VNAME, ';'.join(salt.utils.data.decode(system_path)), VTYPE)
    if result and rehash_:
        return rehash()
    else:
        return result