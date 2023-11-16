"""
Module for listing programs that automatically run on startup
(very alpha...not tested on anything but my Win 7x64)
"""
import os
import salt.utils.platform
__func_alias__ = {'list_': 'list'}
__virtualname__ = 'autoruns'

def __virtual__():
    if False:
        return 10
    '\n    Only works on Windows systems\n    '
    if salt.utils.platform.is_windows():
        return __virtualname__
    return (False, 'Module win_autoruns: module only works on Windows systems')

def _get_dirs(user_dir, startup_dir):
    if False:
        return 10
    '\n    Return a list of startup dirs\n    '
    try:
        users = os.listdir(user_dir)
    except OSError:
        users = []
    full_dirs = []
    for user in users:
        full_dir = os.path.join(user_dir, user, startup_dir)
        if os.path.exists(full_dir):
            full_dirs.append(full_dir)
    return full_dirs

def list_():
    if False:
        while True:
            i = 10
    "\n    Get a list of automatically running programs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' autoruns.list\n    "
    autoruns = {}
    keys = ['HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run', 'HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /reg:64', 'HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run']
    for key in keys:
        autoruns[key] = []
        cmd = ['reg', 'query', key]
        for line in __salt__['cmd.run'](cmd, python_shell=False).splitlines():
            if line and line[0:4] != 'HKEY' and (line[0:5] != 'ERROR'):
                autoruns[key].append(line)
    user_dir = 'C:\\Documents and Settings\\'
    startup_dir = '\\Start Menu\\Programs\\Startup'
    full_dirs = _get_dirs(user_dir, startup_dir)
    if not full_dirs:
        user_dir = 'C:\\Users\\'
        startup_dir = '\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup'
        full_dirs = _get_dirs(user_dir, startup_dir)
    for full_dir in full_dirs:
        files = os.listdir(full_dir)
        autoruns[full_dir] = []
        for single_file in files:
            autoruns[full_dir].append(single_file)
    return autoruns