"""Utilities for determining application-specific dirs.

See <http://github.com/ActiveState/appdirs> for details and usage.
"""
__version_info__ = (1, 3, 0)
__version__ = '.'.join(map(str, __version_info__))
import sys
import os
PY3 = sys.version_info[0] == 3
if PY3:
    unicode = str

def user_data_dir(appname=None, appauthor=None, version=None, roaming=False):
    if False:
        for i in range(10):
            print('nop')
    'Return full path to the user-specific data dir for this application.\n\n        "appname" is the name of application.\n            If None, just the system directory is returned.\n        "appauthor" (only required and used on Windows) is the name of the\n            appauthor or distributing body for this application. Typically\n            it is the owning company name. This falls back to appname.\n        "version" is an optional version path element to append to the\n            path. You might want to use this if you want multiple versions\n            of your app to be able to run independently. If used, this\n            would typically be "<major>.<minor>".\n            Only applied when appname is present.\n        "roaming" (boolean, default False) can be set True to use the Windows\n            roaming appdata directory. That means that for users on a Windows\n            network setup for roaming profiles, this user data will be\n            sync\'d on login. See\n            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>\n            for a discussion of issues.\n\n    Typical user data directories are:\n        Mac OS X:               ~/Library/Application Support/<AppName>\n        Unix:                   ~/.local/share/<AppName>    # or in $XDG_DATA_HOME, if defined\n        Win XP (not roaming):   C:\\Documents and Settings\\<username>\\Application Data\\<AppAuthor>\\<AppName>\n        Win XP (roaming):       C:\\Documents and Settings\\<username>\\Local Settings\\Application Data\\<AppAuthor>\\<AppName>\n        Win 7  (not roaming):   C:\\Users\\<username>\\AppData\\Local\\<AppAuthor>\\<AppName>\n        Win 7  (roaming):       C:\\Users\\<username>\\AppData\\Roaming\\<AppAuthor>\\<AppName>\n\n    For Unix, we follow the XDG spec and support $XDG_DATA_HOME.\n    That means, by deafult "~/.local/share/<AppName>".\n    '
    if sys.platform == 'win32':
        if appauthor is None:
            appauthor = appname
        const = roaming and 'CSIDL_APPDATA' or 'CSIDL_LOCAL_APPDATA'
        path = os.path.normpath(_get_win_folder(const))
        if appname:
            path = os.path.join(path, appauthor, appname)
    elif sys.platform == 'darwin':
        path = os.path.expanduser('~/Library/Application Support/')
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    return path

def site_data_dir(appname=None, appauthor=None, version=None, multipath=False):
    if False:
        return 10
    'Return full path to the user-shared data dir for this application.\n\n        "appname" is the name of application.\n            If None, just the system directory is returned.\n        "appauthor" (only required and used on Windows) is the name of the\n            appauthor or distributing body for this application. Typically\n            it is the owning company name. This falls back to appname.\n        "version" is an optional version path element to append to the\n            path. You might want to use this if you want multiple versions\n            of your app to be able to run independently. If used, this\n            would typically be "<major>.<minor>".\n            Only applied when appname is present.\n        "multipath" is an optional parameter only applicable to *nix\n            which indicates that the entire list of data dirs should be\n            returned. By default, the first item from XDG_DATA_DIRS is\n            returned, or \'/usr/local/share/<AppName>\',\n            if XDG_DATA_DIRS is not set\n\n    Typical user data directories are:\n        Mac OS X:   /Library/Application Support/<AppName>\n        Unix:       /usr/local/share/<AppName> or /usr/share/<AppName>\n        Win XP:     C:\\Documents and Settings\\All Users\\Application Data\\<AppAuthor>\\<AppName>\n        Vista:      (Fail! "C:\\ProgramData" is a hidden *system* directory on Vista.)\n        Win 7:      C:\\ProgramData\\<AppAuthor>\\<AppName>   # Hidden, but writeable on Win 7.\n\n    For Unix, this is using the $XDG_DATA_DIRS[0] default.\n\n    WARNING: Do not use this on Windows. See the Vista-Fail note above for why.\n    '
    if sys.platform == 'win32':
        if appauthor is None:
            appauthor = appname
        path = os.path.normpath(_get_win_folder('CSIDL_COMMON_APPDATA'))
        if appname:
            path = os.path.join(path, appauthor, appname)
    elif sys.platform == 'darwin':
        path = os.path.expanduser('/Library/Application Support')
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv('XDG_DATA_DIRS', os.pathsep.join(['/usr/local/share', '/usr/share']))
        pathlist = [os.path.expanduser(x.rstrip(os.sep)) for x in path.split(os.pathsep)]
        if appname:
            if version:
                appname = os.path.join(appname, version)
            pathlist = [os.sep.join([x, appname]) for x in pathlist]
        if multipath:
            path = os.pathsep.join(pathlist)
        else:
            path = pathlist[0]
        return path
    if appname and version:
        path = os.path.join(path, version)
    return path

def user_config_dir(appname=None, appauthor=None, version=None, roaming=False):
    if False:
        while True:
            i = 10
    'Return full path to the user-specific config dir for this application.\n\n        "appname" is the name of application.\n            If None, just the system directory is returned.\n        "appauthor" (only required and used on Windows) is the name of the\n            appauthor or distributing body for this application. Typically\n            it is the owning company name. This falls back to appname.\n        "version" is an optional version path element to append to the\n            path. You might want to use this if you want multiple versions\n            of your app to be able to run independently. If used, this\n            would typically be "<major>.<minor>".\n            Only applied when appname is present.\n        "roaming" (boolean, default False) can be set True to use the Windows\n            roaming appdata directory. That means that for users on a Windows\n            network setup for roaming profiles, this user data will be\n            sync\'d on login. See\n            <http://technet.microsoft.com/en-us/library/cc766489(WS.10).aspx>\n            for a discussion of issues.\n\n    Typical user data directories are:\n        Mac OS X:               same as user_data_dir\n        Unix:                   ~/.config/<AppName>     # or in $XDG_CONFIG_HOME, if defined\n        Win *:                  same as user_data_dir\n\n    For Unix, we follow the XDG spec and support $XDG_DATA_HOME.\n    That means, by deafult "~/.local/share/<AppName>".\n    '
    if sys.platform in ['win32', 'darwin']:
        path = user_data_dir(appname, appauthor, None, roaming)
    else:
        path = os.getenv('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    return path

def site_config_dir(appname=None, appauthor=None, version=None, multipath=False):
    if False:
        while True:
            i = 10
    'Return full path to the user-shared data dir for this application.\n\n        "appname" is the name of application.\n            If None, just the system directory is returned.\n        "appauthor" (only required and used on Windows) is the name of the\n            appauthor or distributing body for this application. Typically\n            it is the owning company name. This falls back to appname.\n        "version" is an optional version path element to append to the\n            path. You might want to use this if you want multiple versions\n            of your app to be able to run independently. If used, this\n            would typically be "<major>.<minor>".\n            Only applied when appname is present.\n        "multipath" is an optional parameter only applicable to *nix\n            which indicates that the entire list of config dirs should be\n            returned. By default, the first item from XDG_CONFIG_DIRS is\n            returned, or \'/etc/xdg/<AppName>\', if XDG_CONFIG_DIRS is not set\n\n    Typical user data directories are:\n        Mac OS X:   same as site_data_dir\n        Unix:       /etc/xdg/<AppName> or $XDG_CONFIG_DIRS[i]/<AppName> for each value in\n                    $XDG_CONFIG_DIRS\n        Win *:      same as site_data_dir\n        Vista:      (Fail! "C:\\ProgramData" is a hidden *system* directory on Vista.)\n\n    For Unix, this is using the $XDG_CONFIG_DIRS[0] default, if multipath=False\n\n    WARNING: Do not use this on Windows. See the Vista-Fail note above for why.\n    '
    if sys.platform in ['win32', 'darwin']:
        path = site_data_dir(appname, appauthor)
        if appname and version:
            path = os.path.join(path, version)
    else:
        path = os.getenv('XDG_CONFIG_DIRS', '/etc/xdg')
        pathlist = [os.path.expanduser(x.rstrip(os.sep)) for x in path.split(os.pathsep)]
        if appname:
            if version:
                appname = os.path.join(appname, version)
            pathlist = [os.sep.join([x, appname]) for x in pathlist]
        if multipath:
            path = os.pathsep.join(pathlist)
        else:
            path = pathlist[0]
    return path

def user_cache_dir(appname=None, appauthor=None, version=None, opinion=True):
    if False:
        while True:
            i = 10
    'Return full path to the user-specific cache dir for this application.\n\n        "appname" is the name of application.\n            If None, just the system directory is returned.\n        "appauthor" (only required and used on Windows) is the name of the\n            appauthor or distributing body for this application. Typically\n            it is the owning company name. This falls back to appname.\n        "version" is an optional version path element to append to the\n            path. You might want to use this if you want multiple versions\n            of your app to be able to run independently. If used, this\n            would typically be "<major>.<minor>".\n            Only applied when appname is present.\n        "opinion" (boolean) can be False to disable the appending of\n            "Cache" to the base app data dir for Windows. See\n            discussion below.\n\n    Typical user cache directories are:\n        Mac OS X:   ~/Library/Caches/<AppName>\n        Unix:       ~/.cache/<AppName> (XDG default)\n        Win XP:     C:\\Documents and Settings\\<username>\\Local Settings\\Application Data\\<AppAuthor>\\<AppName>\\Cache\n        Vista:      C:\\Users\\<username>\\AppData\\Local\\<AppAuthor>\\<AppName>\\Cache\n\n    On Windows the only suggestion in the MSDN docs is that local settings go in\n    the `CSIDL_LOCAL_APPDATA` directory. This is identical to the non-roaming\n    app data dir (the default returned by `user_data_dir` above). Apps typically\n    put cache data somewhere *under* the given dir here. Some examples:\n        ...\\Mozilla\\Firefox\\Profiles\\<ProfileName>\\Cache\n        ...\\Acme\\SuperApp\\Cache\\1.0\n    OPINION: This function appends "Cache" to the `CSIDL_LOCAL_APPDATA` value.\n    This can be disabled with the `opinion=False` option.\n    '
    if sys.platform == 'win32':
        if appauthor is None:
            appauthor = appname
        path = os.path.normpath(_get_win_folder('CSIDL_LOCAL_APPDATA'))
        if appname:
            path = os.path.join(path, appauthor, appname)
            if opinion:
                path = os.path.join(path, 'Cache')
    elif sys.platform == 'darwin':
        path = os.path.expanduser('~/Library/Caches')
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)
    return path

def user_log_dir(appname=None, appauthor=None, version=None, opinion=True):
    if False:
        print('Hello World!')
    'Return full path to the user-specific log dir for this application.\n\n        "appname" is the name of application.\n            If None, just the system directory is returned.\n        "appauthor" (only required and used on Windows) is the name of the\n            appauthor or distributing body for this application. Typically\n            it is the owning company name. This falls back to appname.\n        "version" is an optional version path element to append to the\n            path. You might want to use this if you want multiple versions\n            of your app to be able to run independently. If used, this\n            would typically be "<major>.<minor>".\n            Only applied when appname is present.\n        "opinion" (boolean) can be False to disable the appending of\n            "Logs" to the base app data dir for Windows, and "log" to the\n            base cache dir for Unix. See discussion below.\n\n    Typical user cache directories are:\n        Mac OS X:   ~/Library/Logs/<AppName>\n        Unix:       ~/.cache/<AppName>/log  # or under $XDG_CACHE_HOME if defined\n        Win XP:     C:\\Documents and Settings\\<username>\\Local Settings\\Application Data\\<AppAuthor>\\<AppName>\\Logs\n        Vista:      C:\\Users\\<username>\\AppData\\Local\\<AppAuthor>\\<AppName>\\Logs\n\n    On Windows the only suggestion in the MSDN docs is that local settings\n    go in the `CSIDL_LOCAL_APPDATA` directory. (Note: I\'m interested in\n    examples of what some windows apps use for a logs dir.)\n\n    OPINION: This function appends "Logs" to the `CSIDL_LOCAL_APPDATA`\n    value for Windows and appends "log" to the user cache dir for Unix.\n    This can be disabled with the `opinion=False` option.\n    '
    if sys.platform == 'darwin':
        path = os.path.join(os.path.expanduser('~/Library/Logs'), appname)
    elif sys.platform == 'win32':
        path = user_data_dir(appname, appauthor, version)
        version = False
        if opinion:
            path = os.path.join(path, 'Logs')
    else:
        path = user_cache_dir(appname, appauthor, version)
        version = False
        if opinion:
            path = os.path.join(path, 'log')
    if appname and version:
        path = os.path.join(path, version)
    return path

class AppDirs(object):
    """Convenience wrapper for getting application dirs."""

    def __init__(self, appname, appauthor=None, version=None, roaming=False, multipath=False):
        if False:
            for i in range(10):
                print('nop')
        self.appname = appname
        self.appauthor = appauthor
        self.version = version
        self.roaming = roaming
        self.multipath = multipath

    @property
    def user_data_dir(self):
        if False:
            i = 10
            return i + 15
        return user_data_dir(self.appname, self.appauthor, version=self.version, roaming=self.roaming)

    @property
    def site_data_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return site_data_dir(self.appname, self.appauthor, version=self.version, multipath=self.multipath)

    @property
    def user_config_dir(self):
        if False:
            i = 10
            return i + 15
        return user_config_dir(self.appname, self.appauthor, version=self.version, roaming=self.roaming)

    @property
    def site_config_dir(self):
        if False:
            i = 10
            return i + 15
        return site_data_dir(self.appname, self.appauthor, version=self.version, multipath=self.multipath)

    @property
    def user_cache_dir(self):
        if False:
            print('Hello World!')
        return user_cache_dir(self.appname, self.appauthor, version=self.version)

    @property
    def user_log_dir(self):
        if False:
            print('Hello World!')
        return user_log_dir(self.appname, self.appauthor, version=self.version)

def _get_win_folder_from_registry(csidl_name):
    if False:
        print('Hello World!')
    "This is a fallback technique at best. I'm not sure if using the\n    registry for this guarantees us the correct answer for all CSIDL_*\n    names.\n    "
    import _winreg
    shell_folder_name = {'CSIDL_APPDATA': 'AppData', 'CSIDL_COMMON_APPDATA': 'Common AppData', 'CSIDL_LOCAL_APPDATA': 'Local AppData'}[csidl_name]
    key = _winreg.OpenKey(_winreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders')
    (dir, type) = _winreg.QueryValueEx(key, shell_folder_name)
    return dir

def _get_win_folder_with_pywin32(csidl_name):
    if False:
        i = 10
        return i + 15
    from win32com.shell import shellcon, shell
    dir = shell.SHGetFolderPath(0, getattr(shellcon, csidl_name), 0, 0)
    try:
        dir = unicode(dir)
        has_high_char = False
        for c in dir:
            if ord(c) > 255:
                has_high_char = True
                break
        if has_high_char:
            try:
                import win32api
                dir = win32api.GetShortPathName(dir)
            except ImportError:
                pass
    except UnicodeError:
        pass
    return dir

def _get_win_folder_with_ctypes(csidl_name):
    if False:
        return 10
    import ctypes
    csidl_const = {'CSIDL_APPDATA': 26, 'CSIDL_COMMON_APPDATA': 35, 'CSIDL_LOCAL_APPDATA': 28}[csidl_name]
    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)
    has_high_char = False
    for c in buf:
        if ord(c) > 255:
            has_high_char = True
            break
    if has_high_char:
        buf2 = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
            buf = buf2
    return buf.value
if sys.platform == 'win32':
    try:
        import win32com.shell
        _get_win_folder = _get_win_folder_with_pywin32
    except ImportError:
        try:
            import ctypes
            _get_win_folder = _get_win_folder_with_ctypes
        except ImportError:
            _get_win_folder = _get_win_folder_from_registry
if __name__ == '__main__':
    appname = 'MyApp'
    appauthor = 'MyCompany'
    props = ('user_data_dir', 'site_data_dir', 'user_config_dir', 'site_config_dir', 'user_cache_dir', 'user_log_dir')
    print("-- app dirs (with optional 'version')")
    dirs = AppDirs(appname, appauthor, version='1.0')
    for prop in props:
        print('%s: %s' % (prop, getattr(dirs, prop)))
    print("\n-- app dirs (without optional 'version')")
    dirs = AppDirs(appname, appauthor)
    for prop in props:
        print('%s: %s' % (prop, getattr(dirs, prop)))
    print("\n-- app dirs (without optional 'appauthor')")
    dirs = AppDirs(appname)
    for prop in props:
        print('%s: %s' % (prop, getattr(dirs, prop)))