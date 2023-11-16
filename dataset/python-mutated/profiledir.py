"""An object for managing IPython profile directories."""
import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe

class ProfileDirError(Exception):
    pass

class ProfileDir(LoggingConfigurable):
    """An object to manage the profile directory and its resources.

    The profile directory is used by all IPython applications, to manage
    configuration, logging and security.

    This object knows how to find, create and manage these directories. This
    should be used by any code that wants to handle profiles.
    """
    security_dir_name = Unicode('security')
    log_dir_name = Unicode('log')
    startup_dir_name = Unicode('startup')
    pid_dir_name = Unicode('pid')
    static_dir_name = Unicode('static')
    security_dir = Unicode(u'')
    log_dir = Unicode(u'')
    startup_dir = Unicode(u'')
    pid_dir = Unicode(u'')
    static_dir = Unicode(u'')
    location = Unicode(u'', help='Set the profile location directly. This overrides the logic used by the\n        `profile` option.').tag(config=True)
    _location_isset = Bool(False)

    @observe('location')
    def _location_changed(self, change):
        if False:
            for i in range(10):
                print('nop')
        if self._location_isset:
            raise RuntimeError('Cannot set profile location more than once.')
        self._location_isset = True
        new = change['new']
        ensure_dir_exists(new)
        self.security_dir = os.path.join(new, self.security_dir_name)
        self.log_dir = os.path.join(new, self.log_dir_name)
        self.startup_dir = os.path.join(new, self.startup_dir_name)
        self.pid_dir = os.path.join(new, self.pid_dir_name)
        self.static_dir = os.path.join(new, self.static_dir_name)
        self.check_dirs()

    def _mkdir(self, path, mode=None):
        if False:
            print('Hello World!')
        'ensure a directory exists at a given path\n\n        This is a version of os.mkdir, with the following differences:\n\n        - returns True if it created the directory, False otherwise\n        - ignores EEXIST, protecting against race conditions where\n          the dir may have been created in between the check and\n          the creation\n        - sets permissions if requested and the dir already exists\n        '
        if os.path.exists(path):
            if mode and os.stat(path).st_mode != mode:
                try:
                    os.chmod(path, mode)
                except OSError:
                    self.log.warning('Could not set permissions on %s', path)
            return False
        try:
            if mode:
                os.mkdir(path, mode)
            else:
                os.mkdir(path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                return False
            else:
                raise
        return True

    @observe('log_dir')
    def check_log_dir(self, change=None):
        if False:
            for i in range(10):
                print('nop')
        self._mkdir(self.log_dir)

    @observe('startup_dir')
    def check_startup_dir(self, change=None):
        if False:
            for i in range(10):
                print('nop')
        self._mkdir(self.startup_dir)
        readme = os.path.join(self.startup_dir, 'README')
        src = os.path.join(get_ipython_package_dir(), u'core', u'profile', u'README_STARTUP')
        if not os.path.exists(src):
            self.log.warning('Could not copy README_STARTUP to startup dir. Source file %s does not exist.', src)
        if os.path.exists(src) and (not os.path.exists(readme)):
            shutil.copy(src, readme)

    @observe('security_dir')
    def check_security_dir(self, change=None):
        if False:
            while True:
                i = 10
        self._mkdir(self.security_dir, 16832)

    @observe('pid_dir')
    def check_pid_dir(self, change=None):
        if False:
            print('Hello World!')
        self._mkdir(self.pid_dir, 16832)

    def check_dirs(self):
        if False:
            print('Hello World!')
        self.check_security_dir()
        self.check_log_dir()
        self.check_pid_dir()
        self.check_startup_dir()

    def copy_config_file(self, config_file: str, path: Path, overwrite=False) -> bool:
        if False:
            return 10
        'Copy a default config file into the active profile directory.\n\n        Default configuration files are kept in :mod:`IPython.core.profile`.\n        This function moves these from that location to the working profile\n        directory.\n        '
        dst = Path(os.path.join(self.location, config_file))
        if dst.exists() and (not overwrite):
            return False
        if path is None:
            path = os.path.join(get_ipython_package_dir(), u'core', u'profile', u'default')
        assert isinstance(path, Path)
        src = path / config_file
        shutil.copy(src, dst)
        return True

    @classmethod
    def create_profile_dir(cls, profile_dir, config=None):
        if False:
            print('Hello World!')
        'Create a new profile directory given a full path.\n\n        Parameters\n        ----------\n        profile_dir : str\n            The full path to the profile directory.  If it does exist, it will\n            be used.  If not, it will be created.\n        '
        return cls(location=profile_dir, config=config)

    @classmethod
    def create_profile_dir_by_name(cls, path, name=u'default', config=None):
        if False:
            print('Hello World!')
        'Create a profile dir by profile name and path.\n\n        Parameters\n        ----------\n        path : unicode\n            The path (directory) to put the profile directory in.\n        name : unicode\n            The name of the profile.  The name of the profile directory will\n            be "profile_<profile>".\n        '
        if not os.path.isdir(path):
            raise ProfileDirError('Directory not found: %s' % path)
        profile_dir = os.path.join(path, u'profile_' + name)
        return cls(location=profile_dir, config=config)

    @classmethod
    def find_profile_dir_by_name(cls, ipython_dir, name=u'default', config=None):
        if False:
            i = 10
            return i + 15
        'Find an existing profile dir by profile name, return its ProfileDir.\n\n        This searches through a sequence of paths for a profile dir.  If it\n        is not found, a :class:`ProfileDirError` exception will be raised.\n\n        The search path algorithm is:\n        1. ``os.getcwd()`` # removed for security reason.\n        2. ``ipython_dir``\n\n        Parameters\n        ----------\n        ipython_dir : unicode or str\n            The IPython directory to use.\n        name : unicode or str\n            The name of the profile.  The name of the profile directory\n            will be "profile_<profile>".\n        '
        dirname = u'profile_' + name
        paths = [ipython_dir]
        for p in paths:
            profile_dir = os.path.join(p, dirname)
            if os.path.isdir(profile_dir):
                return cls(location=profile_dir, config=config)
        else:
            raise ProfileDirError('Profile directory not found in paths: %s' % dirname)

    @classmethod
    def find_profile_dir(cls, profile_dir, config=None):
        if False:
            return 10
        "Find/create a profile dir and return its ProfileDir.\n\n        This will create the profile directory if it doesn't exist.\n\n        Parameters\n        ----------\n        profile_dir : unicode or str\n            The path of the profile directory.\n        "
        profile_dir = expand_path(profile_dir)
        if not os.path.isdir(profile_dir):
            raise ProfileDirError('Profile directory not found: %s' % profile_dir)
        return cls(location=profile_dir, config=config)