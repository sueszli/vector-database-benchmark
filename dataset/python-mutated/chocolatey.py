"""
A module that wraps calls to the Chocolatey package manager
(http://chocolatey.org)

.. versionadded:: 2014.1.0
"""
import logging
import os
import re
import tempfile
from requests.structures import CaseInsensitiveDict
import salt.utils.data
import salt.utils.platform
from salt.exceptions import CommandExecutionError, CommandNotFoundError, MinionError, SaltInvocationError
from salt.utils.data import CaseInsensitiveDict
from salt.utils.versions import Version
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}
__virtualname__ = 'chocolatey'

def __virtual__():
    if False:
        return 10
    '\n    Confirm this module is on a Windows system running Vista or later.\n\n    While it is possible to make Chocolatey run under XP and Server 2003 with\n    an awful lot of hassle (e.g. SSL is completely broken), the PowerShell shim\n    for simulating UAC forces a GUI prompt, and is not compatible with\n    salt-minion running as SYSTEM.\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'Chocolatey: Requires Windows')
    if __grains__['osrelease'] in ('XP', '2003Server'):
        return (False, 'Chocolatey: Requires Windows Vista or later')
    return __virtualname__

def _clear_context():
    if False:
        while True:
            i = 10
    '\n    Clear variables stored in __context__. Run this function when a new version\n    of chocolatey is installed.\n    '
    choco_items = [x for x in __context__ if x.startswith('chocolatey.')]
    for var in choco_items:
        __context__.pop(var)

def _yes():
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns ['--yes'] if on v0.9.9.0 or later, otherwise returns an empty list\n    Confirm all prompts (--yes_ is available on v0.9.9.0 or later\n    "
    if 'chocolatey._yes' in __context__:
        return __context__['chocolatey._yes']
    if Version(chocolatey_version()) >= Version('0.9.9'):
        answer = ['--yes']
    else:
        answer = []
    __context__['chocolatey._yes'] = answer
    return __context__['chocolatey._yes']

def _no_progress():
    if False:
        while True:
            i = 10
    "\n    Returns ['--no-progress'] if on v0.10.4 or later, otherwise returns an\n    empty list\n    "
    if 'chocolatey._no_progress' in __context__:
        return __context__['chocolatey._no_progress']
    if Version(chocolatey_version()) >= Version('0.10.4'):
        answer = ['--no-progress']
    else:
        log.warning('--no-progress unsupported in choco < 0.10.4')
        answer = []
    __context__['chocolatey._no_progress'] = answer
    return __context__['chocolatey._no_progress']

def _find_chocolatey():
    if False:
        while True:
            i = 10
    '\n    Returns the full path to chocolatey.bat on the host.\n    '
    if 'chocolatey._path' in __context__:
        return __context__['chocolatey._path']
    choc_path = __salt__['cmd.which']('chocolatey.exe')
    if choc_path:
        __context__['chocolatey._path'] = choc_path
        return __context__['chocolatey._path']
    choc_defaults = [os.path.join(os.environ.get('ProgramData'), 'Chocolatey', 'bin', 'chocolatey.exe'), os.path.join(os.environ.get('ProgramData'), 'Chocolatey', 'bin', 'choco.exe'), os.path.join(os.environ.get('SystemDrive'), 'Chocolatey', 'bin', 'chocolatey.bat')]
    for choc_exe in choc_defaults:
        if os.path.isfile(choc_exe):
            __context__['chocolatey._path'] = choc_exe
            return __context__['chocolatey._path']
    err = 'Chocolatey not installed. Use chocolatey.bootstrap to install the Chocolatey package manager.'
    raise CommandExecutionError(err)

def chocolatey_version():
    if False:
        i = 10
        return i + 15
    "\n    Returns the version of Chocolatey installed on the minion.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.chocolatey_version\n    "
    if 'chocolatey._version' in __context__:
        return __context__['chocolatey._version']
    cmd = [_find_chocolatey()]
    cmd.append('-v')
    out = __salt__['cmd.run'](cmd, python_shell=False)
    __context__['chocolatey._version'] = out
    return __context__['chocolatey._version']

def bootstrap(force=False, source=None):
    if False:
        i = 10
        return i + 15
    "\n    Download and install the latest version of the Chocolatey package manager\n    via the official bootstrap.\n\n    Chocolatey requires Windows PowerShell and the .NET v4.0 runtime. Depending\n    on the host's version of Windows, chocolatey.bootstrap will attempt to\n    ensure these prerequisites are met by downloading and executing the\n    appropriate installers from Microsoft.\n\n    .. note::\n        If PowerShell is installed, you may have to restart the host machine for\n        Chocolatey to work.\n\n    .. note::\n        If you're installing offline using the source parameter, the PowerShell\n        and .NET requirements must already be met on the target. This shouldn't\n        be a problem on Windows versions 2012/8 and later\n\n    Args:\n\n        force (bool):\n            Run the bootstrap process even if Chocolatey is found in the path.\n\n        source (str):\n            The location of the ``.nupkg`` file or ``.ps1`` file to run from an\n            alternate location. This can be one of the following types of URLs:\n\n            - salt://\n            - http(s)://\n            - ftp://\n            - file:// - A local file on the system\n\n            .. versionadded:: 3001\n\n    Returns:\n        str: The stdout of the Chocolatey installation script\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # To bootstrap Chocolatey\n        salt '*' chocolatey.bootstrap\n        salt '*' chocolatey.bootstrap force=True\n\n        # To bootstrap Chocolatey offline from a file on the salt master\n        salt '*' chocolatey.bootstrap source=salt://files/chocolatey.nupkg\n\n        # To bootstrap Chocolatey from a file on C:\\Temp\n        salt '*' chocolatey.bootstrap source=C:\\Temp\\chocolatey.nupkg\n    "
    try:
        choc_path = _find_chocolatey()
    except CommandExecutionError:
        choc_path = None
    if choc_path and (not force):
        return f'Chocolatey found at {choc_path}'
    temp_dir = tempfile.gettempdir()
    powershell_info = __salt__['cmd.shell_info'](shell='powershell')
    if not powershell_info['installed']:
        ps_downloads = {('Vista', 'x86'): 'http://download.microsoft.com/download/A/7/5/A75BC017-63CE-47D6-8FA4-AFB5C21BAC54/Windows6.0-KB968930-x86.msu', ('Vista', 'AMD64'): 'http://download.microsoft.com/download/3/C/8/3C8CF51E-1D9D-4DAA-AAEA-5C48D1CD055C/Windows6.0-KB968930-x64.msu', ('2008Server', 'x86'): 'http://download.microsoft.com/download/F/9/E/F9EF6ACB-2BA8-4845-9C10-85FC4A69B207/Windows6.0-KB968930-x86.msu', ('2008Server', 'AMD64'): 'http://download.microsoft.com/download/2/8/6/28686477-3242-4E96-9009-30B16BED89AF/Windows6.0-KB968930-x64.msu'}
        if (__grains__['osrelease'], __grains__['cpuarch']) in ps_downloads:
            url = ps_downloads[__grains__['osrelease'], __grains__['cpuarch']]
            dest = os.path.join(temp_dir, os.path.basename(url))
            try:
                log.debug('Downloading PowerShell...')
                __salt__['cp.get_url'](path=url, dest=dest)
            except MinionError:
                err = 'Failed to download PowerShell KB for {}'.format(__grains__['osrelease'])
                if source:
                    raise CommandExecutionError('{}: PowerShell is required to bootstrap Chocolatey with Source'.format(err))
                raise CommandExecutionError(err)
            cmd = [dest, '/quiet', '/norestart']
            log.debug('Installing PowerShell...')
            result = __salt__['cmd.run_all'](cmd, python_shell=False)
            if result['retcode'] != 0:
                err = 'Failed to install PowerShell KB. For more information run the installer manually on the host'
                raise CommandExecutionError(err)
        else:
            err = 'Windows PowerShell Installation not available'
            raise CommandNotFoundError(err)
    if not __utils__['dotnet.version_at_least'](version='4'):
        url = 'http://download.microsoft.com/download/1/B/E/1BE39E79-7E39-46A3-96FF-047F95396215/dotNetFx40_Full_setup.exe'
        dest = os.path.join(temp_dir, os.path.basename(url))
        try:
            log.debug('Downloading .NET v4.0...')
            __salt__['cp.get_url'](path=url, dest=dest)
        except MinionError:
            err = 'Failed to download .NET v4.0 Web Installer'
            if source:
                err = '{}: .NET v4.0+ is required to bootstrap Chocolatey with Source'.format(err)
            raise CommandExecutionError(err)
        cmd = [dest, '/q', '/norestart']
        log.debug('Installing .NET v4.0...')
        result = __salt__['cmd.run_all'](cmd, python_shell=False)
        if result['retcode'] != 0:
            err = 'Failed to install .NET v4.0 failed. For more information run the installer manually on the host'
            raise CommandExecutionError(err)
    if source:
        url = source
    else:
        url = 'https://chocolatey.org/install.ps1'
    dest = os.path.join(temp_dir, os.path.basename(url))
    try:
        log.debug('Downloading Chocolatey: %s', os.path.basename(url))
        script = __salt__['cp.get_url'](path=url, dest=dest)
        log.debug('Script: %s', script)
    except MinionError:
        err = 'Failed to download Chocolatey Installer'
        if source:
            err = '{0} from source'
        raise CommandExecutionError(err)
    if os.path.splitext(os.path.basename(dest))[1] == '.nupkg':
        log.debug('Unzipping Chocolatey: %s', dest)
        __salt__['archive.unzip'](zip_file=dest, dest=os.path.join(os.path.dirname(dest), 'chocolatey'), extract_perms=False)
        script = os.path.join(os.path.dirname(dest), 'chocolatey', 'tools', 'chocolateyInstall.ps1')
    if not os.path.exists(script):
        raise CommandExecutionError(f'Failed to find Chocolatey installation script: {script}')
    log.debug('Installing Chocolatey: %s', script)
    result = __salt__['cmd.script'](script, cwd=os.path.dirname(script), shell='powershell', python_shell=True)
    if result['retcode'] != 0:
        err = 'Bootstrapping Chocolatey failed: {}'.format(result['stderr'])
        raise CommandExecutionError(err)
    return result['stdout']

def unbootstrap():
    if False:
        return 10
    '\n    Uninstall chocolatey from the system by doing the following:\n\n    - Delete the Chocolatey Directory\n    - Remove Chocolatey from the path\n    - Remove Chocolatey environment variables\n\n    .. versionadded:: 3001\n\n    Returns:\n        list: A list of items that were removed, otherwise an empty list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt * chocolatey.unbootstrap\n    '
    removed = []
    choco_dir = os.environ.get('ChocolateyInstall', False)
    if choco_dir:
        if os.path.exists(choco_dir):
            log.debug('Removing Chocolatey directory: %s', choco_dir)
            __salt__['file.remove'](path=choco_dir, force=True)
            removed.append(f'Removed Directory: {choco_dir}')
    else:
        known_paths = [os.path.join(os.environ.get('ProgramData'), 'Chocolatey'), os.path.join(os.environ.get('SystemDrive'), 'Chocolatey')]
        for path in known_paths:
            if os.path.exists(path):
                log.debug('Removing Chocolatey directory: %s', path)
                __salt__['file.remove'](path=path, force=True)
                removed.append(f'Removed Directory: {path}')
    for env_var in __salt__['environ.items']():
        if env_var.lower().startswith('chocolatey'):
            log.debug('Removing Chocolatey environment variable: %s', env_var)
            __salt__['environ.setval'](key=env_var, val=False, false_unsets=True, permanent='HKLM')
            __salt__['environ.setval'](key=env_var, val=False, false_unsets=True, permanent='HKCU')
            removed.append(f'Removed Environment Var: {env_var}')
    for path in __salt__['win_path.get_path']():
        if 'chocolatey' in path.lower():
            log.debug('Removing Chocolatey path item: %s', path)
            __salt__['win_path.remove'](path=path, rehash=True)
            removed.append(f'Removed Path Item: {path}')
    return removed

def list_(narrow=None, all_versions=False, pre_versions=False, source=None, local_only=False, exact=False):
    if False:
        while True:
            i = 10
    "\n    Instructs Chocolatey to pull a vague package list from the repository.\n\n    Args:\n\n        narrow (str):\n            Term used to narrow down results. Searches against\n            name/description/tag. Default is None.\n\n        all_versions (bool):\n            Display all available package versions in results. Default is False.\n\n        pre_versions (bool):\n            Display pre-release packages in results. Default is False.\n\n        source (str):\n            Chocolatey repository (directory, share or remote URL feed) the\n            package comes from. Defaults to the official Chocolatey feed if\n            None is passed. Default is None.\n\n        local_only (bool):\n            Only display packages that are installed locally. Default is False.\n\n        exact (bool):\n            Only display packages that match ``narrow`` exactly. Default is\n            False.\n\n            .. versionadded:: 2017.7.0\n\n    Returns:\n        dict: A dictionary of results.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.list <narrow>\n        salt '*' chocolatey.list <narrow> all_versions=True\n    "
    choc_path = _find_chocolatey()
    if Version(chocolatey_version()) < Version('2.0.0'):
        cmd = [choc_path, 'list']
        if local_only:
            cmd.append('--local-only')
    elif local_only:
        cmd = [choc_path, 'list']
    else:
        cmd = [choc_path, 'search']
    if narrow:
        cmd.append(narrow)
    if salt.utils.data.is_true(all_versions):
        cmd.append('--allversions')
    if salt.utils.data.is_true(pre_versions):
        cmd.append('--prerelease')
    if source:
        cmd.extend(['--source', source])
    if exact:
        cmd.append('--exact')
    cmd.append('--limit-output')
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] not in [0, 2]:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    ret = CaseInsensitiveDict({})
    pkg_re = re.compile('(\\S+)\\|(\\S+)')
    for line in result['stdout'].split('\n'):
        if line.startswith('No packages'):
            return ret
        for (name, ver) in pkg_re.findall(line):
            if 'chocolatey' in name:
                continue
            if name not in ret:
                ret[name] = []
            ret[name].append(ver)
    return ret

def list_webpi():
    if False:
        for i in range(10):
            print('nop')
    "\n    Instructs Chocolatey to pull a full package list from the Microsoft Web PI\n    repository.\n\n    Returns:\n        str: List of webpi packages\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.list_webpi\n    "
    choc_path = _find_chocolatey()
    if Version(chocolatey_version()) < Version('2.0.0'):
        cmd = [choc_path, 'list', '--source', 'webpi']
    else:
        cmd = [choc_path, 'search', '--source', 'webpi']
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] != 0:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    return result['stdout']

def list_windowsfeatures():
    if False:
        return 10
    "\n    Instructs Chocolatey to pull a full package list from the Windows Features\n    list, via the Deployment Image Servicing and Management tool.\n\n    Returns:\n        str: List of Windows Features\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.list_windowsfeatures\n    "
    choc_path = _find_chocolatey()
    if Version(chocolatey_version()) < Version('2.0.0'):
        cmd = [choc_path, 'list', '--source', 'windowsfeatures']
    else:
        cmd = [choc_path, 'search', '--source', 'windowsfeatures']
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] != 0:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    return result['stdout']

def install(name, version=None, source=None, force=False, pre_versions=False, install_args=None, override_args=False, force_x86=False, package_args=None, allow_multiple=False, execution_timeout=None):
    if False:
        print('Hello World!')
    "\n    Instructs Chocolatey to install a package.\n\n    Args:\n\n        name (str):\n            The name of the package to be installed. Only accepts a single\n            argument. Required.\n\n        version (str):\n            Install a specific version of the package. Defaults to latest\n            version. Default is ``None``.\n\n        source (str):\n            Chocolatey repository (directory, share or remote URL feed) the\n            package comes from. Defaults to the official Chocolatey feed.\n            Default is ``None``.\n\n            Alternate Sources:\n\n            - cygwin\n            - python\n            - ruby\n            - webpi\n            - windowsfeatures\n\n        force (bool):\n            Reinstall the current version of an existing package. Do not use\n            with ``allow_multiple``. Default is ``False``.\n\n        pre_versions (bool):\n            Include pre-release packages. Default is ``False``.\n\n        install_args (str):\n            A list of install arguments you want to pass to the installation\n            process, i.e. product key or feature list. Default is ``None``.\n\n        override_args (bool):\n            Set to true if you want to override the original install arguments\n            (for the native installer) in the package and use your own. When\n            this is set to ``False`` install_args will be appended to the end of\n            the default arguments. Default is ``None``.\n\n        force_x86 (bool):\n            Force x86 (32bit) installation on 64bit systems. Default is\n            ``False``.\n\n        package_args (str):\n            Arguments you want to pass to the package. Default is ``None``.\n\n        allow_multiple (bool):\n            Allow multiple versions of the package to be installed. Do not use\n            with ``force``. Does not work with all packages. Default is\n            ``False``.\n\n            .. versionadded:: 2017.7.0\n\n        execution_timeout (str):\n            Chocolatey execution timeout value you want to pass to the\n            installation process. Default is ``None``.\n\n            .. versionadded:: 2018.3.0\n\n    Returns:\n        str: The output of the ``chocolatey`` command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.install <package name>\n        salt '*' chocolatey.install <package name> version=<package version>\n        salt '*' chocolatey.install <package name> install_args=<args> override_args=True\n    "
    if force and allow_multiple:
        raise SaltInvocationError("Cannot use 'force' in conjunction with 'allow_multiple'")
    choc_path = _find_chocolatey()
    cmd = [choc_path, 'install', name]
    if version:
        cmd.extend(['--version', version])
    if source:
        cmd.extend(['--source', source])
    if salt.utils.data.is_true(force):
        cmd.append('--force')
    if salt.utils.data.is_true(pre_versions):
        cmd.append('--prerelease')
    if install_args:
        cmd.extend(['--installarguments', install_args])
    if override_args:
        cmd.append('--overridearguments')
    if force_x86:
        cmd.append('--forcex86')
    if package_args:
        cmd.extend(['--packageparameters', package_args])
    if allow_multiple:
        cmd.append('--allow-multiple')
    if execution_timeout:
        cmd.extend(['--execution-timeout', execution_timeout])
    cmd.extend(_no_progress())
    cmd.extend(_yes())
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] not in [0, 1641, 3010]:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    if name == 'chocolatey':
        _clear_context()
    return result['stdout']

def install_cygwin(name, install_args=None, override_args=False):
    if False:
        return 10
    "\n    Instructs Chocolatey to install a package via Cygwin.\n\n    Args:\n\n        name (str):\n            The name of the package to be installed. Only accepts a single\n            argument.\n\n        install_args (str):\n            A list of install arguments you want to pass to the installation\n            process, i.e. product key or feature list\n\n        override_args (bool):\n            Set to ``True`` if you want to override the original install\n            arguments (for the native installer) in the package and use your\n            own. When this is set to ``False`` install_args will be appended to\n            the end of the default arguments\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.install_cygwin <package name>\n        salt '*' chocolatey.install_cygwin <package name> install_args=<args> override_args=True\n    "
    return install(name, source='cygwin', install_args=install_args, override_args=override_args)

def install_gem(name, version=None, install_args=None, override_args=False):
    if False:
        print('Hello World!')
    "\n    Instructs Chocolatey to install a package via Ruby's Gems.\n\n    Args:\n\n        name (str):\n            The name of the package to be installed. Only accepts a single\n            argument.\n\n        version (str):\n            Install a specific version of the package. Defaults to the latest\n            version available.\n\n        install_args (str):\n            A list of install arguments you want to pass to the installation\n            process, i.e. product key or feature list\n\n        override_args (bool):\n            Set to ``True`` if you want to override the original install\n            arguments (for the native installer) in the package and use your\n            own. When this is set to ``False`` install_args will be appended to\n            the end of the default arguments\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.install_gem <package name>\n        salt '*' chocolatey.install_gem <package name> version=<package version>\n        salt '*' chocolatey.install_gem <package name> install_args=<args> override_args=True\n    "
    return install(name, version=version, source='ruby', install_args=install_args, override_args=override_args)

def install_missing(name, version=None, source=None):
    if False:
        print('Hello World!')
    "\n    Instructs Chocolatey to install a package if it doesn't already exist.\n\n    .. versionchanged:: 2014.7.0\n        If the minion has Chocolatey >= 0.9.8.24 installed, this function calls\n        :mod:`chocolatey.install <salt.modules.chocolatey.install>` instead, as\n        ``installmissing`` is deprecated as of that version and will be removed\n        in Chocolatey 1.0.\n\n    Args:\n\n        name (str):\n            The name of the package to be installed. Only accepts a single\n            argument.\n\n        version (str):\n            Install a specific version of the package. Defaults to the latest\n            version available.\n\n        source (str):\n            Chocolatey repository (directory, share or remote URL feed) the\n            package comes from. Defaults to the official Chocolatey feed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.install_missing <package name>\n        salt '*' chocolatey.install_missing <package name> version=<package version>\n    "
    if Version(chocolatey_version()) >= Version('0.9.8.24'):
        log.warning('installmissing is deprecated, using install')
        return install(name, version=version)
    cmd = [_find_chocolatey(), 'installmissing', name]
    if version:
        cmd.extend(['--version', version])
    if source:
        cmd.extend(['--source', source])
    cmd.extend(_yes())
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] != 0:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    return result['stdout']

def install_python(name, version=None, install_args=None, override_args=False):
    if False:
        while True:
            i = 10
    "\n    Instructs Chocolatey to install a package via Python's easy_install.\n\n    Args:\n\n        name (str):\n            The name of the package to be installed. Only accepts a single\n            argument.\n\n        version (str):\n            Install a specific version of the package. Defaults to the latest\n            version available.\n\n        install_args (str):\n            A list of install arguments you want to pass to the installation\n            process, i.e. product key or feature list.\n\n        override_args (bool):\n            Set to ``True`` if you want to override the original install\n            arguments (for the native installer) in the package and use your\n            own. When this is set to ``False`` install_args will be appended to\n            the end of the default arguments.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.install_python <package name>\n        salt '*' chocolatey.install_python <package name> version=<package version>\n        salt '*' chocolatey.install_python <package name> install_args=<args> override_args=True\n    "
    return install(name, version=version, source='python', install_args=install_args, override_args=override_args)

def install_windowsfeatures(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Instructs Chocolatey to install a Windows Feature via the Deployment Image\n    Servicing and Management tool.\n\n    Args:\n\n        name (str):\n            The name of the feature to be installed. Only accepts a single\n            argument.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.install_windowsfeatures <package name>\n    "
    return install(name, source='windowsfeatures')

def install_webpi(name, install_args=None, override_args=False):
    if False:
        i = 10
        return i + 15
    "\n    Instructs Chocolatey to install a package via the Microsoft Web PI service.\n\n    Args:\n\n        name (str):\n            The name of the package to be installed. Only accepts a single\n            argument.\n\n        install_args (str):\n            A list of install arguments you want to pass to the installation\n            process, i.e. product key or feature list.\n\n        override_args (bool):\n            Set to ``True`` if you want to override the original install\n            arguments (for the native installer) in the package and use your\n            own. When this is set to ``False`` install_args will be appended to\n            the end of the default arguments.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.install_webpi <package name>\n        salt '*' chocolatey.install_webpi <package name> install_args=<args> override_args=True\n    "
    return install(name, source='webpi', install_args=install_args, override_args=override_args)

def uninstall(name, version=None, uninstall_args=None, override_args=False, force=False):
    if False:
        return 10
    "\n    Instructs Chocolatey to uninstall a package.\n\n    Args:\n\n        name (str):\n            The name of the package to be uninstalled. Only accepts a single\n            argument.\n\n        version (str):\n            Uninstalls a specific version of the package. Defaults to the latest\n            version installed.\n\n        uninstall_args (str):\n            A list of uninstall arguments you want to pass to the uninstallation\n            process, i.e. product key or feature list.\n\n        override_args\n            Set to ``True`` if you want to override the original uninstall\n            arguments (for the native uninstaller) in the package and use your\n            own. When this is set to ``False`` uninstall_args will be appended\n            to the end of the default arguments.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.uninstall <package name>\n        salt '*' chocolatey.uninstall <package name> version=<package version>\n        salt '*' chocolatey.uninstall <package name> version=<package version> uninstall_args=<args> override_args=True\n    "
    cmd = [_find_chocolatey(), 'uninstall', name]
    if version:
        cmd.extend(['--version', version])
    if uninstall_args:
        cmd.extend(['--uninstallarguments', uninstall_args])
    if override_args:
        cmd.append('--overridearguments')
    if force:
        cmd.append('--force')
    cmd.extend(_yes())
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] not in [0, 1, 1605, 1614, 1641]:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    return result['stdout']

def upgrade(name, version=None, source=None, force=False, pre_versions=False, install_args=None, override_args=False, force_x86=False, package_args=None):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2016.3.4\n\n    Instructs Chocolatey to upgrade packages on the system. (update is being\n    deprecated). This command will install the package if not installed.\n\n    Args:\n\n        name (str):\n            The name of the package to update, or "all" to update everything\n            installed on the system.\n\n        version (str):\n            Install a specific version of the package. Defaults to latest\n            version.\n\n        source (str):\n            Chocolatey repository (directory, share or remote URL feed) the\n            package comes from. Defaults to the official Chocolatey feed.\n\n        force (bool):\n            Reinstall the **same** version already installed.\n\n        pre_versions (bool):\n            Include pre-release packages in comparison. Defaults to ``False``.\n\n        install_args (str):\n            A list of install arguments you want to pass to the installation\n            process, i.e. product key or feature list.\n\n        override_args (bool):\n            Set to ``True`` if you want to override the original install\n            arguments (for the native installer) in the package and use your\n            own. When this is set to ``False`` install_args will be appended to\n            the end of the default arguments.\n\n        force_x86 (bool):\n            Force x86 (32bit) installation on 64bit systems. Defaults to\n            ``False``.\n\n        package_args (str):\n            A list of arguments you want to pass to the package.\n\n    Returns:\n        str: Results of the ``chocolatey`` command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt "*" chocolatey.upgrade all\n        salt "*" chocolatey.upgrade <package name> pre_versions=True\n    '
    cmd = [_find_chocolatey(), 'upgrade', name]
    if version:
        cmd.extend(['--version', version])
    if source:
        cmd.extend(['--source', source])
    if salt.utils.data.is_true(force):
        cmd.append('--force')
    if salt.utils.data.is_true(pre_versions):
        cmd.append('--prerelease')
    if install_args:
        cmd.extend(['--installarguments', install_args])
    if override_args:
        cmd.append('--overridearguments')
    if force_x86:
        cmd.append('--forcex86')
    if package_args:
        cmd.extend(['--packageparameters', package_args])
    cmd.extend(_no_progress())
    cmd.extend(_yes())
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] not in [0, 1641, 3010]:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    return result['stdout']

def update(name, source=None, pre_versions=False):
    if False:
        while True:
            i = 10
    '\n    Instructs Chocolatey to update packages on the system.\n\n    Args:\n\n        name (str):\n            The name of the package to update, or "all" to update everything\n            installed on the system.\n\n        source (str):\n            Chocolatey repository (directory, share or remote URL feed) the\n            package comes from. Defaults to the official Chocolatey feed.\n\n        pre_versions (bool):\n            Include pre-release packages in comparison. Defaults to ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt "*" chocolatey.update all\n        salt "*" chocolatey.update <package name> pre_versions=True\n    '
    if Version(chocolatey_version()) >= Version('0.9.8.24'):
        log.warning('update is deprecated, using upgrade')
        return upgrade(name, source=source, pre_versions=pre_versions)
    cmd = [_find_chocolatey(), 'update', name]
    if source:
        cmd.extend(['--source', source])
    if salt.utils.data.is_true(pre_versions):
        cmd.append('--prerelease')
    cmd.extend(_no_progress())
    cmd.extend(_yes())
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] not in [0, 1641, 3010]:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    return result['stdout']

def version(name, check_remote=False, source=None, pre_versions=False):
    if False:
        return 10
    '\n    Instructs Chocolatey to check an installed package version, and optionally\n    compare it to one available from a remote feed.\n\n    Args:\n\n        name (str):\n            The name of the package to check. Required.\n\n        check_remote (bool):\n            Get the version number of the latest package from the remote feed.\n            Default is ``False``.\n\n        source (str):\n            Chocolatey repository (directory, share or remote URL feed) the\n            package comes from. Defaults to the official Chocolatey feed.\n            Default is ``None``.\n\n        pre_versions (bool):\n            Include pre-release packages in comparison. Default is ``False``.\n\n    Returns:\n        dict: A dictionary of currently installed software and versions\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt "*" chocolatey.version <package name>\n        salt "*" chocolatey.version <package name> check_remote=True\n    '
    installed = list_(narrow=name, local_only=True)
    packages = {}
    lower_name = name.lower()
    if installed:
        for pkg in installed:
            if lower_name == pkg.lower():
                packages.setdefault(pkg, {})
                packages[pkg]['installed'] = installed[pkg]
    if check_remote:
        available = list_(narrow=name, local_only=False, pre_versions=pre_versions, source=source)
        if available:
            for pkg in available:
                if lower_name == pkg.lower():
                    packages.setdefault(pkg, {})
                    packages[pkg]['available'] = available[pkg]
    return packages

def add_source(name, source_location, username=None, password=None, priority=None):
    if False:
        print('Hello World!')
    "\n    Instructs Chocolatey to add a source.\n\n    Args:\n\n        name (str):\n            The name of the source to be added as a chocolatey repository.\n\n        source (str):\n            Location of the source you want to work with.\n\n        username (str):\n            Provide username for chocolatey sources that need authentication\n            credentials.\n\n        password (str):\n            Provide password for chocolatey sources that need authentication\n            credentials.\n\n        priority (int):\n            The priority order of this source as compared to other sources,\n            lower is better. Defaults to 0 (no priority). All priorities\n            above 0 will be evaluated first, then zero-based values will be\n            evaluated in config file order.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.add_source <source name> <source_location>\n        salt '*' chocolatey.add_source <source name> <source_location> priority=100\n        salt '*' chocolatey.add_source <source name> <source_location> user=<user> password=<password>\n\n    "
    cmd = [_find_chocolatey(), 'sources', 'add', '--name', name, '--source', source_location]
    if username:
        cmd.extend(['--user', username])
    if password:
        cmd.extend(['--password', password])
    if priority:
        cmd.extend(['--priority', priority])
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] != 0:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    return result['stdout']

def _change_source_state(name, state):
    if False:
        return 10
    '\n    Instructs Chocolatey to change the state of a source.\n\n    Args:\n\n        name (str):\n            Name of the repository to affect.\n\n        state (str):\n            State in which you want the chocolatey repository.\n    '
    cmd = [_find_chocolatey(), 'source', state, '--name', name]
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] != 0:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    return result['stdout']

def enable_source(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Instructs Chocolatey to enable a source.\n\n    Args:\n\n        name (str):\n            Name of the source repository to enable.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.enable_source <name>\n\n    "
    return _change_source_state(name, 'enable')

def disable_source(name):
    if False:
        i = 10
        return i + 15
    "\n    Instructs Chocolatey to disable a source.\n\n    Args:\n\n        name (str):\n            Name of the source repository to disable.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.disable_source <name>\n    "
    return _change_source_state(name, 'disable')

def list_sources():
    if False:
        return 10
    "\n    Returns the list of installed sources.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' chocolatey.list_sources\n    "
    choc_path = _find_chocolatey()
    cmd = [choc_path, 'source']
    cmd.append('--limit-output')
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if result['retcode'] not in [0, 2]:
        err = 'Running chocolatey failed: {}'.format(result['stdout'])
        raise CommandExecutionError(err)
    ret = CaseInsensitiveDict({})
    pkg_re = re.compile('(.*)\\|(.*)\\|(.*)\\|(.*)\\|.*\\|.*\\|.*\\|.*\\|.*')
    for line in result['stdout'].split('\n'):
        for (name, url, disabled, user) in pkg_re.findall(line):
            if name not in ret:
                ret[name] = {'URL: ': url, 'Disabled': disabled, 'User: ': user}
    return ret