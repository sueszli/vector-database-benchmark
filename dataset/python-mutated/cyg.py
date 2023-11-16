"""
Manage cygwin packages.

Module file to accompany the cyg state.
"""
import bz2
import logging
import os
import re
import urllib.request
import salt.utils.files
import salt.utils.platform
import salt.utils.stringutils
from salt.exceptions import SaltInvocationError
LOG = logging.getLogger(__name__)
DEFAULT_MIRROR = 'ftp://mirrors.kernel.org/sourceware/cygwin/'
DEFAULT_MIRROR_KEY = ''
__virtualname__ = 'cyg'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only works on Windows systems\n    '
    if salt.utils.platform.is_windows():
        return __virtualname__
    return (False, 'Module cyg: module only works on Windows systems.')
__func_alias__ = {'list_': 'list'}

def _get_cyg_dir(cyg_arch='x86_64'):
    if False:
        print('Hello World!')
    '\n    Return the cygwin install directory based on the architecture.\n    '
    if cyg_arch == 'x86_64':
        return 'cygwin64'
    elif cyg_arch == 'x86':
        return 'cygwin'
    raise SaltInvocationError('Invalid architecture {arch}'.format(arch=cyg_arch))

def _check_cygwin_installed(cyg_arch='x86_64'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return True or False if cygwin is installed.\n\n    Use the cygcheck executable to check install. It is installed as part of\n    the base package, and we use it to check packages\n    '
    path_to_cygcheck = os.sep.join(['C:', _get_cyg_dir(cyg_arch), 'bin', 'cygcheck.exe'])
    LOG.debug('Path to cygcheck.exe: %s', path_to_cygcheck)
    if not os.path.exists(path_to_cygcheck):
        LOG.debug('Could not find cygcheck.exe')
        return False
    return True

def _get_all_packages(mirror=DEFAULT_MIRROR, cyg_arch='x86_64'):
    if False:
        i = 10
        return i + 15
    '\n    Return the list of packages based on the mirror provided.\n    '
    if 'cyg.all_packages' not in __context__:
        __context__['cyg.all_packages'] = {}
    if mirror not in __context__['cyg.all_packages']:
        __context__['cyg.all_packages'][mirror] = []
    if not __context__['cyg.all_packages'][mirror]:
        pkg_source = '/'.join([mirror, cyg_arch, 'setup.bz2'])
        file_data = urllib.request.urlopen(pkg_source).read()
        file_lines = bz2.decompress(file_data).decode('utf_8', errors='replace').splitlines()
        packages = [re.search('^@ ([^ ]+)', line).group(1) for line in file_lines if re.match('^@ [^ ]+', line)]
        __context__['cyg.all_packages'][mirror] = packages
    return __context__['cyg.all_packages'][mirror]

def check_valid_package(package, cyg_arch='x86_64', mirrors=None):
    if False:
        print('Hello World!')
    "\n    Check if the package is valid on the given mirrors.\n\n    Args:\n        package: The name of the package\n        cyg_arch: The cygwin architecture\n        mirrors: any mirrors to check\n\n    Returns (bool): True if Valid, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cyg.check_valid_package <package name>\n    "
    if mirrors is None:
        mirrors = [{DEFAULT_MIRROR: DEFAULT_MIRROR_KEY}]
    LOG.debug('Checking Valid Mirrors: %s', mirrors)
    for mirror in mirrors:
        for (mirror_url, key) in mirror.items():
            if package in _get_all_packages(mirror_url, cyg_arch):
                return True
    return False

def _run_silent_cygwin(cyg_arch='x86_64', args=None, mirrors=None):
    if False:
        print('Hello World!')
    '\n    Retrieve the correct setup.exe.\n\n    Run it with the correct arguments to get the bare minimum cygwin\n    installation up and running.\n    '
    cyg_cache_dir = os.sep.join(['c:', 'cygcache'])
    cyg_setup = 'setup-{}.exe'.format(cyg_arch)
    cyg_setup_path = os.sep.join([cyg_cache_dir, cyg_setup])
    cyg_setup_source = 'http://cygwin.com/{}'.format(cyg_setup)
    if not os.path.exists(cyg_cache_dir):
        os.mkdir(cyg_cache_dir)
    elif os.path.exists(cyg_setup_path):
        os.remove(cyg_setup_path)
    file_data = urllib.request.urlopen(cyg_setup_source)
    with salt.utils.files.fopen(cyg_setup_path, 'wb') as fhw:
        fhw.write(file_data.read())
    setup_command = cyg_setup_path
    options = []
    options.append('--local-package-dir {}'.format(cyg_cache_dir))
    if mirrors is None:
        mirrors = [{DEFAULT_MIRROR: DEFAULT_MIRROR_KEY}]
    for mirror in mirrors:
        for (mirror_url, key) in mirror.items():
            options.append('--site {}'.format(mirror_url))
            if key:
                options.append('--pubkey {}'.format(key))
    options.append('--no-desktop')
    options.append('--quiet-mode')
    options.append('--disable-buggy-antivirus')
    if args is not None:
        for arg in args:
            options.append(arg)
    cmdline_args = ' '.join(options)
    setup_command = ' '.join([cyg_setup_path, cmdline_args])
    ret = __salt__['cmd.run_all'](setup_command)
    if ret['retcode'] == 0:
        return ret['stdout']
    else:
        return False

def _cygcheck(args, cyg_arch='x86_64'):
    if False:
        while True:
            i = 10
    '\n    Run the cygcheck executable.\n    '
    cmd = ' '.join([os.sep.join(['c:', _get_cyg_dir(cyg_arch), 'bin', 'cygcheck']), '-c', args])
    ret = __salt__['cmd.run_all'](cmd)
    if ret['retcode'] == 0:
        return ret['stdout']
    else:
        return False

def install(packages=None, cyg_arch='x86_64', mirrors=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Install one or several packages.\n\n    packages : None\n        The packages to install\n\n    cyg_arch : x86_64\n        Specify the architecture to install the package under\n        Current options are x86 and x86_64\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' cyg.install dos2unix\n        salt \'*\' cyg.install dos2unix mirrors="[{\'http://mirror\': \'http://url/to/public/key}]\'\n    '
    args = []
    if packages is not None:
        args.append('--packages {pkgs}'.format(pkgs=packages))
        if not _check_cygwin_installed(cyg_arch):
            _run_silent_cygwin(cyg_arch=cyg_arch)
    return _run_silent_cygwin(cyg_arch=cyg_arch, args=args, mirrors=mirrors)

def uninstall(packages, cyg_arch='x86_64', mirrors=None):
    if False:
        i = 10
        return i + 15
    '\n    Uninstall one or several packages.\n\n    packages\n        The packages to uninstall.\n\n    cyg_arch : x86_64\n        Specify the architecture to remove the package from\n        Current options are x86 and x86_64\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' cyg.uninstall dos2unix\n        salt \'*\' cyg.uninstall dos2unix mirrors="[{\'http://mirror\': \'http://url/to/public/key}]"\n    '
    args = []
    if packages is not None:
        args.append('--remove-packages {pkgs}'.format(pkgs=packages))
        LOG.debug('args: %s', args)
        if not _check_cygwin_installed(cyg_arch):
            LOG.debug("We're convinced cygwin isn't installed")
            return True
    return _run_silent_cygwin(cyg_arch=cyg_arch, args=args, mirrors=mirrors)

def update(cyg_arch='x86_64', mirrors=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Update all packages.\n\n    cyg_arch : x86_64\n        Specify the cygwin architecture update\n        Current options are x86 and x86_64\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' cyg.update\n        salt \'*\' cyg.update dos2unix mirrors="[{\'http://mirror\': \'http://url/to/public/key}]"\n    '
    args = []
    args.append('--upgrade-also')
    if not _check_cygwin_installed(cyg_arch):
        LOG.debug('Cygwin (%s) not installed, could not update', cyg_arch)
        return False
    return _run_silent_cygwin(cyg_arch=cyg_arch, args=args, mirrors=mirrors)

def list_(package='', cyg_arch='x86_64'):
    if False:
        for i in range(10):
            print('nop')
    "\n    List locally installed packages.\n\n    package : ''\n        package name to check. else all\n\n    cyg_arch :\n        Cygwin architecture to use\n        Options are x86 and x86_64\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' cyg.list\n    "
    pkgs = {}
    args = ' '.join(['-c', '-d', package])
    stdout = _cygcheck(args, cyg_arch=cyg_arch)
    lines = []
    if isinstance(stdout, str):
        lines = salt.utils.stringutils.to_unicode(stdout).splitlines()
    for line in lines:
        match = re.match('^([^ ]+) *([^ ]+)', line)
        if match:
            pkg = match.group(1)
            version = match.group(2)
            pkgs[pkg] = version
    return pkgs