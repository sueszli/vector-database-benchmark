"""
Installer support for macOS.

Installer is the native .pkg/.mpkg package manager for macOS.
"""
import os.path
import urllib
import salt.utils.itertools
import salt.utils.mac_utils
import salt.utils.path
import salt.utils.platform
from salt.exceptions import SaltInvocationError
__func_alias__ = {'list_': 'list'}
__virtualname__ = 'pkgutil'

def __virtual__():
    if False:
        while True:
            i = 10
    if not salt.utils.platform.is_darwin():
        return (False, 'Only available on Mac OS systems')
    if not salt.utils.path.which('pkgutil'):
        return (False, 'Missing pkgutil binary')
    return __virtualname__

def list_():
    if False:
        return 10
    "\n    List the installed packages.\n\n    :return: A list of installed packages\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkgutil.list\n    "
    cmd = 'pkgutil --pkgs'
    ret = salt.utils.mac_utils.execute_return_result(cmd)
    return ret.splitlines()

def is_installed(package_id):
    if False:
        while True:
            i = 10
    "\n    Returns whether a given package id is installed.\n\n    :return: True if installed, otherwise False\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkgutil.is_installed com.apple.pkg.gcc4.2Leo\n    "
    return package_id in list_()

def _install_from_path(path):
    if False:
        return 10
    '\n    Internal function to install a package from the given path\n    '
    if not os.path.exists(path):
        msg = 'File not found: {}'.format(path)
        raise SaltInvocationError(msg)
    cmd = 'installer -pkg "{}" -target /'.format(path)
    return salt.utils.mac_utils.execute_return_success(cmd)

def install(source, package_id):
    if False:
        print('Hello World!')
    "\n    Install a .pkg from an URI or an absolute path.\n\n    :param str source: The path to a package.\n\n    :param str package_id: The package ID\n\n    :return: True if successful, otherwise False\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkgutil.install source=/vagrant/build_essentials.pkg package_id=com.apple.pkg.gcc4.2Leo\n    "
    if is_installed(package_id):
        return True
    uri = urllib.parse.urlparse(source)
    if not uri.scheme == '':
        msg = 'Unsupported scheme for source uri: {}'.format(uri.scheme)
        raise SaltInvocationError(msg)
    _install_from_path(source)
    return is_installed(package_id)

def forget(package_id):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2016.3.0\n\n    Remove the receipt data about the specified package. Does not remove files.\n\n    .. warning::\n        DO NOT use this command to fix broken package design\n\n    :param str package_id: The name of the package to forget\n\n    :return: True if successful, otherwise False\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkgutil.forget com.apple.pkg.gcc4.2Leo\n    "
    cmd = 'pkgutil --forget {}'.format(package_id)
    salt.utils.mac_utils.execute_return_success(cmd)
    return not is_installed(package_id)