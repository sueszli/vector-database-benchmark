""" Python flavors specifics.

This abstracts the Python variants from different people. There is not just
CPython, but Anaconda, Debian, pyenv, Apple, lots of people who make Python
in a way the requires technical differences, e.g. static linking, LTO, or
DLL presence, link paths, etc.

"""
import os
import sys
from nuitka.utils.FileOperations import areSamePaths, isFilenameBelowPath, isFilenameSameAsOrBelowPath
from nuitka.utils.Utils import isAndroidBasedLinux, isFedoraBasedLinux, isLinux, isMacOS, isPosixWindows, isWin32Windows, withNoDeprecationWarning
from .PythonVersions import getInstalledPythonRegistryPaths, getRunningPythonDLLPath, getSystemPrefixPath, isStaticallyLinkedPython, python_version, python_version_str

def isNuitkaPython():
    if False:
        print('Hello World!')
    'Is this our own fork of CPython named Nuitka-Python.'
    if python_version >= 768:
        return sys.implementation.name == 'nuitkapython'
    else:
        return sys.subversion[0] == 'nuitkapython'
_is_anaconda = None

def isAnacondaPython():
    if False:
        for i in range(10):
            print('nop')
    'Detect if Python variant Anaconda'
    global _is_anaconda
    if _is_anaconda is None:
        _is_anaconda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
    return _is_anaconda

def isApplePython():
    if False:
        while True:
            i = 10
    if not isMacOS():
        return False
    if '+internal-os' in sys.version:
        return True
    if isFilenameSameAsOrBelowPath(path='/usr/bin/', filename=getSystemPrefixPath()):
        return True
    if isFilenameSameAsOrBelowPath(path='/Library/Developer/CommandLineTools/', filename=getSystemPrefixPath()):
        return True
    if isFilenameSameAsOrBelowPath(path='/Applications/Xcode.app/Contents/Developer/', filename=getSystemPrefixPath()):
        return True
    return False

def isHomebrewPython():
    if False:
        print('Hello World!')
    if not isMacOS():
        return False
    candidate = os.path.join(getSystemPrefixPath(), 'lib', 'python' + python_version_str, 'sitecustomize.py')
    if os.path.exists(candidate):
        with open(candidate, 'rb') as site_file:
            line = site_file.readline()
        if b'Homebrew' in line:
            return True
    return False

def isPyenvPython():
    if False:
        return 10
    if isWin32Windows():
        return False
    return os.environ.get('PYENV_ROOT') and isFilenameSameAsOrBelowPath(path=os.environ['PYENV_ROOT'], filename=getSystemPrefixPath())

def isMSYS2MingwPython():
    if False:
        for i in range(10):
            print('nop')
    'MSYS2 the MinGW64 variant that is more Win32 compatible.'
    if not isWin32Windows() or 'GCC' not in sys.version:
        return False
    import sysconfig
    if python_version >= 944:
        return '-mingw_' in sysconfig.get_config_var('EXT_SUFFIX')
    else:
        return '-mingw_' in sysconfig.get_config_var('SO')

def isTermuxPython():
    if False:
        return 10
    'Is this Termux Android Python.'
    if not isAndroidBasedLinux():
        return False
    return 'com.termux' in getSystemPrefixPath().split('/')

def isUninstalledPython():
    if False:
        for i in range(10):
            print('nop')
    if isDebianPackagePython():
        return False
    if isStaticallyLinkedPython():
        return False
    if os.name == 'nt':
        import ctypes.wintypes
        GetSystemDirectory = ctypes.windll.kernel32.GetSystemDirectoryW
        GetSystemDirectory.argtypes = (ctypes.wintypes.LPWSTR, ctypes.wintypes.DWORD)
        GetSystemDirectory.restype = ctypes.wintypes.DWORD
        MAX_PATH = 4096
        buf = ctypes.create_unicode_buffer(MAX_PATH)
        res = GetSystemDirectory(buf, MAX_PATH)
        assert res != 0
        system_path = os.path.normcase(buf.value)
        return not getRunningPythonDLLPath().startswith(system_path)
    return isAnacondaPython() or 'WinPython' in sys.version
_is_win_python = None

def isWinPython():
    if False:
        i = 10
        return i + 15
    'Is this Python from WinPython.'
    if 'WinPython' in sys.version:
        return True
    global _is_win_python
    if _is_win_python is None:
        for element in sys.path:
            if os.path.basename(element) == 'site-packages':
                if os.path.exists(os.path.join(element, 'winpython')):
                    _is_win_python = True
                    break
        else:
            _is_win_python = False
    return _is_win_python

def isDebianPackagePython():
    if False:
        return 10
    'Is this Python from a debian package.'
    if not isLinux():
        return False
    if python_version < 768:
        return hasattr(sys, '_multiarch')
    else:
        with withNoDeprecationWarning():
            try:
                from distutils.dir_util import _multiarch
            except ImportError:
                return False
            else:
                return True

def isFedoraPackagePython():
    if False:
        for i in range(10):
            print('nop')
    'Is the Python from a Fedora package.'
    if not isFedoraBasedLinux():
        return False
    system_prefix_path = getSystemPrefixPath()
    return system_prefix_path == '/usr'

def isCPythonOfficialPackage():
    if False:
        i = 10
        return i + 15
    "Official CPython download, kind of hard to detect since self-compiled doesn't change much."
    sys_prefix = getSystemPrefixPath()
    if isMacOS() and isFilenameBelowPath(path='/Library/Frameworks/Python.framework/Versions/', filename=sys_prefix):
        return True
    if isWin32Windows():
        for registry_python_exe in getInstalledPythonRegistryPaths(python_version_str):
            if areSamePaths(sys_prefix, os.path.dirname(registry_python_exe)):
                return True
    return False

def isGithubActionsPython():
    if False:
        while True:
            i = 10
    return os.environ.get('GITHUB_ACTIONS', '') == 'true' and getSystemPrefixPath().startswith('/opt/hostedtoolcache/Python')

def getPythonFlavorName():
    if False:
        print('Hello World!')
    'For output to the user only.'
    if isNuitkaPython():
        return 'Nuitka Python'
    elif isAnacondaPython():
        return 'Anaconda Python'
    elif isWinPython():
        return 'WinPython'
    elif isDebianPackagePython():
        return 'Debian Python'
    elif isFedoraPackagePython():
        return 'Fedora Python'
    elif isHomebrewPython():
        return 'Homebrew Python'
    elif isApplePython():
        return 'Apple Python'
    elif isPyenvPython():
        return 'pyenv'
    elif isPosixWindows():
        return 'MSYS2 Posix'
    elif isMSYS2MingwPython():
        return 'MSYS2 MinGW'
    elif isTermuxPython():
        return 'Android Termux'
    elif isCPythonOfficialPackage():
        return 'CPython Official'
    elif isGithubActionsPython():
        return 'GitHub Actions Python'
    else:
        return 'Unknown'