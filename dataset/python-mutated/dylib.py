"""
Manipulating with dynamic libraries.
"""
import os.path
from PyInstaller.utils.win32 import winutils
__all__ = ['exclude_list', 'include_list', 'include_library']
import os
import re
import PyInstaller.log as logging
from PyInstaller import compat
logger = logging.getLogger(__name__)
_excludes = {'advapi32\\.dll', 'ws2_32\\.dll', 'gdi32\\.dll', 'oleaut32\\.dll', 'shell32\\.dll', 'ole32\\.dll', 'coredll\\.dll', 'crypt32\\.dll', 'kernel32', 'kernel32\\.dll', 'msvcrt\\.dll', 'rpcrt4\\.dll', 'user32\\.dll', 'python\\%s\\%s'}
_includes = set()
_win_includes = {'atl100\\.dll', 'msvcr100\\.dll', 'msvcp100\\.dll', 'mfc100\\.dll', 'mfc100u\\.dll', 'mfcmifc80\\.dll', 'mfcm100\\.dll', 'mfcm100u\\.dll', 'atl110\\.dll', 'msvcp110\\.dll', 'msvcr110\\.dll', 'vccorlib110\\.dll', 'vcamp110\\.dll', 'mfc110\\.dll', 'mfc110u\\.dll', 'mfcm110\\.dll', 'mfcm110u\\.dll', 'mfc110chs\\.dll', 'mfc110cht\\.dll', 'mfc110enu\\.dll', 'mfc110esn\\.dll', 'mfc110deu\\.dll', 'mfc110fra\\.dll', 'mfc110ita\\.dll', 'mfc110jpn\\.dll', 'mfc110kor\\.dll', 'mfc110rus\\.dll', 'vcomp110\\.dll', 'msdia110\\.dll', 'msvcp120\\.dll', 'msvcr120\\.dll', 'vccorlib120\\.dll', 'vcamp120\\.dll', 'mfc120\\.dll', 'mfc120u\\.dll', 'mfcm120\\.dll', 'mfcm120u\\.dll', 'mfc120chs\\.dll', 'mfc120cht\\.dll', 'mfc120deu\\.dll', 'mfc120enu\\.dll', 'mfc120esn\\.dll', 'mfc120fra\\.dll', 'mfc120ita\\.dll', 'mfc120jpn\\.dll', 'mfc120kor\\.dll', 'mfc120rus\\.dll', 'vcomp120\\.dll', 'msdia120\\.dll', 'casablanca120.winrt\\.dll', 'zumosdk120.winrt\\.dll', 'casablanca120\\.dll', 'api-ms-win-core.*', 'api-ms-win-crt.*', 'ucrtbase\\.dll', 'concrt140\\.dll', 'msvcp140\\.dll', 'msvcp140_1\\.dll', 'msvcp140_2\\.dll', 'msvcp140_atomic_wait\\.dll', 'msvcp140_codecvt_ids\\.dll', 'vccorlib140\\.dll', 'vcruntime140\\.dll', 'vcruntime140_1\\.dll', 'vcamp140\\.dll', 'vcomp140\\.dll', 'msdia140\\.dll', 'py(?:thon(?:com(?:loader)?)?|wintypes)\\d+\\.dll'}
_win_excludes = {'.*\\.so', '.*\\.dylib', 'Microsoft\\.Windows\\.Common-Controls'}
_unix_excludes = {'libc\\.so(\\..*)?', 'libdl\\.so(\\..*)?', 'libm\\.so(\\..*)?', 'libpthread\\.so(\\..*)?', 'librt\\.so(\\..*)?', 'libthread_db\\.so(\\..*)?', 'ld-linux\\.so(\\..*)?', 'libBrokenLocale\\.so(\\..*)?', 'libanl\\.so(\\..*)?', 'libcidn\\.so(\\..*)?', 'libcrypt\\.so(\\..*)?', 'libnsl\\.so(\\..*)?', 'libnss_compat.*\\.so(\\..*)?', 'libnss_dns.*\\.so(\\..*)?', 'libnss_files.*\\.so(\\..*)?', 'libnss_hesiod.*\\.so(\\..*)?', 'libnss_nis.*\\.so(\\..*)?', 'libnss_nisplus.*\\.so(\\..*)?', 'libresolv\\.so(\\..*)?', 'libutil\\.so(\\..*)?', 'libE?(Open)?GLX?(ESv1_CM|ESv2)?(dispatch)?\\.so(\\..*)?', 'libdrm\\.so(\\..*)?', 'nvidia_drv\\.so', 'libglxserver_nvidia\\.so(\\..*)?', 'libnvidia-egl-(gbm|wayland)\\.so(\\..*)?', 'libnvidia-(cfg|compiler|e?glcore|glsi|glvkspirv|rtcore|allocator|tls|ml)\\.so(\\..*)?', 'lib(EGL|GLX)_nvidia\\.so(\\..*)?', 'libxcb\\.so(\\..*)?', 'libxcb-dri.*\\.so(\\..*)?'}
_aix_excludes = {'libbz2\\.a', 'libc\\.a', 'libC\\.a', 'libcrypt\\.a', 'libdl\\.a', 'libintl\\.a', 'libpthreads\\.a', 'librt\\\\.a', 'librtl\\.a', 'libz\\.a'}
if compat.is_win:
    _includes |= _win_includes
    _excludes |= _win_excludes
elif compat.is_aix:
    _excludes |= _aix_excludes
elif compat.is_unix:
    _excludes |= _unix_excludes

class ExcludeList:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.regex = re.compile('|'.join(_excludes), re.I)

    def search(self, libname):
        if False:
            print('Hello World!')
        if _excludes:
            return self.regex.match(os.path.basename(libname))
        else:
            return False

class IncludeList:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.regex = re.compile('|'.join(_includes), re.I)

    def search(self, libname):
        if False:
            return 10
        if _includes:
            return self.regex.match(os.path.basename(libname))
        else:
            return False
exclude_list = ExcludeList()
include_list = IncludeList()
if compat.is_darwin:
    from macholib import util

    class MacExcludeList:

        def __init__(self, global_exclude_list):
            if False:
                i = 10
                return i + 15
            self._exclude_list = global_exclude_list

        def search(self, libname):
            if False:
                print('Hello World!')
            result = self._exclude_list.search(libname)
            if result:
                return result
            else:
                return util.in_system_path(libname)
    exclude_list = MacExcludeList(exclude_list)
elif compat.is_win:

    class WinExcludeList:

        def __init__(self, global_exclude_list):
            if False:
                print('Hello World!')
            self._exclude_list = global_exclude_list
            self._windows_dir = os.path.normpath(winutils.get_windows_dir().lower())

        def search(self, libname):
            if False:
                print('Hello World!')
            libname = libname.lower()
            result = self._exclude_list.search(libname)
            if result:
                return result
            else:
                fn = os.path.normpath(os.path.realpath(libname).lower())
                return fn.startswith(self._windows_dir)
    exclude_list = WinExcludeList(exclude_list)
_seen_wine_dlls = set()

def include_library(libname):
    if False:
        while True:
            i = 10
    '\n    Check if the dynamic library should be included with application or not.\n    '
    if exclude_list:
        if exclude_list.search(libname) and (not include_list.search(libname)):
            return False
    if compat.is_win_wine and compat.is_wine_dll(libname):
        if libname not in _seen_wine_dlls:
            logger.warning('Excluding Wine built-in DLL: %s', libname)
            _seen_wine_dlls.add(libname)
        return False
    return True
_warning_suppressions = []
if compat.is_linux:
    _warning_suppressions.append('ldd')
if compat.is_win_10:
    _warning_suppressions.append('api-ms-win-.*\\.dll')

class MissingLibWarningSuppressionList:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.regex = re.compile('|'.join(_warning_suppressions), re.I)

    def search(self, libname):
        if False:
            for i in range(10):
                print('nop')
        if _warning_suppressions:
            return self.regex.match(os.path.basename(libname))
        else:
            return False
missing_lib_warning_suppression_list = MissingLibWarningSuppressionList()

def warn_missing_lib(libname):
    if False:
        print('Hello World!')
    '\n    Check if a missing-library warning should be displayed for the given library name (or full path).\n    '
    return not missing_lib_warning_suppression_list.search(libname)