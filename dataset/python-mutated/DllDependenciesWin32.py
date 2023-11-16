"""DLL dependency scan methods for Win32 Windows.

Note: MSYS2, aka POSIX Windows is dealt with in the "DllDependenciesPosix" module.
"""
import os
import sys
from nuitka.__past__ import iterItems
from nuitka.build.SconsUtils import readSconsReport
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.Options import isShowProgress
from nuitka.plugins.Plugins import Plugins
from nuitka.Tracing import inclusion_logger
from nuitka.utils.AppDirs import getCacheDir
from nuitka.utils.FileOperations import areSamePaths, getDirectoryRealPath, getFileContentByLine, getSubDirectoriesWithDlls, listDllFilesFromDirectory, makePath, putTextFileContents, withFileLock
from nuitka.utils.Hashing import Hash
from nuitka.utils.SharedLibraries import getPyWin32Dir
from nuitka.Version import version_string
from .DependsExe import detectDLLsWithDependencyWalker
from .DllDependenciesCommon import getPackageSpecificDLLDirectories
_scan_dir_cache = {}

def detectBinaryPathDLLsWin32(is_main_executable, source_dir, original_dir, binary_filename, package_name, use_cache, update_cache):
    if False:
        print('Hello World!')
    if use_cache or update_cache:
        cache_filename = _getCacheFilename(dependency_tool='depends.exe', is_main_executable=is_main_executable, source_dir=source_dir, original_dir=original_dir, binary_filename=binary_filename, package_name=package_name)
        if use_cache:
            with withFileLock():
                if not os.path.exists(cache_filename):
                    use_cache = False
        if use_cache:
            result = OrderedSet()
            for line in getFileContentByLine(cache_filename):
                line = line.strip()
                if not os.path.exists(line):
                    break
                result.add(line)
            else:
                return result
    if isShowProgress():
        inclusion_logger.info("Analyzing dependencies of '%s'." % binary_filename)
    scan_dirs = _getScanDirectories(package_name, original_dir)
    result = detectDLLsWithDependencyWalker(binary_filename=binary_filename, source_dir=source_dir, scan_dirs=scan_dirs)
    if update_cache:
        putTextFileContents(filename=cache_filename, contents=result)
    return result

def _getScanDirectories(package_name, original_dir):
    if False:
        while True:
            i = 10
    cache_key = (package_name, original_dir)
    if cache_key in _scan_dir_cache:
        return _scan_dir_cache[cache_key]
    scan_dirs = [sys.prefix]
    if package_name is not None:
        scan_dirs.extend(getPackageSpecificDLLDirectories(package_name))
    if original_dir is not None:
        scan_dirs.append(original_dir)
        scan_dirs.extend(getSubDirectoriesWithDlls(original_dir))
    if package_name is not None and package_name.isBelowNamespace('win32com'):
        py_win32_dir = getPyWin32Dir()
        if py_win32_dir is not None:
            scan_dirs.append(py_win32_dir)
    for path_dir in os.environ['PATH'].split(';'):
        if not os.path.isdir(path_dir):
            continue
        if areSamePaths(path_dir, os.path.join(os.environ['SYSTEMROOT'])):
            continue
        if areSamePaths(path_dir, os.path.join(os.environ['SYSTEMROOT'], 'System32')):
            continue
        if areSamePaths(path_dir, os.path.join(os.environ['SYSTEMROOT'], 'SysWOW64')):
            continue
        scan_dirs.append(path_dir)
    result = []
    for scan_dir in scan_dirs:
        scan_dir = getDirectoryRealPath(scan_dir)
        try:
            if not os.path.isdir(scan_dir) or not any(listDllFilesFromDirectory(scan_dir)):
                continue
        except OSError:
            continue
        result.append(os.path.realpath(scan_dir))
    _scan_dir_cache[cache_key] = result
    return result

def _getCacheFilename(dependency_tool, is_main_executable, source_dir, original_dir, binary_filename, package_name):
    if False:
        for i in range(10):
            print('nop')
    original_filename = os.path.join(original_dir, os.path.basename(binary_filename))
    original_filename = os.path.normcase(original_filename)
    hash_value = Hash()
    if is_main_executable:
        hash_value.updateFromValues(''.join((key + value for (key, value) in iterItems(readSconsReport(source_dir=source_dir)) if key not in ('CLCACHE_STATS', 'CCACHE_LOGFILE', 'CCACHE_DIR'))))
    else:
        hash_value.updateFromValues(original_filename)
        hash_value.updateFromFile(filename=original_filename)
    hash_value.updateFromValues(sys.version, sys.executable)
    hash_value.updateFromValues(*Plugins.getCacheContributionValues(package_name))
    hash_value.updateFromValues(version_string)
    cache_dir = os.path.join(getCacheDir(), 'library_dependencies', dependency_tool)
    makePath(cache_dir)
    return os.path.join(cache_dir, hash_value.asHexDigest())