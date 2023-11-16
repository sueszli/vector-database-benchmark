""" Interface to depends.exe on Windows.

We use depends.exe to investigate needed DLLs of Python DLLs.

"""
import os
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.Options import assumeYesForDownloads
from nuitka.Tracing import inclusion_logger
from nuitka.utils.Download import getCachedDownload
from nuitka.utils.Execution import executeProcess, withEnvironmentVarOverridden
from nuitka.utils.FileOperations import deleteFile, getExternalUsePath, getFileContentByLine, getWindowsLongPathName, isFilenameBelowPath, putTextFileContents, withFileLock
from nuitka.utils.SharedLibraries import getWindowsRunningProcessDLLPaths
from nuitka.utils.Utils import getArchitecture

def getDependsExePath():
    if False:
        for i in range(10):
            print('nop')
    'Return the path of depends.exe (for Windows).\n\n    Will prompt the user to download if not already cached in AppData\n    directory for Nuitka.\n    '
    if getArchitecture() == 'x86':
        depends_url = 'https://dependencywalker.com/depends22_x86.zip'
    else:
        depends_url = 'https://dependencywalker.com/depends22_x64.zip'
    return getCachedDownload(name='dependency walker', url=depends_url, is_arch_specific=getArchitecture(), binary='depends.exe', flatten=True, specificity='', message='Nuitka will make use of Dependency Walker (https://dependencywalker.com) tool\nto analyze the dependencies of Python extension modules.', reject='Nuitka does not work in --standalone or --onefile on Windows without.', assume_yes_for_downloads=assumeYesForDownloads())

def _attemptToFindNotFoundDLL(dll_filename):
    if False:
        print('Hello World!')
    'Some heuristics and tricks to find DLLs that dependency walker did not find.'
    currently_loaded_dlls = getWindowsRunningProcessDLLPaths()
    if dll_filename in currently_loaded_dlls:
        return currently_loaded_dlls[dll_filename]
    dll_filename = os.path.join(os.environ['SYSTEMROOT'], 'SysWOW64' if getArchitecture() == 'x86_64' else 'System32', dll_filename)
    dll_filename = os.path.normcase(dll_filename)
    if os.path.exists(dll_filename):
        return dll_filename
    return None

def _parseDependsExeOutput2(lines):
    if False:
        i = 10
        return i + 15
    result = OrderedSet()
    inside = False
    first = False
    for line in lines:
        if '| Module Dependency Tree |' in line:
            inside = True
            first = True
            continue
        if not inside:
            continue
        if '| Module List |' in line:
            break
        if ']' not in line:
            continue
        dll_filename = line[line.find(']') + 2:].rstrip()
        dll_filename = os.path.normcase(dll_filename)
        if isFilenameBelowPath(path=os.path.join(os.environ['SYSTEMROOT'], 'WinSxS'), filename=dll_filename):
            continue
        if 'E' in line[:line.find(']')]:
            continue
        if '?' in line[:line.find(']')]:
            if dll_filename.startswith('python') and dll_filename.endswith('.dll'):
                dll_filename = _attemptToFindNotFoundDLL(dll_filename)
                if dll_filename is None:
                    continue
            else:
                continue
        assert os.path.basename(dll_filename) != 'kernel32.dll'
        if first:
            first = False
            continue
        dll_filename = os.path.abspath(dll_filename)
        dll_filename = getWindowsLongPathName(dll_filename)
        dll_name = os.path.basename(dll_filename)
        if dll_name in ('msvcr90.dll',):
            continue
        if dll_name.startswith('api-ms-win-'):
            continue
        if dll_name == 'ucrtbase.dll':
            continue
        assert os.path.isfile(dll_filename), (dll_filename, line)
        result.add(os.path.normcase(os.path.abspath(dll_filename)))
    return result

def parseDependsExeOutput(filename):
    if False:
        return 10
    return _parseDependsExeOutput2(getFileContentByLine(filename, encoding='latin1'))

def detectDLLsWithDependencyWalker(binary_filename, source_dir, scan_dirs):
    if False:
        return 10
    dwp_filename = os.path.join(source_dir, os.path.basename(binary_filename) + '.dwp')
    output_filename = os.path.join(source_dir, os.path.basename(binary_filename) + '.depends')
    with withFileLock('Finding out dependency walker path and creating DWP file for %s' % binary_filename):
        depends_exe = getDependsExePath()
        putTextFileContents(dwp_filename, contents='SxS\n%(scan_dirs)s\n' % {'scan_dirs': '\n'.join(('UserDir %s' % getExternalUsePath(dirname) for dirname in scan_dirs))})
    with withEnvironmentVarOverridden('PATH', ''):
        (_stdout, _stderr, _exit_code) = executeProcess(command=(depends_exe, '-c', '-ot%s' % output_filename, '-d:%s' % dwp_filename, '-f1', '-pa1', '-ps1', binary_filename), external_cwd=True)
    if not os.path.exists(output_filename):
        inclusion_logger.sysexit("Error, 'depends.exe' failed to produce expected output.")
    result = parseDependsExeOutput(output_filename)
    deleteFile(output_filename, must_exist=True)
    deleteFile(dwp_filename, must_exist=True)
    return result