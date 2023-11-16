""" Included entry points for standalone mode.

This keeps track of entry points for standalone. These should be extension
modules, added by core code, the main binary, added by core code, and from
plugins in their getExtraDlls implementation, where they provide DLLs to be
added, and whose dependencies will also be included.
"""
import collections
import fnmatch
import os
from nuitka.Options import getShallNotIncludeDllFilePatterns, isShowInclusion
from nuitka.Tracing import general, inclusion_logger
from nuitka.utils.FileOperations import areSamePaths, getReportPath, hasFilenameExtension, haveSameFileContents, isRelativePath
from nuitka.utils.Importing import getSharedLibrarySuffix
from nuitka.utils.ModuleNames import ModuleName, checkModuleName
from nuitka.utils.SharedLibraries import getDLLVersion
IncludedEntryPoint = collections.namedtuple('IncludedEntryPoint', ('logger', 'kind', 'source_path', 'dest_path', 'module_name', 'package_name', 'executable', 'reason'))

def _makeIncludedEntryPoint(logger, kind, source_path, dest_path, module_name, package_name, reason, executable):
    if False:
        i = 10
        return i + 15
    if package_name is not None:
        package_name = ModuleName(package_name)
    assert type(executable) is bool, executable
    assert source_path == os.path.normpath(source_path), source_path
    assert not hasFilenameExtension(path=source_path, extensions=('.qml', '.json'))
    return IncludedEntryPoint(logger=logger, kind=kind, source_path=source_path, dest_path=os.path.normpath(dest_path), module_name=module_name, package_name=package_name, executable=executable, reason=reason)

def _makeDllOrExeEntryPoint(logger, kind, source_path, dest_path, module_name, package_name, reason, executable):
    if False:
        for i in range(10):
            print('nop')
    assert type(dest_path) not in (tuple, list)
    assert type(source_path) not in (tuple, list)
    assert isRelativePath(dest_path), dest_path
    assert '.dist' not in dest_path, dest_path
    if module_name is not None:
        assert checkModuleName(module_name), module_name
        module_name = ModuleName(module_name)
    if package_name is not None:
        assert checkModuleName(package_name), package_name
        package_name = ModuleName(package_name)
    if not os.path.isfile(source_path):
        logger.sysexit("Error, attempting to include file '%s' (%s) that does not exist." % (getReportPath(source_path), reason))
    return _makeIncludedEntryPoint(logger=logger, kind=kind, source_path=source_path, dest_path=dest_path, module_name=module_name, package_name=package_name, reason=reason, executable=executable)

def makeExtensionModuleEntryPoint(logger, source_path, dest_path, module_name, package_name, reason):
    if False:
        print('Hello World!')
    return _makeDllOrExeEntryPoint(logger=logger, kind='extension', source_path=source_path, dest_path=dest_path, module_name=module_name, package_name=package_name, reason=reason, executable=False)

def makeDllEntryPoint(logger, source_path, dest_path, module_name, package_name, reason):
    if False:
        print('Hello World!')
    return _makeDllOrExeEntryPoint(logger=logger, kind='dll', source_path=source_path, dest_path=dest_path, module_name=module_name, package_name=package_name, reason=reason, executable=False)

def makeExeEntryPoint(logger, source_path, dest_path, module_name, package_name, reason):
    if False:
        print('Hello World!')
    return _makeDllOrExeEntryPoint(logger=logger, kind='exe', source_path=source_path, dest_path=dest_path, module_name=module_name, package_name=package_name, reason=reason, executable=True)

def makeMainExecutableEntryPoint(dest_path):
    if False:
        for i in range(10):
            print('nop')
    return _makeDllOrExeEntryPoint(logger=general, kind='executable', source_path=dest_path, dest_path=os.path.basename(dest_path), module_name=None, package_name=None, reason='main binary', executable=True)

def _makeIgnoredEntryPoint(entry_point):
    if False:
        i = 10
        return i + 15
    return _makeDllOrExeEntryPoint(logger=entry_point.logger, kind=entry_point.kind + '_ignored', source_path=entry_point.source_path, dest_path=entry_point.dest_path, module_name=entry_point.module_name, package_name=entry_point.package_name, reason=entry_point.reason, executable=entry_point.executable)
standalone_entry_points = []

def _getTopLevelPackageName(package_name):
    if False:
        for i in range(10):
            print('nop')
    if package_name is None:
        return None
    else:
        return package_name.getTopLevelPackageName()

def _warnNonIdenticalEntryPoints(entry_point1, entry_point2):
    if False:
        i = 10
        return i + 15
    if frozenset((_getTopLevelPackageName(entry_point1.package_name), _getTopLevelPackageName(entry_point2.package_name))) == frozenset(('numpy', 'scipy')):
        return
    if frozenset((_getTopLevelPackageName(entry_point1.package_name), _getTopLevelPackageName(entry_point2.package_name))) == frozenset(('av', 'cv2')):
        return

    def _describe(entry_point):
        if False:
            i = 10
            return i + 15
        if entry_point.package_name:
            return "'%s' of package '%s'" % (entry_point.source_path, entry_point.package_name)
        else:
            return "'%s'" % entry_point.source_path
    inclusion_logger.warning('Ignoring non-identical DLLs for %s, %s different from %s. Using first one and hoping for the best.' % (entry_point1.dest_path, _describe(entry_point1), _describe(entry_point2)))

def addIncludedEntryPoint(entry_point):
    if False:
        while True:
            i = 10
    for (count, standalone_entry_point) in enumerate(standalone_entry_points):
        if standalone_entry_point.kind.endswith('_ignored'):
            continue
        if areSamePaths(entry_point.dest_path, standalone_entry_point.dest_path):
            if areSamePaths(entry_point.source_path, standalone_entry_point.source_path):
                return
            if isShowInclusion():
                inclusion_logger.info("Colliding DLL names for %s, checking identity of '%s' <-> '%s'." % (entry_point.dest_path, entry_point.source_path, standalone_entry_point.source_path))
            if haveSameFileContents(entry_point.source_path, standalone_entry_point.source_path):
                entry_point = _makeIgnoredEntryPoint(entry_point)
                break
            old_dll_version = getDLLVersion(standalone_entry_point.source_path)
            new_dll_version = getDLLVersion(entry_point.source_path)
            if old_dll_version is None and new_dll_version is None:
                _warnNonIdenticalEntryPoints(standalone_entry_point, entry_point)
                entry_point = _makeIgnoredEntryPoint(entry_point)
                break
            if old_dll_version is None and new_dll_version is not None:
                standalone_entry_points[count] = _makeIgnoredEntryPoint(standalone_entry_point)
                break
            if old_dll_version is not None and new_dll_version is None:
                entry_point = _makeIgnoredEntryPoint(entry_point)
                break
            if old_dll_version < new_dll_version:
                standalone_entry_points[count] = _makeIgnoredEntryPoint(standalone_entry_point)
                break
            if old_dll_version >= new_dll_version:
                entry_point = _makeIgnoredEntryPoint(entry_point)
                break
            assert False, (old_dll_version, new_dll_version)
    if not entry_point.kind.endswith('_ignored'):
        for noinclude_dll_pattern in getShallNotIncludeDllFilePatterns():
            if fnmatch.fnmatch(entry_point.dest_path, noinclude_dll_pattern):
                entry_point = _makeIgnoredEntryPoint(entry_point)
    standalone_entry_points.append(entry_point)

def addIncludedEntryPoints(entry_points):
    if False:
        while True:
            i = 10
    for entry_point in entry_points:
        addIncludedEntryPoint(entry_point)

def setMainEntryPoint(binary_filename):
    if False:
        for i in range(10):
            print('nop')
    entry_point = makeMainExecutableEntryPoint(binary_filename)
    standalone_entry_points.insert(0, entry_point)

def addExtensionModuleEntryPoint(module):
    if False:
        for i in range(10):
            print('nop')
    standalone_entry_points.append(makeExtensionModuleEntryPoint(logger=general, source_path=module.getFilename(), dest_path=module.getFullName().asPath() + getSharedLibrarySuffix(preferred=False), module_name=module.getFullName(), package_name=module.getFullName().getPackageName(), reason='required extension module for CPython library startup' if module.isTechnical() else 'used extension module'))

def getStandaloneEntryPoints():
    if False:
        for i in range(10):
            print('nop')
    return tuple(standalone_entry_points)