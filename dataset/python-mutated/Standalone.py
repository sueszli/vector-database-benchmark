""" Pack and copy files for standalone mode.

This is expected to work for macOS, Windows, and Linux. Other things like
FreeBSD are also very welcome, but might break with time and need your
help.
"""
import os
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.Errors import NuitkaForbiddenDLLEncounter
from nuitka.importing.Importing import getPythonUnpackedSearchPath, locateModule
from nuitka.importing.StandardLibrary import isStandardLibraryPath
from nuitka.Options import isShowProgress, shallNotStoreDependsExeCachedResults, shallNotUseDependsExeCachedResults
from nuitka.plugins.Plugins import Plugins
from nuitka.Progress import closeProgressBar, reportProgressBar, setupProgressBar
from nuitka.Tracing import general, inclusion_logger
from nuitka.utils.FileOperations import areInSamePaths, isFilenameBelowPath
from nuitka.utils.SharedLibraries import copyDllFile, setSharedLibraryRPATH
from nuitka.utils.Signing import addMacOSCodeSignature
from nuitka.utils.Timing import TimerReport
from nuitka.utils.Utils import getOS, isDebianBasedLinux, isMacOS, isPosixWindows, isWin32Windows
from .DllDependenciesMacOS import detectBinaryPathDLLsMacOS, fixupBinaryDLLPathsMacOS
from .DllDependenciesPosix import detectBinaryPathDLLsPosix
from .DllDependenciesWin32 import detectBinaryPathDLLsWin32
from .IncludedEntryPoints import addIncludedEntryPoint, makeDllEntryPoint

def checkFreezingModuleSet():
    if False:
        i = 10
        return i + 15
    'Check the module set for troubles.\n\n    Typically Linux OS specific packages must be avoided, e.g. Debian packaging\n    does make sure the packages will not run on other OSes.\n    '
    from nuitka import ModuleRegistry
    problem_modules = OrderedSet()
    if isDebianBasedLinux():
        message = 'Standalone with Python package from Debian installation may not be working.'
        mnemonic = 'debian-dist-packages'

        def checkModulePath(module):
            if False:
                print('Hello World!')
            module_filename = module.getCompileTimeFilename()
            module_filename_parts = module_filename.split('/')
            if 'dist-packages' in module_filename_parts and 'local' not in module_filename_parts:
                module_name = module.getFullName()
                package_name = module_name.getTopLevelPackageName()
                if package_name is not None:
                    problem_modules.add(package_name)
                else:
                    problem_modules.add(module_name)
    else:
        checkModulePath = None
        message = None
        mnemonic = None
    if checkModulePath is not None:
        for module in ModuleRegistry.getDoneModules():
            if not module.getFullName().isFakeModuleName():
                checkModulePath(module)
    if problem_modules:
        general.info("Using Debian packages for '%s'." % ','.join(problem_modules))
        general.warning(message=message, mnemonic=mnemonic)

def _detectBinaryDLLs(is_main_executable, source_dir, original_filename, binary_filename, package_name, use_cache, update_cache):
    if False:
        while True:
            i = 10
    'Detect the DLLs used by a binary.\n\n    Using "ldd" (Linux), "depends.exe" (Windows), or\n    "otool" (macOS) the list of used DLLs is retrieved.\n    '
    if getOS() in ('Linux', 'NetBSD', 'FreeBSD', 'OpenBSD') or isPosixWindows():
        return detectBinaryPathDLLsPosix(dll_filename=original_filename, package_name=package_name, original_dir=os.path.dirname(original_filename))
    elif isWin32Windows():
        with TimerReport(message="Running 'depends.exe' for %s took %%.2f seconds" % binary_filename, decider=isShowProgress):
            return detectBinaryPathDLLsWin32(is_main_executable=is_main_executable, source_dir=source_dir, original_dir=os.path.dirname(original_filename), binary_filename=binary_filename, package_name=package_name, use_cache=use_cache, update_cache=update_cache)
    elif isMacOS():
        return detectBinaryPathDLLsMacOS(original_dir=os.path.dirname(original_filename), binary_filename=original_filename, package_name=package_name, keep_unresolved=False, recursive=True)
    else:
        assert False, getOS()

def copyDllsUsed(dist_dir, standalone_entry_points):
    if False:
        return 10
    copy_standalone_entry_points = [standalone_entry_point for standalone_entry_point in standalone_entry_points[1:] if not standalone_entry_point.kind.endswith('_ignored')]
    main_standalone_entry_point = standalone_entry_points[0]
    if isMacOS():
        fixupBinaryDLLPathsMacOS(binary_filename=os.path.join(dist_dir, main_standalone_entry_point.dest_path), package_name=main_standalone_entry_point.package_name, original_location=main_standalone_entry_point.source_path, standalone_entry_points=standalone_entry_points)
        setSharedLibraryRPATH(os.path.join(dist_dir, standalone_entry_points[0].dest_path), '$ORIGIN')
    setupProgressBar(stage='Copying used DLLs', unit='DLL', total=len(copy_standalone_entry_points))
    for standalone_entry_point in copy_standalone_entry_points:
        reportProgressBar(standalone_entry_point.dest_path)
        copyDllFile(source_path=standalone_entry_point.source_path, dist_dir=dist_dir, dest_path=standalone_entry_point.dest_path, executable=standalone_entry_point.executable)
        if isMacOS():
            fixupBinaryDLLPathsMacOS(binary_filename=os.path.join(dist_dir, standalone_entry_point.dest_path), package_name=standalone_entry_point.package_name, original_location=standalone_entry_point.source_path, standalone_entry_points=standalone_entry_points)
    closeProgressBar()
    if isMacOS():
        addMacOSCodeSignature(filenames=[os.path.join(dist_dir, standalone_entry_point.dest_path) for standalone_entry_point in [main_standalone_entry_point] + copy_standalone_entry_points])
    Plugins.onCopiedDLLs(dist_dir=dist_dir, standalone_entry_points=copy_standalone_entry_points)

def _reduceToPythonPath(used_dlls):
    if False:
        return 10
    inside_paths = getPythonUnpackedSearchPath()

    def decideInside(dll_filename):
        if False:
            print('Hello World!')
        return any((isFilenameBelowPath(path=inside_path, filename=dll_filename) for inside_path in inside_paths))
    used_dlls = set((dll_filename for dll_filename in used_dlls if decideInside(dll_filename)))
    return used_dlls

def _detectUsedDLLs(standalone_entry_point, source_dir):
    if False:
        while True:
            i = 10
    binary_filename = standalone_entry_point.source_path
    try:
        used_dlls = _detectBinaryDLLs(is_main_executable=standalone_entry_point.kind == 'executable', source_dir=source_dir, original_filename=standalone_entry_point.source_path, binary_filename=standalone_entry_point.source_path, package_name=standalone_entry_point.package_name, use_cache=not shallNotUseDependsExeCachedResults(), update_cache=not shallNotStoreDependsExeCachedResults())
    except NuitkaForbiddenDLLEncounter:
        inclusion_logger.info("Not including forbidden DLL '%s'." % binary_filename)
    else:
        if standalone_entry_point.module_name is not None and used_dlls:
            (module_name, module_filename, _kind, finding) = locateModule(standalone_entry_point.module_name, parent_package=None, level=0)
            assert module_name == standalone_entry_point.module_name, standalone_entry_point.module_name
            assert finding == 'absolute', standalone_entry_point.module_name
            if isStandardLibraryPath(module_filename):
                allow_outside_dependencies = True
            else:
                allow_outside_dependencies = Plugins.decideAllowOutsideDependencies(standalone_entry_point.module_name)
            if allow_outside_dependencies is False:
                used_dlls = _reduceToPythonPath(used_dlls)
        removed_dlls = Plugins.removeDllDependencies(dll_filename=binary_filename, dll_filenames=used_dlls)
        used_dlls = tuple(OrderedSet(used_dlls) - OrderedSet(removed_dlls))
        for used_dll in used_dlls:
            dest_path = os.path.basename(used_dll)
            if standalone_entry_point.package_name in ('openvino', 'av') and areInSamePaths(standalone_entry_point.source_path, used_dll):
                dest_path = os.path.normpath(os.path.join(os.path.dirname(standalone_entry_point.dest_path), dest_path))
            dll_entry_point = makeDllEntryPoint(logger=inclusion_logger, source_path=used_dll, dest_path=dest_path, module_name=standalone_entry_point.module_name, package_name=standalone_entry_point.package_name, reason="Used by '%s'" % standalone_entry_point.dest_path)
            addIncludedEntryPoint(dll_entry_point)

def detectUsedDLLs(standalone_entry_points, source_dir):
    if False:
        for i in range(10):
            print('nop')
    setupProgressBar(stage='Detecting used DLLs', unit='DLL', total=len(standalone_entry_points))
    for standalone_entry_point in standalone_entry_points:
        reportProgressBar(standalone_entry_point.dest_path)
        _detectUsedDLLs(standalone_entry_point=standalone_entry_point, source_dir=source_dir)
    closeProgressBar()