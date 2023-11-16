"""
Build packages using spec files.

NOTE: All global variables, classes and imported modules create API for .spec files.
"""
import glob
import os
import pathlib
import pprint
import shutil
import enum
import sys
from PyInstaller import DEFAULT_DISTPATH, DEFAULT_WORKPATH, HOMEPATH, compat
from PyInstaller import log as logging
from PyInstaller.building.api import COLLECT, EXE, MERGE, PYZ
from PyInstaller.building.datastruct import TOC, Target, Tree, _check_guts_eq, normalize_toc, normalize_pyz_toc, toc_process_symbolic_links
from PyInstaller.building.osx import BUNDLE
from PyInstaller.building.splash import Splash
from PyInstaller.building.utils import _check_guts_toc, _check_guts_toc_mtime, _should_include_system_binary, format_binaries_and_datas, compile_pymodule, add_suffix_to_extension, postprocess_binaries_toc_pywin32, postprocess_binaries_toc_pywin32_anaconda
from PyInstaller.compat import is_win, is_conda, is_darwin
from PyInstaller.depend import bindepend
from PyInstaller.depend.analysis import initialize_modgraph
from PyInstaller.depend.utils import create_py3_base_library, scan_code_for_ctypes
from PyInstaller import isolated
from PyInstaller.utils.misc import absnormpath, get_path_to_toplevel_modules, mtime
from PyInstaller.utils.hooks import get_package_paths
from PyInstaller.utils.hooks.gi import compile_glib_schema_files
if is_darwin:
    from PyInstaller.utils import osx as osxutils
logger = logging.getLogger(__name__)
STRINGTYPE = type('')
TUPLETYPE = type((None,))
rthooks = {}
_init_code_path = os.path.join(HOMEPATH, 'PyInstaller', 'loader')
IMPORT_TYPES = ['top-level', 'conditional', 'delayed', 'delayed, conditional', 'optional', 'conditional, optional', 'delayed, optional', 'delayed, conditional, optional']
WARNFILE_HEADER = "\nThis file lists modules PyInstaller was not able to find. This does not\nnecessarily mean this module is required for running your program. Python and\nPython 3rd-party packages include a lot of conditional or optional modules. For\nexample the module 'ntpath' only exists on Windows, whereas the module\n'posixpath' only exists on Posix systems.\n\nTypes if import:\n* top-level: imported at the top-level - look at these first\n* conditional: imported within an if-statement\n* delayed: imported within a function\n* optional: imported within a try-except-statement\n\nIMPORTANT: Do NOT post this list to the issue-tracker. Use it as a basis for\n            tracking down the missing module yourself. Thanks!\n\n"

@isolated.decorate
def discover_hook_directories():
    if False:
        return 10
    '\n    Discover hook directories via pyinstaller40 entry points. Perform the discovery in an isolated subprocess\n    to avoid importing the package(s) in the main process.\n\n    :return: list of discovered hook directories.\n    '
    from traceback import format_exception_only
    from PyInstaller.log import logger
    from PyInstaller.compat import importlib_metadata
    entry_points = importlib_metadata.entry_points(group='pyinstaller40', name='hook-dirs')
    entry_points = sorted(entry_points, key=lambda x: x.module == '_pyinstaller_hooks_contrib.hooks')
    hook_directories = []
    for entry_point in entry_points:
        try:
            hook_directories.extend(entry_point.load()())
        except Exception as e:
            msg = ''.join(format_exception_only(type(e), e)).strip()
            logger.warning("discover_hook_directories: Failed to process hook entry point '%s': %s", entry_point, msg)
    logger.debug('discover_hook_directories: Hook directories: %s', hook_directories)
    return hook_directories

def find_binary_dependencies(binaries, import_packages):
    if False:
        while True:
            i = 10
    '\n    Find dynamic dependencies (linked shared libraries) for the provided list of binaries.\n\n    On Windows, this function performs additional pre-processing in an isolated environment in an attempt to handle\n    dynamic library search path modifications made by packages during their import. The packages from the given list\n    of collected packages are imported one by one, while keeping track of modifications made by `os.add_dll_directory`\n    calls and additions to the `PATH`  environment variable. The recorded additional search paths are then passed to\n    the binary dependency analysis step.\n\n    binaries\n            List of binaries to scan for dynamic dependencies.\n    import_packages\n            List of packages to import prior to scanning binaries.\n\n    :return: expanded list of binaries and then dependencies.\n    '
    extra_libdirs = []
    if compat.is_win:
        extra_libdirs.append(compat.base_prefix)
        pywin32_system32_dir = None
        try:
            (_, pywin32_system32_dir) = get_package_paths('pywin32_system32')
        except Exception:
            pass
        if pywin32_system32_dir:
            pywin32_base_dir = os.path.dirname(pywin32_system32_dir)
            extra_libdirs += [pywin32_system32_dir, os.path.join(pywin32_base_dir, 'win32'), os.path.join(pywin32_base_dir, 'win32', 'lib'), os.path.join(pywin32_base_dir, 'Pythonwin')]
    if compat.is_win:

        def setup():
            if False:
                print('Hello World!')
            '\n            Prepare environment for change tracking\n            '
            import os
            os._added_dll_directories = []
            os._original_path_env = os.environ.get('PATH', '')
            _original_add_dll_directory = os.add_dll_directory

            def _pyi_add_dll_directory(path):
                if False:
                    for i in range(10):
                        print('nop')
                os._added_dll_directories.append(path)
                return _original_add_dll_directory(path)
            os.add_dll_directory = _pyi_add_dll_directory

        def import_library(package):
            if False:
                print('Hello World!')
            '\n            Import collected package to set up environment.\n            '
            try:
                __import__(package)
            except Exception:
                pass

        def process_search_paths():
            if False:
                print('Hello World!')
            '\n            Obtain lists of added search paths.\n            '
            import os
            dll_directories = [str(path) for path in os._added_dll_directories]
            orig_path = set(os._original_path_env.split(os.pathsep))
            modified_path = os.environ.get('PATH', '').split(os.pathsep)
            path_additions = [path for path in modified_path if path and path not in orig_path]
            return (dll_directories, path_additions)
        with isolated.Python() as child:
            child.call(setup)
            for package in import_packages:
                child.call(import_library, package)
            (added_dll_directories, added_path_directories) = child.call(process_search_paths)
        logger.info('Extra DLL search directories (AddDllDirectory): %r', added_dll_directories)
        extra_libdirs += added_dll_directories
        logger.info('Extra DLL search directories (PATH): %r', added_path_directories)
        extra_libdirs += added_path_directories
    extra_libdirs = list(dict.fromkeys(extra_libdirs).keys())
    return bindepend.binary_dependency_analysis(binaries, search_paths=extra_libdirs)

class _ModuleCollectionMode(enum.IntFlag):
    """
    Module collection mode flags.
    """
    PYZ = enum.auto()
    PYC = enum.auto()
    PY = enum.auto()
_MODULE_COLLECTION_MODES = {'pyz': _ModuleCollectionMode.PYZ, 'pyc': _ModuleCollectionMode.PYC, 'py': _ModuleCollectionMode.PY, 'pyz+py': _ModuleCollectionMode.PYZ | _ModuleCollectionMode.PY, 'py+pyz': _ModuleCollectionMode.PYZ | _ModuleCollectionMode.PY}

def _get_module_collection_mode(mode_dict, name, noarchive=False):
    if False:
        while True:
            i = 10
    '\n    Determine the module/package collection mode for the given module name, based on the provided collection\n    mode settings dictionary.\n    '
    mode_flags = _ModuleCollectionMode.PYC if noarchive else _ModuleCollectionMode.PYZ
    if not mode_dict:
        return mode_flags
    mode = 'pyz'
    name_parts = name.split('.')
    for i in range(len(name_parts)):
        modlevel = '.'.join(name_parts[:i + 1])
        modlevel_mode = mode_dict.get(modlevel, None)
        if modlevel_mode is not None:
            mode = modlevel_mode
    try:
        mode_flags = _MODULE_COLLECTION_MODES[mode]
    except KeyError:
        raise ValueError(f'Unknown module collection mode for {name!r}: {mode!r}!')
    if noarchive and _ModuleCollectionMode.PYZ in mode_flags:
        mode_flags ^= _ModuleCollectionMode.PYZ
        mode_flags |= _ModuleCollectionMode.PYC
    return mode_flags

class Analysis(Target):
    """
    Class that performs analysis of the user's main Python scripts.

    An Analysis contains multiple TOC (Table of Contents) lists, accessed as attributes of the analysis object.

    scripts
            The scripts you gave Analysis as input, with any runtime hook scripts prepended.
    pure
            The pure Python modules.
    binaries
            The extension modules and their dependencies.
    datas
            Data files collected from packages.
    zipfiles
            Deprecated - always empty.
    zipped_data
            Deprecated - always empty.
    """
    _old_scripts = {absnormpath(os.path.join(HOMEPATH, 'support', '_mountzlib.py')), absnormpath(os.path.join(HOMEPATH, 'support', 'useUnicode.py')), absnormpath(os.path.join(HOMEPATH, 'support', 'useTK.py')), absnormpath(os.path.join(HOMEPATH, 'support', 'unpackTK.py')), absnormpath(os.path.join(HOMEPATH, 'support', 'removeTK.py'))}

    def __init__(self, scripts, pathex=None, binaries=None, datas=None, hiddenimports=None, hookspath=None, hooksconfig=None, excludes=None, runtime_hooks=None, cipher=None, win_no_prefer_redirects=False, win_private_assemblies=False, noarchive=False, module_collection_mode=None, **_kwargs):
        if False:
            while True:
                i = 10
        "\n        scripts\n                A list of scripts specified as file names.\n        pathex\n                An optional list of paths to be searched before sys.path.\n        binaries\n                An optional list of additional binaries (dlls, etc.) to include.\n        datas\n                An optional list of additional data files to include.\n        hiddenimport\n                An optional list of additional (hidden) modules to include.\n        hookspath\n                An optional list of additional paths to search for hooks. (hook-modules).\n        hooksconfig\n                An optional dict of config settings for hooks. (hook-modules).\n        excludes\n                An optional list of module or package names (their Python names, not path names) that will be\n                ignored (as though they were not found).\n        runtime_hooks\n                An optional list of scripts to use as users' runtime hooks. Specified as file names.\n        cipher\n                Deprecated. Raises an error if not None.\n        win_no_prefer_redirects\n                Deprecated. Raises an error if not False.\n        win_private_assemblies\n                Deprecated. Raises an error if not False.\n        noarchive\n                If True, do not place source files in a archive, but keep them as individual files.\n        module_collection_mode\n                An optional dict of package/module names and collection mode strings. Valid collection mode strings:\n                'pyz' (default), 'pyc', 'py', 'pyz+py' (or 'py+pyz')\n        "
        if cipher is not None:
            from PyInstaller.exceptions import RemovedCipherFeatureError
            raise RemovedCipherFeatureError("Please remove the 'cipher' arguments to PYZ() and Analysis() in your spec file.")
        if win_no_prefer_redirects:
            from PyInstaller.exceptions import RemovedWinSideBySideSupportError
            raise RemovedWinSideBySideSupportError("Please remove the 'win_no_prefer_redirects' argument to Analysis() in your spec file.")
        if win_private_assemblies:
            from PyInstaller.exceptions import RemovedWinSideBySideSupportError
            raise RemovedWinSideBySideSupportError("Please remove the 'win_private_assemblies' argument to Analysis() in your spec file.")
        super().__init__()
        from PyInstaller.config import CONF
        self.inputs = []
        spec_dir = os.path.dirname(CONF['spec'])
        for script in scripts:
            if not os.path.isabs(script):
                script = os.path.join(spec_dir, script)
            if absnormpath(script) in self._old_scripts:
                logger.warning('Ignoring obsolete auto-added script %s', script)
                continue
            script = os.path.normpath(script)
            if not os.path.exists(script):
                raise SystemExit("script '%s' not found" % script)
            self.inputs.append(script)
        CONF['main_script'] = self.inputs[0]
        self.pathex = self._extend_pathex(pathex, self.inputs)
        CONF['pathex'] = self.pathex
        logger.info('Extending PYTHONPATH with paths\n' + pprint.pformat(self.pathex))
        sys.path.extend(self.pathex)
        self.hiddenimports = hiddenimports or []
        self.hiddenimports.extend(CONF.get('hiddenimports', []))
        self.hookspath = []
        if hookspath:
            self.hookspath.extend(hookspath)
        self.hookspath += discover_hook_directories()
        self.hooksconfig = {}
        if hooksconfig:
            self.hooksconfig.update(hooksconfig)
        self.custom_runtime_hooks = runtime_hooks or []
        self._input_binaries = []
        self._input_datas = []
        self.excludes = excludes or []
        self.scripts = []
        self.pure = []
        self.binaries = []
        self.zipfiles = []
        self.zipped_data = []
        self.datas = []
        self.dependencies = []
        self._python_version = sys.version
        self.noarchive = noarchive
        self.module_collection_mode = module_collection_mode or {}
        if binaries:
            logger.info("Appending 'binaries' from .spec")
            self._input_binaries = [(dest_name, src_name, 'BINARY') for (dest_name, src_name) in format_binaries_and_datas(binaries, workingdir=spec_dir)]
            self._input_binaries = sorted(normalize_toc(self._input_binaries))
        if datas:
            logger.info("Appending 'datas' from .spec")
            self._input_datas = [(dest_name, src_name, 'DATA') for (dest_name, src_name) in format_binaries_and_datas(datas, workingdir=spec_dir)]
            self._input_datas = sorted(normalize_toc(self._input_datas))
        self.__postinit__()
    _GUTS = (('inputs', _check_guts_eq), ('pathex', _check_guts_eq), ('hiddenimports', _check_guts_eq), ('hookspath', _check_guts_eq), ('hooksconfig', _check_guts_eq), ('excludes', _check_guts_eq), ('custom_runtime_hooks', _check_guts_eq), ('noarchive', _check_guts_eq), ('module_collection_mode', _check_guts_eq), ('_input_binaries', _check_guts_toc), ('_input_datas', _check_guts_toc), ('_python_version', _check_guts_eq), ('scripts', _check_guts_toc_mtime), ('pure', _check_guts_toc_mtime), ('binaries', _check_guts_toc_mtime), ('zipfiles', _check_guts_toc_mtime), ('zipped_data', None), ('datas', _check_guts_toc_mtime))

    def _extend_pathex(self, spec_pathex, scripts):
        if False:
            print('Hello World!')
        '\n        Normalize additional paths where PyInstaller will look for modules and add paths with scripts to the list of\n        paths.\n\n        :param spec_pathex: Additional paths defined defined in .spec file.\n        :param scripts: Scripts to create executable from.\n        :return: list of updated paths\n        '
        pathex = []
        for script in scripts:
            logger.debug('script: %s' % script)
            script_toplevel_dir = get_path_to_toplevel_modules(script)
            if script_toplevel_dir:
                pathex.append(script_toplevel_dir)
        if spec_pathex is not None:
            pathex.extend(spec_pathex)
        return [absnormpath(p) for p in pathex]

    def _check_guts(self, data, last_build):
        if False:
            print('Hello World!')
        if Target._check_guts(self, data, last_build):
            return True
        for filename in self.inputs:
            if mtime(filename) > last_build:
                logger.info('Building because %s changed', filename)
                return True
        self.scripts = data['scripts']
        self.pure = data['pure']
        self.binaries = data['binaries']
        self.zipfiles = data['zipfiles']
        self.zipped_data = data['zipped_data']
        self.datas = data['datas']
        return False

    def assemble(self):
        if False:
            print('Hello World!')
        '\n        This method is the MAIN method for finding all necessary files to be bundled.\n        '
        from PyInstaller.config import CONF
        for m in self.excludes:
            logger.debug("Excluding module '%s'" % m)
        self.graph = initialize_modgraph(excludes=self.excludes, user_hook_dirs=self.hookspath)
        self.datas = [entry for entry in self._input_datas]
        self.binaries = [entry for entry in self._input_binaries]
        libzip_filename = os.path.join(CONF['workpath'], 'base_library.zip')
        create_py3_base_library(libzip_filename, graph=self.graph)
        self.datas.append((os.path.basename(libzip_filename), libzip_filename, 'DATA'))
        self.graph.path = self.pathex + self.graph.path
        self.graph.scan_legacy_namespace_packages()
        logger.info('Running Analysis %s', self.tocbasename)
        logger.info('Looking for Python shared library...')
        python_lib = bindepend.get_python_library_path()
        if python_lib is None:
            from PyInstaller.exceptions import PythonLibraryNotFoundError
            raise PythonLibraryNotFoundError()
        logger.info('Using Python shared library: %s', python_lib)
        if is_darwin and osxutils.is_framework_bundle_lib(python_lib):
            src_path = pathlib.PurePath(python_lib)
            dst_path = pathlib.PurePath(src_path.relative_to(src_path.parent.parent.parent.parent))
            self.binaries.append((str(dst_path), str(src_path), 'BINARY'))
            self.binaries.append((os.path.basename(python_lib), str(dst_path), 'SYMLINK'))
        else:
            self.binaries.append((os.path.basename(python_lib), python_lib, 'BINARY'))
        priority_scripts = []
        for script in self.inputs:
            logger.info('Analyzing %s', script)
            priority_scripts.append(self.graph.add_script(script))
        self.graph.add_hiddenimports(self.hiddenimports)
        self.graph.process_post_graph_hooks(self)
        self.binaries += self.graph.make_hook_binaries_toc()
        self.datas += self.graph.make_hook_datas_toc()
        self.zipped_data = []
        self.zipfiles = []
        combined_toc = normalize_toc(self.datas + self.binaries)
        self.datas = []
        self.binaries = []
        for (dest_name, src_name, typecode) in combined_toc:
            detected_typecode = bindepend.classify_binary_vs_data(src_name)
            if detected_typecode is not None:
                if detected_typecode != typecode:
                    logger.debug('Reclassifying collected file %r from %s to %s...', src_name, typecode, detected_typecode)
                typecode = detected_typecode
            if typecode in {'BINARY', 'EXTENSION'}:
                self.binaries.append((dest_name, src_name, typecode))
            else:
                self.datas.append((dest_name, src_name, typecode))
        logger.info('Looking for ctypes DLLs')
        ctypes_code_objs = self.graph.get_code_using('ctypes')
        for (name, co) in ctypes_code_objs.items():
            logger.debug('Scanning %s for ctypes-based references to shared libraries', name)
            try:
                ctypes_binaries = scan_code_for_ctypes(co)
                for (dest_name, src_name, typecode) in set(ctypes_binaries):
                    if bindepend.classify_binary_vs_data(src_name) not in (None, 'BINARY'):
                        logger.warning('Ignoring %s found via ctypes - not a valid binary!', src_name)
                        continue
                    self.binaries.append((dest_name, src_name, typecode))
            except Exception as ex:
                raise RuntimeError(f"Failed to scan the module '{name}'. This is a bug. Please report it.") from ex
        self.datas.extend(((dest, source, 'DATA') for (dest, source) in format_binaries_and_datas(self.graph.metadata_required())))
        priority_scripts = self.graph.analyze_runtime_hooks(self.custom_runtime_hooks) + priority_scripts
        self.scripts = self.graph.nodes_to_toc(priority_scripts)
        self.scripts = normalize_toc(self.scripts)
        self.binaries += self.graph.make_binaries_toc()
        for (idx, (dest, source, typecode)) in enumerate(self.binaries):
            if typecode != 'EXTENSION':
                continue
            (dest, source, typecode) = add_suffix_to_extension(dest, source, typecode)
            src_parent = os.path.basename(os.path.dirname(source))
            if src_parent == 'lib-dynload' and (not os.path.dirname(os.path.normpath(dest))):
                dest = os.path.join('lib-dynload', dest)
            self.binaries[idx] = (dest, source, typecode)
        self.datas = normalize_toc(self.datas)
        self.binaries = normalize_toc(self.binaries)
        self.datas = compile_glib_schema_files(self.datas, os.path.join(CONF['workpath'], '_pyi_gschema_compilation'))
        self.datas = normalize_toc(self.datas)
        assert len(self.pure) == 0
        pure_pymodules_toc = self.graph.make_pure_toc()
        self.graph._module_collection_mode.update(self.module_collection_mode)
        logger.debug('Module collection settings: %r', self.graph._module_collection_mode)
        pycs_dir = os.path.join(CONF['workpath'], 'localpycs')
        code_cache = self.graph.get_code_objects()
        for (name, src_path, typecode) in pure_pymodules_toc:
            assert typecode == 'PYMODULE'
            collect_mode = _get_module_collection_mode(self.graph._module_collection_mode, name, self.noarchive)
            if _ModuleCollectionMode.PYZ in collect_mode:
                self.pure.append((name, src_path, typecode))
            if src_path in (None, '-'):
                continue
            if _ModuleCollectionMode.PY in collect_mode:
                dest_path = name.replace('.', os.sep)
                (basename, ext) = os.path.splitext(os.path.basename(src_path))
                if basename == '__init__':
                    dest_path += os.sep + '__init__' + ext
                else:
                    dest_path += ext
                self.datas.append((dest_path, src_path, 'DATA'))
            if _ModuleCollectionMode.PYC in collect_mode:
                dest_path = name.replace('.', os.sep)
                (basename, ext) = os.path.splitext(os.path.basename(src_path))
                if basename == '__init__':
                    dest_path += os.sep + '__init__'
                dest_path += '.pyc'
                obj_path = compile_pymodule(name, src_path, workpath=pycs_dir, code_cache=code_cache)
                self.datas.append((dest_path, obj_path, 'DATA'))
        self.pure = normalize_pyz_toc(self.pure)
        from PyInstaller.config import CONF
        global_code_cache_map = CONF['code_cache']
        global_code_cache_map[id(self.pure)] = code_cache
        logger.info('Looking for dynamic libraries')
        collected_packages = self.graph.get_collected_packages()
        self.binaries.extend(find_binary_dependencies(self.binaries, collected_packages))
        if is_win:
            self.binaries = postprocess_binaries_toc_pywin32(self.binaries)
            if is_conda:
                self.binaries = postprocess_binaries_toc_pywin32_anaconda(self.binaries)
        combined_toc = normalize_toc(self.datas + self.binaries)
        combined_toc = toc_process_symbolic_links(combined_toc)
        if is_darwin:
            combined_toc += osxutils.collect_files_from_framework_bundles(combined_toc)
        self.datas = []
        self.binaries = []
        for entry in combined_toc:
            (dest_name, src_name, typecode) = entry
            if typecode in {'BINARY', 'EXTENSION'}:
                self.binaries.append(entry)
            else:
                self.datas.append(entry)
        if is_darwin:
            self.datas = [(dest_name, src_name, typecode) for (dest_name, src_name, typecode) in self.datas if os.path.basename(src_name) != '.DS_Store']
        self._write_warnings()
        self._write_graph_debug()
        if is_darwin:
            binaries_with_invalid_sdk = []
            for (dest_name, src_name, typecode) in self.binaries:
                sdk_version = osxutils.get_macos_sdk_version(src_name)
                if sdk_version < (10, 9, 0):
                    binaries_with_invalid_sdk.append((dest_name, src_name, sdk_version))
            if binaries_with_invalid_sdk:
                logger.warning('Found one or more binaries with invalid or incompatible macOS SDK version:')
                for (dest_name, src_name, sdk_version) in binaries_with_invalid_sdk:
                    logger.warning(' * %r, collected as %r; version: %r', src_name, dest_name, sdk_version)
                logger.warning('These binaries will likely cause issues with code-signing and hardened runtime!')

    def _write_warnings(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write warnings about missing modules. Get them from the graph and use the graph to figure out who tried to\n        import them.\n        '

        def dependency_description(name, dep_info):
            if False:
                return 10
            if not dep_info or dep_info == 'direct':
                imptype = 0
            else:
                imptype = dep_info.conditional + 2 * dep_info.function + 4 * dep_info.tryexcept
            return '%s (%s)' % (name, IMPORT_TYPES[imptype])
        from PyInstaller.config import CONF
        miss_toc = self.graph.make_missing_toc()
        with open(CONF['warnfile'], 'w', encoding='utf-8') as wf:
            wf.write(WARNFILE_HEADER)
            for (n, p, status) in miss_toc:
                importers = self.graph.get_importers(n)
                print(status, 'module named', n, '- imported by', ', '.join((dependency_description(name, data) for (name, data) in importers)), file=wf)
        logger.info('Warnings written to %s', CONF['warnfile'])

    def _write_graph_debug(self):
        if False:
            while True:
                i = 10
        '\n        Write a xref (in html) and with `--log-level DEBUG` a dot-drawing of the graph.\n        '
        from PyInstaller.config import CONF
        with open(CONF['xref-file'], 'w', encoding='utf-8') as fh:
            self.graph.create_xref(fh)
            logger.info('Graph cross-reference written to %s', CONF['xref-file'])
        if logger.getEffectiveLevel() > logging.DEBUG:
            return
        with open(CONF['dot-file'], 'w', encoding='utf-8') as fh:
            self.graph.graphreport(fh)
            logger.info('Graph drawing written to %s', CONF['dot-file'])

    def exclude_system_libraries(self, list_of_exceptions=None):
        if False:
            print('Hello World!')
        "\n        This method may be optionally called from the spec file to exclude any system libraries from the list of\n        binaries other than those containing the shell-style wildcards in list_of_exceptions. Those that match\n        '*python*' or are stored under 'lib-dynload' are always treated as exceptions and not excluded.\n        "
        self.binaries = [entry for entry in self.binaries if _should_include_system_binary(entry, list_of_exceptions or [])]

class ExecutableBuilder:
    """
    Class that constructs the executable.
    """

def build(spec, distpath, workpath, clean_build):
    if False:
        for i in range(10):
            print('nop')
    '\n    Build the executable according to the created SPEC file.\n    '
    from PyInstaller.config import CONF
    distpath = os.path.abspath(compat.expand_path(distpath))
    workpath = os.path.abspath(compat.expand_path(workpath))
    CONF['spec'] = os.path.abspath(compat.expand_path(spec))
    (CONF['specpath'], CONF['specnm']) = os.path.split(CONF['spec'])
    CONF['specnm'] = os.path.splitext(CONF['specnm'])[0]
    if os.path.dirname(distpath) == HOMEPATH:
        distpath = os.path.join(HOMEPATH, CONF['specnm'], os.path.basename(distpath))
    CONF['distpath'] = distpath
    if os.path.dirname(workpath) == HOMEPATH:
        workpath = os.path.join(HOMEPATH, CONF['specnm'], os.path.basename(workpath), CONF['specnm'])
    else:
        workpath = os.path.join(workpath, CONF['specnm'])
    CONF['workpath'] = workpath
    CONF['warnfile'] = os.path.join(workpath, 'warn-%s.txt' % CONF['specnm'])
    CONF['dot-file'] = os.path.join(workpath, 'graph-%s.dot' % CONF['specnm'])
    CONF['xref-file'] = os.path.join(workpath, 'xref-%s.html' % CONF['specnm'])
    CONF['code_cache'] = dict()
    if clean_build:
        logger.info('Removing temporary files and cleaning cache in %s', CONF['cachedir'])
        for pth in (CONF['cachedir'], workpath):
            if os.path.exists(pth):
                for f in glob.glob(pth + '/*'):
                    if os.path.isdir(f):
                        shutil.rmtree(f)
                    else:
                        os.remove(f)
    for pth in (CONF['distpath'], CONF['workpath']):
        os.makedirs(pth, exist_ok=True)
    spec_namespace = {'DISTPATH': CONF['distpath'], 'HOMEPATH': HOMEPATH, 'SPEC': CONF['spec'], 'specnm': CONF['specnm'], 'SPECPATH': CONF['specpath'], 'WARNFILE': CONF['warnfile'], 'workpath': CONF['workpath'], 'TOC': TOC, 'Analysis': Analysis, 'BUNDLE': BUNDLE, 'COLLECT': COLLECT, 'EXE': EXE, 'MERGE': MERGE, 'PYZ': PYZ, 'Tree': Tree, 'Splash': Splash, 'os': os}
    try:
        with open(spec, 'rb') as f:
            code = compile(f.read(), spec, 'exec')
    except FileNotFoundError:
        raise SystemExit(f'Spec file "{spec}" not found!')
    exec(code, spec_namespace)

def __add_options(parser):
    if False:
        for i in range(10):
            print('nop')
    parser.add_argument('--distpath', metavar='DIR', default=DEFAULT_DISTPATH, help='Where to put the bundled app (default: ./dist)')
    parser.add_argument('--workpath', default=DEFAULT_WORKPATH, help='Where to put all the temporary work files, .log, .pyz and etc. (default: ./build)')
    parser.add_argument('-y', '--noconfirm', action='store_true', default=False, help='Replace output directory (default: %s) without asking for confirmation' % os.path.join('SPECPATH', 'dist', 'SPECNAME'))
    parser.add_argument('--upx-dir', default=None, help='Path to UPX utility (default: search the execution path)')
    parser.add_argument('--clean', dest='clean_build', action='store_true', default=False, help='Clean PyInstaller cache and remove temporary files before building.')

def main(pyi_config, specfile, noconfirm=False, distpath=DEFAULT_DISTPATH, workpath=DEFAULT_WORKPATH, upx_dir=None, clean_build=False, **kw):
    if False:
        for i in range(10):
            print('nop')
    from PyInstaller.config import CONF
    CONF['noconfirm'] = noconfirm
    if pyi_config is None:
        import PyInstaller.configure as configure
        CONF.update(configure.get_config(upx_dir=upx_dir))
    else:
        CONF.update(pyi_config)
    CONF['ui_admin'] = kw.get('ui_admin', False)
    CONF['ui_access'] = kw.get('ui_uiaccess', False)
    build(specfile, distpath, workpath, clean_build)