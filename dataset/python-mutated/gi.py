import os
import pathlib
import shutil
import subprocess
import hashlib
import re
from PyInstaller.depend.utils import _resolveCtypesImports
from PyInstaller.utils.hooks import collect_submodules, collect_system_data_files, get_hook_config
from PyInstaller import isolated
from PyInstaller import log as logging
from PyInstaller import compat
from PyInstaller.depend.bindepend import findSystemLibrary
logger = logging.getLogger(__name__)

class GiModuleInfo:

    def __init__(self, module, version, hook_api=None):
        if False:
            print('Hello World!')
        self.name = module
        self.version = version
        self.available = False
        self.sharedlibs = []
        self.typelib = None
        self.dependencies = []
        if hook_api is not None:
            module_versions = get_hook_config(hook_api, 'gi', 'module-versions')
            if module_versions:
                self.version = module_versions.get(module, version)
        logger.debug('Gathering GI module info for %s %s', module, self.version)

        @isolated.decorate
        def _get_module_info(module, version):
            if False:
                print('Hello World!')
            import gi
            gi.require_version('GIRepository', '2.0')
            from gi.repository import GIRepository
            repo = GIRepository.Repository.get_default()
            repo.require(module, version, GIRepository.RepositoryLoadFlags.IREPOSITORY_LOAD_FLAG_LAZY)
            sharedlibs = repo.get_shared_library(module)
            sharedlibs = [lib.strip() for lib in sharedlibs.split(',')] if sharedlibs else []
            typelib = repo.get_typelib_path(module)
            if hasattr(repo, 'get_immediate_dependencies'):
                dependencies = repo.get_immediate_dependencies(module)
            else:
                dependencies = repo.get_dependencies(module)
            return {'sharedlibs': sharedlibs, 'typelib': typelib, 'dependencies': dependencies}
        try:
            info = _get_module_info(module, self.version)
            self.sharedlibs = info['sharedlibs']
            self.typelib = info['typelib']
            self.dependencies = info['dependencies']
            self.available = True
        except Exception as e:
            logger.debug('Failed to query GI module %s %s: %s', module, self.version, e)
            self.available = False

    def get_libdir(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the path to shared library used by the module. If no libraries are associated with the typelib, None is\n        returned. If multiple library names are associated with the typelib, the path to the first resolved shared\n        library is returned. Raises exception if module is unavailable or none of the shared libraries could be\n        resolved.\n        '
        if not self.available:
            raise ValueError(f'Module {self.name} {self.version} is unavailable!')
        if not self.sharedlibs:
            return None
        for lib in self.sharedlibs:
            path = findSystemLibrary(lib)
            if path:
                return os.path.normpath(os.path.dirname(path))
        raise ValueError(f'Could not resolve any shared library of {self.name} {self.version}: {self.sharedlibs}!')

    def collect_typelib_data(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a tuple of (binaries, datas, hiddenimports) to be used by PyGObject related hooks.\n        '
        datas = []
        binaries = []
        hiddenimports = []
        logger.debug('Collecting module data for %s %s', self.name, self.version)
        if not self.available:
            raise ValueError(f'Module {self.name} {self.version} is unavailable!')
        resolved_libs = _resolveCtypesImports(self.sharedlibs)
        for resolved_lib in resolved_libs:
            logger.debug('Collecting shared library %s at %s', resolved_lib[0], resolved_lib[1])
            binaries.append((resolved_lib[1], '.'))
        typelib_entry = gir_library_path_fix(self.typelib)
        if typelib_entry:
            logger.debug('Collecting gir typelib at %s', typelib_entry[0])
            datas.append(typelib_entry)
        hiddenimports += collect_submodules('gi.overrides', lambda name: name.endswith('.' + self.name))
        for dep in self.dependencies:
            (dep_module, _) = dep.rsplit('-', 1)
            hiddenimports += [f'gi.repository.{dep_module}']
        return (binaries, datas, hiddenimports)

def get_gi_libdir(module, version):
    if False:
        print('Hello World!')
    module_info = GiModuleInfo(module, version)
    return module_info.get_libdir()

def get_gi_typelibs(module, version):
    if False:
        return 10
    "\n    Return a tuple of (binaries, datas, hiddenimports) to be used by PyGObject related hooks. Searches for and adds\n    dependencies recursively.\n\n    :param module: GI module name, as passed to 'gi.require_version()'\n    :param version: GI module version, as passed to 'gi.require_version()'\n    "
    module_info = GiModuleInfo(module, version)
    return module_info.collect_typelib_data()

def gir_library_path_fix(path):
    if False:
        i = 10
        return i + 15
    import subprocess
    from PyInstaller.config import CONF
    path = os.path.abspath(path)
    if compat.is_darwin:
        common_path = os.path.commonprefix([compat.base_prefix, path])
        if common_path == '/':
            logger.debug('virtualenv detected? fixing the gir path...')
            common_path = os.path.abspath(os.path.join(path, '..', '..', '..'))
        gir_path = os.path.join(common_path, 'share', 'gir-1.0')
        typelib_name = os.path.basename(path)
        gir_name = os.path.splitext(typelib_name)[0] + '.gir'
        gir_file = os.path.join(gir_path, gir_name)
        if not os.path.exists(gir_path):
            logger.error("Unable to find gir directory: %s.\nTry installing your platform's gobject-introspection package.", gir_path)
            return None
        if not os.path.exists(gir_file):
            logger.error("Unable to find gir file: %s.\nTry installing your platform's gobject-introspection package.", gir_file)
            return None
        with open(gir_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(os.path.join(CONF['workpath'], gir_name), 'w', encoding='utf-8') as f:
            for line in lines:
                if 'shared-library' in line:
                    split = re.split('(=)', line)
                    files = re.split('(["|,])', split[2])
                    for (count, item) in enumerate(files):
                        if 'lib' in item:
                            files[count] = '@loader_path/' + os.path.basename(item)
                    line = ''.join(split[0:2]) + ''.join(files)
                f.write(line)
        command = subprocess.Popen(('g-ir-compiler', os.path.join(CONF['workpath'], gir_name), '-o', os.path.join(CONF['workpath'], typelib_name)))
        command.wait()
        return (os.path.join(CONF['workpath'], typelib_name), 'gi_typelibs')
    else:
        return (path, 'gi_typelibs')

@isolated.decorate
def get_glib_system_data_dirs():
    if False:
        print('Hello World!')
    import gi
    gi.require_version('GLib', '2.0')
    from gi.repository import GLib
    return GLib.get_system_data_dirs()

def get_glib_sysconf_dirs():
    if False:
        for i in range(10):
            print('nop')
    '\n    Try to return the sysconf directories (e.g., /etc).\n    '
    if compat.is_win:
        return [os.path.join(get_gi_libdir('GLib', '2.0'), 'etc')]

    @isolated.call
    def data_dirs():
        if False:
            print('Hello World!')
        import gi
        gi.require_version('GLib', '2.0')
        from gi.repository import GLib
        return GLib.get_system_config_dirs()
    return data_dirs

def collect_glib_share_files(*path):
    if False:
        print('Hello World!')
    '\n    Path is relative to the system data directory (e.g., /usr/share).\n    '
    glib_data_dirs = get_glib_system_data_dirs()
    if glib_data_dirs is None:
        return []
    destdir = os.path.join('share', *path)
    collected = []
    for data_dir in glib_data_dirs:
        p = os.path.join(data_dir, *path)
        collected += collect_system_data_files(p, destdir=destdir, include_py_files=False)
    return collected

def collect_glib_etc_files(*path):
    if False:
        i = 10
        return i + 15
    '\n    Path is relative to the system config directory (e.g., /etc).\n    '
    glib_config_dirs = get_glib_sysconf_dirs()
    if glib_config_dirs is None:
        return []
    destdir = os.path.join('etc', *path)
    collected = []
    for config_dir in glib_config_dirs:
        p = os.path.join(config_dir, *path)
        collected += collect_system_data_files(p, destdir=destdir, include_py_files=False)
    return collected
_glib_translations = None

def collect_glib_translations(prog, lang_list=None):
    if False:
        print('Hello World!')
    '\n    Return a list of translations in the system locale directory whose names equal prog.mo.\n    '
    global _glib_translations
    if _glib_translations is None:
        if lang_list is not None:
            trans = []
            for lang in lang_list:
                trans += collect_glib_share_files(os.path.join('locale', lang))
            _glib_translations = trans
        else:
            _glib_translations = collect_glib_share_files('locale')
    names = [os.sep + prog + '.mo', os.sep + prog + '.po']
    namelen = len(names[0])
    return [(src, dst) for (src, dst) in _glib_translations if src[-namelen:] in names]

def compile_glib_schema_files(datas_toc, workdir, collect_source_files=False):
    if False:
        while True:
            i = 10
    '\n    Compile collected GLib schema files. Extracts the list of GLib schema files from the given input datas TOC, copies\n    them to temporary working directory, and compiles them. The resulting `gschemas.compiled` file is added to the\n    output TOC, replacing any existing entry with that name. If `collect_source_files` flag is set, the source XML\n    schema files are also (re)added to the output TOC; by default, they are not. This function is no-op (returns the\n    original TOC) if no GLib schemas are found in TOC or if `glib-compile-schemas` executable is not found in `PATH`.\n    '
    SCHEMA_DEST_DIR = pathlib.PurePath('share/glib-2.0/schemas')
    workdir = pathlib.Path(workdir)
    schema_files = []
    output_toc = []
    for toc_entry in datas_toc:
        (dest_name, src_name, typecode) = toc_entry
        dest_name = pathlib.PurePath(dest_name)
        src_name = pathlib.PurePath(src_name)
        if dest_name.parent != SCHEMA_DEST_DIR:
            output_toc.append(toc_entry)
            continue
        if src_name.name in {'gschema.dtd', 'gschemas.compiled'}:
            continue
        schema_files.append(src_name)
    if not schema_files:
        return datas_toc
    schema_compiler_exe = shutil.which('glib-compile-schemas')
    if not schema_compiler_exe:
        logger.warning('GLib schema compiler (glib-compile-schemas) not found! Skipping GLib schema recompilation...')
        return datas_toc
    compiled_file = workdir / 'gschemas.compiled'
    old_compiled_file_hash = None
    old_compiled_file_stat = None
    if compiled_file.is_file():
        old_compiled_file_stat = compiled_file.stat()
        old_compiled_file_hash = hashlib.sha1(compiled_file.read_bytes()).digest()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(exist_ok=True)
    for schema_file in schema_files:
        shutil.copy(schema_file, workdir)
    logger.info('Compiling collected GLib schema files in %r...', str(workdir))
    try:
        cmd_args = [schema_compiler_exe, str(workdir), '--targetdir', str(workdir)]
        p = subprocess.run(cmd_args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, text=True, errors='ignore')
        logger.debug('Output from glib-compile-schemas:\n%s', p.stdout)
    except subprocess.CalledProcessError as e:
        logger.warning('Failed to recompile GLib schemas! Returning collected files as-is!', exc_info=True)
        logger.warning('Output from glib-compile-schemas:\n%s', e.stdout)
        return datas_toc
    except Exception:
        logger.warning('Failed to recompile GLib schemas! Returning collected files as-is!', exc_info=True)
        return datas_toc
    if old_compiled_file_hash is not None:
        new_compiled_file_hash = hashlib.sha1(compiled_file.read_bytes()).digest()
        if new_compiled_file_hash == old_compiled_file_hash:
            os.utime(compiled_file, ns=(old_compiled_file_stat.st_atime_ns, old_compiled_file_stat.st_mtime_ns))
    output_toc.append((str(SCHEMA_DEST_DIR / compiled_file.name), str(compiled_file), 'DATA'))
    if collect_source_files:
        for schema_file in schema_files:
            output_toc.append((str(SCHEMA_DEST_DIR / schema_file.name), str(schema_file), 'DATA'))
    return output_toc