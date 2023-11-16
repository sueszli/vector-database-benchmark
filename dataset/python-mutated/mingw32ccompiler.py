"""
Support code for building Python extensions on Windows.

    # NT stuff
    # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
    # 2. Force windows to use gcc (we're struggling with MSVC and g77 support)
    # 3. Force windows to use g77

"""
import os
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import msvc_runtime_library, msvc_runtime_version, msvc_runtime_major, get_build_architecture

def get_msvcr_replacement():
    if False:
        for i in range(10):
            print('nop')
    'Replacement for outdated version of get_msvcr from cygwinccompiler'
    msvcr = msvc_runtime_library()
    return [] if msvcr is None else [msvcr]
_START = re.compile('\\[Ordinal/Name Pointer\\] Table')
_TABLE = re.compile('^\\s+\\[([\\s*[0-9]*)\\] ([a-zA-Z0-9_]*)')

class Mingw32CCompiler(distutils.cygwinccompiler.CygwinCCompiler):
    """ A modified MingW32 compiler compatible with an MSVC built Python.

    """
    compiler_type = 'mingw32'

    def __init__(self, verbose=0, dry_run=0, force=0):
        if False:
            print('Hello World!')
        distutils.cygwinccompiler.CygwinCCompiler.__init__(self, verbose, dry_run, force)
        build_import_library()
        msvcr_success = build_msvcr_library()
        msvcr_dbg_success = build_msvcr_library(debug=True)
        if msvcr_success or msvcr_dbg_success:
            self.define_macro('NPY_MINGW_USE_CUSTOM_MSVCR')
        msvcr_version = msvc_runtime_version()
        if msvcr_version:
            self.define_macro('__MSVCRT_VERSION__', '0x%04i' % msvcr_version)
        if get_build_architecture() == 'AMD64':
            self.set_executables(compiler='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall', compiler_so='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall -Wstrict-prototypes', linker_exe='gcc -g', linker_so='gcc -g -shared')
        else:
            self.set_executables(compiler='gcc -O2 -Wall', compiler_so='gcc -O2 -Wall -Wstrict-prototypes', linker_exe='g++ ', linker_so='g++ -shared')
        self.compiler_cxx = ['g++']
        return

    def link(self, target_desc, objects, output_filename, output_dir, libraries, library_dirs, runtime_library_dirs, export_symbols=None, debug=0, extra_preargs=None, extra_postargs=None, build_temp=None, target_lang=None):
        if False:
            while True:
                i = 10
        runtime_library = msvc_runtime_library()
        if runtime_library:
            if not libraries:
                libraries = []
            libraries.append(runtime_library)
        args = (self, target_desc, objects, output_filename, output_dir, libraries, library_dirs, runtime_library_dirs, None, debug, extra_preargs, extra_postargs, build_temp, target_lang)
        func = UnixCCompiler.link
        func(*args[:func.__code__.co_argcount])
        return

    def object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
        if False:
            return 10
        if output_dir is None:
            output_dir = ''
        obj_names = []
        for src_name in source_filenames:
            (base, ext) = os.path.splitext(os.path.normcase(src_name))
            (drv, base) = os.path.splitdrive(base)
            if drv:
                base = base[1:]
            if ext not in self.src_extensions + ['.rc', '.res']:
                raise UnknownFileError("unknown file type '%s' (from '%s')" % (ext, src_name))
            if strip_dir:
                base = os.path.basename(base)
            if ext == '.res' or ext == '.rc':
                obj_names.append(os.path.join(output_dir, base + ext + self.obj_extension))
            else:
                obj_names.append(os.path.join(output_dir, base + self.obj_extension))
        return obj_names

def find_python_dll():
    if False:
        while True:
            i = 10
    stems = [sys.prefix]
    if sys.base_prefix != sys.prefix:
        stems.append(sys.base_prefix)
    sub_dirs = ['', 'lib', 'bin']
    lib_dirs = []
    for stem in stems:
        for folder in sub_dirs:
            lib_dirs.append(os.path.join(stem, folder))
    if 'SYSTEMROOT' in os.environ:
        lib_dirs.append(os.path.join(os.environ['SYSTEMROOT'], 'System32'))
    (major_version, minor_version) = tuple(sys.version_info[:2])
    implementation = sys.implementation.name
    if implementation == 'cpython':
        dllname = f'python{major_version}{minor_version}.dll'
    elif implementation == 'pypy':
        dllname = f'libpypy{major_version}.{minor_version}-c.dll'
    else:
        dllname = f'Unknown platform {implementation}'
    print('Looking for %s' % dllname)
    for folder in lib_dirs:
        dll = os.path.join(folder, dllname)
        if os.path.exists(dll):
            return dll
    raise ValueError('%s not found in %s' % (dllname, lib_dirs))

def dump_table(dll):
    if False:
        return 10
    st = subprocess.check_output(['objdump.exe', '-p', dll])
    return st.split(b'\n')

def generate_def(dll, dfile):
    if False:
        for i in range(10):
            print('nop')
    'Given a dll file location,  get all its exported symbols and dump them\n    into the given def file.\n\n    The .def file will be overwritten'
    dump = dump_table(dll)
    for i in range(len(dump)):
        if _START.match(dump[i].decode()):
            break
    else:
        raise ValueError('Symbol table not found')
    syms = []
    for j in range(i + 1, len(dump)):
        m = _TABLE.match(dump[j].decode())
        if m:
            syms.append((int(m.group(1).strip()), m.group(2)))
        else:
            break
    if len(syms) == 0:
        log.warn('No symbols found in %s' % dll)
    with open(dfile, 'w') as d:
        d.write('LIBRARY        %s\n' % os.path.basename(dll))
        d.write(';CODE          PRELOAD MOVEABLE DISCARDABLE\n')
        d.write(';DATA          PRELOAD SINGLE\n')
        d.write('\nEXPORTS\n')
        for s in syms:
            d.write('%s\n' % s[1])

def find_dll(dll_name):
    if False:
        while True:
            i = 10
    arch = {'AMD64': 'amd64', 'Intel': 'x86'}[get_build_architecture()]

    def _find_dll_in_winsxs(dll_name):
        if False:
            while True:
                i = 10
        winsxs_path = os.path.join(os.environ.get('WINDIR', 'C:\\WINDOWS'), 'winsxs')
        if not os.path.exists(winsxs_path):
            return None
        for (root, dirs, files) in os.walk(winsxs_path):
            if dll_name in files and arch in root:
                return os.path.join(root, dll_name)
        return None

    def _find_dll_in_path(dll_name):
        if False:
            i = 10
            return i + 15
        for path in [sys.prefix] + os.environ['PATH'].split(';'):
            filepath = os.path.join(path, dll_name)
            if os.path.exists(filepath):
                return os.path.abspath(filepath)
    return _find_dll_in_winsxs(dll_name) or _find_dll_in_path(dll_name)

def build_msvcr_library(debug=False):
    if False:
        return 10
    if os.name != 'nt':
        return False
    msvcr_ver = msvc_runtime_major()
    if msvcr_ver is None:
        log.debug('Skip building import library: Runtime is not compiled with MSVC')
        return False
    if msvcr_ver < 80:
        log.debug('Skip building msvcr library: custom functionality not present')
        return False
    msvcr_name = msvc_runtime_library()
    if debug:
        msvcr_name += 'd'
    out_name = 'lib%s.a' % msvcr_name
    out_file = os.path.join(sys.prefix, 'libs', out_name)
    if os.path.isfile(out_file):
        log.debug('Skip building msvcr library: "%s" exists' % (out_file,))
        return True
    msvcr_dll_name = msvcr_name + '.dll'
    dll_file = find_dll(msvcr_dll_name)
    if not dll_file:
        log.warn('Cannot build msvcr library: "%s" not found' % msvcr_dll_name)
        return False
    def_name = 'lib%s.def' % msvcr_name
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    log.info('Building msvcr library: "%s" (from %s)' % (out_file, dll_file))
    generate_def(dll_file, def_file)
    cmd = ['dlltool', '-d', def_file, '-l', out_file]
    retcode = subprocess.call(cmd)
    os.remove(def_file)
    return not retcode

def build_import_library():
    if False:
        while True:
            i = 10
    if os.name != 'nt':
        return
    arch = get_build_architecture()
    if arch == 'AMD64':
        return _build_import_library_amd64()
    elif arch == 'Intel':
        return _build_import_library_x86()
    else:
        raise ValueError('Unhandled arch %s' % arch)

def _check_for_import_lib():
    if False:
        print('Hello World!')
    'Check if an import library for the Python runtime already exists.'
    (major_version, minor_version) = tuple(sys.version_info[:2])
    patterns = ['libpython%d%d.a', 'libpython%d%d.dll.a', 'libpython%d.%d.dll.a']
    stems = [sys.prefix]
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        stems.append(sys.base_prefix)
    elif hasattr(sys, 'real_prefix') and sys.real_prefix != sys.prefix:
        stems.append(sys.real_prefix)
    sub_dirs = ['libs', 'lib']
    candidates = []
    for pat in patterns:
        filename = pat % (major_version, minor_version)
        for stem_dir in stems:
            for folder in sub_dirs:
                candidates.append(os.path.join(stem_dir, folder, filename))
    for fullname in candidates:
        if os.path.isfile(fullname):
            return (True, fullname)
    return (False, candidates[0])

def _build_import_library_amd64():
    if False:
        for i in range(10):
            print('nop')
    (out_exists, out_file) = _check_for_import_lib()
    if out_exists:
        log.debug('Skip building import library: "%s" exists', out_file)
        return
    dll_file = find_python_dll()
    log.info('Building import library (arch=AMD64): "%s" (from %s)' % (out_file, dll_file))
    def_name = 'python%d%d.def' % tuple(sys.version_info[:2])
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    generate_def(dll_file, def_file)
    cmd = ['dlltool', '-d', def_file, '-l', out_file]
    subprocess.check_call(cmd)

def _build_import_library_x86():
    if False:
        for i in range(10):
            print('nop')
    ' Build the import libraries for Mingw32-gcc on Windows\n    '
    (out_exists, out_file) = _check_for_import_lib()
    if out_exists:
        log.debug('Skip building import library: "%s" exists', out_file)
        return
    lib_name = 'python%d%d.lib' % tuple(sys.version_info[:2])
    lib_file = os.path.join(sys.prefix, 'libs', lib_name)
    if not os.path.isfile(lib_file):
        if hasattr(sys, 'base_prefix'):
            base_lib = os.path.join(sys.base_prefix, 'libs', lib_name)
        elif hasattr(sys, 'real_prefix'):
            base_lib = os.path.join(sys.real_prefix, 'libs', lib_name)
        else:
            base_lib = ''
        if os.path.isfile(base_lib):
            lib_file = base_lib
        else:
            log.warn('Cannot build import library: "%s" not found', lib_file)
            return
    log.info('Building import library (ARCH=x86): "%s"', out_file)
    from numpy.distutils import lib2def
    def_name = 'python%d%d.def' % tuple(sys.version_info[:2])
    def_file = os.path.join(sys.prefix, 'libs', def_name)
    nm_output = lib2def.getnm(lib2def.DEFAULT_NM + [lib_file], shell=False)
    (dlist, flist) = lib2def.parse_nm(nm_output)
    with open(def_file, 'w') as fid:
        lib2def.output_def(dlist, flist, lib2def.DEF_HEADER, fid)
    dll_name = find_python_dll()
    cmd = ['dlltool', '--dllname', dll_name, '--def', def_file, '--output-lib', out_file]
    status = subprocess.check_output(cmd)
    if status:
        log.warn('Failed to build import library for gcc. Linking will fail.')
    return
_MSVCRVER_TO_FULLVER = {}
if sys.platform == 'win32':
    try:
        import msvcrt
        _MSVCRVER_TO_FULLVER['80'] = '8.0.50727.42'
        _MSVCRVER_TO_FULLVER['90'] = '9.0.21022.8'
        _MSVCRVER_TO_FULLVER['100'] = '10.0.30319.460'
        crt_ver = getattr(msvcrt, 'CRT_ASSEMBLY_VERSION', None)
        if crt_ver is not None:
            (maj, min) = re.match('(\\d+)\\.(\\d)', crt_ver).groups()
            _MSVCRVER_TO_FULLVER[maj + min] = crt_ver
            del maj, min
        del crt_ver
    except ImportError:
        log.warn('Cannot import msvcrt: using manifest will not be possible')

def msvc_manifest_xml(maj, min):
    if False:
        return 10
    'Given a major and minor version of the MSVCR, returns the\n    corresponding XML file.'
    try:
        fullver = _MSVCRVER_TO_FULLVER[str(maj * 10 + min)]
    except KeyError:
        raise ValueError('Version %d,%d of MSVCRT not supported yet' % (maj, min)) from None
    template = textwrap.dedent('        <assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\n          <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\n            <security>\n              <requestedPrivileges>\n                <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>\n              </requestedPrivileges>\n            </security>\n          </trustInfo>\n          <dependency>\n            <dependentAssembly>\n              <assemblyIdentity type="win32" name="Microsoft.VC%(maj)d%(min)d.CRT" version="%(fullver)s" processorArchitecture="*" publicKeyToken="1fc8b3b9a1e18e3b"></assemblyIdentity>\n            </dependentAssembly>\n          </dependency>\n        </assembly>')
    return template % {'fullver': fullver, 'maj': maj, 'min': min}

def manifest_rc(name, type='dll'):
    if False:
        i = 10
        return i + 15
    "Return the rc file used to generate the res file which will be embedded\n    as manifest for given manifest file name, of given type ('dll' or\n    'exe').\n\n    Parameters\n    ----------\n    name : str\n            name of the manifest file to embed\n    type : str {'dll', 'exe'}\n            type of the binary which will embed the manifest\n\n    "
    if type == 'dll':
        rctype = 2
    elif type == 'exe':
        rctype = 1
    else:
        raise ValueError('Type %s not supported' % type)
    return '#include "winuser.h"\n%d RT_MANIFEST %s' % (rctype, name)

def check_embedded_msvcr_match_linked(msver):
    if False:
        i = 10
        return i + 15
    'msver is the ms runtime version used for the MANIFEST.'
    maj = msvc_runtime_major()
    if maj:
        if not maj == int(msver):
            raise ValueError('Discrepancy between linked msvcr (%d) and the one about to be embedded (%d)' % (int(msver), maj))

def configtest_name(config):
    if False:
        return 10
    base = os.path.basename(config._gen_temp_sourcefile('yo', [], 'c'))
    return os.path.splitext(base)[0]

def manifest_name(config):
    if False:
        for i in range(10):
            print('nop')
    root = configtest_name(config)
    exext = config.compiler.exe_extension
    return root + exext + '.manifest'

def rc_name(config):
    if False:
        i = 10
        return i + 15
    root = configtest_name(config)
    return root + '.rc'

def generate_manifest(config):
    if False:
        i = 10
        return i + 15
    msver = get_build_msvc_version()
    if msver is not None:
        if msver >= 8:
            check_embedded_msvcr_match_linked(msver)
            (ma_str, mi_str) = str(msver).split('.')
            manxml = msvc_manifest_xml(int(ma_str), int(mi_str))
            with open(manifest_name(config), 'w') as man:
                config.temp_files.append(manifest_name(config))
                man.write(manxml)