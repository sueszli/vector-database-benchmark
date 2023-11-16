import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion
compilers = ['GnuFCompiler', 'Gnu95FCompiler']
TARGET_R = re.compile('Target: ([a-zA-Z0-9_\\-]*)')

def is_win64():
    if False:
        while True:
            i = 10
    return sys.platform == 'win32' and platform.architecture()[0] == '64bit'

class GnuFCompiler(FCompiler):
    compiler_type = 'gnu'
    compiler_aliases = ('g77',)
    description = 'GNU Fortran 77 compiler'

    def gnu_version_match(self, version_string):
        if False:
            while True:
                i = 10
        'Handle the different versions of GNU fortran compilers'
        while version_string.startswith('gfortran: warning'):
            version_string = version_string[version_string.find('\n') + 1:].strip()
        if len(version_string) <= 20:
            m = re.search('([0-9.]+)', version_string)
            if m:
                if version_string.startswith('GNU Fortran'):
                    return ('g77', m.group(1))
                elif m.start() == 0:
                    return ('gfortran', m.group(1))
        else:
            m = re.search('GNU Fortran\\s+95.*?([0-9-.]+)', version_string)
            if m:
                return ('gfortran', m.group(1))
            m = re.search('GNU Fortran.*?\\-?([0-9-.]+\\.[0-9-.]+)', version_string)
            if m:
                v = m.group(1)
                if v.startswith('0') or v.startswith('2') or v.startswith('3'):
                    return ('g77', v)
                else:
                    return ('gfortran', v)
        err = 'A valid Fortran version was not found in this string:\n'
        raise ValueError(err + version_string)

    def version_match(self, version_string):
        if False:
            print('Hello World!')
        v = self.gnu_version_match(version_string)
        if not v or v[0] != 'g77':
            return None
        return v[1]
    possible_executables = ['g77', 'f77']
    executables = {'version_cmd': [None, '-dumpversion'], 'compiler_f77': [None, '-g', '-Wall', '-fno-second-underscore'], 'compiler_f90': None, 'compiler_fix': None, 'linker_so': [None, '-g', '-Wall'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib'], 'linker_exe': [None, '-g', '-Wall']}
    module_dir_switch = None
    module_include_switch = None
    if os.name != 'nt' and sys.platform != 'cygwin':
        pic_flags = ['-fPIC']
    if sys.platform == 'win32':
        for key in ['version_cmd', 'compiler_f77', 'linker_so', 'linker_exe']:
            executables[key].append('-mno-cygwin')
    g2c = 'g2c'
    suggested_f90_compiler = 'gnu95'

    def get_flags_linker_so(self):
        if False:
            for i in range(10):
                print('nop')
        opt = self.linker_so[1:]
        if sys.platform == 'darwin':
            target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', None)
            if not target:
                import sysconfig
                target = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
                if not target:
                    target = '10.9'
                    s = f'Env. variable MACOSX_DEPLOYMENT_TARGET set to {target}'
                    warnings.warn(s, stacklevel=2)
                os.environ['MACOSX_DEPLOYMENT_TARGET'] = str(target)
            opt.extend(['-undefined', 'dynamic_lookup', '-bundle'])
        else:
            opt.append('-shared')
        if sys.platform.startswith('sunos'):
            opt.append('-mimpure-text')
        return opt

    def get_libgcc_dir(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            output = subprocess.check_output(self.compiler_f77 + ['-print-libgcc-file-name'])
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            output = filepath_from_subprocess_output(output)
            return os.path.dirname(output)
        return None

    def get_libgfortran_dir(self):
        if False:
            i = 10
            return i + 15
        if sys.platform[:5] == 'linux':
            libgfortran_name = 'libgfortran.so'
        elif sys.platform == 'darwin':
            libgfortran_name = 'libgfortran.dylib'
        else:
            libgfortran_name = None
        libgfortran_dir = None
        if libgfortran_name:
            find_lib_arg = ['-print-file-name={0}'.format(libgfortran_name)]
            try:
                output = subprocess.check_output(self.compiler_f77 + find_lib_arg)
            except (OSError, subprocess.CalledProcessError):
                pass
            else:
                output = filepath_from_subprocess_output(output)
                libgfortran_dir = os.path.dirname(output)
        return libgfortran_dir

    def get_library_dirs(self):
        if False:
            return 10
        opt = []
        if sys.platform[:5] != 'linux':
            d = self.get_libgcc_dir()
            if d:
                if sys.platform == 'win32' and (not d.startswith('/usr/lib')):
                    d = os.path.normpath(d)
                    path = os.path.join(d, 'lib%s.a' % self.g2c)
                    if not os.path.exists(path):
                        root = os.path.join(d, *(os.pardir,) * 4)
                        d2 = os.path.abspath(os.path.join(root, 'lib'))
                        path = os.path.join(d2, 'lib%s.a' % self.g2c)
                        if os.path.exists(path):
                            opt.append(d2)
                opt.append(d)
        lib_gfortran_dir = self.get_libgfortran_dir()
        if lib_gfortran_dir:
            opt.append(lib_gfortran_dir)
        return opt

    def get_libraries(self):
        if False:
            for i in range(10):
                print('nop')
        opt = []
        d = self.get_libgcc_dir()
        if d is not None:
            g2c = self.g2c + '-pic'
            f = self.static_lib_format % (g2c, self.static_lib_extension)
            if not os.path.isfile(os.path.join(d, f)):
                g2c = self.g2c
        else:
            g2c = self.g2c
        if g2c is not None:
            opt.append(g2c)
        c_compiler = self.c_compiler
        if sys.platform == 'win32' and c_compiler and (c_compiler.compiler_type == 'msvc'):
            opt.append('gcc')
        if sys.platform == 'darwin':
            opt.append('cc_dynamic')
        return opt

    def get_flags_debug(self):
        if False:
            return 10
        return ['-g']

    def get_flags_opt(self):
        if False:
            for i in range(10):
                print('nop')
        v = self.get_version()
        if v and v <= '3.3.3':
            opt = ['-O2']
        else:
            opt = ['-O3']
        opt.append('-funroll-loops')
        return opt

    def _c_arch_flags(self):
        if False:
            print('Hello World!')
        ' Return detected arch flags from CFLAGS '
        import sysconfig
        try:
            cflags = sysconfig.get_config_vars()['CFLAGS']
        except KeyError:
            return []
        arch_re = re.compile('-arch\\s+(\\w+)')
        arch_flags = []
        for arch in arch_re.findall(cflags):
            arch_flags += ['-arch', arch]
        return arch_flags

    def get_flags_arch(self):
        if False:
            return 10
        return []

    def runtime_library_dir_option(self, dir):
        if False:
            print('Hello World!')
        if sys.platform == 'win32' or sys.platform == 'cygwin':
            raise NotImplementedError
        assert ',' not in dir
        if sys.platform == 'darwin':
            return f'-Wl,-rpath,{dir}'
        elif sys.platform.startswith(('aix', 'os400')):
            return f'-Wl,-blibpath:{dir}'
        else:
            return f'-Wl,-rpath={dir}'

class Gnu95FCompiler(GnuFCompiler):
    compiler_type = 'gnu95'
    compiler_aliases = ('gfortran',)
    description = 'GNU Fortran 95 compiler'

    def version_match(self, version_string):
        if False:
            print('Hello World!')
        v = self.gnu_version_match(version_string)
        if not v or v[0] != 'gfortran':
            return None
        v = v[1]
        if LooseVersion(v) >= '4':
            pass
        elif sys.platform == 'win32':
            for key in ['version_cmd', 'compiler_f77', 'compiler_f90', 'compiler_fix', 'linker_so', 'linker_exe']:
                self.executables[key].append('-mno-cygwin')
        return v
    possible_executables = ['gfortran', 'f95']
    executables = {'version_cmd': ['<F90>', '-dumpversion'], 'compiler_f77': [None, '-Wall', '-g', '-ffixed-form', '-fno-second-underscore'], 'compiler_f90': [None, '-Wall', '-g', '-fno-second-underscore'], 'compiler_fix': [None, '-Wall', '-g', '-ffixed-form', '-fno-second-underscore'], 'linker_so': ['<F90>', '-Wall', '-g'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib'], 'linker_exe': [None, '-Wall']}
    module_dir_switch = '-J'
    module_include_switch = '-I'
    if sys.platform.startswith(('aix', 'os400')):
        executables['linker_so'].append('-lpthread')
        if platform.architecture()[0][:2] == '64':
            for key in ['compiler_f77', 'compiler_f90', 'compiler_fix', 'linker_so', 'linker_exe']:
                executables[key].append('-maix64')
    g2c = 'gfortran'

    def _universal_flags(self, cmd):
        if False:
            while True:
                i = 10
        'Return a list of -arch flags for every supported architecture.'
        if not sys.platform == 'darwin':
            return []
        arch_flags = []
        c_archs = self._c_arch_flags()
        if 'i386' in c_archs:
            c_archs[c_archs.index('i386')] = 'i686'
        for arch in ['ppc', 'i686', 'x86_64', 'ppc64', 's390x']:
            if _can_target(cmd, arch) and arch in c_archs:
                arch_flags.extend(['-arch', arch])
        return arch_flags

    def get_flags(self):
        if False:
            return 10
        flags = GnuFCompiler.get_flags(self)
        arch_flags = self._universal_flags(self.compiler_f90)
        if arch_flags:
            flags[:0] = arch_flags
        return flags

    def get_flags_linker_so(self):
        if False:
            print('Hello World!')
        flags = GnuFCompiler.get_flags_linker_so(self)
        arch_flags = self._universal_flags(self.linker_so)
        if arch_flags:
            flags[:0] = arch_flags
        return flags

    def get_library_dirs(self):
        if False:
            while True:
                i = 10
        opt = GnuFCompiler.get_library_dirs(self)
        if sys.platform == 'win32':
            c_compiler = self.c_compiler
            if c_compiler and c_compiler.compiler_type == 'msvc':
                target = self.get_target()
                if target:
                    d = os.path.normpath(self.get_libgcc_dir())
                    root = os.path.join(d, *(os.pardir,) * 4)
                    path = os.path.join(root, 'lib')
                    mingwdir = os.path.normpath(path)
                    if os.path.exists(os.path.join(mingwdir, 'libmingwex.a')):
                        opt.append(mingwdir)
        lib_gfortran_dir = self.get_libgfortran_dir()
        if lib_gfortran_dir:
            opt.append(lib_gfortran_dir)
        return opt

    def get_libraries(self):
        if False:
            print('Hello World!')
        opt = GnuFCompiler.get_libraries(self)
        if sys.platform == 'darwin':
            opt.remove('cc_dynamic')
        if sys.platform == 'win32':
            c_compiler = self.c_compiler
            if c_compiler and c_compiler.compiler_type == 'msvc':
                if 'gcc' in opt:
                    i = opt.index('gcc')
                    opt.insert(i + 1, 'mingwex')
                    opt.insert(i + 1, 'mingw32')
            c_compiler = self.c_compiler
            if c_compiler and c_compiler.compiler_type == 'msvc':
                return []
            else:
                pass
        return opt

    def get_target(self):
        if False:
            while True:
                i = 10
        try:
            p = subprocess.Popen(self.compiler_f77 + ['-v'], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            (stdout, stderr) = p.communicate()
            output = (stdout or b'') + (stderr or b'')
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            output = filepath_from_subprocess_output(output)
            m = TARGET_R.search(output)
            if m:
                return m.group(1)
        return ''

    def _hash_files(self, filenames):
        if False:
            i = 10
            return i + 15
        h = hashlib.sha1()
        for fn in filenames:
            with open(fn, 'rb') as f:
                while True:
                    block = f.read(131072)
                    if not block:
                        break
                    h.update(block)
        text = base64.b32encode(h.digest())
        text = text.decode('ascii')
        return text.rstrip('=')

    def _link_wrapper_lib(self, objects, output_dir, extra_dll_dir, chained_dlls, is_archive):
        if False:
            return 10
        'Create a wrapper shared library for the given objects\n\n        Return an MSVC-compatible lib\n        '
        c_compiler = self.c_compiler
        if c_compiler.compiler_type != 'msvc':
            raise ValueError('This method only supports MSVC')
        object_hash = self._hash_files(list(objects) + list(chained_dlls))
        if is_win64():
            tag = 'win_amd64'
        else:
            tag = 'win32'
        basename = 'lib' + os.path.splitext(os.path.basename(objects[0]))[0][:8]
        root_name = basename + '.' + object_hash + '.gfortran-' + tag
        dll_name = root_name + '.dll'
        def_name = root_name + '.def'
        lib_name = root_name + '.lib'
        dll_path = os.path.join(extra_dll_dir, dll_name)
        def_path = os.path.join(output_dir, def_name)
        lib_path = os.path.join(output_dir, lib_name)
        if os.path.isfile(lib_path):
            return (lib_path, dll_path)
        if is_archive:
            objects = ['-Wl,--whole-archive'] + list(objects) + ['-Wl,--no-whole-archive']
        self.link_shared_object(objects, dll_name, output_dir=extra_dll_dir, extra_postargs=list(chained_dlls) + ['-Wl,--allow-multiple-definition', '-Wl,--output-def,' + def_path, '-Wl,--export-all-symbols', '-Wl,--enable-auto-import', '-static', '-mlong-double-64'])
        if is_win64():
            specifier = '/MACHINE:X64'
        else:
            specifier = '/MACHINE:X86'
        lib_args = ['/def:' + def_path, '/OUT:' + lib_path, specifier]
        if not c_compiler.initialized:
            c_compiler.initialize()
        c_compiler.spawn([c_compiler.lib] + lib_args)
        return (lib_path, dll_path)

    def can_ccompiler_link(self, compiler):
        if False:
            i = 10
            return i + 15
        return compiler.compiler_type not in ('msvc',)

    def wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert a set of object files that are not compatible with the default\n        linker, to a file that is compatible.\n        '
        if self.c_compiler.compiler_type == 'msvc':
            archives = []
            plain_objects = []
            for obj in objects:
                if obj.lower().endswith('.a'):
                    archives.append(obj)
                else:
                    plain_objects.append(obj)
            chained_libs = []
            chained_dlls = []
            for archive in archives[::-1]:
                (lib, dll) = self._link_wrapper_lib([archive], output_dir, extra_dll_dir, chained_dlls=chained_dlls, is_archive=True)
                chained_libs.insert(0, lib)
                chained_dlls.insert(0, dll)
            if not plain_objects:
                return chained_libs
            (lib, dll) = self._link_wrapper_lib(plain_objects, output_dir, extra_dll_dir, chained_dlls=chained_dlls, is_archive=False)
            return [lib] + chained_libs
        else:
            raise ValueError('Unsupported C compiler')

def _can_target(cmd, arch):
    if False:
        print('Hello World!')
    'Return true if the architecture supports the -arch flag'
    newcmd = cmd[:]
    (fid, filename) = tempfile.mkstemp(suffix='.f')
    os.close(fid)
    try:
        d = os.path.dirname(filename)
        output = os.path.splitext(filename)[0] + '.o'
        try:
            newcmd.extend(['-arch', arch, '-c', filename])
            p = Popen(newcmd, stderr=STDOUT, stdout=PIPE, cwd=d)
            p.communicate()
            return p.returncode == 0
        finally:
            if os.path.exists(output):
                os.remove(output)
    finally:
        os.remove(filename)
if __name__ == '__main__':
    from distutils import log
    from numpy.distutils import customized_fcompiler
    log.set_verbosity(2)
    print(customized_fcompiler('gnu').get_version())
    try:
        print(customized_fcompiler('g95').get_version())
    except Exception as e:
        print(e)