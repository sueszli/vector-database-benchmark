import os
from spack.compiler import Compiler, UnsupportedCompilerFlag
from spack.version import Version

class Pgi(Compiler):
    cc_names = ['pgcc']
    cxx_names = ['pgc++', 'pgCC']
    f77_names = ['pgfortran', 'pgf77']
    fc_names = ['pgfortran', 'pgf95', 'pgf90']
    link_paths = {'cc': os.path.join('pgi', 'pgcc'), 'cxx': os.path.join('pgi', 'pgc++'), 'f77': os.path.join('pgi', 'pgfortran'), 'fc': os.path.join('pgi', 'pgfortran')}
    PrgEnv = 'PrgEnv-pgi'
    PrgEnv_compiler = 'pgi'
    version_argument = '-V'
    ignore_version_errors = [2]
    version_regex = 'pg[^ ]* ([0-9.]+)-[0-9]+ (LLVM )?[^ ]+ target on '

    @property
    def verbose_flag(self):
        if False:
            while True:
                i = 10
        return '-v'

    @property
    def debug_flags(self):
        if False:
            print('Hello World!')
        return ['-g', '-gopt']

    @property
    def opt_flags(self):
        if False:
            for i in range(10):
                print('nop')
        return ['-O', '-O0', '-O1', '-O2', '-O3', '-O4']

    @property
    def openmp_flag(self):
        if False:
            return 10
        return '-mp'

    @property
    def cxx11_flag(self):
        if False:
            print('Hello World!')
        return '-std=c++11'

    @property
    def cc_pic_flag(self):
        if False:
            print('Hello World!')
        return '-fpic'

    @property
    def cxx_pic_flag(self):
        if False:
            print('Hello World!')
        return '-fpic'

    @property
    def f77_pic_flag(self):
        if False:
            while True:
                i = 10
        return '-fpic'

    @property
    def fc_pic_flag(self):
        if False:
            for i in range(10):
                print('nop')
        return '-fpic'
    required_libs = ['libpgc', 'libpgf90']

    @property
    def c99_flag(self):
        if False:
            while True:
                i = 10
        if self.real_version >= Version('12.10'):
            return '-c99'
        raise UnsupportedCompilerFlag(self, 'the C99 standard', 'c99_flag', '< 12.10')

    @property
    def c11_flag(self):
        if False:
            for i in range(10):
                print('nop')
        if self.real_version >= Version('15.3'):
            return '-c11'
        raise UnsupportedCompilerFlag(self, 'the C11 standard', 'c11_flag', '< 15.3')

    @property
    def stdcxx_libs(self):
        if False:
            i = 10
            return i + 15
        return ('-pgc++libs',)