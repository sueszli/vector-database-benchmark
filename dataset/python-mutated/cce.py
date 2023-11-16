import os
from spack.compiler import Compiler, UnsupportedCompilerFlag
from spack.version import Version

class Cce(Compiler):
    """Cray compiler environment compiler."""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        if not self.is_clang_based:
            self.version_argument = '-V'
    cc_names = ['craycc', 'cc']
    cxx_names = ['crayCC', 'CC']
    f77_names = ['crayftn', 'ftn']
    fc_names = ['crayftn', 'ftn']
    suffixes = ['-mp-\\d\\.\\d']
    PrgEnv = 'PrgEnv-cray'
    PrgEnv_compiler = 'cce'

    @property
    def link_paths(self):
        if False:
            return 10
        if any((self.PrgEnv in m for m in self.modules)):
            return {'cc': os.path.join('cce', 'cc'), 'cxx': os.path.join('case-insensitive', 'CC'), 'f77': os.path.join('cce', 'ftn'), 'fc': os.path.join('cce', 'ftn')}
        return {'cc': os.path.join('cce', 'craycc'), 'cxx': os.path.join('cce', 'case-insensitive', 'crayCC'), 'f77': os.path.join('cce', 'crayftn'), 'fc': os.path.join('cce', 'crayftn')}

    @property
    def is_clang_based(self):
        if False:
            for i in range(10):
                print('nop')
        version = self._real_version or self.version
        return version >= Version('9.0') and 'classic' not in str(version)
    version_argument = '--version'
    version_regex = '[Cc]ray (?:clang|C :|C\\+\\+ :|Fortran :) [Vv]ersion.*?(\\d+(\\.\\d+)+)'

    @property
    def verbose_flag(self):
        if False:
            i = 10
            return i + 15
        return '-v'

    @property
    def debug_flags(self):
        if False:
            while True:
                i = 10
        return ['-g', '-G0', '-G1', '-G2', '-Gfast']

    @property
    def openmp_flag(self):
        if False:
            i = 10
            return i + 15
        if self.is_clang_based:
            return '-fopenmp'
        return '-h omp'

    @property
    def cxx11_flag(self):
        if False:
            i = 10
            return i + 15
        if self.is_clang_based:
            return '-std=c++11'
        return '-h std=c++11'

    @property
    def cxx14_flag(self):
        if False:
            while True:
                i = 10
        if self.is_clang_based:
            return '-std=c++14'
        return '-h std=c++14'

    @property
    def cxx17_flag(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_clang_based:
            return '-std=c++17'

    @property
    def c99_flag(self):
        if False:
            while True:
                i = 10
        if self.is_clang_based:
            return '-std=c99'
        elif self.real_version >= Version('8.4'):
            return '-h std=c99,noconform,gnu'
        elif self.real_version >= Version('8.1'):
            return '-h c99,noconform,gnu'
        raise UnsupportedCompilerFlag(self, 'the C99 standard', 'c99_flag', '< 8.1')

    @property
    def c11_flag(self):
        if False:
            i = 10
            return i + 15
        if self.is_clang_based:
            return '-std=c11'
        elif self.real_version >= Version('8.5'):
            return '-h std=c11,noconform,gnu'
        raise UnsupportedCompilerFlag(self, 'the C11 standard', 'c11_flag', '< 8.5')

    @property
    def cc_pic_flag(self):
        if False:
            while True:
                i = 10
        if self.is_clang_based:
            return '-fPIC'
        return '-h PIC'

    @property
    def cxx_pic_flag(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_clang_based:
            return '-fPIC'
        return '-h PIC'

    @property
    def f77_pic_flag(self):
        if False:
            return 10
        if self.is_clang_based:
            return '-fPIC'
        return '-h PIC'

    @property
    def fc_pic_flag(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_clang_based:
            return '-fPIC'
        return '-h PIC'

    @property
    def stdcxx_libs(self):
        if False:
            return 10
        return ()