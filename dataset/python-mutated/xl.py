import os
from spack.compiler import Compiler, UnsupportedCompilerFlag
from spack.version import Version

class Xl(Compiler):
    cc_names = ['xlc']
    cxx_names = ['xlC', 'xlc++']
    f77_names = ['xlf']
    fc_names = ['xlf90', 'xlf95', 'xlf2003', 'xlf2008']
    link_paths = {'cc': os.path.join('xl', 'xlc'), 'cxx': os.path.join('xl', 'xlc++'), 'f77': os.path.join('xl', 'xlf'), 'fc': os.path.join('xl', 'xlf90')}
    version_argument = '-qversion'
    version_regex = '([0-9]?[0-9]\\.[0-9])'

    @property
    def verbose_flag(self):
        if False:
            for i in range(10):
                print('nop')
        return '-V'

    @property
    def debug_flags(self):
        if False:
            for i in range(10):
                print('nop')
        return ['-g', '-g0', '-g1', '-g2', '-g8', '-g9']

    @property
    def opt_flags(self):
        if False:
            return 10
        return ['-O', '-O0', '-O1', '-O2', '-O3', '-O4', '-O5', '-Ofast']

    @property
    def openmp_flag(self):
        if False:
            while True:
                i = 10
        return '-qsmp=omp'

    @property
    def cxx11_flag(self):
        if False:
            print('Hello World!')
        if self.real_version < Version('13.1'):
            raise UnsupportedCompilerFlag(self, 'the C++11 standard', 'cxx11_flag', '< 13.1')
        else:
            return '-qlanglvl=extended0x'

    @property
    def c99_flag(self):
        if False:
            for i in range(10):
                print('nop')
        if self.real_version >= Version('13.1.1'):
            return '-std=gnu99'
        if self.real_version >= Version('10.1'):
            return '-qlanglvl=extc99'
        raise UnsupportedCompilerFlag(self, 'the C99 standard', 'c99_flag', '< 10.1')

    @property
    def c11_flag(self):
        if False:
            return 10
        if self.real_version >= Version('13.1.2'):
            return '-std=gnu11'
        if self.real_version >= Version('12.1'):
            return '-qlanglvl=extc1x'
        raise UnsupportedCompilerFlag(self, 'the C11 standard', 'c11_flag', '< 12.1')

    @property
    def cxx14_flag(self):
        if False:
            return 10
        if self.version >= Version('16.1.1.8'):
            return '-std=c++14'
        raise UnsupportedCompilerFlag(self, 'the C++14 standard', 'cxx14_flag', '< 16.1.1.8')

    @property
    def cc_pic_flag(self):
        if False:
            while True:
                i = 10
        return '-qpic'

    @property
    def cxx_pic_flag(self):
        if False:
            for i in range(10):
                print('nop')
        return '-qpic'

    @property
    def f77_pic_flag(self):
        if False:
            while True:
                i = 10
        return '-qpic'

    @property
    def fc_pic_flag(self):
        if False:
            i = 10
            return i + 15
        return '-qpic'

    @property
    def fflags(self):
        if False:
            return 10
        return '-qzerosize'

    @classmethod
    def fc_version(cls, fc):
        if False:
            i = 10
            return i + 15
        fortran_version = cls.default_version(fc)
        if fortran_version >= 16:
            return str(fortran_version)
        c_version = float(fortran_version) - 2
        if c_version < 10:
            c_version = c_version - 0.1
        return str(c_version)

    @classmethod
    def f77_version(cls, f77):
        if False:
            i = 10
            return i + 15
        return cls.fc_version(f77)