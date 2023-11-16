import os
from spack.compiler import Compiler

class Nvhpc(Compiler):
    cc_names = ['nvc']
    cxx_names = ['nvc++']
    f77_names = ['nvfortran']
    fc_names = ['nvfortran']
    link_paths = {'cc': os.path.join('nvhpc', 'nvc'), 'cxx': os.path.join('nvhpc', 'nvc++'), 'f77': os.path.join('nvhpc', 'nvfortran'), 'fc': os.path.join('nvhpc', 'nvfortran')}
    PrgEnv = 'PrgEnv-nvhpc'
    PrgEnv_compiler = 'nvhpc'
    version_argument = '--version'
    version_regex = 'nv[^ ]* (?:[^ ]+ Dev-r)?([0-9.]+)(?:-[0-9]+)?'

    @property
    def verbose_flag(self):
        if False:
            while True:
                i = 10
        return '-v'

    @property
    def debug_flags(self):
        if False:
            for i in range(10):
                print('nop')
        return ['-g', '-gopt']

    @property
    def opt_flags(self):
        if False:
            print('Hello World!')
        return ['-O', '-O0', '-O1', '-O2', '-O3', '-O4']

    @property
    def openmp_flag(self):
        if False:
            i = 10
            return i + 15
        return '-mp'

    @property
    def cc_pic_flag(self):
        if False:
            for i in range(10):
                print('nop')
        return '-fpic'

    @property
    def cxx_pic_flag(self):
        if False:
            i = 10
            return i + 15
        return '-fpic'

    @property
    def f77_pic_flag(self):
        if False:
            i = 10
            return i + 15
        return '-fpic'

    @property
    def fc_pic_flag(self):
        if False:
            while True:
                i = 10
        return '-fpic'

    @property
    def c99_flag(self):
        if False:
            for i in range(10):
                print('nop')
        return '-c99'

    @property
    def c11_flag(self):
        if False:
            print('Hello World!')
        return '-c11'

    @property
    def cxx11_flag(self):
        if False:
            for i in range(10):
                print('nop')
        return '--c++11'

    @property
    def cxx14_flag(self):
        if False:
            while True:
                i = 10
        return '--c++14'

    @property
    def cxx17_flag(self):
        if False:
            print('Hello World!')
        return '--c++17'

    @property
    def stdcxx_libs(self):
        if False:
            i = 10
            return i + 15
        return ('-c++libs',)
    required_libs = ['libnvc', 'libnvf']