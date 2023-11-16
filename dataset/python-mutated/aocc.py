import os
import re
import llnl.util.lang
from spack.compiler import Compiler
from spack.version import ver

class Aocc(Compiler):
    cc_names = ['clang']
    cxx_names = ['clang++']
    f77_names = ['flang']
    fc_names = ['flang']
    PrgEnv = 'PrgEnv-aocc'
    PrgEnv_compiler = 'aocc'
    version_argument = '--version'

    @property
    def debug_flags(self):
        if False:
            while True:
                i = 10
        return ['-gcodeview', '-gdwarf-2', '-gdwarf-3', '-gdwarf-4', '-gdwarf-5', '-gline-tables-only', '-gmodules', '-g']

    @property
    def opt_flags(self):
        if False:
            i = 10
            return i + 15
        return ['-O0', '-O1', '-O2', '-O3', '-Ofast', '-Os', '-Oz', '-Og', '-O', '-O4']

    @property
    def link_paths(self):
        if False:
            i = 10
            return i + 15
        link_paths = {'cc': os.path.join('aocc', 'clang'), 'cxx': os.path.join('aocc', 'clang++'), 'f77': os.path.join('aocc', 'flang'), 'fc': os.path.join('aocc', 'flang')}
        return link_paths

    @property
    def verbose_flag(self):
        if False:
            print('Hello World!')
        return '-v'

    @property
    def openmp_flag(self):
        if False:
            for i in range(10):
                print('nop')
        return '-fopenmp'

    @property
    def cxx11_flag(self):
        if False:
            print('Hello World!')
        return '-std=c++11'

    @property
    def cxx14_flag(self):
        if False:
            i = 10
            return i + 15
        return '-std=c++14'

    @property
    def cxx17_flag(self):
        if False:
            i = 10
            return i + 15
        return '-std=c++17'

    @property
    def c99_flag(self):
        if False:
            return 10
        return '-std=c99'

    @property
    def c11_flag(self):
        if False:
            return 10
        return '-std=c11'

    @property
    def cc_pic_flag(self):
        if False:
            print('Hello World!')
        return '-fPIC'

    @property
    def cxx_pic_flag(self):
        if False:
            print('Hello World!')
        return '-fPIC'

    @property
    def f77_pic_flag(self):
        if False:
            return 10
        return '-fPIC'

    @property
    def fc_pic_flag(self):
        if False:
            return 10
        return '-fPIC'
    required_libs = ['libclang']

    @classmethod
    @llnl.util.lang.memoized
    def extract_version_from_output(cls, output):
        if False:
            print('Hello World!')
        match = re.search('AOCC_(\\d+)[._](\\d+)[._](\\d+)', output)
        if match:
            return '.'.join(match.groups())
        return 'unknown'

    @property
    def stdcxx_libs(self):
        if False:
            for i in range(10):
                print('nop')
        return ('-lstdc++',)

    @property
    def cflags(self):
        if False:
            for i in range(10):
                print('nop')
        return self._handle_default_flag_addtions()

    @property
    def cxxflags(self):
        if False:
            while True:
                i = 10
        return self._handle_default_flag_addtions()

    @property
    def fflags(self):
        if False:
            for i in range(10):
                print('nop')
        return self._handle_default_flag_addtions()

    def _handle_default_flag_addtions(self):
        if False:
            print('Hello World!')
        if self.real_version.satisfies(ver('3.0.0')):
            return '-Wno-unused-command-line-argument -mllvm -eliminate-similar-expr=false'