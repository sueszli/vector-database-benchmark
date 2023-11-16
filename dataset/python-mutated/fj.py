import os
import spack.compiler

class Fj(spack.compiler.Compiler):
    cc_names = ['fcc']
    cxx_names = ['FCC']
    f77_names = ['frt']
    fc_names = ['frt']
    link_paths = {'cc': os.path.join('fj', 'fcc'), 'cxx': os.path.join('fj', 'case-insensitive', 'FCC'), 'f77': os.path.join('fj', 'frt'), 'fc': os.path.join('fj', 'frt')}
    version_argument = '--version'
    version_regex = '\\((?:FCC|FRT)\\) ([a-z\\d.]+)'
    required_libs = ['libfj90i', 'libfj90f', 'libfjsrcinfo']

    @property
    def verbose_flag(self):
        if False:
            print('Hello World!')
        return '-v'

    @property
    def debug_flags(self):
        if False:
            for i in range(10):
                print('nop')
        return '-g'

    @property
    def opt_flags(self):
        if False:
            print('Hello World!')
        return ['-O0', '-O1', '-O2', '-O3', '-Ofast']

    @property
    def openmp_flag(self):
        if False:
            while True:
                i = 10
        return '-Kopenmp'

    @property
    def cxx98_flag(self):
        if False:
            return 10
        return '-std=c++98'

    @property
    def cxx11_flag(self):
        if False:
            while True:
                i = 10
        return '-std=c++11'

    @property
    def cxx14_flag(self):
        if False:
            print('Hello World!')
        return '-std=c++14'

    @property
    def cxx17_flag(self):
        if False:
            print('Hello World!')
        return '-std=c++17'

    @property
    def c99_flag(self):
        if False:
            return 10
        return '-std=c99'

    @property
    def c11_flag(self):
        if False:
            print('Hello World!')
        return '-std=c11'

    @property
    def cc_pic_flag(self):
        if False:
            print('Hello World!')
        return '-KPIC'

    @property
    def cxx_pic_flag(self):
        if False:
            for i in range(10):
                print('nop')
        return '-KPIC'

    @property
    def f77_pic_flag(self):
        if False:
            for i in range(10):
                print('nop')
        return '-KPIC'

    @property
    def fc_pic_flag(self):
        if False:
            i = 10
            return i + 15
        return '-KPIC'