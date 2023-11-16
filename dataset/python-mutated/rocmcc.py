import re
import llnl.util.lang
import spack.compilers.clang

class Rocmcc(spack.compilers.clang.Clang):
    cc_names = ['amdclang']
    cxx_names = ['amdclang++']
    f77_names = ['amdflang']
    fc_names = ['amdflang']
    PrgEnv = 'PrgEnv-amd'
    PrgEnv_compiler = 'amd'

    @property
    def link_paths(self):
        if False:
            while True:
                i = 10
        link_paths = {'cc': 'rocmcc/amdclang', 'cxx': 'rocmcc/amdclang++', 'f77': 'rocmcc/amdflang', 'fc': 'rocmcc/amdflang'}
        return link_paths

    @property
    def cxx11_flag(self):
        if False:
            i = 10
            return i + 15
        return '-std=c++11'

    @property
    def cxx14_flag(self):
        if False:
            print('Hello World!')
        return '-std=c++14'

    @property
    def cxx17_flag(self):
        if False:
            return 10
        return '-std=c++17'

    @property
    def c99_flag(self):
        if False:
            i = 10
            return i + 15
        return '-std=c99'

    @property
    def c11_flag(self):
        if False:
            for i in range(10):
                print('nop')
        return '-std=c11'

    @classmethod
    @llnl.util.lang.memoized
    def extract_version_from_output(cls, output):
        if False:
            for i in range(10):
                print('nop')
        match = re.search('llvm-project roc-(\\d+)[._](\\d+)[._](\\d+)', output)
        if match:
            return '.'.join(match.groups())

    @classmethod
    def fc_version(cls, fortran_compiler):
        if False:
            while True:
                i = 10
        return cls.default_version(fortran_compiler)

    @classmethod
    def f77_version(cls, f77):
        if False:
            print('Hello World!')
        return cls.fc_version(f77)

    @property
    def stdcxx_libs(self):
        if False:
            print('Hello World!')
        return ('-lstdc++',)