import re
import llnl.util.lang
import spack.compiler
import spack.compilers.clang
import spack.util.executable
from spack.version import Version

class AppleClang(spack.compilers.clang.Clang):
    openmp_flag = '-Xpreprocessor -fopenmp'

    @classmethod
    @llnl.util.lang.memoized
    def extract_version_from_output(cls, output):
        if False:
            print('Hello World!')
        ver = 'unknown'
        match = re.search('^Apple (?:LLVM|clang) version ([^ )]+)', output, re.M)
        if match:
            ver = match.group(match.lastindex)
        return ver

    @property
    def cxx11_flag(self):
        if False:
            while True:
                i = 10
        if self.real_version < Version('4.0'):
            raise spack.compiler.UnsupportedCompilerFlag(self, 'the C++11 standard', 'cxx11_flag', 'Xcode < 4.0')
        return '-std=c++11'

    @property
    def cxx14_flag(self):
        if False:
            for i in range(10):
                print('nop')
        if self.real_version < Version('5.1'):
            raise spack.compiler.UnsupportedCompilerFlag(self, 'the C++14 standard', 'cxx14_flag', 'Xcode < 5.1')
        elif self.real_version < Version('6.1'):
            return '-std=c++1y'
        return '-std=c++14'

    @property
    def cxx17_flag(self):
        if False:
            for i in range(10):
                print('nop')
        if self.real_version < Version('6.1'):
            raise spack.compiler.UnsupportedCompilerFlag(self, 'the C++17 standard', 'cxx17_flag', 'Xcode < 6.1')
        elif self.real_version < Version('10.0'):
            return '-std=c++1z'
        return '-std=c++17'

    @property
    def cxx20_flag(self):
        if False:
            return 10
        if self.real_version < Version('10.0'):
            raise spack.compiler.UnsupportedCompilerFlag(self, 'the C++20 standard', 'cxx20_flag', 'Xcode < 10.0')
        elif self.real_version < Version('13.0'):
            return '-std=c++2a'
        return '-std=c++20'

    @property
    def cxx23_flag(self):
        if False:
            return 10
        if self.real_version < Version('13.0'):
            raise spack.compiler.UnsupportedCompilerFlag(self, 'the C++23 standard', 'cxx23_flag', 'Xcode < 13.0')
        return '-std=c++2b'

    @property
    def c99_flag(self):
        if False:
            print('Hello World!')
        if self.real_version < Version('4.0'):
            raise spack.compiler.UnsupportedCompilerFlag(self, 'the C99 standard', 'c99_flag', '< 4.0')
        return '-std=c99'

    @property
    def c11_flag(self):
        if False:
            while True:
                i = 10
        if self.real_version < Version('4.0'):
            raise spack.compiler.UnsupportedCompilerFlag(self, 'the C11 standard', 'c11_flag', '< 4.0')
        return '-std=c11'

    @property
    def c17_flag(self):
        if False:
            print('Hello World!')
        if self.real_version < Version('11.0'):
            raise spack.compiler.UnsupportedCompilerFlag(self, 'the C17 standard', 'c17_flag', '< 11.0')
        return '-std=c17'

    @property
    def c23_flag(self):
        if False:
            while True:
                i = 10
        if self.real_version < Version('11.0.3'):
            raise spack.compiler.UnsupportedCompilerFlag(self, 'the C23 standard', 'c23_flag', '< 11.0.3')
        return '-std=c2x'