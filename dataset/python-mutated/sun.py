from numpy.distutils.ccompiler import simple_version_match
from numpy.distutils.fcompiler import FCompiler
compilers = ['SunFCompiler']

class SunFCompiler(FCompiler):
    compiler_type = 'sun'
    description = 'Sun or Forte Fortran 95 Compiler'
    version_match = simple_version_match(start='f9[05]: (Sun|Forte|WorkShop).*Fortran 95')
    executables = {'version_cmd': ['<F90>', '-V'], 'compiler_f77': ['f90'], 'compiler_fix': ['f90', '-fixed'], 'compiler_f90': ['f90'], 'linker_so': ['<F90>', '-Bdynamic', '-G'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}
    module_dir_switch = '-moddir='
    module_include_switch = '-M'
    pic_flags = ['-xcode=pic32']

    def get_flags_f77(self):
        if False:
            while True:
                i = 10
        ret = ['-ftrap=%none']
        if (self.get_version() or '') >= '7':
            ret.append('-f77')
        else:
            ret.append('-fixed')
        return ret

    def get_opt(self):
        if False:
            for i in range(10):
                print('nop')
        return ['-fast', '-dalign']

    def get_arch(self):
        if False:
            i = 10
            return i + 15
        return ['-xtarget=generic']

    def get_libraries(self):
        if False:
            for i in range(10):
                print('nop')
        opt = []
        opt.extend(['fsu', 'sunmath', 'mvec'])
        return opt

    def runtime_library_dir_option(self, dir):
        if False:
            i = 10
            return i + 15
        return '-R%s' % dir
if __name__ == '__main__':
    from distutils import log
    log.set_verbosity(2)
    from numpy.distutils import customized_fcompiler
    print(customized_fcompiler(compiler='sun').get_version())