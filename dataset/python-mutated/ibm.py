import os
import re
import sys
import subprocess
from numpy.distutils.fcompiler import FCompiler
from numpy.distutils.exec_command import find_executable
from numpy.distutils.misc_util import make_temp_file
from distutils import log
compilers = ['IBMFCompiler']

class IBMFCompiler(FCompiler):
    compiler_type = 'ibm'
    description = 'IBM XL Fortran Compiler'
    version_pattern = '(xlf\\(1\\)\\s*|)IBM XL Fortran ((Advanced Edition |)Version |Enterprise Edition V|for AIX, V)(?P<version>[^\\s*]*)'
    executables = {'version_cmd': ['<F77>', '-qversion'], 'compiler_f77': ['xlf'], 'compiler_fix': ['xlf90', '-qfixed'], 'compiler_f90': ['xlf90'], 'linker_so': ['xlf95'], 'archiver': ['ar', '-cr'], 'ranlib': ['ranlib']}

    def get_version(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        version = FCompiler.get_version(self, *args, **kwds)
        if version is None and sys.platform.startswith('aix'):
            lslpp = find_executable('lslpp')
            xlf = find_executable('xlf')
            if os.path.exists(xlf) and os.path.exists(lslpp):
                try:
                    o = subprocess.check_output([lslpp, '-Lc', 'xlfcmp'])
                except (OSError, subprocess.CalledProcessError):
                    pass
                else:
                    m = re.search('xlfcmp:(?P<version>\\d+([.]\\d+)+)', o)
                    if m:
                        version = m.group('version')
        xlf_dir = '/etc/opt/ibmcmp/xlf'
        if version is None and os.path.isdir(xlf_dir):
            l = sorted(os.listdir(xlf_dir))
            l.reverse()
            l = [d for d in l if os.path.isfile(os.path.join(xlf_dir, d, 'xlf.cfg'))]
            if l:
                from distutils.version import LooseVersion
                self.version = version = LooseVersion(l[0])
        return version

    def get_flags(self):
        if False:
            while True:
                i = 10
        return ['-qextname']

    def get_flags_debug(self):
        if False:
            return 10
        return ['-g']

    def get_flags_linker_so(self):
        if False:
            i = 10
            return i + 15
        opt = []
        if sys.platform == 'darwin':
            opt.append('-Wl,-bundle,-flat_namespace,-undefined,suppress')
        else:
            opt.append('-bshared')
        version = self.get_version(ok_status=[0, 40])
        if version is not None:
            if sys.platform.startswith('aix'):
                xlf_cfg = '/etc/xlf.cfg'
            else:
                xlf_cfg = '/etc/opt/ibmcmp/xlf/%s/xlf.cfg' % version
            (fo, new_cfg) = make_temp_file(suffix='_xlf.cfg')
            log.info('Creating ' + new_cfg)
            with open(xlf_cfg) as fi:
                crt1_match = re.compile('\\s*crt\\s*=\\s*(?P<path>.*)/crt1.o').match
                for line in fi:
                    m = crt1_match(line)
                    if m:
                        fo.write('crt = %s/bundle1.o\n' % m.group('path'))
                    else:
                        fo.write(line)
            fo.close()
            opt.append('-F' + new_cfg)
        return opt

    def get_flags_opt(self):
        if False:
            while True:
                i = 10
        return ['-O3']
if __name__ == '__main__':
    from numpy.distutils import customized_fcompiler
    log.set_verbosity(2)
    print(customized_fcompiler(compiler='ibm').get_version())