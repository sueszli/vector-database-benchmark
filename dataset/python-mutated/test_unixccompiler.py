"""Tests for distutils.unixccompiler."""
import sys
import unittest
from test.support import run_unittest
from test.support.os_helper import EnvironmentVarGuard
from distutils import sysconfig
from distutils.unixccompiler import UnixCCompiler

class UnixCCompilerTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self._backup_platform = sys.platform
        self._backup_get_config_var = sysconfig.get_config_var
        self._backup_config_vars = dict(sysconfig._config_vars)

        class CompilerWrapper(UnixCCompiler):

            def rpath_foo(self):
                if False:
                    while True:
                        i = 10
                return self.runtime_library_dir_option('/foo')
        self.cc = CompilerWrapper()

    def tearDown(self):
        if False:
            while True:
                i = 10
        sys.platform = self._backup_platform
        sysconfig.get_config_var = self._backup_get_config_var
        sysconfig._config_vars.clear()
        sysconfig._config_vars.update(self._backup_config_vars)

    @unittest.skipIf(sys.platform == 'win32', "can't test on Windows")
    def test_runtime_libdir_option(self):
        if False:
            return 10
        sys.platform = 'darwin'
        self.assertEqual(self.cc.rpath_foo(), '-L/foo')
        sys.platform = 'hp-ux'
        old_gcv = sysconfig.get_config_var

        def gcv(v):
            if False:
                print('Hello World!')
            return 'xxx'
        sysconfig.get_config_var = gcv
        self.assertEqual(self.cc.rpath_foo(), ['+s', '-L/foo'])

        def gcv(v):
            if False:
                for i in range(10):
                    print('nop')
            return 'gcc'
        sysconfig.get_config_var = gcv
        self.assertEqual(self.cc.rpath_foo(), ['-Wl,+s', '-L/foo'])

        def gcv(v):
            if False:
                i = 10
                return i + 15
            return 'g++'
        sysconfig.get_config_var = gcv
        self.assertEqual(self.cc.rpath_foo(), ['-Wl,+s', '-L/foo'])
        sysconfig.get_config_var = old_gcv
        sys.platform = 'bar'

        def gcv(v):
            if False:
                return 10
            if v == 'CC':
                return 'gcc'
            elif v == 'GNULD':
                return 'yes'
        sysconfig.get_config_var = gcv
        self.assertEqual(self.cc.rpath_foo(), '-Wl,--enable-new-dtags,-R/foo')
        sys.platform = 'bar'

        def gcv(v):
            if False:
                print('Hello World!')
            if v == 'CC':
                return 'gcc'
            elif v == 'GNULD':
                return 'no'
        sysconfig.get_config_var = gcv
        self.assertEqual(self.cc.rpath_foo(), '-Wl,-R/foo')
        sys.platform = 'bar'

        def gcv(v):
            if False:
                for i in range(10):
                    print('nop')
            if v == 'CC':
                return 'x86_64-pc-linux-gnu-gcc-4.4.2'
            elif v == 'GNULD':
                return 'yes'
        sysconfig.get_config_var = gcv
        self.assertEqual(self.cc.rpath_foo(), '-Wl,--enable-new-dtags,-R/foo')
        sys.platform = 'bar'

        def gcv(v):
            if False:
                return 10
            if v == 'CC':
                return 'cc'
            elif v == 'GNULD':
                return 'yes'
        sysconfig.get_config_var = gcv
        self.assertEqual(self.cc.rpath_foo(), '-R/foo')
        sys.platform = 'bar'

        def gcv(v):
            if False:
                return 10
            if v == 'CC':
                return 'cc'
            elif v == 'GNULD':
                return 'no'
        sysconfig.get_config_var = gcv
        self.assertEqual(self.cc.rpath_foo(), '-R/foo')

    @unittest.skipUnless(sys.platform == 'darwin', 'test only relevant for OS X')
    def test_osx_cc_overrides_ldshared(self):
        if False:
            while True:
                i = 10

        def gcv(v):
            if False:
                for i in range(10):
                    print('nop')
            if v == 'LDSHARED':
                return 'gcc-4.2 -bundle -undefined dynamic_lookup '
            return 'gcc-4.2'
        sysconfig.get_config_var = gcv
        with EnvironmentVarGuard() as env:
            env['CC'] = 'my_cc'
            del env['LDSHARED']
            sysconfig.customize_compiler(self.cc)
        self.assertEqual(self.cc.linker_so[0], 'my_cc')

    @unittest.skipUnless(sys.platform == 'darwin', 'test only relevant for OS X')
    def test_osx_explicit_ldshared(self):
        if False:
            i = 10
            return i + 15

        def gcv(v):
            if False:
                i = 10
                return i + 15
            if v == 'LDSHARED':
                return 'gcc-4.2 -bundle -undefined dynamic_lookup '
            return 'gcc-4.2'
        sysconfig.get_config_var = gcv
        with EnvironmentVarGuard() as env:
            env['CC'] = 'my_cc'
            env['LDSHARED'] = 'my_ld -bundle -dynamic'
            sysconfig.customize_compiler(self.cc)
        self.assertEqual(self.cc.linker_so[0], 'my_ld')

def test_suite():
    if False:
        print('Hello World!')
    return unittest.makeSuite(UnixCCompilerTestCase)
if __name__ == '__main__':
    run_unittest(test_suite())