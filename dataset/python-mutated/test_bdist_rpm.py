"""Tests for distutils.command.bdist_rpm."""
import unittest
import sys
import os
from test.support import run_unittest, requires_zlib
from distutils.core import Distribution
from distutils.command.bdist_rpm import bdist_rpm
from distutils.tests import support
from distutils.spawn import find_executable
SETUP_PY = "from distutils.core import setup\nimport foo\n\nsetup(name='foo', version='0.1', py_modules=['foo'],\n      url='xxx', author='xxx', author_email='xxx')\n\n"

class BuildRpmTestCase(support.TempdirManager, support.EnvironGuard, support.LoggingSilencer, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            sys.executable.encode('UTF-8')
        except UnicodeEncodeError:
            raise unittest.SkipTest('sys.executable is not encodable to UTF-8')
        super(BuildRpmTestCase, self).setUp()
        self.old_location = os.getcwd()
        self.old_sys_argv = (sys.argv, sys.argv[:])

    def tearDown(self):
        if False:
            return 10
        os.chdir(self.old_location)
        sys.argv = self.old_sys_argv[0]
        sys.argv[:] = self.old_sys_argv[1]
        super(BuildRpmTestCase, self).tearDown()

    @unittest.skipUnless(sys.platform.startswith('linux'), 'spurious sdtout/stderr output under Mac OS X')
    @requires_zlib()
    @unittest.skipIf(find_executable('rpm') is None, 'the rpm command is not found')
    @unittest.skipIf(find_executable('rpmbuild') is None, 'the rpmbuild command is not found')
    @unittest.skip('not allowed to use rpmbuild directly on devservers')
    def test_quiet(self):
        if False:
            for i in range(10):
                print('nop')
        tmp_dir = self.mkdtemp()
        os.environ['HOME'] = tmp_dir
        pkg_dir = os.path.join(tmp_dir, 'foo')
        os.mkdir(pkg_dir)
        self.write_file((pkg_dir, 'setup.py'), SETUP_PY)
        self.write_file((pkg_dir, 'foo.py'), '#')
        self.write_file((pkg_dir, 'MANIFEST.in'), 'include foo.py')
        self.write_file((pkg_dir, 'README'), '')
        dist = Distribution({'name': 'foo', 'version': '0.1', 'py_modules': ['foo'], 'url': 'xxx', 'author': 'xxx', 'author_email': 'xxx'})
        dist.script_name = 'setup.py'
        os.chdir(pkg_dir)
        sys.argv = ['setup.py']
        cmd = bdist_rpm(dist)
        cmd.fix_python = True
        cmd.quiet = 1
        cmd.ensure_finalized()
        cmd.run()
        dist_created = os.listdir(os.path.join(pkg_dir, 'dist'))
        self.assertIn('foo-0.1-1.noarch.rpm', dist_created)
        self.assertIn(('bdist_rpm', 'any', 'dist/foo-0.1-1.src.rpm'), dist.dist_files)
        self.assertIn(('bdist_rpm', 'any', 'dist/foo-0.1-1.noarch.rpm'), dist.dist_files)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'spurious sdtout/stderr output under Mac OS X')
    @requires_zlib()
    @unittest.skipIf(find_executable('rpm') is None, 'the rpm command is not found')
    @unittest.skipIf(find_executable('rpmbuild') is None, 'the rpmbuild command is not found')
    @unittest.skip('not allowed to use rpmbuild directly on devservers')
    def test_no_optimize_flag(self):
        if False:
            print('Hello World!')
        tmp_dir = self.mkdtemp()
        os.environ['HOME'] = tmp_dir
        pkg_dir = os.path.join(tmp_dir, 'foo')
        os.mkdir(pkg_dir)
        self.write_file((pkg_dir, 'setup.py'), SETUP_PY)
        self.write_file((pkg_dir, 'foo.py'), '#')
        self.write_file((pkg_dir, 'MANIFEST.in'), 'include foo.py')
        self.write_file((pkg_dir, 'README'), '')
        dist = Distribution({'name': 'foo', 'version': '0.1', 'py_modules': ['foo'], 'url': 'xxx', 'author': 'xxx', 'author_email': 'xxx'})
        dist.script_name = 'setup.py'
        os.chdir(pkg_dir)
        sys.argv = ['setup.py']
        cmd = bdist_rpm(dist)
        cmd.fix_python = True
        cmd.quiet = 1
        cmd.ensure_finalized()
        cmd.run()
        dist_created = os.listdir(os.path.join(pkg_dir, 'dist'))
        self.assertIn('foo-0.1-1.noarch.rpm', dist_created)
        self.assertIn(('bdist_rpm', 'any', 'dist/foo-0.1-1.src.rpm'), dist.dist_files)
        self.assertIn(('bdist_rpm', 'any', 'dist/foo-0.1-1.noarch.rpm'), dist.dist_files)
        os.remove(os.path.join(pkg_dir, 'dist', 'foo-0.1-1.noarch.rpm'))

def test_suite():
    if False:
        return 10
    return unittest.makeSuite(BuildRpmTestCase)
if __name__ == '__main__':
    run_unittest(test_suite())