"""Tests for distutils.command.build_py."""
import os
import sys
import unittest
from distutils.command.build_py import build_py
from distutils.core import Distribution
from distutils.errors import DistutilsFileError
from distutils.tests import support
from test.support import run_unittest

class BuildPyTestCase(support.TempdirManager, support.LoggingSilencer, unittest.TestCase):

    def test_package_data(self):
        if False:
            for i in range(10):
                print('nop')
        sources = self.mkdtemp()
        f = open(os.path.join(sources, '__init__.py'), 'w')
        try:
            f.write('# Pretend this is a package.')
        finally:
            f.close()
        f = open(os.path.join(sources, 'README.txt'), 'w')
        try:
            f.write('Info about this package')
        finally:
            f.close()
        destination = self.mkdtemp()
        dist = Distribution({'packages': ['pkg'], 'package_dir': {'pkg': sources}})
        dist.script_name = os.path.join(sources, 'setup.py')
        dist.command_obj['build'] = support.DummyCommand(force=0, build_lib=destination)
        dist.packages = ['pkg']
        dist.package_data = {'pkg': ['README.txt']}
        dist.package_dir = {'pkg': sources}
        cmd = build_py(dist)
        cmd.compile = 1
        cmd.ensure_finalized()
        self.assertEqual(cmd.package_data, dist.package_data)
        cmd.run()
        self.assertEqual(len(cmd.get_outputs()), 3)
        pkgdest = os.path.join(destination, 'pkg')
        files = os.listdir(pkgdest)
        pycache_dir = os.path.join(pkgdest, '__pycache__')
        self.assertIn('__init__.py', files)
        self.assertIn('README.txt', files)
        if sys.dont_write_bytecode:
            self.assertFalse(os.path.exists(pycache_dir))
        else:
            pyc_files = os.listdir(pycache_dir)
            self.assertIn('__init__.%s.pyc' % sys.implementation.cache_tag, pyc_files)

    def test_empty_package_dir(self):
        if False:
            print('Hello World!')
        sources = self.mkdtemp()
        open(os.path.join(sources, '__init__.py'), 'w').close()
        testdir = os.path.join(sources, 'doc')
        os.mkdir(testdir)
        open(os.path.join(testdir, 'testfile'), 'w').close()
        os.chdir(sources)
        dist = Distribution({'packages': ['pkg'], 'package_dir': {'pkg': ''}, 'package_data': {'pkg': ['doc/*']}})
        dist.script_name = os.path.join(sources, 'setup.py')
        dist.script_args = ['build']
        dist.parse_command_line()
        try:
            dist.run_commands()
        except DistutilsFileError:
            self.fail("failed package_data test when package_dir is ''")

    @unittest.skipIf(sys.dont_write_bytecode, 'byte-compile disabled')
    def test_byte_compile(self):
        if False:
            while True:
                i = 10
        (project_dir, dist) = self.create_dist(py_modules=['boiledeggs'])
        os.chdir(project_dir)
        self.write_file('boiledeggs.py', 'import antigravity')
        cmd = build_py(dist)
        cmd.compile = 1
        cmd.build_lib = 'here'
        cmd.finalize_options()
        cmd.run()
        found = os.listdir(cmd.build_lib)
        self.assertEqual(sorted(found), ['__pycache__', 'boiledeggs.py'])
        found = os.listdir(os.path.join(cmd.build_lib, '__pycache__'))
        self.assertEqual(found, ['boiledeggs.%s.pyc' % sys.implementation.cache_tag])

    @unittest.skipIf(sys.dont_write_bytecode, 'byte-compile disabled')
    def test_byte_compile_optimized(self):
        if False:
            return 10
        (project_dir, dist) = self.create_dist(py_modules=['boiledeggs'])
        os.chdir(project_dir)
        self.write_file('boiledeggs.py', 'import antigravity')
        cmd = build_py(dist)
        cmd.compile = 0
        cmd.optimize = 1
        cmd.build_lib = 'here'
        cmd.finalize_options()
        cmd.run()
        found = os.listdir(cmd.build_lib)
        self.assertEqual(sorted(found), ['__pycache__', 'boiledeggs.py'])
        found = os.listdir(os.path.join(cmd.build_lib, '__pycache__'))
        expect = 'boiledeggs.{}.opt-1.pyc'.format(sys.implementation.cache_tag)
        self.assertEqual(sorted(found), [expect])

    def test_dir_in_package_data(self):
        if False:
            print('Hello World!')
        '\n        A directory in package_data should not be added to the filelist.\n        '
        sources = self.mkdtemp()
        pkg_dir = os.path.join(sources, 'pkg')
        os.mkdir(pkg_dir)
        open(os.path.join(pkg_dir, '__init__.py'), 'w').close()
        docdir = os.path.join(pkg_dir, 'doc')
        os.mkdir(docdir)
        open(os.path.join(docdir, 'testfile'), 'w').close()
        os.mkdir(os.path.join(docdir, 'otherdir'))
        os.chdir(sources)
        dist = Distribution({'packages': ['pkg'], 'package_data': {'pkg': ['doc/*']}})
        dist.script_name = os.path.join(sources, 'setup.py')
        dist.script_args = ['build']
        dist.parse_command_line()
        try:
            dist.run_commands()
        except DistutilsFileError:
            self.fail('failed package_data when data dir includes a dir')

    def test_dont_write_bytecode(self):
        if False:
            i = 10
            return i + 15
        dist = self.create_dist()[1]
        cmd = build_py(dist)
        cmd.compile = 1
        cmd.optimize = 1
        old_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            cmd.byte_compile([])
        finally:
            sys.dont_write_bytecode = old_dont_write_bytecode
        self.assertIn('byte-compiling is disabled', self.logs[0][1] % self.logs[0][2])

def test_suite():
    if False:
        i = 10
        return i + 15
    return unittest.makeSuite(BuildPyTestCase)
if __name__ == '__main__':
    run_unittest(test_suite())