"""Tests for distutils.cygwinccompiler."""
import unittest
import sys
import os
from io import BytesIO
from test.support import run_unittest
from distutils import cygwinccompiler
from distutils.cygwinccompiler import check_config_h, CONFIG_H_OK, CONFIG_H_NOTOK, CONFIG_H_UNCERTAIN, get_versions, get_msvcr
from distutils.tests import support

class FakePopen(object):
    test_class = None

    def __init__(self, cmd, shell, stdout):
        if False:
            for i in range(10):
                print('nop')
        self.cmd = cmd.split()[0]
        exes = self.test_class._exes
        if self.cmd in exes:
            self.stdout = BytesIO(exes[self.cmd])
        else:
            self.stdout = os.popen(cmd, 'r')

class CygwinCCompilerTestCase(support.TempdirManager, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(CygwinCCompilerTestCase, self).setUp()
        self.version = sys.version
        self.python_h = os.path.join(self.mkdtemp(), 'python.h')
        from distutils import sysconfig
        self.old_get_config_h_filename = sysconfig.get_config_h_filename
        sysconfig.get_config_h_filename = self._get_config_h_filename
        self.old_find_executable = cygwinccompiler.find_executable
        cygwinccompiler.find_executable = self._find_executable
        self._exes = {}
        self.old_popen = cygwinccompiler.Popen
        FakePopen.test_class = self
        cygwinccompiler.Popen = FakePopen

    def tearDown(self):
        if False:
            return 10
        sys.version = self.version
        from distutils import sysconfig
        sysconfig.get_config_h_filename = self.old_get_config_h_filename
        cygwinccompiler.find_executable = self.old_find_executable
        cygwinccompiler.Popen = self.old_popen
        super(CygwinCCompilerTestCase, self).tearDown()

    def _get_config_h_filename(self):
        if False:
            i = 10
            return i + 15
        return self.python_h

    def _find_executable(self, name):
        if False:
            print('Hello World!')
        if name in self._exes:
            return name
        return None

    def test_check_config_h(self):
        if False:
            return 10
        sys.version = '2.6.1 (r261:67515, Dec  6 2008, 16:42:21) \n[GCC 4.0.1 (Apple Computer, Inc. build 5370)]'
        self.assertEqual(check_config_h()[0], CONFIG_H_OK)
        sys.version = 'something without the *CC word'
        self.assertEqual(check_config_h()[0], CONFIG_H_UNCERTAIN)
        self.write_file(self.python_h, 'xxx')
        self.assertEqual(check_config_h()[0], CONFIG_H_NOTOK)
        self.write_file(self.python_h, 'xxx __GNUC__ xxx')
        self.assertEqual(check_config_h()[0], CONFIG_H_OK)

    def test_get_versions(self):
        if False:
            return 10
        self.assertEqual(get_versions(), (None, None, None))
        self._exes['gcc'] = b'gcc (GCC) 3.4.5 (mingw special)\nFSF'
        res = get_versions()
        self.assertEqual(str(res[0]), '3.4.5')
        self._exes['gcc'] = b'very strange output'
        res = get_versions()
        self.assertEqual(res[0], None)
        self._exes['ld'] = b'GNU ld version 2.17.50 20060824'
        res = get_versions()
        self.assertEqual(str(res[1]), '2.17.50')
        self._exes['ld'] = b'@(#)PROGRAM:ld  PROJECT:ld64-77'
        res = get_versions()
        self.assertEqual(res[1], None)
        self._exes['dllwrap'] = b'GNU dllwrap 2.17.50 20060824\nFSF'
        res = get_versions()
        self.assertEqual(str(res[2]), '2.17.50')
        self._exes['dllwrap'] = b'Cheese Wrap'
        res = get_versions()
        self.assertEqual(res[2], None)

    def test_get_msvcr(self):
        if False:
            return 10
        sys.version = '2.6.1 (r261:67515, Dec  6 2008, 16:42:21) \n[GCC 4.0.1 (Apple Computer, Inc. build 5370)]'
        self.assertEqual(get_msvcr(), None)
        sys.version = '2.5.1 (r251:54863, Apr 18 2007, 08:51:08) [MSC v.1300 32 bits (Intel)]'
        self.assertEqual(get_msvcr(), ['msvcr70'])
        sys.version = '2.5.1 (r251:54863, Apr 18 2007, 08:51:08) [MSC v.1310 32 bits (Intel)]'
        self.assertEqual(get_msvcr(), ['msvcr71'])
        sys.version = '2.5.1 (r251:54863, Apr 18 2007, 08:51:08) [MSC v.1400 32 bits (Intel)]'
        self.assertEqual(get_msvcr(), ['msvcr80'])
        sys.version = '2.5.1 (r251:54863, Apr 18 2007, 08:51:08) [MSC v.1500 32 bits (Intel)]'
        self.assertEqual(get_msvcr(), ['msvcr90'])
        sys.version = '2.5.1 (r251:54863, Apr 18 2007, 08:51:08) [MSC v.1999 32 bits (Intel)]'
        self.assertRaises(ValueError, get_msvcr)

def test_suite():
    if False:
        print('Hello World!')
    return unittest.makeSuite(CygwinCCompilerTestCase)
if __name__ == '__main__':
    run_unittest(test_suite())