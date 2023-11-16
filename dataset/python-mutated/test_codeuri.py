import os
from unittest import TestCase
from parameterized import parameterized
from pathlib import Path
from samcli.lib.utils.codeuri import resolve_code_path

class TestLocalLambda_get_code_path(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.cwd = '/my/current/working/directory'
        self.relative_codeuri = './my/path'
        self.absolute_codeuri = '/home/foo/bar'
        self.os_cwd = os.getcwd()

    @parameterized.expand(['.', ''])
    def test_must_resolve_present_cwd(self, cwd_path):
        if False:
            for i in range(10):
                print('nop')
        codeuri = self.relative_codeuri
        expected = os.path.normpath(os.path.join(self.os_cwd, codeuri))
        actual = resolve_code_path(cwd_path, codeuri)
        self.assertEqual(expected, actual)
        self.assertTrue(os.path.isabs(actual), 'Result must be an absolute path')

    @parameterized.expand(['var/task', 'some/dir'])
    def test_must_resolve_relative_cwd(self, cwd_path):
        if False:
            for i in range(10):
                print('nop')
        codeuri = self.relative_codeuri
        abs_cwd = os.path.abspath(cwd_path)
        expected = os.path.normpath(os.path.join(abs_cwd, codeuri))
        actual = resolve_code_path(cwd_path, codeuri)
        self.assertEqual(expected, actual)
        self.assertTrue(os.path.isabs(actual), 'Result must be an absolute path')

    @parameterized.expand(['', '.', 'hello', 'a/b/c/d', '../../c/d/e'])
    def test_must_resolve_relative_codeuri(self, codeuri):
        if False:
            for i in range(10):
                print('nop')
        expected = os.path.normpath(os.path.join(self.cwd, codeuri))
        actual = resolve_code_path(self.cwd, codeuri)
        self.assertEqual(str(Path(expected).resolve()), actual)
        self.assertTrue(os.path.isabs(actual), 'Result must be an absolute path')

    @parameterized.expand(['/a/b/c', '/'])
    def test_must_resolve_absolute_codeuri(self, codeuri):
        if False:
            print('Hello World!')
        expected = codeuri
        actual = resolve_code_path(None, codeuri)
        self.assertEqual(expected, actual)
        self.assertTrue(os.path.isabs(actual), 'Result must be an absolute path')