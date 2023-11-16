"""Tests for parse_args.py."""
import os
import sys
import types
from pytype import file_utils
from pytype.platform_utils import path_utils
from pytype.tests import test_utils
from pytype.tools.analyze_project import config
from pytype.tools.analyze_project import parse_args
import unittest

class TestConvertString(unittest.TestCase):
    """Test parse_args.convert_string."""

    def test_int(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(parse_args.convert_string('3'), 3)

    def test_bool(self):
        if False:
            i = 10
            return i + 15
        self.assertIs(parse_args.convert_string('True'), True)
        self.assertIs(parse_args.convert_string('False'), False)

    def test_whitespace(self):
        if False:
            return 10
        self.assertEqual(parse_args.convert_string('err1,\nerr2'), 'err1,err2')

class TestParser(unittest.TestCase):
    """Test parse_args.Parser."""

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        cls.parser = parse_args.make_parser()

    def test_parse_filenames(self):
        if False:
            return 10
        filenames = ['a.py', 'b.py']
        args = self.parser.parse_args(filenames)
        self.assertEqual(args.inputs, {path_utils.realpath(f) for f in filenames})

    def test_parse_no_filename(self):
        if False:
            return 10
        args = self.parser.parse_args([])
        self.assertFalse(hasattr(args, 'inputs'))

    def test_parse_bad_filename(self):
        if False:
            for i in range(10):
                print('nop')
        args = self.parser.parse_args(['this_file_should_not_exist'])
        self.assertEqual(args.inputs, set())

    def test_parse_filenames_default(self):
        if False:
            print('Hello World!')
        args = self.parser.config_from_defaults()
        self.assertEqual(args.inputs, set())

    def test_parse_exclude(self):
        if False:
            for i in range(10):
                print('nop')
        filenames = ['a.py', 'b.py']
        args = self.parser.parse_args(['--exclude'] + filenames)
        self.assertEqual(args.exclude, {path_utils.realpath(f) for f in filenames})

    def test_parse_single_exclude(self):
        if False:
            for i in range(10):
                print('nop')
        filenames = ['a.py', 'b/c.py']
        with test_utils.Tempdir() as d:
            for f in filenames:
                d.create_file(f)
            with file_utils.cd(d.path):
                args = self.parser.parse_args(['--exclude=**/*.py'])
                self.assertEqual(args.exclude, {path_utils.realpath(f) for f in filenames})

    def test_parse_exclude_dir(self):
        if False:
            while True:
                i = 10
        filenames = ['foo/f1.py', 'foo/f2.py']
        with test_utils.Tempdir() as d:
            for f in filenames:
                d.create_file(f)
            with file_utils.cd(d.path):
                args = self.parser.parse_args(['--exclude=foo/'])
                self.assertEqual(args.exclude, {path_utils.realpath(f) for f in filenames})

    def test_parse_bad_exclude(self):
        if False:
            i = 10
            return i + 15
        args = self.parser.parse_args(['-x', 'this_file_should_not_exist'])
        self.assertEqual(args.exclude, set())

    def test_verbosity(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.parser.parse_args(['--verbosity', '0']).verbosity, 0)
        self.assertEqual(self.parser.parse_args(['-v1']).verbosity, 1)

    def test_version(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.parser.parse_args(['--version']).version)

    def test_config(self):
        if False:
            print('Hello World!')
        args = self.parser.parse_args(['--config=test.cfg'])
        self.assertEqual(args.config, 'test.cfg')

    def test_tree(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.parser.parse_args(['--tree']).tree)
        with self.assertRaises(SystemExit):
            self.parser.parse_args(['--tree', '--unresolved'])

    def test_unresolved(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.parser.parse_args(['--unresolved']).unresolved)

    def test_generate_config(self):
        if False:
            while True:
                i = 10
        args = self.parser.parse_args(['--generate-config', 'test.cfg'])
        self.assertEqual(args.generate_config, 'test.cfg')
        with self.assertRaises(SystemExit):
            self.parser.parse_args(['--generate-config', 'test.cfg', '--tree'])

    def test_python_version(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.parser.parse_args(['-V3.7']).python_version, '3.7')
        self.assertEqual(self.parser.parse_args(['--python-version', '3.7']).python_version, '3.7')

    def test_python_version_default(self):
        if False:
            return 10
        self.assertEqual(self.parser.config_from_defaults().python_version, f'{sys.version_info.major}.{sys.version_info.minor}')

    def test_output(self):
        if False:
            return 10
        self.assertEqual(self.parser.parse_args(['-o', 'pyi']).output, path_utils.join(path_utils.getcwd(), 'pyi'))
        self.assertEqual(self.parser.parse_args(['--output', 'pyi']).output, path_utils.join(path_utils.getcwd(), 'pyi'))

    def test_no_cache(self):
        if False:
            print('Hello World!')
        self.assertFalse(self.parser.parse_args([]).no_cache)
        self.assertTrue(self.parser.parse_args(['-n']).no_cache)
        self.assertTrue(self.parser.parse_args(['--no-cache']).no_cache)
        with self.assertRaises(SystemExit):
            self.parser.parse_args(['--output', 'pyi', '--no-cache'])

    def test_pythonpath(self):
        if False:
            print('Hello World!')
        d = path_utils.getcwd()
        self.assertSequenceEqual(self.parser.parse_args(['-P', f'{os.pathsep}foo']).pythonpath, [d, path_utils.join(d, 'foo')])
        self.assertSequenceEqual(self.parser.parse_args(['--pythonpath', f'{os.pathsep}foo']).pythonpath, [d, path_utils.join(d, 'foo')])

    def test_keep_going(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.parser.parse_args(['-k']).keep_going)

    def test_keep_going_default(self):
        if False:
            while True:
                i = 10
        self.assertIsInstance(self.parser.config_from_defaults().keep_going, bool)

    def test_defaults(self):
        if False:
            i = 10
            return i + 15
        args = self.parser.parse_args([])
        for arg in config.ITEMS:
            self.assertFalse(hasattr(args, arg))

    def test_pytype_single_args(self):
        if False:
            for i in range(10):
                print('nop')
        args = self.parser.parse_args(['--disable=import-error'])
        self.assertSequenceEqual(args.disable, ['import-error'])

    def test_config_file(self):
        if False:
            for i in range(10):
                print('nop')
        conf = self.parser.config_from_defaults()
        self.assertEqual(conf.output, path_utils.join(path_utils.getcwd(), '.pytype'))
        self.assertIsInstance(conf.disable, list)
        self.assertFalse(conf.disable)

    def test_postprocess(self):
        if False:
            i = 10
            return i + 15
        args = types.SimpleNamespace(disable='import-error')
        self.parser.postprocess(args)
        self.assertSequenceEqual(args.disable, ['import-error'])

    def test_postprocess_from_strings(self):
        if False:
            i = 10
            return i + 15
        args = types.SimpleNamespace(report_errors='False', protocols='True')
        self.parser.convert_strings(args)
        self.parser.postprocess(args)
        self.assertFalse(args.report_errors)
        self.assertTrue(args.protocols)
if __name__ == '__main__':
    unittest.main()