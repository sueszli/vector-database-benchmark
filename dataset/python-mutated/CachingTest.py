import unittest
import os
from unittest.mock import patch
from pyprint.NullPrinter import NullPrinter
from pyprint.ConsolePrinter import ConsolePrinter
from coalib.misc.Caching import FileCache, FileDictFileCache, ProxyMapFileCache
from coalib.processes.Processing import get_file_dict
from coalib.io.FileProxy import FileProxy, FileProxyMap
from coalib.misc.CachingUtilities import pickle_load, pickle_dump
from coalib.output.printers.LogPrinter import LogPrinter
from coalib import coala
from coalib.coala_main import run_coala
from coala_utils.ContextManagers import make_temp, prepare_file
from coala_utils.ContextManagers import simulate_console_inputs
from tests.TestUtilities import execute_coala, bear_test_module

class CachingTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        current_dir = os.path.split(__file__)[0]
        self.caching_test_dir = os.path.join(current_dir, 'caching_testfiles')
        self.log_printer = LogPrinter(NullPrinter())
        self.cache = FileCache(self.log_printer, 'coala_test', flush_cache=True)

    def test_file_tracking(self):
        if False:
            while True:
                i = 10
        self.cache.track_files({'test.c', 'file.py'})
        self.assertEqual(self.cache.data, {'test.c': -1, 'file.py': -1})
        self.cache.untrack_files({'test.c'})
        self.cache.track_files({'test.c'})
        self.cache.write()
        self.assertFalse('test.c' in self.cache.data)
        self.assertTrue('file.py' in self.cache.data)
        self.cache.untrack_files({'test.c', 'file.py'})
        self.cache.write()
        self.assertFalse('test.c' in self.cache.data)
        self.assertFalse('file.py' in self.cache.data)

    def test_write(self):
        if False:
            i = 10
            return i + 15
        self.cache.track_files({'test2.c'})
        self.assertEqual(self.cache.data['test2.c'], -1)
        self.cache.write()
        self.assertNotEqual(self.cache.data['test2.c'], -1)

    @patch('coalib.misc.Caching.os')
    def test_get_uncached_files(self, mock_os):
        if False:
            for i in range(10):
                print('nop')
        file_path = os.path.join(self.caching_test_dir, 'test.c')
        cache = FileCache(self.log_printer, 'coala_test3', flush_cache=True)
        cache.current_time = 0
        mock_os.path.getmtime.return_value = 0
        self.assertEqual(cache.get_uncached_files({file_path}), {file_path})
        cache.track_files({file_path})
        self.assertEqual(cache.get_uncached_files({file_path}), {file_path})
        cache.write()
        self.assertEqual(cache.get_uncached_files({file_path}), set())
        cache.current_time = 1
        mock_os.path.getmtime.return_value = 1
        cache.track_files({file_path})
        self.assertEqual(cache.get_uncached_files({file_path}), {file_path})
        cache.write()
        cache.current_time = 2
        self.assertEqual(cache.get_uncached_files({file_path}), set())

    def test_persistence(self):
        if False:
            i = 10
            return i + 15
        with FileCache(self.log_printer, 'test3', flush_cache=True) as cache:
            cache.track_files({'file.c'})
        self.assertTrue('file.c' in cache.data)
        with FileCache(self.log_printer, 'test3', flush_cache=False) as cache:
            self.assertTrue('file.c' in cache.data)

    def test_time_travel(self):
        if False:
            return 10
        cache = FileCache(self.log_printer, 'coala_test2', flush_cache=True)
        cache.track_files({'file.c'})
        cache.write()
        self.assertTrue('file.c' in cache.data)
        cache_data = pickle_load(self.log_printer, 'coala_test2', {})
        cache_data['time'] = 2000000000
        pickle_dump(self.log_printer, 'coala_test2', cache_data)
        cache = FileCache(self.log_printer, 'coala_test2', flush_cache=False)
        self.assertFalse('file.c' in cache.data)

    def test_caching_results(self):
        if False:
            i = 10
            return i + 15
        '\n        A simple integration test to assert that results are not dropped\n        when coala is ran multiple times with caching enabled.\n        '
        with bear_test_module():
            with prepare_file(['a=(5,6)'], None) as (lines, filename):
                with simulate_console_inputs('n'):
                    (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '-c', os.devnull, '--disable-caching', '--flush-cache', '-f', filename, '-b', 'LineCountTestBear', '-L', 'DEBUG')
                    self.assertIn('This file has', stdout)
                    self.assertIn('Running bear LineCountTestBear', stderr)
                (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '--no-color', '-c', os.devnull, '-f', filename, '-b', 'LineCountTestBear')
                self.assertIn('This file has', stdout)
                self.assertEqual(1, len(stderr.splitlines()))
                self.assertIn('LineCountTestBear: This result has no patch attached.', stderr)
                (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '--no-color', '-c', os.devnull, '-f', filename, '-b', 'LineCountTestBear')
                self.assertIn('This file has', stdout)
                self.assertEqual(1, len(stderr.splitlines()))
                self.assertIn('LineCountTestBear: This result has no patch attached.', stderr)

    def test_caching_multi_results(self):
        if False:
            while True:
                i = 10
        "\n        Integration test to assert that results are not dropped when coala is\n        ran multiple times with caching enabled and one section yields a result\n        and second one doesn't.\n        "
        filename = 'tests/misc/test_caching_multi_results/'
        with bear_test_module():
            with simulate_console_inputs('n'):
                (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '-c', filename + '.coafile', '-f', filename + 'test.py')
                self.assertIn('This file has', stdout)
                self.assertIn("Implicit 'Default' section inheritance is deprecated", stderr)
            (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '--no-color', '-c', filename + '.coafile', '-f', filename + 'test.py')
            self.assertIn('This file has', stdout)
            self.assertEqual(2, len(stderr.splitlines()))
            self.assertIn('LineCountTestBear: This result has no patch attached.', stderr)
            self.assertIn("Implicit 'Default' section inheritance is deprecated", stderr)

class FileDictFileCacheTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        log_printer = LogPrinter(NullPrinter())
        self.cache = FileDictFileCache(log_printer, 'coala_test', flush_cache=True)

    def test_build_as_file_cache(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(isinstance(self.cache, FileCache))
        self.cache.track_files({'test.c', 'file.py'})
        self.assertEqual(self.cache.data, {'test.c': -1, 'file.py': -1})

    def test_get_file_dict(self):
        if False:
            return 10
        with prepare_file(['line\n'], None) as (_, file):
            self.assertEqual(self.cache.get_file_dict([file]), get_file_dict([file]))

class ProxyMapFileCacheTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        log_printer = LogPrinter(NullPrinter())
        self.cache = ProxyMapFileCache(log_printer, 'coala_test', flush_cache=True)

    def test_build_as_file_cache(self):
        if False:
            while True:
                i = 10
        self.assertTrue(isinstance(self.cache, FileCache))

    def test_get_dict_no_file_proxy_map(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError) as context:
            self.cache.get_file_dict([])
        self.assertIn('set_proxymap() should be called to set proxymapof ProxyMapFileCache instance', str(context.exception))

    def test_get_file_dict_with_proxy_map(self):
        if False:
            print('Hello World!')
        with prepare_file([], None) as (_, file):
            proxy = FileProxy(file, None, 'coala\n')
            proxymap = FileProxyMap([proxy])
            self.cache.set_proxymap(proxymap)
            file_dict = self.cache.get_file_dict([file])
            self.assertEqual(file_dict, {file: ('coala\n',)})

    def test_get_file_dict_unicode_error(self):
        if False:
            print('Hello World!')
        with make_temp() as filename:
            with open(filename, 'wb') as file:
                file.write(bytearray([120, 3, 255, 0, 100]))
            self.cache.set_proxymap(FileProxyMap())
            file_dict = self.cache.get_file_dict([filename])
            self.assertEqual(len(file_dict), 0)
            self.cache.set_proxymap(FileProxyMap())
            file_dict = self.cache.get_file_dict([filename], allow_raw_files=True)
            self.assertEqual(len(file_dict), 1)
            self.assertIn(filename, file_dict)

    def test_get_file_dict_file_not_found(self):
        if False:
            print('Hello World!')
        self.cache.set_proxymap(FileProxyMap())
        file_dict = self.cache.get_file_dict(['nofile.pycoala'])
        self.assertEqual(len(file_dict), 0)

    def test_file_cache_proxy_integration(self, debug=False):
        if False:
            for i in range(10):
                print('nop')
        with bear_test_module():
            with prepare_file(['disk-copy\n'], None) as (_, filename):
                memory_data = 'in-memory\n'
                proxy = FileProxy(filename, None, memory_data)
                proxymap = FileProxyMap([proxy])
                self.cache.set_proxymap(proxymap)
                (results, exitcode, file_dicts) = run_coala(console_printer=ConsolePrinter(), log_printer=LogPrinter(), arg_list=('-c', os.devnull, '-f', filename, '-b', 'TestBear'), autoapply=False, debug=debug, cache=self.cache)
                self.assertEqual(exitcode, 0)
                self.assertEqual(len(results), 1)
                self.assertEqual(file_dicts['cli'][filename.lower()], (memory_data,))