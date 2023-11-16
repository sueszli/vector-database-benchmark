import test.support
import os
import os.path
import sys
import textwrap
import zipfile
import zipimport
import doctest
import inspect
import linecache
import unittest
from test.support import os_helper
from test.support.script_helper import spawn_python, kill_python, assert_python_ok, make_script, make_zip_script
verbose = test.support.verbose
from test import test_doctest, sample_doctest, sample_doctest_no_doctests, sample_doctest_no_docstrings

def _run_object_doctest(obj, module):
    if False:
        return 10
    finder = doctest.DocTestFinder(verbose=verbose, recurse=False)
    runner = doctest.DocTestRunner(verbose=verbose)
    try:
        name = '%s.%s' % (obj.__module__, obj.__qualname__)
    except AttributeError:
        name = module.__name__
    for example in finder.find(obj, name, module):
        runner.run(example)
    (f, t) = (runner.failures, runner.tries)
    if f:
        raise test.support.TestFailed('%d of %d doctests failed' % (f, t))
    if verbose:
        print('doctest (%s) ... %d tests with zero failures' % (module.__name__, t))
    return (f, t)

class ZipSupportTests(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        linecache.clearcache()
        zipimport._zip_directory_cache.clear()
        self.path = sys.path[:]
        self.meta_path = sys.meta_path[:]
        self.path_hooks = sys.path_hooks[:]
        sys.path_importer_cache.clear()

    def tearDown(self):
        if False:
            while True:
                i = 10
        sys.path[:] = self.path
        sys.meta_path[:] = self.meta_path
        sys.path_hooks[:] = self.path_hooks
        sys.path_importer_cache.clear()

    def test_inspect_getsource_issue4223(self):
        if False:
            for i in range(10):
                print('nop')
        test_src = 'def foo(): pass\n'
        with os_helper.temp_dir() as d:
            init_name = make_script(d, '__init__', test_src)
            name_in_zip = os.path.join('zip_pkg', os.path.basename(init_name))
            (zip_name, run_name) = make_zip_script(d, 'test_zip', init_name, name_in_zip)
            os.remove(init_name)
            sys.path.insert(0, zip_name)
            import zip_pkg
            try:
                self.assertEqual(inspect.getsource(zip_pkg.foo), test_src)
            finally:
                del sys.modules['zip_pkg']

    def test_doctest_issue4197(self):
        if False:
            while True:
                i = 10
        test_src = inspect.getsource(test_doctest)
        test_src = test_src.replace('from test import test_doctest', 'import test_zipped_doctest as test_doctest')
        test_src = test_src.replace('test.test_doctest', 'test_zipped_doctest')
        test_src = test_src.replace('test.sample_doctest', 'sample_zipped_doctest')
        sample_sources = {}
        for mod in [sample_doctest, sample_doctest_no_doctests, sample_doctest_no_docstrings]:
            src = inspect.getsource(mod)
            src = src.replace('test.test_doctest', 'test_zipped_doctest')
            mod_name = mod.__name__.split('.')[-1]
            mod_name = mod_name.replace('sample_', 'sample_zipped_')
            sample_sources[mod_name] = src
        with os_helper.temp_dir() as d:
            script_name = make_script(d, 'test_zipped_doctest', test_src)
            (zip_name, run_name) = make_zip_script(d, 'test_zip', script_name)
            with zipfile.ZipFile(zip_name, 'a') as z:
                for (mod_name, src) in sample_sources.items():
                    z.writestr(mod_name + '.py', src)
            if verbose:
                with zipfile.ZipFile(zip_name, 'r') as zip_file:
                    print('Contents of %r:' % zip_name)
                    zip_file.printdir()
            os.remove(script_name)
            sys.path.insert(0, zip_name)
            import test_zipped_doctest
            try:
                known_good_tests = [test_zipped_doctest.SampleClass, test_zipped_doctest.SampleClass.NestedClass, test_zipped_doctest.SampleClass.NestedClass.__init__, test_zipped_doctest.SampleClass.__init__, test_zipped_doctest.SampleClass.a_classmethod, test_zipped_doctest.SampleClass.a_property, test_zipped_doctest.SampleClass.a_staticmethod, test_zipped_doctest.SampleClass.double, test_zipped_doctest.SampleClass.get, test_zipped_doctest.SampleNewStyleClass, test_zipped_doctest.SampleNewStyleClass.__init__, test_zipped_doctest.SampleNewStyleClass.double, test_zipped_doctest.SampleNewStyleClass.get, test_zipped_doctest.sample_func, test_zipped_doctest.test_DocTest, test_zipped_doctest.test_DocTestParser, test_zipped_doctest.test_DocTestRunner.basics, test_zipped_doctest.test_DocTestRunner.exceptions, test_zipped_doctest.test_DocTestRunner.option_directives, test_zipped_doctest.test_DocTestRunner.optionflags, test_zipped_doctest.test_DocTestRunner.verbose_flag, test_zipped_doctest.test_Example, test_zipped_doctest.test_debug, test_zipped_doctest.test_testsource, test_zipped_doctest.test_trailing_space_in_test, test_zipped_doctest.test_DocTestSuite, test_zipped_doctest.test_DocTestFinder]
                fail_due_to_missing_data_files = [test_zipped_doctest.test_DocFileSuite, test_zipped_doctest.test_testfile, test_zipped_doctest.test_unittest_reportflags]
                for obj in known_good_tests:
                    _run_object_doctest(obj, test_zipped_doctest)
            finally:
                del sys.modules['test_zipped_doctest']

    def test_doctest_main_issue4197(self):
        if False:
            i = 10
            return i + 15
        test_src = textwrap.dedent('                    class Test:\n                        ">>> \'line 2\'"\n                        pass\n\n                    import doctest\n                    doctest.testmod()\n                    ')
        pattern = 'File "%s", line 2, in %s'
        with os_helper.temp_dir() as d:
            script_name = make_script(d, 'script', test_src)
            (rc, out, err) = assert_python_ok(script_name)
            expected = pattern % (script_name, '__main__.Test')
            if verbose:
                print('Expected line', expected)
                print('Got stdout:')
                print(ascii(out))
            self.assertIn(expected.encode('utf-8'), out)
            (zip_name, run_name) = make_zip_script(d, 'test_zip', script_name, '__main__.py')
            (rc, out, err) = assert_python_ok(zip_name)
            expected = pattern % (run_name, '__main__.Test')
            if verbose:
                print('Expected line', expected)
                print('Got stdout:')
                print(ascii(out))
            self.assertIn(expected.encode('utf-8'), out)

    def test_pdb_issue4201(self):
        if False:
            while True:
                i = 10
        test_src = textwrap.dedent('                    def f():\n                        pass\n\n                    import pdb\n                    pdb.Pdb(nosigint=True).runcall(f)\n                    ')
        with os_helper.temp_dir() as d:
            script_name = make_script(d, 'script', test_src)
            p = spawn_python(script_name)
            p.stdin.write(b'l\n')
            data = kill_python(p)
            self.assertIn(os.path.normcase(script_name.encode('utf-8')), data)
            (zip_name, run_name) = make_zip_script(d, 'test_zip', script_name, '__main__.py')
            p = spawn_python(zip_name)
            p.stdin.write(b'l\n')
            data = kill_python(p)
            self.assertIn(os.path.normcase(run_name.encode('utf-8')), data)

def tearDownModule():
    if False:
        i = 10
        return i + 15
    test.support.reap_children()
if __name__ == '__main__':
    unittest.main()