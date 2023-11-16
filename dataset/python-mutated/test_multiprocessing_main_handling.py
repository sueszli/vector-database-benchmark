from test import support
from test.support import import_helper
import_helper.import_module('_multiprocessing')
import importlib
import importlib.machinery
import unittest
import sys
import os
import os.path
import py_compile
from test.support import os_helper
from test.support.script_helper import make_pkg, make_script, make_zip_pkg, make_zip_script, assert_python_ok
if support.PGO:
    raise unittest.SkipTest('test is not helpful for PGO')
import multiprocessing
AVAILABLE_START_METHODS = set(multiprocessing.get_all_start_methods())
support.skip_if_broken_multiprocessing_synchronize()
verbose = support.verbose
test_source = '# multiprocessing includes all sorts of shenanigans to make __main__\n# attributes accessible in the subprocess in a pickle compatible way.\n\n# We run the "doesn\'t work in the interactive interpreter" example from\n# the docs to make sure it *does* work from an executed __main__,\n# regardless of the invocation mechanism\n\nimport sys\nimport time\nfrom multiprocessing import Pool, set_start_method\n\n# We use this __main__ defined function in the map call below in order to\n# check that multiprocessing in correctly running the unguarded\n# code in child processes and then making it available as __main__\ndef f(x):\n    return x*x\n\n# Check explicit relative imports\nif "check_sibling" in __file__:\n    # We\'re inside a package and not in a __main__.py file\n    # so make sure explicit relative imports work correctly\n    from . import sibling\n\nif __name__ == \'__main__\':\n    start_method = sys.argv[1]\n    set_start_method(start_method)\n    results = []\n    with Pool(5) as pool:\n        pool.map_async(f, [1, 2, 3], callback=results.extend)\n        start_time = time.monotonic()\n        while not results:\n            time.sleep(0.05)\n            # up to 1 min to report the results\n            dt = time.monotonic() - start_time\n            if dt > 60.0:\n                raise RuntimeError("Timed out waiting for results (%.1f sec)" % dt)\n\n    results.sort()\n    print(start_method, "->", results)\n\n    pool.join()\n'
test_source_main_skipped_in_children = '# __main__.py files have an implied "if __name__ == \'__main__\'" so\n# multiprocessing should always skip running them in child processes\n\n# This means we can\'t use __main__ defined functions in child processes,\n# so we just use "int" as a passthrough operation below\n\nif __name__ != "__main__":\n    raise RuntimeError("Should only be called as __main__!")\n\nimport sys\nimport time\nfrom multiprocessing import Pool, set_start_method\n\nstart_method = sys.argv[1]\nset_start_method(start_method)\nresults = []\nwith Pool(5) as pool:\n    pool.map_async(int, [1, 4, 9], callback=results.extend)\n    start_time = time.monotonic()\n    while not results:\n        time.sleep(0.05)\n        # up to 1 min to report the results\n        dt = time.monotonic() - start_time\n        if dt > 60.0:\n            raise RuntimeError("Timed out waiting for results (%.1f sec)" % dt)\n\nresults.sort()\nprint(start_method, "->", results)\n\npool.join()\n'

def _make_test_script(script_dir, script_basename, source=test_source, omit_suffix=False):
    if False:
        for i in range(10):
            print('nop')
    to_return = make_script(script_dir, script_basename, source, omit_suffix)
    if script_basename == 'check_sibling':
        make_script(script_dir, 'sibling', '')
    importlib.invalidate_caches()
    return to_return

def _make_test_zip_pkg(zip_dir, zip_basename, pkg_name, script_basename, source=test_source, depth=1):
    if False:
        while True:
            i = 10
    to_return = make_zip_pkg(zip_dir, zip_basename, pkg_name, script_basename, source, depth)
    importlib.invalidate_caches()
    return to_return
launch_source = 'import sys, os.path, runpy\nsys.path.insert(0, %s)\nrunpy._run_module_as_main(%r)\n'

def _make_launch_script(script_dir, script_basename, module_name, path=None):
    if False:
        return 10
    if path is None:
        path = 'os.path.dirname(__file__)'
    else:
        path = repr(path)
    source = launch_source % (path, module_name)
    to_return = make_script(script_dir, script_basename, source)
    importlib.invalidate_caches()
    return to_return

class MultiProcessingCmdLineMixin:
    maxDiff = None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        if self.start_method not in AVAILABLE_START_METHODS:
            self.skipTest('%r start method not available' % self.start_method)

    def _check_output(self, script_name, exit_code, out, err):
        if False:
            print('Hello World!')
        if verbose > 1:
            print('Output from test script %r:' % script_name)
            print(repr(out))
        self.assertEqual(exit_code, 0)
        self.assertEqual(err.decode('utf-8'), '')
        expected_results = '%s -> [1, 4, 9]' % self.start_method
        self.assertEqual(out.decode('utf-8').strip(), expected_results)

    def _check_script(self, script_name, *cmd_line_switches):
        if False:
            while True:
                i = 10
        if not __debug__:
            cmd_line_switches += ('-' + 'O' * sys.flags.optimize,)
        run_args = cmd_line_switches + (script_name, self.start_method)
        (rc, out, err) = assert_python_ok(*run_args, __isolated=False)
        self._check_output(script_name, rc, out, err)

    def test_basic_script(self):
        if False:
            i = 10
            return i + 15
        with os_helper.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, 'script')
            self._check_script(script_name)

    def test_basic_script_no_suffix(self):
        if False:
            while True:
                i = 10
        with os_helper.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, 'script', omit_suffix=True)
            self._check_script(script_name)

    def test_ipython_workaround(self):
        if False:
            print('Hello World!')
        source = test_source_main_skipped_in_children
        with os_helper.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, 'ipython', source=source)
            self._check_script(script_name)
            script_no_suffix = _make_test_script(script_dir, 'ipython', source=source, omit_suffix=True)
            self._check_script(script_no_suffix)

    def test_script_compiled(self):
        if False:
            return 10
        with os_helper.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, 'script')
            py_compile.compile(script_name, doraise=True)
            os.remove(script_name)
            pyc_file = import_helper.make_legacy_pyc(script_name)
            self._check_script(pyc_file)

    def test_directory(self):
        if False:
            i = 10
            return i + 15
        source = self.main_in_children_source
        with os_helper.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, '__main__', source=source)
            self._check_script(script_dir)

    def test_directory_compiled(self):
        if False:
            return 10
        source = self.main_in_children_source
        with os_helper.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, '__main__', source=source)
            py_compile.compile(script_name, doraise=True)
            os.remove(script_name)
            pyc_file = import_helper.make_legacy_pyc(script_name)
            self._check_script(script_dir)

    def test_zipfile(self):
        if False:
            for i in range(10):
                print('nop')
        source = self.main_in_children_source
        with os_helper.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, '__main__', source=source)
            (zip_name, run_name) = make_zip_script(script_dir, 'test_zip', script_name)
            self._check_script(zip_name)

    def test_zipfile_compiled(self):
        if False:
            i = 10
            return i + 15
        source = self.main_in_children_source
        with os_helper.temp_dir() as script_dir:
            script_name = _make_test_script(script_dir, '__main__', source=source)
            compiled_name = py_compile.compile(script_name, doraise=True)
            (zip_name, run_name) = make_zip_script(script_dir, 'test_zip', compiled_name)
            self._check_script(zip_name)

    def test_module_in_package(self):
        if False:
            print('Hello World!')
        with os_helper.temp_dir() as script_dir:
            pkg_dir = os.path.join(script_dir, 'test_pkg')
            make_pkg(pkg_dir)
            script_name = _make_test_script(pkg_dir, 'check_sibling')
            launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg.check_sibling')
            self._check_script(launch_name)

    def test_module_in_package_in_zipfile(self):
        if False:
            return 10
        with os_helper.temp_dir() as script_dir:
            (zip_name, run_name) = _make_test_zip_pkg(script_dir, 'test_zip', 'test_pkg', 'script')
            launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg.script', zip_name)
            self._check_script(launch_name)

    def test_module_in_subpackage_in_zipfile(self):
        if False:
            print('Hello World!')
        with os_helper.temp_dir() as script_dir:
            (zip_name, run_name) = _make_test_zip_pkg(script_dir, 'test_zip', 'test_pkg', 'script', depth=2)
            launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg.test_pkg.script', zip_name)
            self._check_script(launch_name)

    def test_package(self):
        if False:
            while True:
                i = 10
        source = self.main_in_children_source
        with os_helper.temp_dir() as script_dir:
            pkg_dir = os.path.join(script_dir, 'test_pkg')
            make_pkg(pkg_dir)
            script_name = _make_test_script(pkg_dir, '__main__', source=source)
            launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg')
            self._check_script(launch_name)

    def test_package_compiled(self):
        if False:
            return 10
        source = self.main_in_children_source
        with os_helper.temp_dir() as script_dir:
            pkg_dir = os.path.join(script_dir, 'test_pkg')
            make_pkg(pkg_dir)
            script_name = _make_test_script(pkg_dir, '__main__', source=source)
            compiled_name = py_compile.compile(script_name, doraise=True)
            os.remove(script_name)
            pyc_file = import_helper.make_legacy_pyc(script_name)
            launch_name = _make_launch_script(script_dir, 'launch', 'test_pkg')
            self._check_script(launch_name)

class SpawnCmdLineTest(MultiProcessingCmdLineMixin, unittest.TestCase):
    start_method = 'spawn'
    main_in_children_source = test_source_main_skipped_in_children

class ForkCmdLineTest(MultiProcessingCmdLineMixin, unittest.TestCase):
    start_method = 'fork'
    main_in_children_source = test_source

class ForkServerCmdLineTest(MultiProcessingCmdLineMixin, unittest.TestCase):
    start_method = 'forkserver'
    main_in_children_source = test_source_main_skipped_in_children

def tearDownModule():
    if False:
        i = 10
        return i + 15
    support.reap_children()
if __name__ == '__main__':
    unittest.main()