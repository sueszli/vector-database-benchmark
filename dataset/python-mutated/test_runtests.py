import os
import sys
import subprocess
from numba import cuda
import unittest
import itertools
try:
    import git
except ImportError:
    has_gitpython = False
else:
    has_gitpython = True
try:
    import yaml
except ImportError:
    has_pyyaml = False
else:
    has_pyyaml = True

class TestCase(unittest.TestCase):
    """These test cases are meant to test the Numba test infrastructure itself.
    Therefore, the logic used here shouldn't use numba.testing, but only the
    upstream unittest, and run the numba test suite only in a subprocess."""

    def get_testsuite_listing(self, args, *, subp_kwargs=None):
        if False:
            while True:
                i = 10
        '\n        Use `subp_kwargs` to pass extra argument to `subprocess.check_output`.\n        '
        subp_kwargs = subp_kwargs or {}
        cmd = [sys.executable, '-m', 'numba.runtests', '-l'] + list(args)
        out_bytes = subprocess.check_output(cmd, **subp_kwargs)
        lines = out_bytes.decode('UTF-8').splitlines()
        lines = [line for line in lines if line.strip()]
        return lines

    def check_listing_prefix(self, prefix):
        if False:
            print('Hello World!')
        listing = self.get_testsuite_listing([prefix])
        for ln in listing[:-1]:
            errmsg = '{!r} not startswith {!r}'.format(ln, prefix)
            self.assertTrue(ln.startswith(prefix), msg=errmsg)

    def check_testsuite_size(self, args, minsize):
        if False:
            return 10
        '\n        Check that the reported numbers of tests are at least *minsize*.\n        '
        lines = self.get_testsuite_listing(args)
        last_line = lines[-1]
        self.assertTrue('tests found' in last_line)
        number = int(last_line.split(' ')[0])
        self.assertIn(len(lines), range(number + 1, number + 20))
        self.assertGreaterEqual(number, minsize)
        return lines

    def check_all(self, ids):
        if False:
            while True:
                i = 10
        lines = self.check_testsuite_size(ids, 5000)
        self.assertTrue(any(('numba.cuda.tests.' in line for line in lines)))
        self.assertTrue(any(('numba.tests.npyufunc.test_' in line for line in lines)))

    def _get_numba_tests_from_listing(self, listing):
        if False:
            while True:
                i = 10
        "returns a filter on strings starting with 'numba.', useful for\n        selecting the 'numba' test names from a test listing."
        return filter(lambda x: x.startswith('numba.'), listing)

    def test_default(self):
        if False:
            print('Hello World!')
        self.check_all([])

    def test_all(self):
        if False:
            print('Hello World!')
        self.check_all(['numba.tests'])

    def test_cuda(self):
        if False:
            print('Hello World!')
        minsize = 100 if cuda.is_available() else 1
        self.check_testsuite_size(['numba.cuda.tests'], minsize)

    @unittest.skipIf(not cuda.is_available(), 'NO CUDA')
    def test_cuda_submodules(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_listing_prefix('numba.cuda.tests.cudadrv')
        self.check_listing_prefix('numba.cuda.tests.cudapy')
        self.check_listing_prefix('numba.cuda.tests.nocuda')
        self.check_listing_prefix('numba.cuda.tests.cudasim')

    def test_module(self):
        if False:
            print('Hello World!')
        self.check_testsuite_size(['numba.tests.test_storeslice'], 2)
        self.check_testsuite_size(['numba.tests.test_nested_calls'], 10)
        self.check_testsuite_size(['numba.tests.test_nested_calls', 'numba.tests.test_storeslice'], 12)

    def test_subpackage(self):
        if False:
            i = 10
            return i + 15
        self.check_testsuite_size(['numba.tests.npyufunc'], 50)

    def test_random(self):
        if False:
            return 10
        self.check_testsuite_size(['--random', '0.1', 'numba.tests.npyufunc'], 5)

    def test_include_exclude_tags(self):
        if False:
            return 10

        def get_count(arg_list):
            if False:
                while True:
                    i = 10
            lines = self.get_testsuite_listing(arg_list)
            self.assertIn('tests found', lines[-1])
            count = int(lines[-1].split()[0])
            self.assertTrue(count > 0)
            return count
        tags = ['long_running', 'long_running, important']
        total = get_count(['numba.tests'])
        for tag in tags:
            included = get_count(['--tags', tag, 'numba.tests'])
            excluded = get_count(['--exclude-tags', tag, 'numba.tests'])
            self.assertEqual(total, included + excluded)
            included = get_count(['--tags=%s' % tag, 'numba.tests'])
            excluded = get_count(['--exclude-tags=%s' % tag, 'numba.tests'])
            self.assertEqual(total, included + excluded)

    def test_check_shard(self):
        if False:
            print('Hello World!')
        tmpAll = self.get_testsuite_listing([])
        tmp1 = self.get_testsuite_listing(['-j', '0:2'])
        tmp2 = self.get_testsuite_listing(['-j', '1:2'])
        lAll = set(self._get_numba_tests_from_listing(tmpAll))
        l1 = set(self._get_numba_tests_from_listing(tmp1))
        l2 = set(self._get_numba_tests_from_listing(tmp2))
        self.assertLess(abs(len(l2) - len(l1)), len(lAll) / 20)
        self.assertLess(len(l1), len(lAll))
        self.assertLess(len(l2), len(lAll))

    def test_check_sharding_equivalent(self):
        if False:
            print('Hello World!')
        sharded = list()
        for i in range(3):
            subset = self.get_testsuite_listing(['-j', '{}:3'.format(i)])
            slist = [*self._get_numba_tests_from_listing(subset)]
            sharded.append(slist)
        tmp = self.get_testsuite_listing(['--tag', 'always_test'])
        always_running = set(self._get_numba_tests_from_listing(tmp))
        self.assertGreaterEqual(len(always_running), 1)
        sharded_sets = [set(x) for x in sharded]
        for i in range(len(sharded)):
            self.assertEqual(len(sharded_sets[i]), len(sharded[i]))
        for shard in sharded_sets:
            for test in always_running:
                self.assertIn(test, shard)
                shard.remove(test)
                self.assertNotIn(test, shard)
        for (a, b) in itertools.combinations(sharded_sets, 2):
            self.assertFalse(a & b)
        sum_of_parts = set()
        for x in sharded_sets:
            sum_of_parts.update(x)
        sum_of_parts.update(always_running)
        full_listing = set(self._get_numba_tests_from_listing(self.get_testsuite_listing([])))
        self.assertEqual(sum_of_parts, full_listing)

    @unittest.skipUnless(has_gitpython, 'Requires gitpython')
    def test_gitdiff(self):
        if False:
            return 10
        try:
            subprocess.call('git', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            self.skipTest('no git available')
        outs = self.get_testsuite_listing(['-g'])
        self.assertNotIn('Git diff by common ancestor', outs)
        outs = self.get_testsuite_listing(['-g=ancestor'])
        self.assertIn('Git diff by common ancestor', outs)
        subp_kwargs = dict(stderr=subprocess.DEVNULL)
        with self.assertRaises(subprocess.CalledProcessError):
            self.get_testsuite_listing(['-g=ancest'], subp_kwargs=subp_kwargs)

    @unittest.skipUnless(has_pyyaml, 'Requires pyyaml')
    def test_azure_config(self):
        if False:
            i = 10
            return i + 15
        from yaml import Loader
        base_path = os.path.dirname(os.path.abspath(__file__))
        azure_pipe = os.path.join(base_path, '..', '..', 'azure-pipelines.yml')
        if not os.path.isfile(azure_pipe):
            self.skipTest("'azure-pipelines.yml' is not available")
        with open(os.path.abspath(azure_pipe), 'rt') as f:
            data = f.read()
        pipe_yml = yaml.load(data, Loader=Loader)
        templates = pipe_yml['jobs']
        start_indexes = []
        for tmplt in templates[:2]:
            matrix = tmplt['parameters']['matrix']
            for setup in matrix.values():
                start_indexes.append(setup['TEST_START_INDEX'])
        winpath = ['..', '..', 'buildscripts', 'azure', 'azure-windows.yml']
        azure_windows = os.path.join(base_path, *winpath)
        if not os.path.isfile(azure_windows):
            self.skipTest("'azure-windows.yml' is not available")
        with open(os.path.abspath(azure_windows), 'rt') as f:
            data = f.read()
        windows_yml = yaml.load(data, Loader=Loader)
        matrix = windows_yml['jobs'][0]['strategy']['matrix']
        for setup in matrix.values():
            start_indexes.append(setup['TEST_START_INDEX'])
        self.assertEqual(len(start_indexes), len(set(start_indexes)))
        lim_start_index = max(start_indexes) + 1
        expected = [*range(lim_start_index)]
        self.assertEqual(sorted(start_indexes), expected)
        self.assertEqual(lim_start_index, pipe_yml['variables']['TEST_COUNT'])
if __name__ == '__main__':
    unittest.main()