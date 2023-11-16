import os.path
import subprocess
import sys
import numpy as np
from numba.tests.support import skip_parfors_unsupported
from .test_caching import DispatcherCacheUsecasesTest

@skip_parfors_unsupported
class TestParforsCache(DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, 'parfors_cache_usecases.py')
    modname = 'parfors_caching_test_fodder'

    def run_test(self, fname, num_funcs=1):
        if False:
            for i in range(10):
                print('nop')
        mod = self.import_module()
        self.check_pycache(0)
        f = getattr(mod, fname)
        ary = np.ones(10)
        np.testing.assert_allclose(f(ary), f.py_func(ary))
        dynamic_globals = [cres.library.has_dynamic_globals for cres in f.overloads.values()]
        [cres] = f.overloads.values()
        self.assertEqual(dynamic_globals, [False])
        self.check_pycache(num_funcs * 2)
        self.run_in_separate_process()

    def test_arrayexprs(self):
        if False:
            return 10
        f = 'arrayexprs_case'
        self.run_test(f)

    def test_prange(self):
        if False:
            while True:
                i = 10
        f = 'prange_case'
        self.run_test(f)

    def test_caller(self):
        if False:
            i = 10
            return i + 15
        f = 'caller_case'
        self.run_test(f, num_funcs=3)

@skip_parfors_unsupported
class TestParforsCacheChangingThreads(DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, 'parfors_cache_usecases.py')
    modname = 'parfors_caching_test_fodder'

    def run_in_separate_process(self, thread_count):
        if False:
            return 10
        code = 'if 1:\n            import sys\n\n            sys.path.insert(0, %(tempdir)r)\n            mod = __import__(%(modname)r)\n            mod.self_run()\n            ' % dict(tempdir=self.tempdir, modname=self.modname)
        new_env = {**os.environ, 'NUMBA_NUM_THREADS': str(thread_count)}
        popen = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=new_env)
        (out, err) = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError(f'process failed with code {popen.returncode}:stderr follows\n{err.decode()}\n')

    def test_caching(self):
        if False:
            print('Hello World!')
        self.check_pycache(0)
        self.run_in_separate_process(1)
        self.check_pycache(3 * 2)
        self.run_in_separate_process(2)
        self.check_pycache(3 * 2)