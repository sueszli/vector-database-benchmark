"""
Tests the parallel backend
"""
import faulthandler
import itertools
import multiprocessing
import os
import random
import subprocess
import sys
import textwrap
import threading
import unittest
import numpy as np
from numba import jit, vectorize, guvectorize, set_num_threads
from numba.tests.support import temp_directory, override_config, TestCase, tag, skip_parfors_unsupported, linux_only
import queue as t_queue
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
from numba.core import config
_TEST_TIMEOUT = _RUNNER_TIMEOUT - 60.0
try:
    from numba.np.ufunc.parallel import _check_tbb_version_compatible
    _check_tbb_version_compatible()
    from numba.np.ufunc import tbbpool
    _HAVE_TBB_POOL = True
except ImportError:
    _HAVE_TBB_POOL = False
try:
    from numba.np.ufunc import omppool
    _HAVE_OMP_POOL = True
except ImportError:
    _HAVE_OMP_POOL = False
try:
    import scipy.linalg.cython_lapack
    _HAVE_LAPACK = True
except ImportError:
    _HAVE_LAPACK = False
skip_no_omp = unittest.skipUnless(_HAVE_OMP_POOL, 'OpenMP threadpool required')
skip_no_tbb = unittest.skipUnless(_HAVE_TBB_POOL, 'TBB threadpool required')
_gnuomp = _HAVE_OMP_POOL and omppool.openmp_vendor == 'GNU'
skip_unless_gnu_omp = unittest.skipUnless(_gnuomp, 'GNU OpenMP only tests')
_windows = sys.platform.startswith('win')
_osx = sys.platform.startswith('darwin')
_32bit = sys.maxsize <= 2 ** 32
_parfors_unsupported = _32bit
_HAVE_OS_FORK = not _windows

def foo(n, v):
    if False:
        while True:
            i = 10
    return np.ones(n) + v
if _HAVE_LAPACK:

    def linalg(n, v):
        if False:
            while True:
                i = 10
        x = np.dot(np.ones((n, n)), np.ones((n, n)))
        return x + np.arange(n) + v
else:

    def linalg(n, v):
        if False:
            i = 10
            return i + 15
        return np.arange(n) + v

def ufunc_foo(a, b):
    if False:
        print('Hello World!')
    return a + b

def gufunc_foo(a, b, out):
    if False:
        while True:
            i = 10
    out[0] = a + b

class runnable(object):

    def __init__(self, **options):
        if False:
            print('Hello World!')
        self._options = options

class jit_runner(runnable):

    def __call__(self):
        if False:
            print('Hello World!')
        cfunc = jit(**self._options)(foo)
        a = 4
        b = 10
        expected = foo(a, b)
        got = cfunc(a, b)
        np.testing.assert_allclose(expected, got)

class mask_runner(object):

    def __init__(self, runner, mask, **options):
        if False:
            return 10
        self.runner = runner
        self.mask = mask

    def __call__(self):
        if False:
            while True:
                i = 10
        if self.mask:
            set_num_threads(self.mask)
        self.runner()

class linalg_runner(runnable):

    def __call__(self):
        if False:
            print('Hello World!')
        cfunc = jit(**self._options)(linalg)
        a = 4
        b = 10
        expected = linalg(a, b)
        got = cfunc(a, b)
        np.testing.assert_allclose(expected, got)

class vectorize_runner(runnable):

    def __call__(self):
        if False:
            return 10
        cfunc = vectorize(['(f4, f4)'], **self._options)(ufunc_foo)
        a = b = np.random.random(10).astype(np.float32)
        expected = ufunc_foo(a, b)
        got = cfunc(a, b)
        np.testing.assert_allclose(expected, got)

class guvectorize_runner(runnable):

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        sig = ['(f4, f4, f4[:])']
        cfunc = guvectorize(sig, '(),()->()', **self._options)(gufunc_foo)
        a = b = np.random.random(10).astype(np.float32)
        expected = ufunc_foo(a, b)
        got = cfunc(a, b)
        np.testing.assert_allclose(expected, got)

def chooser(fnlist, **kwargs):
    if False:
        while True:
            i = 10
    q = kwargs.get('queue')
    try:
        faulthandler.enable()
        for _ in range(int(len(fnlist) * 1.5)):
            fn = random.choice(fnlist)
            fn()
    except Exception as e:
        q.put(e)

def compile_factory(parallel_class, queue_impl):
    if False:
        return 10

    def run_compile(fnlist):
        if False:
            for i in range(10):
                print('nop')
        q = queue_impl()
        kws = {'queue': q}
        ths = [parallel_class(target=chooser, args=(fnlist,), kwargs=kws) for i in range(4)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        if not q.empty():
            errors = []
            while not q.empty():
                errors.append(q.get(False))
            _msg = 'Error(s) occurred in delegated runner:\n%s'
            raise RuntimeError(_msg % '\n'.join([repr(x) for x in errors]))
    return run_compile
_thread_class = threading.Thread

class _proc_class_impl(object):

    def __init__(self, method):
        if False:
            while True:
                i = 10
        self._method = method

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        ctx = multiprocessing.get_context(self._method)
        return ctx.Process(*args, **kwargs)

def _get_mp_classes(method):
    if False:
        while True:
            i = 10
    if method == 'default':
        method = None
    ctx = multiprocessing.get_context(method)
    proc = _proc_class_impl(method)
    queue = ctx.Queue
    return (proc, queue)
thread_impl = compile_factory(_thread_class, t_queue.Queue)
spawn_proc_impl = compile_factory(*_get_mp_classes('spawn'))
if not _windows:
    fork_proc_impl = compile_factory(*_get_mp_classes('fork'))
    forkserver_proc_impl = compile_factory(*_get_mp_classes('forkserver'))
default_proc_impl = compile_factory(*_get_mp_classes('default'))

class TestParallelBackendBase(TestCase):
    """
    Base class for testing the parallel backends
    """
    all_impls = [jit_runner(nopython=True), jit_runner(nopython=True, cache=True), jit_runner(nopython=True, nogil=True), linalg_runner(nopython=True), linalg_runner(nopython=True, nogil=True), vectorize_runner(nopython=True), vectorize_runner(nopython=True, target='parallel'), vectorize_runner(nopython=True, target='parallel', cache=True), guvectorize_runner(nopython=True), guvectorize_runner(nopython=True, target='parallel'), guvectorize_runner(nopython=True, target='parallel', cache=True)]
    if not _parfors_unsupported:
        parfor_impls = [jit_runner(nopython=True, parallel=True), jit_runner(nopython=True, parallel=True, cache=True), linalg_runner(nopython=True, parallel=True), linalg_runner(nopython=True, parallel=True, cache=True)]
        all_impls.extend(parfor_impls)
    if config.NUMBA_NUM_THREADS < 2:
        masks = []
    else:
        masks = [1, 2]
    mask_impls = []
    for impl in all_impls:
        for mask in masks:
            mask_impls.append(mask_runner(impl, mask))
    parallelism = ['threading', 'random']
    parallelism.append('multiprocessing_spawn')
    if _HAVE_OS_FORK:
        parallelism.append('multiprocessing_fork')
        parallelism.append('multiprocessing_forkserver')
    runners = {'concurrent_jit': [jit_runner(nopython=True, parallel=not _parfors_unsupported)], 'concurrent_vectorize': [vectorize_runner(nopython=True, target='parallel')], 'concurrent_guvectorize': [guvectorize_runner(nopython=True, target='parallel')], 'concurrent_mix_use': all_impls, 'concurrent_mix_use_masks': mask_impls}
    safe_backends = {'omp', 'tbb'}

    def run_compile(self, fnlist, parallelism='threading'):
        if False:
            print('Hello World!')
        self._cache_dir = temp_directory(self.__class__.__name__)
        with override_config('CACHE_DIR', self._cache_dir):
            if parallelism == 'threading':
                thread_impl(fnlist)
            elif parallelism == 'multiprocessing_fork':
                fork_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_forkserver':
                forkserver_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_spawn':
                spawn_proc_impl(fnlist)
            elif parallelism == 'multiprocessing_default':
                default_proc_impl(fnlist)
            elif parallelism == 'random':
                ps = [thread_impl, spawn_proc_impl]
                if _HAVE_OS_FORK:
                    ps.append(fork_proc_impl)
                    ps.append(forkserver_proc_impl)
                random.shuffle(ps)
                for impl in ps:
                    impl(fnlist)
            else:
                raise ValueError('Unknown parallelism supplied %s' % parallelism)
_specific_backends = config.THREADING_LAYER in ('omp', 'tbb', 'workqueue')

@unittest.skipUnless(_specific_backends, 'Threading layer not explicit')
class TestParallelBackend(TestParallelBackendBase):
    """ These are like the numba.tests.test_threadsafety tests but designed
    instead to torture the parallel backend.
    If a suitable backend is supplied via NUMBA_THREADING_LAYER these tests
    can be run directly. This test class cannot be run using the multiprocessing
    option to the test runner (i.e. `./runtests -m`) as daemon processes cannot
    have children.
    """

    @classmethod
    def generate(cls):
        if False:
            for i in range(10):
                print('nop')
        for p in cls.parallelism:
            for (name, impl) in cls.runners.items():
                methname = 'test_' + p + '_' + name

                def methgen(impl, p):
                    if False:
                        return 10

                    def test_method(self):
                        if False:
                            return 10
                        selfproc = multiprocessing.current_process()
                        if selfproc.daemon:
                            _msg = 'daemonized processes cannot have children'
                            self.skipTest(_msg)
                        else:
                            self.run_compile(impl, parallelism=p)
                    return test_method
                fn = methgen(impl, p)
                fn.__name__ = methname
                setattr(cls, methname, fn)
TestParallelBackend.generate()

class TestInSubprocess(object):
    backends = {'tbb': skip_no_tbb, 'omp': skip_no_omp, 'workqueue': unittest.skipIf(False, '')}

    def run_cmd(self, cmdline, env):
        if False:
            print('Hello World!')
        popen = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        timeout = threading.Timer(_TEST_TIMEOUT, popen.kill)
        try:
            timeout.start()
            (out, err) = popen.communicate()
            if popen.returncode != 0:
                raise AssertionError('process failed with code %s: stderr follows\n%s\n' % (popen.returncode, err.decode()))
            return (out.decode(), err.decode())
        finally:
            timeout.cancel()
        return (None, None)

    def run_test_in_separate_process(self, test, threading_layer):
        if False:
            return 10
        env_copy = os.environ.copy()
        env_copy['NUMBA_THREADING_LAYER'] = str(threading_layer)
        cmdline = [sys.executable, '-m', 'numba.runtests', test]
        return self.run_cmd(cmdline, env_copy)

class TestSpecificBackend(TestInSubprocess, TestParallelBackendBase):
    """
    This is quite contrived, for each test in the TestParallelBackend tests it
    generates a test that will run the TestParallelBackend test in a new python
    process with an environment modified to ensure a specific threadsafe backend
    is used. This is with view of testing the backends independently and in an
    isolated manner such that if they hang/crash/have issues, it doesn't kill
    the test suite.
    """
    _DEBUG = False

    @classmethod
    def _inject(cls, p, name, backend, backend_guard):
        if False:
            for i in range(10):
                print('nop')
        themod = cls.__module__
        thecls = TestParallelBackend.__name__
        methname = 'test_' + p + '_' + name
        injected_method = '%s.%s.%s' % (themod, thecls, methname)

        def test_template(self):
            if False:
                return 10
            (o, e) = self.run_test_in_separate_process(injected_method, backend)
            if self._DEBUG:
                print('stdout:\n "%s"\n stderr:\n "%s"' % (o, e))
            self.assertIn('OK', e)
            self.assertTrue('FAIL' not in e)
            self.assertTrue('ERROR' not in e)
        injected_test = 'test_%s_%s_%s' % (p, name, backend)
        setattr(cls, injected_test, tag('long_running')(backend_guard(test_template)))

    @classmethod
    def generate(cls):
        if False:
            print('Hello World!')
        for (backend, backend_guard) in cls.backends.items():
            for p in cls.parallelism:
                for name in cls.runners.keys():
                    if p in ('multiprocessing_fork', 'random') and backend == 'omp' and sys.platform.startswith('linux'):
                        continue
                    if p in ('threading', 'random') and backend == 'workqueue':
                        continue
                    cls._inject(p, name, backend, backend_guard)
TestSpecificBackend.generate()

class ThreadLayerTestHelper(TestCase):
    """
    Helper class for running an isolated piece of code based on a template
    """
    _here = '%r' % os.path.dirname(__file__)
    template = 'if 1:\n    import sys\n    sys.path.insert(0, "%(here)r")\n    import multiprocessing\n    import numpy as np\n    from numba import njit\n    import numba\n    try:\n        import threading_backend_usecases\n    except ImportError as e:\n        print("DEBUG:", sys.path)\n        raise e\n    import os\n\n    sigterm_handler = threading_backend_usecases.sigterm_handler\n    busy_func = threading_backend_usecases.busy_func\n\n    def the_test():\n        %%s\n\n    if __name__ == "__main__":\n        the_test()\n    ' % {'here': _here}

    def run_cmd(self, cmdline, env=None):
        if False:
            print('Hello World!')
        if env is None:
            env = os.environ.copy()
            env['NUMBA_THREADING_LAYER'] = str('omp')
        popen = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        timeout = threading.Timer(_TEST_TIMEOUT, popen.kill)
        try:
            timeout.start()
            (out, err) = popen.communicate()
            if popen.returncode != 0:
                raise AssertionError('process failed with code %s: stderr follows\n%s\n' % (popen.returncode, err.decode()))
        finally:
            timeout.cancel()
        return (out.decode(), err.decode())

@skip_parfors_unsupported
class TestThreadingLayerSelection(ThreadLayerTestHelper):
    """
    Checks that numba.threading_layer() reports correctly.
    """
    _DEBUG = False
    backends = {'tbb': skip_no_tbb, 'omp': skip_no_omp, 'workqueue': unittest.skipIf(False, '')}

    @classmethod
    def _inject(cls, backend, backend_guard):
        if False:
            return 10

        def test_template(self):
            if False:
                i = 10
                return i + 15
            body = "if 1:\n                X = np.arange(1000000.)\n                Y = np.arange(1000000.)\n                Z = busy_func(X, Y)\n                assert numba.threading_layer() == '%s'\n            "
            runme = self.template % (body % backend)
            cmdline = [sys.executable, '-c', runme]
            env = os.environ.copy()
            env['NUMBA_THREADING_LAYER'] = str(backend)
            (out, err) = self.run_cmd(cmdline, env=env)
            if self._DEBUG:
                print(out, err)
        injected_test = 'test_threading_layer_selector_%s' % backend
        setattr(cls, injected_test, tag('important')(backend_guard(test_template)))

    @classmethod
    def generate(cls):
        if False:
            return 10
        for (backend, backend_guard) in cls.backends.items():
            cls._inject(backend, backend_guard)
TestThreadingLayerSelection.generate()

@skip_parfors_unsupported
class TestThreadingLayerPriority(ThreadLayerTestHelper):

    def each_env_var(self, env_var: str):
        if False:
            for i in range(10):
                print('nop')
        'Test setting priority via env var NUMBA_THREADING_LAYER_PRIORITY.\n        '
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = 'default'
        env['NUMBA_THREADING_LAYER_PRIORITY'] = env_var
        code = f'''\n                import numba\n\n                # trigger threading layer decision\n                # hence catching invalid THREADING_LAYER_PRIORITY\n                @numba.jit(\n                    'float64[::1](float64[::1], float64[::1])',\n                    nopython=True,\n                    parallel=True,\n                )\n                def plus(x, y):\n                    return x + y\n\n                captured_envvar = list("{env_var}".split())\n                assert numba.config.THREADING_LAYER_PRIORITY ==                     captured_envvar, "priority mismatch"\n                assert numba.threading_layer() == captured_envvar[0],                    "selected backend mismatch"\n                '''
        cmd = [sys.executable, '-c', textwrap.dedent(code)]
        self.run_cmd(cmd, env=env)

    @skip_no_omp
    @skip_no_tbb
    def test_valid_env_var(self):
        if False:
            i = 10
            return i + 15
        default = ['tbb', 'omp', 'workqueue']
        for p in itertools.permutations(default):
            env_var = ' '.join(p)
            self.each_env_var(env_var)

    @skip_no_omp
    @skip_no_tbb
    def test_invalid_env_var(self):
        if False:
            i = 10
            return i + 15
        env_var = 'tbb omp workqueue notvalidhere'
        with self.assertRaises(AssertionError) as raises:
            self.each_env_var(env_var)
        for msg in ('THREADING_LAYER_PRIORITY invalid:', 'It must be a permutation of'):
            self.assertIn(f'{msg}', str(raises.exception))

    @skip_no_omp
    def test_omp(self):
        if False:
            for i in range(10):
                print('nop')
        for env_var in ('omp tbb workqueue', 'omp workqueue tbb'):
            self.each_env_var(env_var)

    @skip_no_tbb
    def test_tbb(self):
        if False:
            print('Hello World!')
        for env_var in ('tbb omp workqueue', 'tbb workqueue omp'):
            self.each_env_var(env_var)

    def test_workqueue(self):
        if False:
            while True:
                i = 10
        for env_var in ('workqueue tbb omp', 'workqueue omp tbb'):
            self.each_env_var(env_var)

@skip_parfors_unsupported
class TestMiscBackendIssues(ThreadLayerTestHelper):
    """
    Checks fixes for the issues with threading backends implementation
    """
    _DEBUG = False

    @skip_no_omp
    def test_omp_stack_overflow(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that OMP does not overflow stack\n        '
        runme = 'if 1:\n            from numba import vectorize, threading_layer\n            import numpy as np\n\n            @vectorize([\'f4(f4,f4,f4,f4,f4,f4,f4,f4)\'], target=\'parallel\')\n            def foo(a, b, c, d, e, f, g, h):\n                return a+b+c+d+e+f+g+h\n\n            x = np.ones(2**20, np.float32)\n            foo(*([x]*8))\n            assert threading_layer() == "omp", "omp not found"\n        '
        cmdline = [sys.executable, '-c', runme]
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = 'omp'
        env['OMP_STACKSIZE'] = '100K'
        self.run_cmd(cmdline, env=env)

    @skip_no_tbb
    def test_single_thread_tbb(self):
        if False:
            print('Hello World!')
        '\n        Tests that TBB works well with single thread\n        https://github.com/numba/numba/issues/3440\n        '
        runme = 'if 1:\n            from numba import njit, prange, threading_layer\n\n            @njit(parallel=True)\n            def foo(n):\n                acc = 0\n                for i in prange(n):\n                    acc += i\n                return acc\n\n            foo(100)\n            assert threading_layer() == "tbb", "tbb not found"\n        '
        cmdline = [sys.executable, '-c', runme]
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = 'tbb'
        env['NUMBA_NUM_THREADS'] = '1'
        self.run_cmd(cmdline, env=env)

    def test_workqueue_aborts_on_nested_parallelism(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests workqueue raises sigabrt if a nested parallel call is performed\n        '
        runme = 'if 1:\n            from numba import njit, prange\n            import numpy as np\n\n            @njit(parallel=True)\n            def nested(x):\n                for i in prange(len(x)):\n                    x[i] += 1\n\n\n            @njit(parallel=True)\n            def main():\n                Z = np.zeros((5, 10))\n                for i in prange(Z.shape[0]):\n                    nested(Z[i])\n                return Z\n\n            main()\n        '
        cmdline = [sys.executable, '-c', runme]
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = 'workqueue'
        env['NUMBA_NUM_THREADS'] = '4'
        try:
            (out, err) = self.run_cmd(cmdline, env=env)
        except AssertionError as e:
            if self._DEBUG:
                print(out, err)
            e_msg = str(e)
            self.assertIn('failed with code', e_msg)
            expected = 'Numba workqueue threading layer is terminating: Concurrent access has been detected.'
            self.assertIn(expected, e_msg)

    @unittest.skipUnless(_HAVE_OS_FORK, 'Test needs fork(2)')
    def test_workqueue_handles_fork_from_non_main_thread(self):
        if False:
            i = 10
            return i + 15
        runme = 'if 1:\n            from numba import njit, prange, threading_layer\n            import numpy as np\n            import multiprocessing\n\n            if __name__ == "__main__":\n                # Need for force fork context (OSX default is "spawn")\n                multiprocessing.set_start_method(\'fork\')\n\n                @njit(parallel=True)\n                def func(x):\n                    return 10. * x\n\n                arr = np.arange(2.)\n\n                # run in single process to start Numba\'s thread pool\n                np.testing.assert_allclose(func(arr), func.py_func(arr))\n\n                # now run in a multiprocessing pool to get a fork from a\n                # non-main thread\n                with multiprocessing.Pool(10) as p:\n                    result = p.map(func, [arr])\n                np.testing.assert_allclose(result,\n                                           func.py_func(np.expand_dims(arr, 0)))\n\n                assert threading_layer() == "workqueue"\n        '
        cmdline = [sys.executable, '-c', runme]
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = 'workqueue'
        env['NUMBA_NUM_THREADS'] = '4'
        self.run_cmd(cmdline, env=env)

@skip_parfors_unsupported
@skip_unless_gnu_omp
class TestForkSafetyIssues(ThreadLayerTestHelper):
    """
    Checks Numba's behaviour in various situations involving GNU OpenMP and fork
    """
    _DEBUG = False

    def test_check_threading_layer_is_gnu(self):
        if False:
            i = 10
            return i + 15
        runme = "if 1:\n            from numba.np.ufunc import omppool\n            assert omppool.openmp_vendor == 'GNU'\n            "
        cmdline = [sys.executable, '-c', runme]
        (out, err) = self.run_cmd(cmdline)

    def test_par_parent_os_fork_par_child(self):
        if False:
            i = 10
            return i + 15
        "\n        Whilst normally valid, this actually isn't for Numba invariant of OpenMP\n        Checks SIGABRT is received.\n        "
        body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            Z = busy_func(X, Y)\n            pid = os.fork()\n            if pid  == 0:\n                Z = busy_func(X, Y)\n            else:\n                os.wait()\n        '
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        try:
            (out, err) = self.run_cmd(cmdline)
        except AssertionError as e:
            self.assertIn('failed with code -6', str(e))

    def test_par_parent_implicit_mp_fork_par_child(self):
        if False:
            return 10
        '\n        Implicit use of multiprocessing fork context.\n        Does this:\n        1. Start with OpenMP\n        2. Fork to processes using OpenMP (this is invalid)\n        3. Joins fork\n        4. Check the exception pushed onto the queue that is a result of\n           catching SIGTERM coming from the C++ aborting on illegal fork\n           pattern for GNU OpenMP\n        '
        body = 'if 1:\n            mp = multiprocessing.get_context(\'fork\')\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            q = mp.Queue()\n\n            # Start OpenMP runtime on parent via parallel function\n            Z = busy_func(X, Y, q)\n\n            # fork() underneath with no exec, will abort\n            proc = mp.Process(target = busy_func, args=(X, Y, q))\n            proc.start()\n\n            err = q.get()\n            assert "Caught SIGTERM" in str(err)\n        '
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        (out, err) = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

    @linux_only
    def test_par_parent_explicit_mp_fork_par_child(self):
        if False:
            return 10
        '\n        Explicit use of multiprocessing fork context.\n        Does this:\n        1. Start with OpenMP\n        2. Fork to processes using OpenMP (this is invalid)\n        3. Joins fork\n        4. Check the exception pushed onto the queue that is a result of\n           catching SIGTERM coming from the C++ aborting on illegal fork\n           pattern for GNU OpenMP\n        '
        body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            ctx = multiprocessing.get_context(\'fork\')\n            q = ctx.Queue()\n\n            # Start OpenMP runtime on parent via parallel function\n            Z = busy_func(X, Y, q)\n\n            # fork() underneath with no exec, will abort\n            proc = ctx.Process(target = busy_func, args=(X, Y, q))\n            proc.start()\n            proc.join()\n\n            err = q.get()\n            assert "Caught SIGTERM" in str(err)\n        '
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        (out, err) = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

    def test_par_parent_mp_spawn_par_child_par_parent(self):
        if False:
            i = 10
            return i + 15
        '\n        Explicit use of multiprocessing spawn, this is safe.\n        Does this:\n        1. Start with OpenMP\n        2. Spawn to processes using OpenMP\n        3. Join spawns\n        4. Run some more OpenMP\n        '
        body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            ctx = multiprocessing.get_context(\'spawn\')\n            q = ctx.Queue()\n\n            # Start OpenMP runtime and run on parent via parallel function\n            Z = busy_func(X, Y, q)\n            procs = []\n            for x in range(20): # start a lot to try and get overlap\n                ## fork() + exec() to run some OpenMP on children\n                proc = ctx.Process(target = busy_func, args=(X, Y, q))\n                procs.append(proc)\n                sys.stdout.flush()\n                sys.stderr.flush()\n                proc.start()\n\n            [p.join() for p in procs]\n\n            try:\n                q.get(False)\n            except multiprocessing.queues.Empty:\n                pass\n            else:\n                raise RuntimeError("Queue was not empty")\n\n            # Run some more OpenMP on parent\n            Z = busy_func(X, Y, q)\n        '
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        (out, err) = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

    def test_serial_parent_implicit_mp_fork_par_child_then_par_parent(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Implicit use of multiprocessing (will be fork, but cannot declare that\n        in Py2.7 as there's no process launch context).\n        Does this:\n        1. Start with no OpenMP\n        2. Fork to processes using OpenMP\n        3. Join forks\n        4. Run some OpenMP\n        "
        body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            q = multiprocessing.Queue()\n\n            # this is ok\n            procs = []\n            for x in range(10):\n                # fork() underneath with but no OpenMP in parent, this is ok\n                proc = multiprocessing.Process(target = busy_func,\n                                               args=(X, Y, q))\n                procs.append(proc)\n                proc.start()\n\n            [p.join() for p in procs]\n\n            # and this is still ok as the OpenMP happened in forks\n            Z = busy_func(X, Y, q)\n            try:\n                q.get(False)\n            except multiprocessing.queues.Empty:\n                pass\n            else:\n                raise RuntimeError("Queue was not empty")\n        '
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        (out, err) = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

    @linux_only
    def test_serial_parent_explicit_mp_fork_par_child_then_par_parent(self):
        if False:
            print('Hello World!')
        "\n        Explicit use of multiprocessing 'fork'.\n        Does this:\n        1. Start with no OpenMP\n        2. Fork to processes using OpenMP\n        3. Join forks\n        4. Run some OpenMP\n        "
        body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            ctx = multiprocessing.get_context(\'fork\')\n            q = ctx.Queue()\n\n            # this is ok\n            procs = []\n            for x in range(10):\n                # fork() underneath with but no OpenMP in parent, this is ok\n                proc = ctx.Process(target = busy_func, args=(X, Y, q))\n                procs.append(proc)\n                proc.start()\n\n            [p.join() for p in procs]\n\n            # and this is still ok as the OpenMP happened in forks\n            Z = busy_func(X, Y, q)\n            try:\n                q.get(False)\n            except multiprocessing.queues.Empty:\n                pass\n            else:\n                raise RuntimeError("Queue was not empty")\n        '
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        (out, err) = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

@skip_parfors_unsupported
@skip_no_tbb
class TestTBBSpecificIssues(ThreadLayerTestHelper):
    _DEBUG = False

    @linux_only
    def test_fork_from_non_main_thread(self):
        if False:
            return 10
        runme = "if 1:\n            import threading\n            import numba\n            numba.config.THREADING_LAYER='tbb'\n            from numba import njit, prange, objmode\n            from numba.core.serialize import PickleCallableByPath\n            import os\n\n            e_running = threading.Event()\n            e_proceed = threading.Event()\n\n            def indirect_core():\n                e_running.set()\n                # wait for forker() to have forked\n                while not e_proceed.isSet():\n                    pass\n\n            indirect = PickleCallableByPath(indirect_core)\n\n            @njit\n            def obj_mode_func():\n                with objmode():\n                    indirect()\n\n            @njit(parallel=True, nogil=True)\n            def work():\n                acc = 0\n                for x in prange(10):\n                    acc += x\n                obj_mode_func()\n                return acc\n\n            def runner():\n                work()\n\n            def forker():\n                # wait for the jit function to say it's running\n                while not e_running.isSet():\n                    pass\n                # then fork\n                os.fork()\n                # now fork is done signal the runner to proceed to exit\n                e_proceed.set()\n\n            numba_runner = threading.Thread(target=runner,)\n            fork_runner =  threading.Thread(target=forker,)\n\n            threads = (numba_runner, fork_runner)\n            for t in threads:\n                t.start()\n            for t in threads:\n                t.join()\n        "
        cmdline = [sys.executable, '-c', runme]
        (out, err) = self.run_cmd(cmdline)
        msg_head = 'Attempted to fork from a non-main thread, the TBB library'
        self.assertIn(msg_head, err)
        if self._DEBUG:
            print('OUT:', out)
            print('ERR:', err)

    @linux_only
    def test_lifetime_of_task_scheduler_handle(self):
        if False:
            for i in range(10):
                print('nop')
        self.skip_if_no_external_compiler()
        BROKEN_COMPILERS = 'SKIP: COMPILATION FAILED'
        runme = 'if 1:\n            import ctypes\n            import sys\n            import multiprocessing as mp\n            from tempfile import TemporaryDirectory, NamedTemporaryFile\n            from numba.pycc.platform import Toolchain, external_compiler_works\n            from numba import njit, prange, threading_layer\n            import faulthandler\n            faulthandler.enable()\n            if not external_compiler_works():\n                raise AssertionError(\'External compilers are not found.\')\n            with TemporaryDirectory() as tmpdir:\n                with NamedTemporaryFile(dir=tmpdir) as tmpfile:\n                    try:\n                        src = """\n                        #define TBB_PREVIEW_WAITING_FOR_WORKERS 1\n                        #include <tbb/tbb.h>\n                        static tbb::task_scheduler_handle tsh;\n                        extern "C"\n                        {\n                        void launch(void)\n                        {\n                            tsh = tbb::task_scheduler_handle::get();\n                        }\n                        }\n                        """\n                        cxxfile = f"{tmpfile.name}.cxx"\n                        with open(cxxfile, \'wt\') as f:\n                            f.write(src)\n                        tc = Toolchain()\n                        object_files = tc.compile_objects([cxxfile,],\n                                                           output_dir=tmpdir)\n                        dso_name = f"{tmpfile.name}.so"\n                        tc.link_shared(dso_name, object_files,\n                                       libraries=[\'tbb\',],\n                                       export_symbols=[\'launch\'])\n                        # Load into the process, it doesn\'t matter whether the\n                        # DSO exists on disk once it\'s loaded in.\n                        DLL = ctypes.CDLL(dso_name)\n                    except Exception as e:\n                        # Something is broken in compilation, could be one of\n                        # many things including, but not limited to: missing tbb\n                        # headers, incorrect permissions, compilers that don\'t\n                        # work for the above\n                        print(e)\n                        print(\'BROKEN_COMPILERS\')\n                        sys.exit(0)\n\n                    # Do the test, launch this library and also execute a\n                    # function with the TBB threading layer.\n\n                    DLL.launch()\n\n                    @njit(parallel=True)\n                    def foo(n):\n                        acc = 0\n                        for i in prange(n):\n                            acc += i\n                        return acc\n\n                    foo(1)\n\n            # Check the threading layer used was TBB\n            assert threading_layer() == \'tbb\'\n\n            # Use mp context for a controlled version of fork, this triggers the\n            # reported bug.\n\n            ctx = mp.get_context(\'fork\')\n            def nowork():\n                pass\n            p = ctx.Process(target=nowork)\n            p.start()\n            p.join(10)\n            print("SUCCESS")\n            '.replace('BROKEN_COMPILERS', BROKEN_COMPILERS)
        cmdline = [sys.executable, '-c', runme]
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = 'tbb'
        (out, err) = self.run_cmd(cmdline, env=env)
        if BROKEN_COMPILERS in out:
            self.skipTest('Compilation of DSO failed. Check output for details')
        else:
            self.assertIn('SUCCESS', out)
        if self._DEBUG:
            print('OUT:', out)
            print('ERR:', err)

@skip_parfors_unsupported
class TestInitSafetyIssues(TestCase):
    _DEBUG = False

    def run_cmd(self, cmdline):
        if False:
            print('Hello World!')
        popen = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timeout = threading.Timer(_TEST_TIMEOUT, popen.kill)
        try:
            timeout.start()
            (out, err) = popen.communicate()
            if popen.returncode != 0:
                raise AssertionError('process failed with code %s: stderr follows\n%s\n' % (popen.returncode, err.decode()))
        finally:
            timeout.cancel()
        return (out.decode(), err.decode())

    @linux_only
    def test_orphaned_semaphore(self):
        if False:
            while True:
                i = 10
        test_file = os.path.join(os.path.dirname(__file__), 'orphaned_semaphore_usecase.py')
        cmdline = [sys.executable, test_file]
        (out, err) = self.run_cmd(cmdline)
        self.assertNotIn('leaked semaphore', err)
        if self._DEBUG:
            print('OUT:', out)
            print('ERR:', err)

    def test_lazy_lock_init(self):
        if False:
            for i in range(10):
                print('nop')
        for meth in ('fork', 'spawn', 'forkserver'):
            try:
                multiprocessing.get_context(meth)
            except ValueError:
                continue
            cmd = "import numba; import multiprocessing;multiprocessing.set_start_method('{}');print(multiprocessing.get_context().get_start_method())"
            cmdline = [sys.executable, '-c', cmd.format(meth)]
            (out, err) = self.run_cmd(cmdline)
            if self._DEBUG:
                print('OUT:', out)
                print('ERR:', err)
            self.assertIn(meth, out)

@skip_parfors_unsupported
@skip_no_omp
class TestOpenMPVendors(TestCase):

    def test_vendors(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks the OpenMP vendor strings are correct\n        '
        expected = dict()
        expected['win32'] = 'MS'
        expected['darwin'] = 'Intel'
        expected['linux'] = 'GNU'
        for k in expected.keys():
            if sys.platform.startswith(k):
                self.assertEqual(expected[k], omppool.openmp_vendor)
if __name__ == '__main__':
    unittest.main()