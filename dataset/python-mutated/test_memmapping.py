import os
import mmap
import sys
import platform
import gc
import pickle
import itertools
from time import sleep
import subprocess
import threading
import faulthandler
import pytest
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.test.common import with_dev_shm
from joblib.testing import raises, parametrize, skipif
from joblib.backports import make_memmap
from joblib.parallel import Parallel, delayed
from joblib.pool import MemmappingPool
from joblib.executor import _TestingMemmappingExecutor as TestExecutor
from joblib._memmapping_reducer import has_shareable_memory
from joblib._memmapping_reducer import ArrayMemmapForwardReducer
from joblib._memmapping_reducer import _strided_from_memmap
from joblib._memmapping_reducer import _get_temp_dir
from joblib._memmapping_reducer import _WeakArrayKeyMap
from joblib._memmapping_reducer import _get_backing_memmap
import joblib._memmapping_reducer as jmr

def setup_module():
    if False:
        for i in range(10):
            print('nop')
    faulthandler.dump_traceback_later(timeout=300, exit=True)

def teardown_module():
    if False:
        i = 10
        return i + 15
    faulthandler.cancel_dump_traceback_later()

def check_memmap_and_send_back(array):
    if False:
        return 10
    assert _get_backing_memmap(array) is not None
    return array

def check_array(args):
    if False:
        print('Hello World!')
    'Dummy helper function to be executed in subprocesses\n\n    Check that the provided array has the expected values in the provided\n    range.\n\n    '
    (data, position, expected) = args
    np.testing.assert_array_equal(data[position], expected)

def inplace_double(args):
    if False:
        print('Hello World!')
    'Dummy helper function to be executed in subprocesses\n\n\n    Check that the input array has the right values in the provided range\n    and perform an inplace modification to double the values in the range by\n    two.\n\n    '
    (data, position, expected) = args
    assert data[position] == expected
    data[position] *= 2
    np.testing.assert_array_equal(data[position], 2 * expected)

@with_numpy
@with_multiprocessing
def test_memmap_based_array_reducing(tmpdir):
    if False:
        return 10
    'Check that it is possible to reduce a memmap backed array'
    assert_array_equal = np.testing.assert_array_equal
    filename = tmpdir.join('test.mmap').strpath
    buffer = np.memmap(filename, dtype=np.float64, shape=500, mode='w+')
    buffer[:] = -1.0 * np.arange(buffer.shape[0], dtype=buffer.dtype)
    buffer.flush()
    a = np.memmap(filename, dtype=np.float64, shape=(3, 5, 4), mode='r+', order='F', offset=4)
    a[:] = np.arange(60).reshape(a.shape)
    b = a[1:-1, 2:-1, 2:4]
    c = np.asarray(b)
    d = c.T
    reducer = ArrayMemmapForwardReducer(None, tmpdir.strpath, 'c', True)

    def reconstruct_array_or_memmap(x):
        if False:
            for i in range(10):
                print('nop')
        (cons, args) = reducer(x)
        return cons(*args)
    a_reconstructed = reconstruct_array_or_memmap(a)
    assert has_shareable_memory(a_reconstructed)
    assert isinstance(a_reconstructed, np.memmap)
    assert_array_equal(a_reconstructed, a)
    b_reconstructed = reconstruct_array_or_memmap(b)
    assert has_shareable_memory(b_reconstructed)
    assert_array_equal(b_reconstructed, b)
    c_reconstructed = reconstruct_array_or_memmap(c)
    assert not isinstance(c_reconstructed, np.memmap)
    assert has_shareable_memory(c_reconstructed)
    assert_array_equal(c_reconstructed, c)
    d_reconstructed = reconstruct_array_or_memmap(d)
    assert not isinstance(d_reconstructed, np.memmap)
    assert has_shareable_memory(d_reconstructed)
    assert_array_equal(d_reconstructed, d)
    a3 = a * 3
    assert not has_shareable_memory(a3)
    a3_reconstructed = reconstruct_array_or_memmap(a3)
    assert not has_shareable_memory(a3_reconstructed)
    assert not isinstance(a3_reconstructed, np.memmap)
    assert_array_equal(a3_reconstructed, a * 3)
    b3 = np.asarray(a3)
    assert not has_shareable_memory(b3)
    b3_reconstructed = reconstruct_array_or_memmap(b3)
    assert isinstance(b3_reconstructed, np.ndarray)
    assert not has_shareable_memory(b3_reconstructed)
    assert_array_equal(b3_reconstructed, b3)

@with_multiprocessing
@skipif(sys.platform != 'win32' or (), reason='PermissionError only easily triggerable on Windows')
def test_resource_tracker_retries_when_permissionerror(tmpdir):
    if False:
        return 10
    filename = tmpdir.join('test.mmap').strpath
    cmd = 'if 1:\n    import os\n    import numpy as np\n    import time\n    from joblib.externals.loky.backend import resource_tracker\n    resource_tracker.VERBOSE = 1\n\n    # Start the resource tracker\n    resource_tracker.ensure_running()\n    time.sleep(1)\n\n    # Create a file containing numpy data\n    memmap = np.memmap(r"{filename}", dtype=np.float64, shape=10, mode=\'w+\')\n    memmap[:] = np.arange(10).astype(np.int8).data\n    memmap.flush()\n    assert os.path.exists(r"{filename}")\n    del memmap\n\n    # Create a np.memmap backed by this file\n    memmap = np.memmap(r"{filename}", dtype=np.float64, shape=10, mode=\'w+\')\n    resource_tracker.register(r"{filename}", "file")\n\n    # Ask the resource_tracker to delete the file backing the np.memmap , this\n    # should raise PermissionError that the resource_tracker will log.\n    resource_tracker.maybe_unlink(r"{filename}", "file")\n\n    # Wait for the resource_tracker to process the maybe_unlink before cleaning\n    # up the memmap\n    time.sleep(2)\n    '.format(filename=filename)
    p = subprocess.Popen([sys.executable, '-c', cmd], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    p.wait()
    (out, err) = p.communicate()
    assert p.returncode == 0
    assert out == b''
    msg = 'tried to unlink {}, got PermissionError'.format(filename)
    assert msg in err.decode()

@with_numpy
@with_multiprocessing
def test_high_dimension_memmap_array_reducing(tmpdir):
    if False:
        while True:
            i = 10
    assert_array_equal = np.testing.assert_array_equal
    filename = tmpdir.join('test.mmap').strpath
    a = np.memmap(filename, dtype=np.float64, shape=(100, 15, 15, 3), mode='w+')
    a[:] = np.arange(100 * 15 * 15 * 3).reshape(a.shape)
    b = a[0:10]
    c = a[:, 5:10]
    d = a[:, :, :, 0]
    e = a[1:3:4]
    reducer = ArrayMemmapForwardReducer(None, tmpdir.strpath, 'c', True)

    def reconstruct_array_or_memmap(x):
        if False:
            while True:
                i = 10
        (cons, args) = reducer(x)
        return cons(*args)
    a_reconstructed = reconstruct_array_or_memmap(a)
    assert has_shareable_memory(a_reconstructed)
    assert isinstance(a_reconstructed, np.memmap)
    assert_array_equal(a_reconstructed, a)
    b_reconstructed = reconstruct_array_or_memmap(b)
    assert has_shareable_memory(b_reconstructed)
    assert_array_equal(b_reconstructed, b)
    c_reconstructed = reconstruct_array_or_memmap(c)
    assert has_shareable_memory(c_reconstructed)
    assert_array_equal(c_reconstructed, c)
    d_reconstructed = reconstruct_array_or_memmap(d)
    assert has_shareable_memory(d_reconstructed)
    assert_array_equal(d_reconstructed, d)
    e_reconstructed = reconstruct_array_or_memmap(e)
    assert has_shareable_memory(e_reconstructed)
    assert_array_equal(e_reconstructed, e)

@with_numpy
def test__strided_from_memmap(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    fname = tmpdir.join('test.mmap').strpath
    size = 5 * mmap.ALLOCATIONGRANULARITY
    offset = mmap.ALLOCATIONGRANULARITY + 1
    memmap_obj = np.memmap(fname, mode='w+', shape=size + offset)
    memmap_obj = _strided_from_memmap(fname, dtype='uint8', mode='r', offset=offset, order='C', shape=size, strides=None, total_buffer_len=None, unlink_on_gc_collect=False)
    assert isinstance(memmap_obj, np.memmap)
    assert memmap_obj.offset == offset
    memmap_backed_obj = _strided_from_memmap(fname, dtype='uint8', mode='r', offset=offset, order='C', shape=(size // 2,), strides=(2,), total_buffer_len=size, unlink_on_gc_collect=False)
    assert _get_backing_memmap(memmap_backed_obj).offset == offset

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_pool_with_memmap(factory, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Check that subprocess can access and update shared memory memmap'
    assert_array_equal = np.testing.assert_array_equal
    pool_temp_folder = tmpdir.mkdir('pool').strpath
    p = factory(10, max_nbytes=2, temp_folder=pool_temp_folder)
    try:
        filename = tmpdir.join('test.mmap').strpath
        a = np.memmap(filename, dtype=np.float32, shape=(3, 5), mode='w+')
        a.fill(1.0)
        p.map(inplace_double, [(a, (i, j), 1.0) for i in range(a.shape[0]) for j in range(a.shape[1])])
        assert_array_equal(a, 2 * np.ones(a.shape))
        b = np.memmap(filename, dtype=np.float32, shape=(5, 3), mode='c')
        p.map(inplace_double, [(b, (i, j), 2.0) for i in range(b.shape[0]) for j in range(b.shape[1])])
        assert os.listdir(pool_temp_folder) == []
        assert_array_equal(a, 2 * np.ones(a.shape))
        assert_array_equal(b, 2 * np.ones(b.shape))
        c = np.memmap(filename, dtype=np.float32, shape=(10,), mode='r', offset=5 * 4)
        with raises(AssertionError):
            p.map(check_array, [(c, i, 3.0) for i in range(c.shape[0])])
        with raises((RuntimeError, ValueError)):
            p.map(inplace_double, [(c, i, 2.0) for i in range(c.shape[0])])
    finally:
        p.terminate()
        del p

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_pool_with_memmap_array_view(factory, tmpdir):
    if False:
        print('Hello World!')
    'Check that subprocess can access and update shared memory array'
    assert_array_equal = np.testing.assert_array_equal
    pool_temp_folder = tmpdir.mkdir('pool').strpath
    p = factory(10, max_nbytes=2, temp_folder=pool_temp_folder)
    try:
        filename = tmpdir.join('test.mmap').strpath
        a = np.memmap(filename, dtype=np.float32, shape=(3, 5), mode='w+')
        a.fill(1.0)
        a_view = np.asarray(a)
        assert not isinstance(a_view, np.memmap)
        assert has_shareable_memory(a_view)
        p.map(inplace_double, [(a_view, (i, j), 1.0) for i in range(a.shape[0]) for j in range(a.shape[1])])
        assert_array_equal(a, 2 * np.ones(a.shape))
        assert_array_equal(a_view, 2 * np.ones(a.shape))
        assert os.listdir(pool_temp_folder) == []
    finally:
        p.terminate()
        del p

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_permission_error_windows_reference_cycle(backend):
    if False:
        return 10
    cmd = 'if 1:\n        import numpy as np\n        from joblib import Parallel, delayed\n\n\n        data = np.random.rand(int(2e6)).reshape((int(1e6), 2))\n\n        # Build a complex cyclic reference that is likely to delay garbage\n        # collection of the memmapped array in the worker processes.\n        first_list = current_list = [data]\n        for i in range(10):\n            current_list = [current_list]\n        first_list.append(current_list)\n\n        if __name__ == "__main__":\n            results = Parallel(n_jobs=2, backend="{b}")(\n                delayed(len)(current_list) for i in range(10))\n            assert results == [1] * 10\n    '.format(b=backend)
    p = subprocess.Popen([sys.executable, '-c', cmd], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    p.wait()
    (out, err) = p.communicate()
    assert p.returncode == 0, out.decode() + '\n\n' + err.decode()

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_permission_error_windows_memmap_sent_to_parent(backend):
    if False:
        for i in range(10):
            print('nop')
    cmd = "if 1:\n        import os\n        import time\n\n        import numpy as np\n\n        from joblib import Parallel, delayed\n        from testutils import return_slice_of_data\n\n        data = np.ones(int(2e6))\n\n        if __name__ == '__main__':\n            # warm-up call to launch the workers and start the resource_tracker\n            _ = Parallel(n_jobs=2, verbose=5, backend='{b}')(\n                delayed(id)(i) for i in range(20))\n\n            time.sleep(0.5)\n\n            slice_of_data = Parallel(n_jobs=2, verbose=5, backend='{b}')(\n                delayed(return_slice_of_data)(data, 0, 20) for _ in range(10))\n    ".format(b=backend)
    for _ in range(3):
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.dirname(__file__)
        p = subprocess.Popen([sys.executable, '-c', cmd], stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        p.wait()
        (out, err) = p.communicate()
        assert p.returncode == 0, err
        assert out == b''
        if sys.version_info[:3] not in [(3, 8, 0), (3, 8, 1)]:
            assert b'resource_tracker' not in err

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_parallel_isolated_temp_folders(backend):
    if False:
        i = 10
        return i + 15
    array = np.arange(int(100.0))
    [filename_1] = Parallel(n_jobs=2, backend=backend, max_nbytes=10)((delayed(getattr)(array, 'filename') for _ in range(1)))
    [filename_2] = Parallel(n_jobs=2, backend=backend, max_nbytes=10)((delayed(getattr)(array, 'filename') for _ in range(1)))
    assert os.path.dirname(filename_2) != os.path.dirname(filename_1)

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_managed_backend_reuse_temp_folder(backend):
    if False:
        return 10
    array = np.arange(int(100.0))
    with Parallel(n_jobs=2, backend=backend, max_nbytes=10) as p:
        [filename_1] = p((delayed(getattr)(array, 'filename') for _ in range(1)))
        [filename_2] = p((delayed(getattr)(array, 'filename') for _ in range(1)))
    assert os.path.dirname(filename_2) == os.path.dirname(filename_1)

@with_numpy
@with_multiprocessing
def test_memmapping_temp_folder_thread_safety():
    if False:
        for i in range(10):
            print('nop')
    array = np.arange(int(100.0))
    temp_dirs_thread_1 = set()
    temp_dirs_thread_2 = set()

    def concurrent_get_filename(array, temp_dirs):
        if False:
            print('Hello World!')
        with Parallel(backend='loky', n_jobs=2, max_nbytes=10) as p:
            for i in range(10):
                [filename] = p((delayed(getattr)(array, 'filename') for _ in range(1)))
                temp_dirs.add(os.path.dirname(filename))
    t1 = threading.Thread(target=concurrent_get_filename, args=(array, temp_dirs_thread_1))
    t2 = threading.Thread(target=concurrent_get_filename, args=(array, temp_dirs_thread_2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(temp_dirs_thread_1) == 1
    assert len(temp_dirs_thread_2) == 1
    assert temp_dirs_thread_1 != temp_dirs_thread_2

@with_numpy
@with_multiprocessing
def test_multithreaded_parallel_termination_resource_tracker_silent():
    if False:
        while True:
            i = 10
    cmd = 'if 1:\n        import os\n        import numpy as np\n        from joblib import Parallel, delayed\n        from joblib.externals.loky.backend import resource_tracker\n        from concurrent.futures import ThreadPoolExecutor, wait\n\n        resource_tracker.VERBOSE = 0\n\n        array = np.arange(int(1e2))\n\n        temp_dirs_thread_1 = set()\n        temp_dirs_thread_2 = set()\n\n\n        def raise_error(array):\n            raise ValueError\n\n\n        def parallel_get_filename(array, temp_dirs):\n            with Parallel(backend="loky", n_jobs=2, max_nbytes=10) as p:\n                for i in range(10):\n                    [filename] = p(\n                        delayed(getattr)(array, "filename") for _ in range(1)\n                    )\n                    temp_dirs.add(os.path.dirname(filename))\n\n\n        def parallel_raise(array, temp_dirs):\n            with Parallel(backend="loky", n_jobs=2, max_nbytes=10) as p:\n                for i in range(10):\n                    [filename] = p(\n                        delayed(raise_error)(array) for _ in range(1)\n                    )\n                    temp_dirs.add(os.path.dirname(filename))\n\n\n        executor = ThreadPoolExecutor(max_workers=2)\n\n        # both function calls will use the same loky executor, but with a\n        # different Parallel object.\n        future_1 = executor.submit({f1}, array, temp_dirs_thread_1)\n        future_2 = executor.submit({f2}, array, temp_dirs_thread_2)\n\n        # Wait for both threads to terminate their backend\n        wait([future_1, future_2])\n\n        future_1.result()\n        future_2.result()\n    '
    functions_and_returncodes = [('parallel_get_filename', 'parallel_get_filename', 0), ('parallel_get_filename', 'parallel_raise', 1), ('parallel_raise', 'parallel_raise', 1)]
    for (f1, f2, returncode) in functions_and_returncodes:
        p = subprocess.Popen([sys.executable, '-c', cmd.format(f1=f1, f2=f2)], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        p.wait()
        (out, err) = p.communicate()
        assert p.returncode == returncode, out.decode()
        assert b'resource_tracker' not in err, err.decode()

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_many_parallel_calls_on_same_object(backend):
    if False:
        while True:
            i = 10
    cmd = "if 1:\n        import os\n        import time\n\n        import numpy as np\n\n        from joblib import Parallel, delayed\n        from testutils import return_slice_of_data\n\n        data = np.ones(100)\n\n        if __name__ == '__main__':\n            for i in range(5):\n                slice_of_data = Parallel(\n                    n_jobs=2, max_nbytes=1, backend='{b}')(\n                        delayed(return_slice_of_data)(data, 0, 20)\n                        for _ in range(10)\n                    )\n    ".format(b=backend)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(__file__)
    p = subprocess.Popen([sys.executable, '-c', cmd], stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    p.wait()
    (out, err) = p.communicate()
    assert p.returncode == 0, err
    assert out == b''
    if sys.version_info[:3] not in [(3, 8, 0), (3, 8, 1)]:
        assert b'resource_tracker' not in err

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_memmap_returned_as_regular_array(backend):
    if False:
        i = 10
        return i + 15
    data = np.ones(int(1000.0))
    [result] = Parallel(n_jobs=2, backend=backend, max_nbytes=100)((delayed(check_memmap_and_send_back)(data) for _ in range(1)))
    assert _get_backing_memmap(result) is None

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_resource_tracker_silent_when_reference_cycles(backend):
    if False:
        print('Hello World!')
    if backend == 'loky' and sys.platform.startswith('win'):
        pytest.xfail('The temporary folder cannot be deleted on Windows in the presence of a reference cycle')
    cmd = 'if 1:\n        import numpy as np\n        from joblib import Parallel, delayed\n\n\n        data = np.random.rand(int(2e6)).reshape((int(1e6), 2))\n\n        # Build a complex cyclic reference that is likely to delay garbage\n        # collection of the memmapped array in the worker processes.\n        first_list = current_list = [data]\n        for i in range(10):\n            current_list = [current_list]\n        first_list.append(current_list)\n\n        if __name__ == "__main__":\n            results = Parallel(n_jobs=2, backend="{b}")(\n                delayed(len)(current_list) for i in range(10))\n            assert results == [1] * 10\n    '.format(b=backend)
    p = subprocess.Popen([sys.executable, '-c', cmd], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    p.wait()
    (out, err) = p.communicate()
    out = out.decode()
    err = err.decode()
    assert p.returncode == 0, out + '\n\n' + err
    assert 'resource_tracker' not in err, err

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_pool_for_large_arrays(factory, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Check that large arrays are not copied in memory'
    assert os.listdir(tmpdir.strpath) == []
    p = factory(3, max_nbytes=40, temp_folder=tmpdir.strpath, verbose=2)
    try:
        assert os.listdir(tmpdir.strpath) == []
        assert not os.path.exists(p._temp_folder)
        small = np.ones(5, dtype=np.float32)
        assert small.nbytes == 20
        p.map(check_array, [(small, i, 1.0) for i in range(small.shape[0])])
        assert os.listdir(tmpdir.strpath) == []
        large = np.ones(100, dtype=np.float64)
        assert large.nbytes == 800
        p.map(check_array, [(large, i, 1.0) for i in range(large.shape[0])])
        assert os.path.isdir(p._temp_folder)
        dumped_filenames = os.listdir(p._temp_folder)
        assert len(dumped_filenames) == 1
        objects = np.array(['abc'] * 100, dtype='object')
        results = p.map(has_shareable_memory, [objects])
        assert not results[0]
    finally:
        p.terminate()
        for i in range(10):
            sleep(0.1)
            if not os.path.exists(p._temp_folder):
                break
        else:
            raise AssertionError('temporary folder {} was not deleted'.format(p._temp_folder))
        del p

@with_numpy
@with_multiprocessing
@parametrize('backend', [pytest.param('multiprocessing', marks=pytest.mark.xfail(reason='https://github.com/joblib/joblib/issues/1086')), 'loky'])
def test_child_raises_parent_exits_cleanly(backend):
    if False:
        while True:
            i = 10
    cmd = 'if 1:\n        import os\n        from pathlib import Path\n        from time import sleep\n\n        import numpy as np\n        from joblib import Parallel, delayed\n        from testutils import print_filename_and_raise\n\n        data = np.random.rand(1000)\n\n        def get_temp_folder(parallel_obj, backend):\n            if "{b}" == "loky":\n                return Path(parallel_obj._backend._workers._temp_folder)\n            else:\n                return Path(parallel_obj._backend._pool._temp_folder)\n\n\n        if __name__ == "__main__":\n            try:\n                with Parallel(n_jobs=2, backend="{b}", max_nbytes=100) as p:\n                    temp_folder = get_temp_folder(p, "{b}")\n                    p(delayed(print_filename_and_raise)(data)\n                              for i in range(1))\n            except ValueError as e:\n                # the temporary folder should be deleted by the end of this\n                # call but apparently on some file systems, this takes\n                # some time to be visible.\n                #\n                # We attempt to write into the temporary folder to test for\n                # its existence and we wait for a maximum of 10 seconds.\n                for i in range(100):\n                    try:\n                        with open(temp_folder / "some_file.txt", "w") as f:\n                            f.write("some content")\n                    except FileNotFoundError:\n                        # temp_folder has been deleted, all is fine\n                        break\n\n                    # ... else, wait a bit and try again\n                    sleep(.1)\n                else:\n                    raise AssertionError(\n                        str(temp_folder) + " was not deleted"\n                    ) from e\n    '.format(b=backend)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.dirname(__file__)
    p = subprocess.Popen([sys.executable, '-c', cmd], stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    p.wait()
    (out, err) = p.communicate()
    (out, err) = (out.decode(), err.decode())
    filename = out.split('\n')[0]
    assert p.returncode == 0, err or out
    assert err == ''
    assert not os.path.exists(filename)

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_pool_for_large_arrays_disabled(factory, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Check that large arrays memmapping can be disabled'
    p = factory(3, max_nbytes=None, temp_folder=tmpdir.strpath)
    try:
        assert os.listdir(tmpdir.strpath) == []
        large = np.ones(100, dtype=np.float64)
        assert large.nbytes == 800
        p.map(check_array, [(large, i, 1.0) for i in range(large.shape[0])])
        assert os.listdir(tmpdir.strpath) == []
    finally:
        p.terminate()
        del p

@with_numpy
@with_multiprocessing
@with_dev_shm
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_on_large_enough_dev_shm(factory):
    if False:
        for i in range(10):
            print('nop')
    'Check that memmapping uses /dev/shm when possible'
    orig_size = jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE
    try:
        jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE = int(32000000.0)
        p = factory(3, max_nbytes=10)
        try:
            pool_temp_folder = p._temp_folder
            folder_prefix = '/dev/shm/joblib_memmapping_folder_'
            assert pool_temp_folder.startswith(folder_prefix)
            assert os.path.exists(pool_temp_folder)
            a = np.ones(100, dtype=np.float64)
            assert a.nbytes == 800
            p.map(id, [a] * 10)
            assert len(os.listdir(pool_temp_folder)) == 1
            b = np.ones(100, dtype=np.float64) * 2
            assert b.nbytes == 800
            p.map(id, [b] * 10)
            assert len(os.listdir(pool_temp_folder)) == 2
        finally:
            p.terminate()
            del p
        for i in range(100):
            if not os.path.exists(pool_temp_folder):
                break
            sleep(0.1)
        else:
            raise AssertionError('temporary folder of pool was not deleted')
    finally:
        jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE = orig_size

@with_numpy
@with_multiprocessing
@with_dev_shm
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_on_too_small_dev_shm(factory):
    if False:
        while True:
            i = 10
    orig_size = jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE
    try:
        jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE = int(4.2e+19)
        p = factory(3, max_nbytes=10)
        try:
            pool_temp_folder = p._temp_folder
            assert not pool_temp_folder.startswith('/dev/shm')
        finally:
            p.terminate()
            del p
        assert not os.path.exists(pool_temp_folder)
    finally:
        jmr.SYSTEM_SHARED_MEM_FS_MIN_SIZE = orig_size

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_pool_for_large_arrays_in_return(factory, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Check that large arrays are not copied in memory in return'
    assert_array_equal = np.testing.assert_array_equal
    p = factory(3, max_nbytes=10, temp_folder=tmpdir.strpath)
    try:
        res = p.apply_async(np.ones, args=(1000,))
        large = res.get()
        assert not has_shareable_memory(large)
        assert_array_equal(large, np.ones(1000))
    finally:
        p.terminate()
        del p

def _worker_multiply(a, n_times):
    if False:
        i = 10
        return i + 15
    'Multiplication function to be executed by subprocess'
    assert has_shareable_memory(a)
    return a * n_times

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_workaround_against_bad_memmap_with_copied_buffers(factory, tmpdir):
    if False:
        return 10
    'Check that memmaps with a bad buffer are returned as regular arrays\n\n    Unary operations and ufuncs on memmap instances return a new memmap\n    instance with an in-memory buffer (probably a numpy bug).\n    '
    assert_array_equal = np.testing.assert_array_equal
    p = factory(3, max_nbytes=10, temp_folder=tmpdir.strpath)
    try:
        a = np.asarray(np.arange(6000).reshape((1000, 2, 3)), order='F')[:, :1, :]
        b = p.apply_async(_worker_multiply, args=(a, 3)).get()
        assert not has_shareable_memory(b)
        assert_array_equal(b, 3 * a)
    finally:
        p.terminate()
        del p

def identity(arg):
    if False:
        print('Hello World!')
    return arg

@with_numpy
@with_multiprocessing
@parametrize('factory,retry_no', list(itertools.product([MemmappingPool, TestExecutor.get_memmapping_executor], range(3))), ids=['{}, {}'.format(x, y) for (x, y) in itertools.product(['multiprocessing', 'loky'], map(str, range(3)))])
def test_pool_memmap_with_big_offset(factory, retry_no, tmpdir):
    if False:
        print('Hello World!')
    fname = tmpdir.join('test.mmap').strpath
    size = 5 * mmap.ALLOCATIONGRANULARITY
    offset = mmap.ALLOCATIONGRANULARITY + 1
    obj = make_memmap(fname, mode='w+', shape=size, dtype='uint8', offset=offset)
    p = factory(2, temp_folder=tmpdir.strpath)
    result = p.apply_async(identity, args=(obj,)).get()
    assert isinstance(result, np.memmap)
    assert result.offset == offset
    np.testing.assert_array_equal(obj, result)
    p.terminate()

def test_pool_get_temp_dir(tmpdir):
    if False:
        print('Hello World!')
    pool_folder_name = 'test.tmpdir'
    (pool_folder, shared_mem) = _get_temp_dir(pool_folder_name, tmpdir.strpath)
    assert shared_mem is False
    assert pool_folder == tmpdir.join('test.tmpdir').strpath
    (pool_folder, shared_mem) = _get_temp_dir(pool_folder_name, temp_folder=None)
    if sys.platform.startswith('win'):
        assert shared_mem is False
    assert pool_folder.endswith(pool_folder_name)

def test_pool_get_temp_dir_no_statvfs(tmpdir, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Check that _get_temp_dir works when os.statvfs is not defined\n\n    Regression test for #902\n    '
    pool_folder_name = 'test.tmpdir'
    import joblib._memmapping_reducer
    if hasattr(joblib._memmapping_reducer.os, 'statvfs'):
        monkeypatch.delattr(joblib._memmapping_reducer.os, 'statvfs')
    (pool_folder, shared_mem) = _get_temp_dir(pool_folder_name, temp_folder=None)
    if sys.platform.startswith('win'):
        assert shared_mem is False
    assert pool_folder.endswith(pool_folder_name)

@with_numpy
@skipif(sys.platform == 'win32', reason='This test fails with a PermissionError on Windows')
@parametrize('mmap_mode', ['r+', 'w+'])
def test_numpy_arrays_use_different_memory(mmap_mode):
    if False:
        return 10

    def func(arr, value):
        if False:
            while True:
                i = 10
        arr[:] = value
        return arr
    arrays = [np.zeros((10, 10), dtype='float64') for i in range(10)]
    results = Parallel(mmap_mode=mmap_mode, max_nbytes=0, n_jobs=2)((delayed(func)(arr, i) for (i, arr) in enumerate(arrays)))
    for (i, arr) in enumerate(results):
        np.testing.assert_array_equal(arr, i)

@with_numpy
def test_weak_array_key_map():
    if False:
        for i in range(10):
            print('nop')

    def assert_empty_after_gc_collect(container, retries=100):
        if False:
            for i in range(10):
                print('nop')
        for i in range(retries):
            if len(container) == 0:
                return
            gc.collect()
            sleep(0.1)
        assert len(container) == 0
    a = np.ones(42)
    m = _WeakArrayKeyMap()
    m.set(a, 'a')
    assert m.get(a) == 'a'
    b = a
    assert m.get(b) == 'a'
    m.set(b, 'b')
    assert m.get(a) == 'b'
    del a
    gc.collect()
    assert len(m._data) == 1
    assert m.get(b) == 'b'
    del b
    assert_empty_after_gc_collect(m._data)
    c = np.ones(42)
    m.set(c, 'c')
    assert len(m._data) == 1
    assert m.get(c) == 'c'
    with raises(KeyError):
        m.get(np.ones(42))
    del c
    assert_empty_after_gc_collect(m._data)

    def get_set_get_collect(m, i):
        if False:
            for i in range(10):
                print('nop')
        a = np.ones(42)
        with raises(KeyError):
            m.get(a)
        m.set(a, i)
        assert m.get(a) == i
        return id(a)
    unique_ids = set([get_set_get_collect(m, i) for i in range(1000)])
    if platform.python_implementation() == 'CPython':
        max_len_unique_ids = 400 if getattr(sys.flags, 'nogil', False) else 100
        assert len(unique_ids) < max_len_unique_ids

def test_weak_array_key_map_no_pickling():
    if False:
        return 10
    m = _WeakArrayKeyMap()
    with raises(pickle.PicklingError):
        pickle.dumps(m)

@with_numpy
@with_multiprocessing
def test_direct_mmap(tmpdir):
    if False:
        while True:
            i = 10
    testfile = str(tmpdir.join('arr.dat'))
    a = np.arange(10, dtype='uint8')
    a.tofile(testfile)

    def _read_array():
        if False:
            i = 10
            return i + 15
        with open(testfile) as fd:
            mm = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ, offset=0)
        return np.ndarray((10,), dtype=np.uint8, buffer=mm, offset=0)

    def func(x):
        if False:
            for i in range(10):
                print('nop')
        return x ** 2
    arr = _read_array()
    ref = Parallel(n_jobs=2)((delayed(func)(x) for x in [a]))
    results = Parallel(n_jobs=2)((delayed(func)(x) for x in [arr]))
    np.testing.assert_array_equal(results, ref)

    def worker():
        if False:
            while True:
                i = 10
        return _read_array()
    results = Parallel(n_jobs=2)((delayed(worker)() for _ in range(1)))
    np.testing.assert_array_equal(results[0], arr)