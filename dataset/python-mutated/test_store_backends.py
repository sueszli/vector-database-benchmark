try:
    import cPickle as cpickle
except ImportError:
    import pickle as cpickle
import functools
from pickle import PicklingError
import time
import pytest
from joblib.testing import parametrize, timeout
from joblib.test.common import with_multiprocessing
from joblib.backports import concurrency_safe_rename
from joblib import Parallel, delayed
from joblib._store_backends import concurrency_safe_write, FileSystemStoreBackend, CacheWarning

def write_func(output, filename):
    if False:
        i = 10
        return i + 15
    with open(filename, 'wb') as f:
        cpickle.dump(output, f)

def load_func(expected, filename):
    if False:
        while True:
            i = 10
    for i in range(10):
        try:
            with open(filename, 'rb') as f:
                reloaded = cpickle.load(f)
            break
        except (OSError, IOError):
            time.sleep(0.1)
    else:
        raise
    assert expected == reloaded

def concurrency_safe_write_rename(to_write, filename, write_func):
    if False:
        for i in range(10):
            print('nop')
    temporary_filename = concurrency_safe_write(to_write, filename, write_func)
    concurrency_safe_rename(temporary_filename, filename)

@timeout(0)
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky', 'threading'])
def test_concurrency_safe_write(tmpdir, backend):
    if False:
        while True:
            i = 10
    filename = tmpdir.join('test.pkl').strpath
    obj = {str(i): i for i in range(int(100000.0))}
    funcs = [functools.partial(concurrency_safe_write_rename, write_func=write_func) if i % 3 != 2 else load_func for i in range(12)]
    Parallel(n_jobs=2, backend=backend)((delayed(func)(obj, filename) for func in funcs))

def test_warning_on_dump_failure(tmpdir):
    if False:
        print('Hello World!')

    class UnpicklableObject(object):

        def __reduce__(self):
            if False:
                i = 10
                return i + 15
            raise RuntimeError('some exception')
    backend = FileSystemStoreBackend()
    backend.location = tmpdir.join('test_warning_on_pickling_error').strpath
    backend.compress = None
    with pytest.warns(CacheWarning, match='some exception'):
        backend.dump_item('testpath', UnpicklableObject())

def test_warning_on_pickling_error(tmpdir):
    if False:
        i = 10
        return i + 15

    class UnpicklableObject(object):

        def __reduce__(self):
            if False:
                for i in range(10):
                    print('nop')
            raise PicklingError('not picklable')
    backend = FileSystemStoreBackend()
    backend.location = tmpdir.join('test_warning_on_pickling_error').strpath
    backend.compress = None
    with pytest.warns(FutureWarning, match='not picklable'):
        backend.dump_item('testpath', UnpicklableObject())