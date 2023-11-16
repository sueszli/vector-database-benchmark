from contextlib import contextmanager
import shelve
from filelock import FileLock
_locks = {}

def _lock(filename):
    if False:
        return 10
    try:
        return _locks[filename]
    except KeyError:
        _locks[filename] = FileLock(filename + '.lock')
    return _locks[filename]

@contextmanager
def atomic_shelve(filename):
    if False:
        for i in range(10):
            print('nop')
    with _lock(filename):
        shelved = shelve.open(filename)
        yield shelved
        shelved.close()