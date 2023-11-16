import gc
import sys
from pyo3_pytests.objstore import ObjStore

def test_objstore_doesnot_leak_memory():
    if False:
        for i in range(10):
            print('nop')
    N = 10000
    message = b'\\(-"-;) Praying that memory leak would not happen..'
    getrefcount = getattr(sys, 'getrefcount', lambda obj: 0)
    before = getrefcount(message)
    store = ObjStore()
    for _ in range(N):
        store.push(message)
    del store
    gc.collect()
    after = getrefcount(message)
    assert after - before == 0