import os
import random
import pytest
from ..hashindex import NSIndex

def verify_hash_table(kv, idx):
    if False:
        print('Hello World!')
    'kv should be a python dictionary and idx an NSIndex.  Check that idx\n    has the expected entries and the right number of entries.\n    '
    for (k, v) in kv.items():
        assert k in idx and idx[k] == (v, v, v)
    assert len(idx) == len(kv)

def make_hashtables(*, entries, loops):
    if False:
        i = 10
        return i + 15
    idx = NSIndex()
    kv = {}
    for i in range(loops):
        for j in range(entries):
            k = random.randbytes(32)
            v = random.randint(0, NSIndex.MAX_VALUE - 1)
            idx[k] = (v, v, v)
            kv[k] = v
        delete_keys = random.sample(list(kv), k=random.randint(0, len(kv)))
        for k in delete_keys:
            v = kv.pop(k)
            assert idx.pop(k) == (v, v, v)
        verify_hash_table(kv, idx)
    return (idx, kv)

@pytest.mark.skipif('BORG_TESTS_SLOW' not in os.environ, reason='slow tests not enabled, use BORG_TESTS_SLOW=1')
def test_hashindex_stress():
    if False:
        return 10
    "checks if the hashtable behaves as expected\n\n    This can be used in _hashindex.c before running this test to provoke more collisions (don't forget to compile):\n    #define HASH_MAX_LOAD .99\n    #define HASH_MAX_EFF_LOAD .999\n    "
    make_hashtables(entries=10000, loops=1000)

def test_hashindex_compact():
    if False:
        while True:
            i = 10
    'test that we do not lose or corrupt data by the compaction nor by expanding/rebuilding'
    (idx, kv) = make_hashtables(entries=5000, loops=5)
    size_noncompact = idx.size()
    saved_space = idx.compact()
    size_compact = idx.size()
    assert saved_space > 0
    assert size_noncompact - size_compact == saved_space
    verify_hash_table(kv, idx)
    k = b'x' * 32
    idx[k] = (0, 0, 0)
    kv[k] = 0
    size_rebuilt = idx.size()
    assert size_rebuilt > size_compact + 1
    verify_hash_table(kv, idx)

@pytest.mark.skipif('BORG_TESTS_SLOW' not in os.environ, reason='slow tests not enabled, use BORG_TESTS_SLOW=1')
def test_hashindex_compact_stress():
    if False:
        while True:
            i = 10
    for _ in range(100):
        test_hashindex_compact()