import pytest
from wal_e import exception
from wal_e import worker

class FakeTarPartition(object):
    """Implements enough protocol to test concurrency semantics."""

    def __init__(self, num_members, explosive=False):
        if False:
            return 10
        self._explosive = explosive
        self.num_members = num_members

    def __len__(self):
        if False:
            return 10
        return self.num_members

class FakeUploader(object):
    """A no-op uploader that makes affordance for fault injection."""

    def __call__(self, tpart):
        if False:
            return 10
        if tpart._explosive:
            raise tpart._explosive
        return tpart

class Explosion(Exception):
    """Marker type of injected faults."""
    pass

def make_pool(max_concurrency, max_members):
    if False:
        return 10
    'Set up a pool with a FakeUploader'
    return worker.TarUploadPool(FakeUploader(), max_concurrency, max_members)

def test_simple():
    if False:
        for i in range(10):
            print('nop')
    'Simple case of uploading one partition.'
    pool = make_pool(1, 1)
    pool.put(FakeTarPartition(1))
    pool.join()

def test_not_enough_resources():
    if False:
        for i in range(10):
            print('nop')
    'Detect if a too-large segment can never complete.'
    pool = make_pool(1, 1)
    with pytest.raises(exception.UserCritical):
        pool.put(FakeTarPartition(2))
    pool.join()

def test_simple_concurrency():
    if False:
        return 10
    'Try a pool that cannot execute all submitted jobs at once.'
    pool = make_pool(1, 1)
    for i in range(3):
        pool.put(FakeTarPartition(1))
    pool.join()

def test_fault_midstream():
    if False:
        for i in range(10):
            print('nop')
    'Test if a previous upload fault is detected in calling .put.\n\n    This case is seen while pipelining many uploads in excess of the\n    maximum concurrency.\n\n    NB: This test is critical as to prevent failed uploads from\n    failing to notify a caller that the entire backup is incomplete.\n    '
    pool = make_pool(1, 1)
    tpart = FakeTarPartition(1, explosive=Explosion('Boom'))
    pool.put(tpart)
    tpart = FakeTarPartition(1)
    with pytest.raises(Explosion):
        pool.put(tpart)

def test_fault_join():
    if False:
        return 10
    'Test if a fault is detected when .join is used.\n\n    This case is seen at the end of a series of uploads.\n\n    NB: This test is critical as to prevent failed uploads from\n    failing to notify a caller that the entire backup is incomplete.\n    '
    pool = make_pool(1, 1)
    tpart = FakeTarPartition(1, explosive=Explosion('Boom'))
    pool.put(tpart)
    with pytest.raises(Explosion):
        pool.join()

def test_put_after_join():
    if False:
        print('Hello World!')
    'New jobs cannot be submitted after a .join\n\n    This is mostly a re-check to detect programming errors.\n    '
    pool = make_pool(1, 1)
    pool.join()
    with pytest.raises(exception.UserCritical):
        pool.put(FakeTarPartition(1))

def test_pool_concurrent_success():
    if False:
        i = 10
        return i + 15
    pool = make_pool(4, 4)
    for i in range(30):
        pool.put(FakeTarPartition(1))
    pool.join()

def test_pool_concurrent_failure():
    if False:
        for i in range(10):
            print('nop')
    pool = make_pool(4, 4)
    parts = [FakeTarPartition(1) for i in range(30)]
    exc = Explosion('boom')
    parts[27]._explosive = exc
    with pytest.raises(Explosion) as e:
        for part in parts:
            pool.put(part)
        pool.join()
    assert e.value is exc