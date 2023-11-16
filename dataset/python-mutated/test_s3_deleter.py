import gevent
import pytest
from gevent import lock
from boto.s3 import bucket
from boto.s3 import key
from fast_wait import fast_wait
from wal_e import exception
from wal_e.worker.s3 import s3_deleter
assert fast_wait

class BucketDeleteKeysCollector(object):
    """A callable to stand-in for bucket.delete_keys

    Used to test that given keys are bulk-deleted.

    Also can inject an exception.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.deleted_keys = []
        self.aborted_keys = []
        self.exc = None
        self._exc_protect = lock.RLock()

    def inject(self, exc):
        if False:
            for i in range(10):
                print('nop')
        self._exc_protect.acquire()
        self.exc = exc
        self._exc_protect.release()

    def __call__(self, keys):
        if False:
            while True:
                i = 10
        self._exc_protect.acquire()
        try:
            if self.exc:
                self.aborted_keys.extend(keys)
                gevent.sleep(0.1)
                raise self.exc
        finally:
            self._exc_protect.release()
        self.deleted_keys.extend(keys)

@pytest.fixture
def collect(monkeypatch):
    if False:
        while True:
            i = 10
    'Instead of performing bulk delete, collect key names deleted.\n\n    This is to test invariants, as to ensure deleted keys are passed\n    to boto properly.\n    '
    collect = BucketDeleteKeysCollector()
    monkeypatch.setattr(bucket.Bucket, 'delete_keys', collect)
    return collect

@pytest.fixture
def b():
    if False:
        for i in range(10):
            print('nop')
    return bucket.Bucket(name='test-bucket-name')

@pytest.fixture(autouse=True)
def never_use_single_delete(monkeypatch):
    if False:
        return 10
    'Detect any mistaken uses of single-key deletion.\n\n    Older wal-e versions used one-at-a-time deletions.  This is just\n    to help ensure that use of this API (through the nominal boto\n    symbol) is detected.\n    '

    def die():
        if False:
            while True:
                i = 10
        assert False
    monkeypatch.setattr(key.Key, 'delete', die)
    monkeypatch.setattr(bucket.Bucket, 'delete_key', die)

def test_construction():
    if False:
        i = 10
        return i + 15
    'The constructor basically works.'
    s3_deleter.Deleter()

def test_close_error():
    if False:
        for i in range(10):
            print('nop')
    'Ensure that attempts to use a closed Deleter results in an error.'
    d = s3_deleter.Deleter()
    d.close()
    with pytest.raises(exception.UserCritical):
        d.delete('no value should work')

def test_processes_one_deletion(b, collect):
    if False:
        while True:
            i = 10
    key_name = 'test-key-name'
    k = key.Key(bucket=b, name=key_name)
    d = s3_deleter.Deleter()
    d.delete(k)
    d.close()
    assert collect.deleted_keys == [key_name]

def test_processes_many_deletions(b, collect):
    if False:
        for i in range(10):
            print('nop')
    target = sorted(['test-key-' + str(x) for x in range(20001)])
    keys = [key.Key(bucket=b, name=key_name) for key_name in target]
    d = s3_deleter.Deleter()
    for k in keys:
        d.delete(k)
    d.close()
    assert sorted(collect.deleted_keys) == target

def test_retry_on_normal_error(b, collect):
    if False:
        print('Hello World!')
    'Ensure retries are processed for most errors.'
    key_name = 'test-key-name'
    k = key.Key(bucket=b, name=key_name)
    collect.inject(Exception('Normal error'))
    d = s3_deleter.Deleter()
    d.delete(k)
    while len(collect.aborted_keys) < 2:
        gevent.sleep(0.1)
    assert not collect.deleted_keys
    collect.inject(None)
    d.close()
    assert collect.deleted_keys == [key_name]

def test_no_retry_on_keyboadinterrupt(b, collect):
    if False:
        return 10
    'Ensure that KeyboardInterrupts are forwarded.'
    key_name = 'test-key-name'
    k = key.Key(bucket=b, name=key_name)

    class MarkedKeyboardInterrupt(KeyboardInterrupt):
        pass
    collect.inject(MarkedKeyboardInterrupt('SIGINT, probably'))
    d = s3_deleter.Deleter()
    with pytest.raises(MarkedKeyboardInterrupt):
        d.delete(k)
        while True:
            gevent.sleep(0.1)
    assert collect.aborted_keys == [key_name]
    collect.inject(None)
    d.close()
    assert not collect.deleted_keys