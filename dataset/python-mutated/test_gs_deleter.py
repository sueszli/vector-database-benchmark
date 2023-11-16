import gevent
import pytest
from gevent import lock
from fast_wait import fast_wait
from google.cloud import storage
from wal_e import exception
from wal_e.worker.gs import gs_deleter
assert fast_wait

class BucketDeleteBlobsCollector(object):
    """A callable to stand-in for bucket.delete_blobs

    Used to test that given blobs are bulk-deleted.

    Also can inject an exception.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.deleted_blobs = []
        self.aborted_blobs = []
        self.exc = None
        self._exc_protect = lock.RLock()

    def inject(self, exc):
        if False:
            while True:
                i = 10
        self._exc_protect.acquire()
        self.exc = exc
        self._exc_protect.release()

    def __call__(self, blobs, on_error=None):
        if False:
            while True:
                i = 10
        self._exc_protect.acquire()
        assert on_error is gs_deleter._on_error
        try:
            if self.exc:
                self.aborted_blobs.extend((blob.name for blob in blobs))
                gevent.sleep(0.1)
                raise self.exc
        finally:
            self._exc_protect.release()
        self.deleted_blobs.extend((blob.name for blob in blobs))

@pytest.fixture
def collect(monkeypatch):
    if False:
        while True:
            i = 10
    'Instead of performing bulk delete, collect blob names deleted.\n\n    This is to test invariants, as to ensure deleted blobs are passed\n    to google cloud properly.\n    '
    collect = BucketDeleteBlobsCollector()
    monkeypatch.setattr(storage.Bucket, 'delete_blobs', collect)
    return collect

@pytest.fixture
def b():
    if False:
        print('Hello World!')
    return storage.Bucket('test-bucket-name')

@pytest.fixture(autouse=True)
def never_use_single_delete(monkeypatch):
    if False:
        print('Hello World!')
    'Detect any mistaken uses of single-blob deletion.\n\n    Older wal-e versions used one-at-a-time deletions.  This is just\n    to help ensure that use of this API (through the nominal boto\n    symbol) is detected.\n    '

    def die():
        if False:
            for i in range(10):
                print('nop')
        assert False
    monkeypatch.setattr(storage.Blob, 'delete', die)
    monkeypatch.setattr(storage.Bucket, 'delete_blob', die)

def test_construction():
    if False:
        return 10
    'The constructor basically works.'
    gs_deleter.Deleter()

def test_close_error():
    if False:
        return 10
    'Ensure that attempts to use a closed Deleter results in an error.'
    d = gs_deleter.Deleter()
    d.close()
    with pytest.raises(exception.UserCritical):
        d.delete('no value should work')

def test_processes_one_deletion(b, collect):
    if False:
        print('Hello World!')
    blob_name = 'test-blob-name'
    k = storage.Blob(blob_name, b)
    d = gs_deleter.Deleter()
    d.delete(k)
    d.close()
    assert collect.deleted_blobs == [blob_name]

def test_processes_many_deletions(b, collect):
    if False:
        while True:
            i = 10
    target = sorted(['test-blob-' + str(x) for x in range(20001)])
    blobs = [storage.Blob(blob_name, b) for blob_name in target]
    d = gs_deleter.Deleter()
    for k in blobs:
        d.delete(k)
    d.close()
    assert sorted(collect.deleted_blobs) == target

def test_retry_on_normal_error(b, collect):
    if False:
        i = 10
        return i + 15
    'Ensure retries are processed for most errors.'
    blob_name = 'test-blob-name'
    k = storage.Blob(blob_name, b)
    collect.inject(Exception('Normal error'))
    d = gs_deleter.Deleter()
    d.delete(k)
    while len(collect.aborted_blobs) < 2:
        gevent.sleep(0.1)
    assert not collect.deleted_blobs
    collect.inject(None)
    d.close()
    assert collect.deleted_blobs == [blob_name]

def test_no_retry_on_keyboadinterrupt(b, collect):
    if False:
        while True:
            i = 10
    'Ensure that KeyboardInterrupts are forwarded.'
    blob_name = 'test-blob-name'
    k = storage.Blob(blob_name, b)

    class MarkedKeyboardInterrupt(KeyboardInterrupt):
        pass
    collect.inject(MarkedKeyboardInterrupt('SIGINT, probably'))
    d = gs_deleter.Deleter()
    with pytest.raises(MarkedKeyboardInterrupt):
        d.delete(k)
        while True:
            gevent.sleep(0.1)
    assert collect.aborted_blobs == [blob_name]
    collect.inject(None)
    d.close()
    assert not collect.deleted_blobs