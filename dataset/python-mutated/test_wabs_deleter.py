import gevent
import pytest
from collections import namedtuple
from fast_wait import fast_wait
from gevent import lock
from wal_e import exception
from azure.storage.blob.blockblobservice import BlockBlobService
from wal_e.worker.wabs import wabs_deleter
assert fast_wait
B = namedtuple('Blob', ['name'])

class ContainerDeleteKeysCollector(object):
    """A callable to stand-in for bucket.delete_keys

    Used to test that given keys are bulk-deleted.

    Also can inject an exception.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.deleted_keys = []
        self.aborted_keys = []
        self.exc = None
        self._exc_protect = lock.RLock()

    def inject(self, exc):
        if False:
            i = 10
            return i + 15
        self._exc_protect.acquire()
        self.exc = exc
        self._exc_protect.release()

    def __call__(self, container, key):
        if False:
            for i in range(10):
                print('nop')
        self._exc_protect.acquire()
        try:
            if self.exc:
                self.aborted_keys.append(key)
                gevent.sleep(0.1)
                raise self.exc
        finally:
            self._exc_protect.release()
        self.deleted_keys.append(key)

@pytest.fixture
def collect(monkeypatch):
    if False:
        return 10
    'Instead of performing bulk delete, collect key names deleted.\n\n    This is to test invariants, as to ensure deleted keys are passed\n    to boto properly.\n    '
    collect = ContainerDeleteKeysCollector()
    monkeypatch.setattr(BlockBlobService, 'delete_blob', collect)
    return collect

def test_fast_wait():
    if False:
        for i in range(10):
            print('nop')
    'Annoy someone who causes fast-sleep test patching to regress.\n\n    Someone could break the test-only monkey-patching of gevent.sleep\n    without noticing and costing quite a bit of aggravation aggregated\n    over time waiting in tests, added bit by bit.\n\n    To avoid that, add this incredibly huge/annoying delay that can\n    only be avoided by monkey-patch to catch the regression.\n    '
    gevent.sleep(300)

def test_construction():
    if False:
        print('Hello World!')
    'The constructor basically works.'
    wabs_deleter.Deleter('test', 'ing')

def test_close_error():
    if False:
        while True:
            i = 10
    'Ensure that attempts to use a closed Deleter results in an error.'
    d = wabs_deleter.Deleter(BlockBlobService('test', 'ing'), 'test-container')
    d.close()
    with pytest.raises(exception.UserCritical):
        d.delete('no value should work')

def test_processes_one_deletion(collect):
    if False:
        i = 10
        return i + 15
    key_name = 'test-key-name'
    b = B(name=key_name)
    d = wabs_deleter.Deleter(BlockBlobService('test', 'ing'), 'test-container')
    d.delete(b)
    d.close()
    assert collect.deleted_keys == [key_name]

def test_processes_many_deletions(collect):
    if False:
        i = 10
        return i + 15
    target = sorted(['test-key-' + str(x) for x in range(20001)])
    blobs = [B(name=key_name) for key_name in target]
    d = wabs_deleter.Deleter(BlockBlobService('test', 'ing'), 'test-container')
    for b in blobs:
        d.delete(b)
    d.close()
    assert sorted(collect.deleted_keys) == target

def test_retry_on_normal_error(collect):
    if False:
        return 10
    'Ensure retries are processed for most errors.'
    key_name = 'test-key-name'
    b = B(name=key_name)
    collect.inject(Exception('Normal error'))
    d = wabs_deleter.Deleter(BlockBlobService('test', 'ing'), 'test-container')
    d.delete(b)
    while len(collect.aborted_keys) < 2:
        gevent.sleep(0.1)
    assert not collect.deleted_keys
    collect.inject(None)
    d.close()
    assert collect.deleted_keys == [key_name]

def test_no_retry_on_keyboadinterrupt(collect):
    if False:
        while True:
            i = 10
    'Ensure that KeyboardInterrupts are forwarded.'
    key_name = 'test-key-name'
    b = B(name=key_name)

    class MarkedKeyboardInterrupt(KeyboardInterrupt):
        pass
    collect.inject(MarkedKeyboardInterrupt('SIGINT, probably'))
    d = wabs_deleter.Deleter(BlockBlobService('test', 'ing'), 'test-container')
    with pytest.raises(MarkedKeyboardInterrupt):
        d.delete(b)
        while True:
            gevent.sleep(0.1)
    assert collect.aborted_keys == [key_name]
    collect.inject(None)
    d.close()
    assert not collect.deleted_keys