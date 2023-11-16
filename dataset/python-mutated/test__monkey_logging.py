import logging
logging.basicConfig()
import threading
import sys
PY2 = sys.version_info[0] == 2

def _inner_lock(lock):
    if False:
        i = 10
        return i + 15
    attr = getattr(lock, '_block' if not PY2 else '_RLock__block', None)
    return attr

def _check_type(root, lock, inner_semaphore, kind):
    if False:
        i = 10
        return i + 15
    if not isinstance(inner_semaphore, kind):
        raise AssertionError('Expected <object>.[_]lock._block to be of type %s, but it was of type %s.\n\t<object>.[_]lock=%r\n\t<object>.[_]lock._block=%r\n\t<object>=%r' % (kind, type(inner_semaphore), lock, inner_semaphore, root))

def checkLocks(kind, ignore_none=True):
    if False:
        while True:
            i = 10
    handlers = logging._handlerList
    assert handlers
    for weakref in handlers:
        handler = weakref() if callable(weakref) else weakref
        block = _inner_lock(handler.lock)
        if block is None and ignore_none:
            continue
        _check_type(handler, handler.lock, block, kind)
    attr = _inner_lock(logging._lock)
    if attr is None and ignore_none:
        return
    _check_type(logging, logging._lock, attr, kind)
checkLocks(type(threading._allocate_lock()))
import gevent.monkey
gevent.monkey.patch_all()
import gevent.lock
checkLocks(type(gevent.thread.allocate_lock()), ignore_none=False)