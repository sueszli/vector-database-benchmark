"""
A module to provide some very basic threading primitives, such as
synchronization.
"""
from functools import wraps

class DummyLock:
    """
    Hack to allow locks to be unpickled on an unthreaded system.
    """

    def __reduce__(self):
        if False:
            print('Hello World!')
        return (unpickle_lock, ())

def unpickle_lock():
    if False:
        print('Hello World!')
    if threadingmodule is not None:
        return XLock()
    else:
        return DummyLock()
unpickle_lock.__safe_for_unpickling__ = True

def _synchPre(self):
    if False:
        while True:
            i = 10
    if '_threadable_lock' not in self.__dict__:
        _synchLockCreator.acquire()
        if '_threadable_lock' not in self.__dict__:
            self.__dict__['_threadable_lock'] = XLock()
        _synchLockCreator.release()
    self._threadable_lock.acquire()

def _synchPost(self):
    if False:
        while True:
            i = 10
    self._threadable_lock.release()

def _sync(klass, function):
    if False:
        return 10

    @wraps(function)
    def sync(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        _synchPre(self)
        try:
            return function(self, *args, **kwargs)
        finally:
            _synchPost(self)
    return sync

def synchronize(*klasses):
    if False:
        i = 10
        return i + 15
    "\n    Make all methods listed in each class' synchronized attribute synchronized.\n\n    The synchronized attribute should be a list of strings, consisting of the\n    names of methods that must be synchronized. If we are running in threaded\n    mode these methods will be wrapped with a lock.\n    "
    if threadingmodule is not None:
        for klass in klasses:
            for methodName in klass.synchronized:
                sync = _sync(klass, klass.__dict__[methodName])
                setattr(klass, methodName, sync)

def init(with_threads=1):
    if False:
        while True:
            i = 10
    "Initialize threading.\n\n    Don't bother calling this.  If it needs to happen, it will happen.\n    "
    global threaded, _synchLockCreator, XLock
    if with_threads:
        if not threaded:
            if threadingmodule is not None:
                threaded = True

                class XLock(threadingmodule._RLock):

                    def __reduce__(self):
                        if False:
                            print('Hello World!')
                        return (unpickle_lock, ())
                _synchLockCreator = XLock()
            else:
                raise RuntimeError('Cannot initialize threading, platform lacks thread support')
    elif threaded:
        raise RuntimeError('Cannot uninitialize threads')
    else:
        pass
_dummyID = object()

def getThreadID():
    if False:
        i = 10
        return i + 15
    if threadingmodule is None:
        return _dummyID
    return threadingmodule.current_thread().ident

def isInIOThread():
    if False:
        print('Hello World!')
    'Are we in the thread responsible for I/O requests (the event loop)?'
    return ioThread == getThreadID()

def registerAsIOThread():
    if False:
        i = 10
        return i + 15
    'Mark the current thread as responsible for I/O requests.'
    global ioThread
    ioThread = getThreadID()
ioThread = None
threaded = False
_synchLockCreator = None
XLock = None
try:
    import threading as _threadingmodule
except ImportError:
    threadingmodule = None
else:
    threadingmodule = _threadingmodule
    init(True)
__all__ = ['isInIOThread', 'registerAsIOThread', 'getThreadID', 'XLock']