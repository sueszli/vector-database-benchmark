import traceback
from ..Qt import QtCore

class Mutex(QtCore.QMutex):
    """
    Subclass of QMutex that provides useful debugging information during
    deadlocks--tracebacks are printed for both the code location that is 
    attempting to lock the mutex as well as the location that has already
    acquired the lock.
    
    Also provides __enter__ and __exit__ methods for use in "with" statements.
    """

    def __init__(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        if kargs.get('recursive', False):
            args = (QtCore.QMutex.Recursive,)
        QtCore.QMutex.__init__(self, *args)
        self.l = QtCore.QMutex()
        self.tb = []
        self.debug = kargs.pop('debug', False)

    def tryLock(self, timeout=None, id=None):
        if False:
            print('Hello World!')
        if timeout is None:
            locked = QtCore.QMutex.tryLock(self)
        else:
            locked = QtCore.QMutex.tryLock(self, timeout)
        if self.debug and locked:
            self.l.lock()
            try:
                if id is None:
                    self.tb.append(''.join(traceback.format_stack()[:-1]))
                else:
                    self.tb.append('  ' + str(id))
            finally:
                self.l.unlock()
        return locked

    def lock(self, id=None):
        if False:
            i = 10
            return i + 15
        c = 0
        waitTime = 5000
        while True:
            if self.tryLock(waitTime, id):
                break
            c += 1
            if self.debug:
                self.l.lock()
                try:
                    print('Waiting for mutex lock (%0.1f sec). Traceback follows:' % (c * waitTime / 1000.0))
                    traceback.print_stack()
                    if len(self.tb) > 0:
                        print('Mutex is currently locked from:\n')
                        print(self.tb[-1])
                    else:
                        print('Mutex is currently locked from [???]')
                finally:
                    self.l.unlock()

    def unlock(self):
        if False:
            return 10
        QtCore.QMutex.unlock(self)
        if self.debug:
            self.l.lock()
            try:
                if len(self.tb) > 0:
                    self.tb.pop()
                else:
                    raise Exception('Attempt to unlock mutex before it has been locked')
            finally:
                self.l.unlock()

    def acquire(self, blocking=True):
        if False:
            while True:
                i = 10
        'Mimics threading.Lock.acquire() to allow this class as a drop-in replacement.\n        '
        return self.tryLock()

    def release(self):
        if False:
            i = 10
            return i + 15
        'Mimics threading.Lock.release() to allow this class as a drop-in replacement.\n        '
        self.unlock()

    def depth(self):
        if False:
            return 10
        self.l.lock()
        n = len(self.tb)
        self.l.unlock()
        return n

    def traceback(self):
        if False:
            for i in range(10):
                print('nop')
        self.l.lock()
        try:
            ret = self.tb[:]
        finally:
            self.l.unlock()
        return ret

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.unlock()

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.lock()
        return self

class RecursiveMutex(Mutex):
    """Mimics threading.RLock class.
    """

    def __init__(self, **kwds):
        if False:
            for i in range(10):
                print('nop')
        kwds['recursive'] = True
        Mutex.__init__(self, **kwds)