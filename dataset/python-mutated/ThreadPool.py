import threading
import time
import queue
import gevent
import gevent.monkey
import gevent.threadpool
import gevent._threading

class ThreadPool:

    def __init__(self, max_size, name=None):
        if False:
            for i in range(10):
                print('nop')
        self.setMaxSize(max_size)
        if name:
            self.name = name
        else:
            self.name = 'ThreadPool#%s' % id(self)

    def setMaxSize(self, max_size):
        if False:
            return 10
        self.max_size = max_size
        if max_size > 0:
            self.pool = gevent.threadpool.ThreadPool(max_size)
        else:
            self.pool = None

    def wrap(self, func):
        if False:
            return 10
        if self.pool is None:
            return func

        def wrapper(*args, **kwargs):
            if False:
                return 10
            if not isMainThread():
                return func(*args, **kwargs)
            res = self.apply(func, args, kwargs)
            return res
        return wrapper

    def spawn(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if not isMainThread() and (not self.pool._semaphore.ready()):
            return main_loop.call(self.spawn, *args, **kwargs)
        res = self.pool.spawn(*args, **kwargs)
        return res

    def apply(self, func, args=(), kwargs={}):
        if False:
            for i in range(10):
                print('nop')
        t = self.spawn(func, *args, **kwargs)
        if self.pool._apply_immediately():
            return main_loop.call(t.get)
        else:
            return t.get()

    def kill(self):
        if False:
            return 10
        if self.pool is not None and self.pool.size > 0 and main_loop:
            main_loop.call(lambda : gevent.spawn(self.pool.kill).join(timeout=1))
        del self.pool
        self.pool = None

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.kill()
lock_pool = gevent.threadpool.ThreadPool(50)
main_thread_id = threading.current_thread().ident

def isMainThread():
    if False:
        return 10
    return threading.current_thread().ident == main_thread_id

class Lock:

    def __init__(self):
        if False:
            print('Hello World!')
        self.lock = gevent._threading.Lock()
        self.locked = self.lock.locked
        self.release = self.lock.release
        self.time_lock = 0

    def acquire(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.time_lock = time.time()
        if self.locked() and isMainThread():
            return lock_pool.apply(self.lock.acquire, args, kwargs)
        else:
            return self.lock.acquire(*args, **kwargs)

    def __del__(self):
        if False:
            print('Hello World!')
        while self.locked():
            self.release()

class Event:

    def __init__(self):
        if False:
            return 10
        self.get_lock = Lock()
        self.res = None
        self.get_lock.acquire(False)
        self.done = False

    def set(self, res):
        if False:
            for i in range(10):
                print('nop')
        if self.done:
            raise Exception('Event already has value')
        self.res = res
        self.get_lock.release()
        self.done = True

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.done:
            self.get_lock.acquire(True)
        if self.get_lock.locked():
            self.get_lock.release()
        back = self.res
        return back

    def __del__(self):
        if False:
            return 10
        self.res = None
        while self.get_lock.locked():
            self.get_lock.release()

class MainLoopCaller:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.queue_call = queue.Queue()
        self.pool = gevent.threadpool.ThreadPool(1)
        self.num_direct = 0
        self.running = True

    def caller(self, func, args, kwargs, event_done):
        if False:
            for i in range(10):
                print('nop')
        try:
            res = func(*args, **kwargs)
            event_done.set((True, res))
        except Exception as err:
            event_done.set((False, err))

    def start(self):
        if False:
            i = 10
            return i + 15
        gevent.spawn(self.run)
        time.sleep(0.001)

    def run(self):
        if False:
            i = 10
            return i + 15
        while self.running:
            if self.queue_call.qsize() == 0:
                (func, args, kwargs, event_done) = self.pool.apply(self.queue_call.get)
            else:
                (func, args, kwargs, event_done) = self.queue_call.get()
            gevent.spawn(self.caller, func, args, kwargs, event_done)
            del func, args, kwargs, event_done
        self.running = False

    def call(self, func, *args, **kwargs):
        if False:
            print('Hello World!')
        if threading.current_thread().ident == main_thread_id:
            return func(*args, **kwargs)
        else:
            event_done = Event()
            self.queue_call.put((func, args, kwargs, event_done))
            (success, res) = event_done.get()
            del event_done
            self.queue_call.task_done()
            if success:
                return res
            else:
                raise res

def patchSleep():
    if False:
        while True:
            i = 10
    real_sleep = gevent.monkey.get_original('time', 'sleep')

    def patched_sleep(seconds):
        if False:
            for i in range(10):
                print('nop')
        if isMainThread():
            gevent.sleep(seconds)
        else:
            real_sleep(seconds)
    time.sleep = patched_sleep
main_loop = MainLoopCaller()
main_loop.start()
patchSleep()