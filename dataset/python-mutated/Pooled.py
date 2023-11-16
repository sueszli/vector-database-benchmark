import gevent.pool

class Pooled(object):

    def __init__(self, size=100):
        if False:
            i = 10
            return i + 15
        self.pool = gevent.pool.Pool(size)
        self.pooler_running = False
        self.queue = []
        self.func = None

    def waiter(self, evt, args, kwargs):
        if False:
            i = 10
            return i + 15
        res = self.func(*args, **kwargs)
        if type(res) == gevent.event.AsyncResult:
            evt.set(res.get())
        else:
            evt.set(res)

    def pooler(self):
        if False:
            print('Hello World!')
        while self.queue:
            (evt, args, kwargs) = self.queue.pop(0)
            self.pool.spawn(self.waiter, evt, args, kwargs)
        self.pooler_running = False

    def __call__(self, func):
        if False:
            while True:
                i = 10

        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            evt = gevent.event.AsyncResult()
            self.queue.append((evt, args, kwargs))
            if not self.pooler_running:
                self.pooler_running = True
                gevent.spawn(self.pooler)
            return evt
        wrapper.__name__ = func.__name__
        self.func = func
        return wrapper
if __name__ == '__main__':
    import gevent
    import gevent.pool
    import gevent.queue
    import gevent.event
    import gevent.monkey
    import time
    gevent.monkey.patch_all()

    def addTask(inner_path):
        if False:
            print('Hello World!')
        evt = gevent.event.AsyncResult()
        gevent.spawn_later(1, lambda : evt.set(True))
        return evt

    def needFile(inner_path):
        if False:
            i = 10
            return i + 15
        return addTask(inner_path)

    @Pooled(10)
    def pooledNeedFile(inner_path):
        if False:
            i = 10
            return i + 15
        return needFile(inner_path)
    threads = []
    for i in range(100):
        threads.append(pooledNeedFile(i))
    s = time.time()
    gevent.joinall(threads)
    print(time.time() - s)