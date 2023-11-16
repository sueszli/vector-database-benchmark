import time

class AsyncResultTimeout(Exception):
    """an exception that represents an :class:`AsyncResult` that has timed out"""
    pass

class AsyncResult(object):
    """*AsyncResult* represents a computation that occurs in the background and
    will eventually have a result. Use the :attr:`value` property to access the
    result (which will block if the result has not yet arrived).
    """
    __slots__ = ['_conn', '_is_ready', '_is_exc', '_callbacks', '_obj', '_ttl']

    def __init__(self, conn):
        if False:
            return 10
        self._conn = conn
        self._is_ready = False
        self._is_exc = None
        self._obj = None
        self._callbacks = []
        self._ttl = None

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self._is_ready:
            state = 'ready'
        elif self._is_exc:
            state = 'error'
        elif self.expired:
            state = 'expired'
        else:
            state = 'pending'
        return '<AsyncResult object (%s) at 0x%08x>' % (state, id(self))

    def __call__(self, is_exc, obj):
        if False:
            print('Hello World!')
        if self.expired:
            return
        self._is_exc = is_exc
        self._obj = obj
        self._is_ready = True
        for cb in self._callbacks:
            cb(self)
        del self._callbacks[:]

    def wait(self):
        if False:
            while True:
                i = 10
        'Waits for the result to arrive. If the AsyncResult object has an\n        expiry set, and the result did not arrive within that timeout,\n        an :class:`AsyncResultTimeout` exception is raised'
        if self._is_ready:
            return
        if self._ttl is None:
            while not self._is_ready:
                self._conn.serve()
        else:
            while True:
                timeout = self._ttl - time.time()
                self._conn.poll(timeout=max(timeout, 0))
                if self._is_ready:
                    break
                if timeout <= 0:
                    raise AsyncResultTimeout('result expired')

    def add_callback(self, func):
        if False:
            i = 10
            return i + 15
        'Adds a callback to be invoked when the result arrives. The callback\n        function takes a single argument, which is the current AsyncResult\n        (``self``). If the result has already arrived, the function is invoked\n        immediately.\n\n        :param func: the callback function to add\n        '
        if self._is_ready:
            func(self)
        else:
            self._callbacks.append(func)

    def set_expiry(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        'Sets the expiry time (in seconds, relative to now) or ``None`` for\n        unlimited time\n\n        :param timeout: the expiry time in seconds or ``None``\n        '
        if timeout is None:
            self._ttl = None
        else:
            self._ttl = time.time() + timeout

    @property
    def ready(self):
        if False:
            return 10
        'Indicates whether the result has arrived'
        if self.expired:
            return False
        if not self._is_ready:
            self._conn.poll_all()
        return self._is_ready

    @property
    def error(self):
        if False:
            return 10
        'Indicates whether the returned result is an exception'
        if self.ready:
            return self._is_exc
        return False

    @property
    def expired(self):
        if False:
            while True:
                i = 10
        'Indicates whether the AsyncResult has expired'
        if self._is_ready or self._ttl is None:
            return False
        else:
            return time.time() > self._ttl

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        'Returns the result of the operation. If the result has not yet\n        arrived, accessing this property will wait for it. If the result does\n        not arrive before the expiry time elapses, :class:`AsyncResultTimeout`\n        is raised. If the returned result is an exception, it will be raised\n        here. Otherwise, the result is returned directly.\n        '
        self.wait()
        if self._is_exc:
            raise self._obj
        else:
            return self._obj