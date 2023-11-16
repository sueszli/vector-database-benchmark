import time
try:
    import threading as _threading
except ImportError:
    import dummy_threading as _threading

class Bucket(object):
    """
    traffic flow control with token bucket
    """
    update_interval = 30

    def __init__(self, rate=1, burst=None):
        if False:
            while True:
                i = 10
        self.rate = float(rate)
        if burst is None:
            self.burst = float(rate) * 10
        else:
            self.burst = float(burst)
        self.mutex = _threading.Lock()
        self.bucket = self.burst
        self.last_update = time.time()

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the number of tokens in bucket'
        now = time.time()
        if self.bucket >= self.burst:
            self.last_update = now
            return self.bucket
        bucket = self.rate * (now - self.last_update)
        self.mutex.acquire()
        if bucket > 1:
            self.bucket += bucket
            if self.bucket > self.burst:
                self.bucket = self.burst
            self.last_update = now
        self.mutex.release()
        return self.bucket

    def set(self, value):
        if False:
            return 10
        'Set number of tokens in bucket'
        self.bucket = value

    def desc(self, value=1):
        if False:
            print('Hello World!')
        'Use value tokens'
        self.bucket -= value