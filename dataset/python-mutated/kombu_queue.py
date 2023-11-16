import time
import umsgpack
from kombu import Connection, enable_insecure_serializers
from kombu.serialization import register
from kombu.exceptions import ChannelError
from six.moves import queue as BaseQueue
register('umsgpack', umsgpack.packb, umsgpack.unpackb, 'application/x-msgpack')
enable_insecure_serializers(['umsgpack'])

class KombuQueue(object):
    """
    kombu is a high-level interface for multiple message queue backends.

    KombuQueue is built on top of kombu API.
    """
    Empty = BaseQueue.Empty
    Full = BaseQueue.Full
    max_timeout = 0.3

    def __init__(self, name, url='amqp://', maxsize=0, lazy_limit=True):
        if False:
            return 10
        '\n        Constructor for KombuQueue\n\n        url:        http://kombu.readthedocs.org/en/latest/userguide/connections.html#urls\n        maxsize:    an integer that sets the upperbound limit on the number of\n                    items that can be placed in the queue.\n        '
        self.name = name
        self.conn = Connection(url)
        self.queue = self.conn.SimpleQueue(self.name, no_ack=True, serializer='umsgpack')
        self.maxsize = maxsize
        self.lazy_limit = lazy_limit
        if self.lazy_limit and self.maxsize:
            self.qsize_diff_limit = int(self.maxsize * 0.1)
        else:
            self.qsize_diff_limit = 0
        self.qsize_diff = 0

    def qsize(self):
        if False:
            i = 10
            return i + 15
        try:
            return self.queue.qsize()
        except ChannelError:
            return 0

    def empty(self):
        if False:
            for i in range(10):
                print('nop')
        if self.qsize() == 0:
            return True
        else:
            return False

    def full(self):
        if False:
            print('Hello World!')
        if self.maxsize and self.qsize() >= self.maxsize:
            return True
        else:
            return False

    def put(self, obj, block=True, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        if not block:
            return self.put_nowait(obj)
        start_time = time.time()
        while True:
            try:
                return self.put_nowait(obj)
            except BaseQueue.Full:
                if timeout:
                    lasted = time.time() - start_time
                    if timeout > lasted:
                        time.sleep(min(self.max_timeout, timeout - lasted))
                    else:
                        raise
                else:
                    time.sleep(self.max_timeout)

    def put_nowait(self, obj):
        if False:
            return 10
        if self.lazy_limit and self.qsize_diff < self.qsize_diff_limit:
            pass
        elif self.full():
            raise BaseQueue.Full
        else:
            self.qsize_diff = 0
        return self.queue.put(obj)

    def get(self, block=True, timeout=None):
        if False:
            i = 10
            return i + 15
        try:
            ret = self.queue.get(block, timeout)
            return ret.payload
        except self.queue.Empty:
            raise BaseQueue.Empty

    def get_nowait(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            ret = self.queue.get_nowait()
            return ret.payload
        except self.queue.Empty:
            raise BaseQueue.Empty

    def delete(self):
        if False:
            i = 10
            return i + 15
        self.queue.queue.delete()

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self.queue.close()
Queue = KombuQueue