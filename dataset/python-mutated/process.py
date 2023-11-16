import warnings
from deeplake.core.compute.provider import ComputeProvider
from pathos.pools import ProcessPool
from pathos.helpers import mp as pathos_multiprocess

class ProcessProvider(ComputeProvider):

    def __init__(self, workers):
        if False:
            return 10
        self.workers = workers
        self.pool = ProcessPool(nodes=workers)
        self.manager = pathos_multiprocess.Manager()
        self._closed = False

    def map(self, func, iterable):
        if False:
            print('Hello World!')
        return self.pool.map(func, iterable)

    def create_queue(self):
        if False:
            return 10
        return self.manager.Queue()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.pool.close()
        self.pool.join()
        self.pool.clear()
        self._closed = True

    def __del__(self):
        if False:
            while True:
                i = 10
        if not self._closed:
            self.close()
            warnings.warn('process pool thread leak. check compute provider is closed after use')