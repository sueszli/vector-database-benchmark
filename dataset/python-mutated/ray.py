import ray
from ray.util.multiprocessing import Pool
from ray.util.queue import Queue
from deeplake.core.compute.provider import ComputeProvider

class RayProvider(ComputeProvider):

    def __init__(self, workers):
        if False:
            i = 10
            return i + 15
        super().__init__(workers)
        if not ray.is_initialized():
            ray.init()
        self.workers = workers
        self.pool = Pool(processes=workers)

    def map(self, func, iterable):
        if False:
            i = 10
            return i + 15
        return self.pool.map(func, iterable)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.pool.close()
        self.pool.join()

    def create_queue(self):
        if False:
            return 10
        return Queue()