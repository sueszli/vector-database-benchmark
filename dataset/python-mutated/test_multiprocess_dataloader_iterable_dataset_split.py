import math
import unittest
import numpy as np
from paddle import base
from paddle.io import DataLoader, IterableDataset, get_worker_info

class RangeIterableDatasetSplit(IterableDataset):

    def __init__(self, start, end):
        if False:
            while True:
                i = 10
        self.start = start
        self.end = end

    def __iter__(self):
        if False:
            print('Hello World!')
        worker_info = get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        for i in range(iter_start, iter_end):
            yield np.array([i])

class TestDynamicDataLoaderIterSplit(unittest.TestCase):

    def test_main(self):
        if False:
            print('Hello World!')
        place = base.CPUPlace()
        with base.dygraph.guard(place):
            dataset = RangeIterableDatasetSplit(0, 10)
            dataloader = DataLoader(dataset, places=place, num_workers=2, batch_size=1, drop_last=True)
            rets = []
            for d in dataloader:
                rets.append(d.numpy()[0][0])
            assert tuple(sorted(rets)) == tuple(range(0, 10))

class RangeIterableDataset(IterableDataset):

    def __init__(self, start, end):
        if False:
            return 10
        self.start = start
        self.end = end

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(self.start, self.end):
            yield np.array([i])

class TestDynamicDataLoaderIterInitFuncSplit(unittest.TestCase):

    def test_main(self):
        if False:
            return 10
        place = base.CPUPlace()
        with base.dygraph.guard(place):
            dataset = RangeIterableDataset(0, 10)

            def worker_spliter(worker_id):
                if False:
                    return 10
                worker_info = get_worker_info()
                dataset = worker_info.dataset
                start = dataset.start
                end = dataset.end
                num_per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
                worker_id = worker_info.id
                dataset.start = start + worker_id * num_per_worker
                dataset.end = min(dataset.start + num_per_worker, end)
            dataloader = DataLoader(dataset, places=place, num_workers=1, batch_size=1, drop_last=True, worker_init_fn=worker_spliter)
            rets = []
            for d in dataloader:
                rets.append(d.numpy()[0][0])
            assert tuple(sorted(rets)) == tuple(range(0, 10))
if __name__ == '__main__':
    unittest.main()