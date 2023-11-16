import multiprocessing
import unittest
import numpy as np
from paddle import base
from paddle.base import core
from paddle.io import BatchSampler, DataLoader, Dataset, IterableDataset
from paddle.io.dataloader.worker import _worker_loop

class RandomDataset(Dataset):

    def __init__(self, sample_num):
        if False:
            i = 10
            return i + 15
        self.sample_num = sample_num

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        np.random.seed(idx)
        image = np.random.random([784]).astype('float32')
        label = np.random.randint(0, 9, (1,)).astype('int64')
        return (image, label)

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.sample_num

class TestDataLoaderAssert(unittest.TestCase):

    def test_main(self):
        if False:
            return 10
        place = base.cpu_places()[0]
        with base.dygraph.guard(place):
            dataset = RandomDataset(100)
            batch_sampler = BatchSampler(dataset=dataset, batch_size=4)
            try:
                loader = DataLoader(dataset=batch_sampler, places=place)
                self.assertTrue(False)
            except AssertionError:
                pass
            try:
                loader = DataLoader(dataset=dataset, places=None)
                self.assertTrue(False)
            except AssertionError:
                pass
            try:
                loader = DataLoader(dataset=dataset, places=place, num_workers=-1)
                self.assertTrue(False)
            except AssertionError:
                pass
            try:
                loader = DataLoader(dataset=dataset, places=place, timeout=-1)
                self.assertTrue(False)
            except AssertionError:
                pass
            try:
                loader = DataLoader(dataset=dataset, places=place, batch_sampler=batch_sampler, shuffle=True, drop_last=True)
                self.assertTrue(False)
            except AssertionError:
                pass
            try:
                loader = DataLoader(dataset=dataset, places=place, batch_sampler=batch_sampler)
                self.assertTrue(True)
            except AssertionError:
                self.assertTrue(False)

class TestDatasetRuntimeError(unittest.TestCase):

    def test_main(self):
        if False:
            while True:
                i = 10
        dataset = Dataset()
        try:
            d = dataset[0]
            self.assertTrue(False)
        except NotImplementedError:
            pass
        try:
            l = len(dataset)
            self.assertTrue(False)
        except NotImplementedError:
            pass
        dataset = IterableDataset()
        try:
            d = iter(dataset)
            self.assertTrue(False)
        except NotImplementedError:
            pass
        try:
            d = dataset[0]
            self.assertTrue(False)
        except RuntimeError:
            pass
        try:
            l = len(dataset)
            self.assertTrue(False)
        except RuntimeError:
            pass

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestDataLoaderWorkerLoop(unittest.TestCase):

    def run_without_worker_done(self, use_shared_memory=True):
        if False:
            while True:
                i = 10
        try:
            place = base.cpu_places()[0]
            with base.dygraph.guard(place):
                dataset = RandomDataset(800)

                def _init_fn(worker_id):
                    if False:
                        for i in range(10):
                            print('nop')
                    pass

                def _collate_fn(sample_list):
                    if False:
                        i = 10
                        return i + 15
                    return [np.stack(s, axis=0) for s in list(zip(*sample_list))]
                loader = DataLoader(dataset, num_workers=1, places=place, use_shared_memory=use_shared_memory)
                assert loader.num_workers > 0, 'go to AssertionError and pass in Mac and Windows'
                loader = iter(loader)
                print('loader length', len(loader))
                indices_queue = multiprocessing.Queue()
                for i in range(10):
                    indices_queue.put([i, i + 10])
                indices_queue.put(None)
                base_seed = 1234
                _worker_loop(loader._dataset, 0, indices_queue, loader._data_queue, loader._workers_done_event, True, _collate_fn, True, _init_fn, 0, 1, loader._use_shared_memory, base_seed)
                self.assertTrue(False)
        except AssertionError:
            pass
        except Exception as e:
            print('Exception', e)
            import sys
            sys.stdout.flush()
            self.assertTrue(False)

    def run_with_worker_done(self, use_shared_memory=True):
        if False:
            return 10
        try:
            place = base.CPUPlace()
            with base.dygraph.guard(place):
                dataset = RandomDataset(800)

                def _init_fn(worker_id):
                    if False:
                        for i in range(10):
                            print('nop')
                    pass

                def _collate_fn(sample_list):
                    if False:
                        i = 10
                        return i + 15
                    return [np.stack(s, axis=0) for s in list(zip(*sample_list))]
                loader = DataLoader(dataset, num_workers=1, places=place, use_shared_memory=use_shared_memory)
                assert loader.num_workers > 0, 'go to AssertionError and pass in Mac and Windows'
                loader = iter(loader)
                print('loader length', len(loader))
                indices_queue = multiprocessing.Queue()
                for i in range(10):
                    indices_queue.put([i, i + 10])
                indices_queue.put(None)
                loader._workers_done_event.set()
                base_seed = 1234
                _worker_loop(loader._dataset, 0, indices_queue, loader._data_queue, loader._workers_done_event, True, _collate_fn, True, _init_fn, 0, 1, loader._use_shared_memory, base_seed)
                self.assertTrue(True)
        except AssertionError:
            pass
        except Exception:
            self.assertTrue(False)

    def test_main(self):
        if False:
            while True:
                i = 10
        for use_shared_memory in [False]:
            self.run_without_worker_done(use_shared_memory)
            self.run_with_worker_done(use_shared_memory)
if __name__ == '__main__':
    unittest.main()