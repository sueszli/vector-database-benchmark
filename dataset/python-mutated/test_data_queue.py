import multiprocessing
import time
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from qlib.rl.utils.data_queue import DataQueue

class DummyDataset(Dataset):

    def __init__(self, length):
        if False:
            return 10
        self.length = length

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        assert 0 <= index < self.length
        return pd.DataFrame(np.random.randint(0, 100, size=(index + 1, 4)), columns=list('ABCD'))

    def __len__(self):
        if False:
            return 10
        return self.length

def _worker(dataloader, collector):
    if False:
        return 10
    for (i, data) in enumerate(dataloader):
        collector.put(len(data))

def _queue_to_list(queue):
    if False:
        print('Hello World!')
    result = []
    while not queue.empty():
        result.append(queue.get())
    return result

def test_pytorch_dataloader():
    if False:
        return 10
    dataset = DummyDataset(100)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=1)
    queue = multiprocessing.Queue()
    _worker(dataloader, queue)
    assert len(set(_queue_to_list(queue))) == 100

def test_multiprocess_shared_dataloader():
    if False:
        return 10
    dataset = DummyDataset(100)
    with DataQueue(dataset, producer_num_workers=1) as data_queue:
        queue = multiprocessing.Queue()
        processes = []
        for _ in range(3):
            processes.append(multiprocessing.Process(target=_worker, args=(data_queue, queue)))
            processes[-1].start()
        for p in processes:
            p.join()
        assert len(set(_queue_to_list(queue))) == 100

def test_exit_on_crash_finite():
    if False:
        while True:
            i = 10

    def _exit_finite():
        if False:
            return 10
        dataset = DummyDataset(100)
        with DataQueue(dataset, producer_num_workers=4) as data_queue:
            time.sleep(3)
            raise ValueError
    process = multiprocessing.Process(target=_exit_finite)
    process.start()
    process.join()

def test_exit_on_crash_infinite():
    if False:
        print('Hello World!')

    def _exit_infinite():
        if False:
            return 10
        dataset = DummyDataset(100)
        with DataQueue(dataset, repeat=-1, queue_maxsize=100) as data_queue:
            time.sleep(3)
            raise ValueError
    process = multiprocessing.Process(target=_exit_infinite)
    process.start()
    process.join()
if __name__ == '__main__':
    test_multiprocess_shared_dataloader()