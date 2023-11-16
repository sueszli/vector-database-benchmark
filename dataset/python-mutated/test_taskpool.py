import unittest
from unittest.mock import patch
import ray
from ray.rllib.utils.actors import TaskPool

def createMockWorkerAndObjectRef(obj_ref):
    if False:
        print('Hello World!')
    return ({obj_ref: 1}, obj_ref)

class TaskPoolTest(unittest.TestCase):

    @patch('ray.wait')
    def test_completed_prefetch_yieldsAllComplete(self, rayWaitMock):
        if False:
            while True:
                i = 10
        task1 = createMockWorkerAndObjectRef(1)
        task2 = createMockWorkerAndObjectRef(2)
        rayWaitMock.return_value = ([2], [1])
        pool = TaskPool()
        pool.add(*task1)
        pool.add(*task2)
        fetched = list(pool.completed_prefetch())
        self.assertListEqual(fetched, [task2])

    @patch('ray.wait')
    def test_completed_prefetch_yieldsAllCompleteUpToDefaultLimit(self, rayWaitMock):
        if False:
            for i in range(10):
                print('nop')
        pool = TaskPool()
        for i in range(1000):
            task = createMockWorkerAndObjectRef(i)
            pool.add(*task)
        rayWaitMock.return_value = (list(range(1000)), [])
        fetched = [pair[1] for pair in pool.completed_prefetch()]
        self.assertListEqual(fetched, list(range(999)))
        fetched = [pair[1] for pair in pool.completed_prefetch()]
        self.assertListEqual(fetched, [999])

    @patch('ray.wait')
    def test_completed_prefetch_yieldsAllCompleteUpToSpecifiedLimit(self, rayWaitMock):
        if False:
            return 10
        pool = TaskPool()
        for i in range(1000):
            task = createMockWorkerAndObjectRef(i)
            pool.add(*task)
        rayWaitMock.return_value = (list(range(1000)), [])
        fetched = [pair[1] for pair in pool.completed_prefetch(max_yield=500)]
        self.assertListEqual(fetched, list(range(500)))
        fetched = [pair[1] for pair in pool.completed_prefetch()]
        self.assertListEqual(fetched, list(range(500, 1000)))

    @patch('ray.wait')
    def test_completed_prefetch_yieldsRemainingIfIterationStops(self, rayWaitMock):
        if False:
            return 10
        pool = TaskPool()
        for i in range(10):
            task = createMockWorkerAndObjectRef(i)
            pool.add(*task)
        rayWaitMock.return_value = (list(range(10)), [])
        try:
            for _ in pool.completed_prefetch():
                raise ray.exceptions.RayError
        except ray.exceptions.RayError:
            pass
        fetched = [pair[1] for pair in pool.completed_prefetch()]
        self.assertListEqual(fetched, list(range(1, 10)))

    @patch('ray.wait')
    def test_reset_workers_pendingFetchesFromFailedWorkersRemoved(self, rayWaitMock):
        if False:
            print('Hello World!')
        pool = TaskPool()
        tasks = []
        for i in range(10):
            task = createMockWorkerAndObjectRef(i)
            pool.add(*task)
            tasks.append(task)
        rayWaitMock.return_value = ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9])
        fetched = [pair[1] for pair in pool.completed_prefetch(max_yield=2)]
        rayWaitMock.return_value = ([], [6, 7, 8, 9])
        pool.reset_workers([tasks[0][0], tasks[1][0], tasks[2][0], tasks[3][0], tasks[5][0], tasks[6][0], tasks[7][0], tasks[8][0], tasks[9][0]])
        fetched = [pair[1] for pair in pool.completed_prefetch()]
        self.assertListEqual(fetched, [2, 3, 5])
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))