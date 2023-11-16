import unittest
from pygame.threads import FuncResult, tmap, WorkerQueue, Empty, STOP
from pygame import threads, Surface, transform
import time

class WorkerQueueTypeTest(unittest.TestCase):

    def test_usage_with_different_functions(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                while True:
                    i = 10
            return x + 1

        def f2(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 2
        wq = WorkerQueue()
        fr = FuncResult(f)
        fr2 = FuncResult(f2)
        wq.do(fr, 1)
        wq.do(fr2, 1)
        wq.wait()
        wq.stop()
        self.assertEqual(fr.result, 2)
        self.assertEqual(fr2.result, 3)

    def test_do(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests function placement on queue and execution after blocking function completion.'

    def test_stop(self):
        if False:
            while True:
                i = 10
        'Ensure stop() stops the worker queue'
        wq = WorkerQueue()
        self.assertGreater(len(wq.pool), 0)
        for t in wq.pool:
            self.assertTrue(t.is_alive())
        for i in range(200):
            wq.do(lambda x: x + 1, i)
        wq.stop()
        for t in wq.pool:
            self.assertFalse(t.is_alive())
        self.assertIs(wq.queue.get(), STOP)

    def test_threadloop(self):
        if False:
            print('Hello World!')
        wq = WorkerQueue(1)
        wq.do(wq.threadloop)
        l = []
        wq.do(l.append, 1)
        time.sleep(0.5)
        self.assertEqual(l[0], 1)
        wq.stop()
        self.assertFalse(wq.pool[0].is_alive())

    def test_wait(self):
        if False:
            return 10
        wq = WorkerQueue()
        for i in range(2000):
            wq.do(lambda x: x + 1, i)
        wq.wait()
        self.assertRaises(Empty, wq.queue.get_nowait)
        wq.stop()

class ThreadsModuleTest(unittest.TestCase):

    def test_benchmark_workers(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure benchmark_workers performance measure functions properly with both default and specified inputs'
        'tags:long_running'
        optimal_workers = threads.benchmark_workers()
        self.assertIsInstance(optimal_workers, int)
        self.assertTrue(0 <= optimal_workers < 64)

        def smooth_scale_bench(data):
            if False:
                for i in range(10):
                    print('nop')
            transform.smoothscale(data, (128, 128))
        surf_data = [Surface((x, x), 0, 32) for x in range(12, 64, 12)]
        best_num_workers = threads.benchmark_workers(smooth_scale_bench, surf_data)
        self.assertIsInstance(best_num_workers, int)

    def test_init(self):
        if False:
            i = 10
            return i + 15
        'Ensure init() sets up the worker queue'
        threads.init(8)
        self.assertIsInstance(threads._wq, WorkerQueue)
        threads.quit()

    def test_quit(self):
        if False:
            while True:
                i = 10
        'Ensure quit() cleans up the worker queue'
        threads.init(8)
        threads.quit()
        self.assertIsNone(threads._wq)

    def test_tmap(self):
        if False:
            print('Hello World!')
        (func, data) = (lambda x: x + 1, range(100))
        tmapped = list(tmap(func, data))
        mapped = list(map(func, data))
        self.assertEqual(tmapped, mapped)
        data2 = range(100)
        always_excepts = lambda x: 1 / 0
        tmapped2 = list(tmap(always_excepts, data2, stop_on_error=False))
        self.assertTrue(all([x is None for x in tmapped2]))

    def todo_test_tmap__None_func_and_multiple_sequences(self):
        if False:
            return 10
        'Using a None as func and multiple sequences'
        self.fail()
        res = tmap(None, [1, 2, 3, 4])
        res2 = tmap(None, [1, 2, 3, 4], [22, 33, 44, 55])
        res3 = tmap(None, [1, 2, 3, 4], [22, 33, 44, 55, 66])
        res4 = tmap(None, [1, 2, 3, 4, 5], [22, 33, 44, 55])
        self.assertEqual([1, 2, 3, 4], res)
        self.assertEqual([(1, 22), (2, 33), (3, 44), (4, 55)], res2)
        self.assertEqual([(1, 22), (2, 33), (3, 44), (4, 55), (None, 66)], res3)
        self.assertEqual([(1, 22), (2, 33), (3, 44), (4, 55), (5, None)], res4)

    def test_tmap__wait(self):
        if False:
            for i in range(10):
                print('nop')
        r = range(1000)
        (wq, results) = tmap(lambda x: x, r, num_workers=5, wait=False)
        wq.wait()
        r2 = map(lambda x: x.result, results)
        self.assertEqual(list(r), list(r2))

    def test_FuncResult(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure FuncResult sets its result and exception attributes'
        fr = FuncResult(lambda x: x + 1)
        fr(2)
        self.assertEqual(fr.result, 3)
        self.assertIsNone(fr.exception, 'no exception should be raised')
        exception = ValueError('rast')

        def x(sdf):
            if False:
                print('Hello World!')
            raise exception
        fr = FuncResult(x)
        fr(None)
        self.assertIs(fr.exception, exception)
if __name__ == '__main__':
    unittest.main()