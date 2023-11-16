import pytest
import unittest
from spiderfoot import SpiderFootThreadPool

@pytest.mark.usefixtures
class TestSpiderFootThreadPool(unittest.TestCase):
    """
    Test SpiderFoot
    """

    def test_threadPool(self):
        if False:
            i = 10
            return i + 15
        '\n        Test ThreadPool(sfp, threads=10)\n        '
        threads = 10

        def callback(x, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return (x, args, list(kwargs.items())[0])
        iterable = ['a', 'b', 'c']
        args = ('arg1',)
        kwargs = {'kwarg1': 'kwarg1'}
        expectedOutput = [('a', ('arg1',), ('kwarg1', 'kwarg1')), ('b', ('arg1',), ('kwarg1', 'kwarg1')), ('c', ('arg1',), ('kwarg1', 'kwarg1'))]
        with SpiderFootThreadPool(threads) as pool:
            map_results = sorted(list(pool.map(callback, iterable, *args, saveResult=True, **kwargs)), key=lambda x: x[0])
        self.assertEqual(map_results, expectedOutput)
        with SpiderFootThreadPool(threads) as pool:
            pool.start()
            for i in iterable:
                pool.submit(callback, *(i,) + args, saveResult=True, **kwargs)
            submit_results = sorted(list(pool.shutdown()['default']), key=lambda x: x[0])
        self.assertEqual(submit_results, expectedOutput)
        threads = 1
        iterable2 = ['d', 'e', 'f']
        expectedOutput2 = [('d', ('arg1',), ('kwarg1', 'kwarg1')), ('e', ('arg1',), ('kwarg1', 'kwarg1')), ('f', ('arg1',), ('kwarg1', 'kwarg1'))]
        pool = SpiderFootThreadPool(threads)
        pool.start()
        for i in iterable2:
            pool.submit(callback, *(i,) + args, taskName='submitTest', saveResult=True, **kwargs)
        map_results = sorted(list(pool.map(callback, iterable, *args, taskName='mapTest', saveResult=True, **kwargs)), key=lambda x: x[0])
        submit_results = sorted(list(pool.shutdown()['submitTest']), key=lambda x: x[0])
        self.assertEqual(map_results, expectedOutput)
        self.assertEqual(submit_results, expectedOutput2)