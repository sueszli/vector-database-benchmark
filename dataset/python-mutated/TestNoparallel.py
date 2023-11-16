import time
import gevent
import pytest
import util
from util import ThreadPool

@pytest.fixture(params=['gevent.spawn', 'thread_pool.spawn'])
def queue_spawn(request):
    if False:
        print('Hello World!')
    thread_pool = ThreadPool.ThreadPool(10)
    if request.param == 'gevent.spawn':
        return gevent.spawn
    else:
        return thread_pool.spawn

class ExampleClass(object):

    def __init__(self):
        if False:
            return 10
        self.counted = 0

    @util.Noparallel()
    def countBlocking(self, num=5):
        if False:
            i = 10
            return i + 15
        for i in range(1, num + 1):
            time.sleep(0.1)
            self.counted += 1
        return 'counted:%s' % i

    @util.Noparallel(queue=True, ignore_class=True)
    def countQueue(self, num=5):
        if False:
            i = 10
            return i + 15
        for i in range(1, num + 1):
            time.sleep(0.1)
            self.counted += 1
        return 'counted:%s' % i

    @util.Noparallel(blocking=False)
    def countNoblocking(self, num=5):
        if False:
            for i in range(10):
                print('nop')
        for i in range(1, num + 1):
            time.sleep(0.01)
            self.counted += 1
        return 'counted:%s' % i

class TestNoparallel:

    def testBlocking(self, queue_spawn):
        if False:
            i = 10
            return i + 15
        obj1 = ExampleClass()
        obj2 = ExampleClass()
        threads = [queue_spawn(obj1.countBlocking), queue_spawn(obj1.countBlocking), queue_spawn(obj1.countBlocking), queue_spawn(obj2.countBlocking)]
        assert obj2.countBlocking() == 'counted:5'
        gevent.joinall(threads)
        assert [thread.value for thread in threads] == ['counted:5', 'counted:5', 'counted:5', 'counted:5']
        obj2.countBlocking()
        assert obj1.counted == 5
        assert obj2.counted == 10

    def testNoblocking(self):
        if False:
            i = 10
            return i + 15
        obj1 = ExampleClass()
        thread1 = obj1.countNoblocking()
        thread2 = obj1.countNoblocking()
        assert obj1.counted == 0
        time.sleep(0.1)
        assert thread1.value == 'counted:5'
        assert thread2.value == 'counted:5'
        assert obj1.counted == 5
        obj1.countNoblocking().join()
        assert obj1.counted == 10

    def testQueue(self, queue_spawn):
        if False:
            for i in range(10):
                print('nop')
        obj1 = ExampleClass()
        queue_spawn(obj1.countQueue, num=1)
        queue_spawn(obj1.countQueue, num=1)
        queue_spawn(obj1.countQueue, num=1)
        time.sleep(0.3)
        assert obj1.counted == 2
        obj2 = ExampleClass()
        queue_spawn(obj2.countQueue, num=10)
        queue_spawn(obj2.countQueue, num=10)
        time.sleep(1.5)
        assert 10 < obj2.counted < 20
        queue_spawn(obj2.countQueue, num=10)
        time.sleep(2.0)
        assert obj2.counted == 30

    def testQueueOverload(self):
        if False:
            print('Hello World!')
        obj1 = ExampleClass()
        threads = []
        for i in range(1000):
            thread = gevent.spawn(obj1.countQueue, num=5)
            threads.append(thread)
        gevent.joinall(threads)
        assert obj1.counted == 5 * 2

    def testIgnoreClass(self, queue_spawn):
        if False:
            i = 10
            return i + 15
        obj1 = ExampleClass()
        obj2 = ExampleClass()
        threads = [queue_spawn(obj1.countQueue), queue_spawn(obj1.countQueue), queue_spawn(obj1.countQueue), queue_spawn(obj2.countQueue), queue_spawn(obj2.countQueue)]
        s = time.time()
        time.sleep(0.001)
        gevent.joinall(threads)
        assert obj1.counted + obj2.counted == 10
        taken = time.time() - s
        assert 1.2 > taken >= 1.0

    def testException(self, queue_spawn):
        if False:
            print('Hello World!')

        class MyException(Exception):
            pass

        @util.Noparallel()
        def raiseException():
            if False:
                print('Hello World!')
            raise MyException('Test error!')
        with pytest.raises(MyException) as err:
            raiseException()
        assert str(err.value) == 'Test error!'
        with pytest.raises(MyException) as err:
            queue_spawn(raiseException).get()
        assert str(err.value) == 'Test error!'

    def testMultithreadMix(self, queue_spawn):
        if False:
            print('Hello World!')
        obj1 = ExampleClass()
        with ThreadPool.ThreadPool(10) as thread_pool:
            s = time.time()
            t1 = queue_spawn(obj1.countBlocking, 5)
            time.sleep(0.01)
            t2 = thread_pool.spawn(obj1.countBlocking, 5)
            time.sleep(0.01)
            t3 = thread_pool.spawn(obj1.countBlocking, 5)
            time.sleep(0.3)
            t4 = gevent.spawn(obj1.countBlocking, 5)
            threads = [t1, t2, t3, t4]
            for thread in threads:
                assert thread.get() == 'counted:5'
            time_taken = time.time() - s
            assert obj1.counted == 5
            assert 0.5 < time_taken < 0.7