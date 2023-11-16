import time
import gevent
from util import RateLimit

def around(t, limit):
    if False:
        while True:
            i = 10
    return t >= limit - 0.05 and t <= limit + 0.05

class ExampleClass(object):

    def __init__(self):
        if False:
            return 10
        self.counted = 0
        self.last_called = None

    def count(self, back='counted'):
        if False:
            for i in range(10):
                print('nop')
        self.counted += 1
        self.last_called = back
        return back

class TestRateLimit:

    def testCall(self):
        if False:
            print('Hello World!')
        obj1 = ExampleClass()
        obj2 = ExampleClass()
        s = time.time()
        assert RateLimit.call('counting', allowed_again=0.1, func=obj1.count) == 'counted'
        assert around(time.time() - s, 0.0)
        assert obj1.counted == 1
        assert not RateLimit.isAllowed('counting', 0.1)
        assert RateLimit.isAllowed('something else', 0.1)
        assert RateLimit.call('counting', allowed_again=0.1, func=obj1.count) == 'counted'
        assert around(time.time() - s, 0.1)
        assert obj1.counted == 2
        time.sleep(0.1)
        s = time.time()
        assert obj2.counted == 0
        threads = [gevent.spawn(lambda : RateLimit.call('counting', allowed_again=0.1, func=obj2.count)), gevent.spawn(lambda : RateLimit.call('counting', allowed_again=0.1, func=obj2.count)), gevent.spawn(lambda : RateLimit.call('counting', allowed_again=0.1, func=obj2.count))]
        gevent.joinall(threads)
        assert [thread.value for thread in threads] == ['counted', 'counted', 'counted']
        assert around(time.time() - s, 0.2)
        assert not RateLimit.isAllowed('counting', 0.1)
        time.sleep(0.11)
        assert RateLimit.isAllowed('counting', 0.1)
        s = time.time()
        assert RateLimit.isAllowed('counting', 0.1)
        assert RateLimit.call('counting', allowed_again=0.1, func=obj2.count) == 'counted'
        assert around(time.time() - s, 0.0)
        assert obj2.counted == 4

    def testCallAsync(self):
        if False:
            print('Hello World!')
        obj1 = ExampleClass()
        obj2 = ExampleClass()
        s = time.time()
        RateLimit.callAsync('counting async', allowed_again=0.1, func=obj1.count, back='call #1').join()
        assert obj1.counted == 1
        assert around(time.time() - s, 0.0)
        s = time.time()
        t1 = RateLimit.callAsync('counting async', allowed_again=0.1, func=obj1.count, back='call #2')
        time.sleep(0.03)
        t2 = RateLimit.callAsync('counting async', allowed_again=0.1, func=obj1.count, back='call #3')
        time.sleep(0.03)
        t3 = RateLimit.callAsync('counting async', allowed_again=0.1, func=obj1.count, back='call #4')
        assert obj1.counted == 1
        t3.join()
        assert t3.value == 'call #4'
        assert around(time.time() - s, 0.1)
        assert obj1.counted == 2
        assert obj1.last_called == 'call #4'
        assert not RateLimit.isAllowed('counting async', 0.1)
        s = time.time()
        t4 = RateLimit.callAsync('counting async', allowed_again=0.1, func=obj1.count, back='call #5').join()
        assert obj1.counted == 3
        assert around(time.time() - s, 0.1)
        assert not RateLimit.isAllowed('counting async', 0.1)
        time.sleep(0.11)
        assert RateLimit.isAllowed('counting async', 0.1)