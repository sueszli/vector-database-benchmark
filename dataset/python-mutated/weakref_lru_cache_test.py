import threading
import time
from absl.testing import absltest
from xla.python import xla_client

class WeakrefLRUCacheTest(absltest.TestCase):

    def testMultiThreaded(self):
        if False:
            for i in range(10):
                print('nop')
        insert_evs = [threading.Event() for _ in range(2)]
        insert_evs_i = 0

        class WRKey:
            pass

        class ClashingKey:

            def __eq__(self, other):
                if False:
                    i = 10
                    return i + 15
                return False

            def __hash__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 333

        class GilReleasingCacheKey:

            def __eq__(self, other):
                if False:
                    print('Hello World!')
                nonlocal insert_evs_i
                if isinstance(other, GilReleasingCacheKey) and insert_evs_i < len(insert_evs):
                    insert_evs[insert_evs_i].set()
                    insert_evs_i += 1
                    time.sleep(0.01)
                return False

            def __hash__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 333

        def CacheFn(obj, gil_releasing_cache_key):
            if False:
                while True:
                    i = 10
            del obj
            del gil_releasing_cache_key
            return None
        cache = xla_client.weakref_lru_cache(lambda : None, CacheFn, 2048)
        wrkey = WRKey()

        def Body():
            if False:
                while True:
                    i = 10
            for insert_ev in insert_evs:
                insert_ev.wait()
                for _ in range(20):
                    cache(wrkey, ClashingKey())
        t = threading.Thread(target=Body)
        t.start()
        for _ in range(3):
            cache(wrkey, GilReleasingCacheKey())
        t.join()

    def testKwargsDictOrder(self):
        if False:
            i = 10
            return i + 15
        miss_id = 0

        class WRKey:
            pass

        def CacheFn(obj, kwkey1, kwkey2):
            if False:
                print('Hello World!')
            del obj, kwkey1, kwkey2
            nonlocal miss_id
            miss_id += 1
            return miss_id
        cache = xla_client.weakref_lru_cache(lambda : None, CacheFn, 4)
        wrkey = WRKey()
        self.assertEqual(cache(wrkey, kwkey1='a', kwkey2='b'), 1)
        self.assertEqual(cache(wrkey, kwkey1='b', kwkey2='a'), 2)
        self.assertEqual(cache(wrkey, kwkey2='b', kwkey1='a'), 1)
if __name__ == '__main__':
    absltest.main()