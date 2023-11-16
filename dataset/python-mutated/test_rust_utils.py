import gc
import threading
from cryptography.hazmat.bindings._rust import FixedPool

class TestFixedPool:

    def test_basic(self):
        if False:
            return 10
        c = 0
        events = []

        def create():
            if False:
                i = 10
                return i + 15
            nonlocal c
            c += 1
            events.append(('create', c))
            return c
        pool = FixedPool(create)
        assert events == [('create', 1)]
        with pool.acquire() as c:
            assert c == 1
            assert events == [('create', 1)]
            with pool.acquire() as c:
                assert c == 2
                assert events == [('create', 1), ('create', 2)]
            assert events == [('create', 1), ('create', 2)]
        assert events == [('create', 1), ('create', 2)]
        del pool
        gc.collect()
        gc.collect()
        gc.collect()
        assert events == [('create', 1), ('create', 2)]

    def test_thread_stress(self):
        if False:
            i = 10
            return i + 15

        def create():
            if False:
                print('Hello World!')
            return None
        pool = FixedPool(create)

        def thread_fn():
            if False:
                while True:
                    i = 10
            with pool.acquire():
                pass
        threads = []
        for i in range(1024):
            t = threading.Thread(target=thread_fn)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()