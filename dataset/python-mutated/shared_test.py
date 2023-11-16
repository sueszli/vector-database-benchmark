"""Test for Shared class."""
import gc
import threading
import time
import unittest
from apache_beam.utils import shared

class Count(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._lock = threading.Lock()
        self._total = 0
        self._active = 0

    def add_ref(self):
        if False:
            while True:
                i = 10
        with self._lock:
            self._total += 1
            self._active += 1

    def release_ref(self):
        if False:
            print('Hello World!')
        with self._lock:
            self._active -= 1

    def get_active(self):
        if False:
            print('Hello World!')
        with self._lock:
            return self._active

    def get_total(self):
        if False:
            return 10
        with self._lock:
            return self._total

class Marker(object):

    def __init__(self, count):
        if False:
            for i in range(10):
                print('nop')
        self._count = count
        self._count.add_ref()

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self._count.release_ref()

class NamedObject(object):

    def __init__(self, name):
        if False:
            print('Hello World!')
        self._name = name

    def get_name(self):
        if False:
            while True:
                i = 10
        return self._name

class Sequence(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self._sequence = 0

    def make_acquire_fn(self):
        if False:
            for i in range(10):
                print('nop')

        def acquire_fn():
            if False:
                for i in range(10):
                    print('nop')
            self._sequence += 1
            return NamedObject('sequence%d' % self._sequence)
        return acquire_fn

class SharedTest(unittest.TestCase):

    def testKeepalive(self):
        if False:
            for i in range(10):
                print('nop')
        count = Count()
        shared_handle = shared.Shared()
        other_shared_handle = shared.Shared()

        def dummy_acquire_fn():
            if False:
                return 10
            return None

        def acquire_fn():
            if False:
                print('Hello World!')
            return Marker(count)
        p1 = shared_handle.acquire(acquire_fn)
        self.assertEqual(1, count.get_total())
        self.assertEqual(1, count.get_active())
        del p1
        gc.collect()
        self.assertEqual(1, count.get_active())
        p2 = shared_handle.acquire(acquire_fn)
        self.assertEqual(1, count.get_total())
        self.assertEqual(1, count.get_active())
        other_shared_handle.acquire(dummy_acquire_fn)
        del p2
        gc.collect()
        self.assertEqual(0, count.get_active())

    def testMultiple(self):
        if False:
            while True:
                i = 10
        count = Count()
        shared_handle = shared.Shared()
        other_shared_handle = shared.Shared()

        def dummy_acquire_fn():
            if False:
                return 10
            return None

        def acquire_fn():
            if False:
                i = 10
                return i + 15
            return Marker(count)
        p = shared_handle.acquire(acquire_fn)
        other_shared_handle.acquire(dummy_acquire_fn)
        self.assertEqual(1, count.get_total())
        self.assertEqual(1, count.get_active())
        del p
        gc.collect()
        self.assertEqual(0, count.get_active())
        p1 = shared_handle.acquire(acquire_fn)
        self.assertEqual(2, count.get_total())
        self.assertEqual(1, count.get_active())
        p2 = shared_handle.acquire(acquire_fn)
        self.assertEqual(2, count.get_total())
        self.assertEqual(1, count.get_active())
        other_shared_handle.acquire(dummy_acquire_fn)
        del p2
        gc.collect()
        self.assertEqual(1, count.get_active())
        del p1
        gc.collect()
        self.assertEqual(0, count.get_active())

    def testConcurrentCallsDeduped(self):
        if False:
            i = 10
            return i + 15
        count = Count()
        shared_handle = shared.Shared()
        other_shared_handle = shared.Shared()
        refs = []
        ref_lock = threading.Lock()

        def dummy_acquire_fn():
            if False:
                print('Hello World!')
            return None

        def acquire_fn():
            if False:
                for i in range(10):
                    print('nop')
            time.sleep(1)
            return Marker(count)

        def thread_fn():
            if False:
                print('Hello World!')
            p = shared_handle.acquire(acquire_fn)
            with ref_lock:
                refs.append(p)
        threads = []
        for _ in range(100):
            t = threading.Thread(target=thread_fn)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(1, count.get_total())
        self.assertEqual(1, count.get_active())
        other_shared_handle.acquire(dummy_acquire_fn)
        with ref_lock:
            del refs[:]
        gc.collect()
        self.assertEqual(0, count.get_active())

    def testDifferentObjects(self):
        if False:
            print('Hello World!')
        sequence = Sequence()

        def dummy_acquire_fn():
            if False:
                while True:
                    i = 10
            return None
        first_handle = shared.Shared()
        second_handle = shared.Shared()
        dummy_handle = shared.Shared()
        f1 = first_handle.acquire(sequence.make_acquire_fn())
        s1 = second_handle.acquire(sequence.make_acquire_fn())
        self.assertEqual('sequence1', f1.get_name())
        self.assertEqual('sequence2', s1.get_name())
        f2 = first_handle.acquire(sequence.make_acquire_fn())
        s2 = second_handle.acquire(sequence.make_acquire_fn())
        self.assertEqual('sequence1', f2.get_name())
        self.assertEqual('sequence2', s2.get_name())
        del f1
        del f2
        del s1
        del s2
        dummy_handle.acquire(dummy_acquire_fn)
        gc.collect()
        f3 = first_handle.acquire(sequence.make_acquire_fn())
        s3 = second_handle.acquire(sequence.make_acquire_fn())
        self.assertEqual('sequence3', f3.get_name())
        self.assertEqual('sequence4', s3.get_name())

    def testTagCacheEviction(self):
        if False:
            while True:
                i = 10
        shared1 = shared.Shared()
        shared2 = shared.Shared()

        def acquire_fn_1():
            if False:
                while True:
                    i = 10
            return NamedObject('obj_1')

        def acquire_fn_2():
            if False:
                print('Hello World!')
            return NamedObject('obj_2')
        p1 = shared1.acquire(acquire_fn_1)
        assert p1.get_name() == 'obj_1'
        p2 = shared1.acquire(acquire_fn_2)
        assert p2.get_name() == 'obj_1'
        p1 = shared2.acquire(acquire_fn_1, tag='1')
        assert p1.get_name() == 'obj_1'
        p2 = shared2.acquire(acquire_fn_2, tag='2')
        assert p2.get_name() == 'obj_2'

    def testTagReturnsCached(self):
        if False:
            for i in range(10):
                print('nop')
        sequence = Sequence()
        handle = shared.Shared()
        f1 = handle.acquire(sequence.make_acquire_fn(), tag='1')
        self.assertEqual('sequence1', f1.get_name())
        f1 = handle.acquire(sequence.make_acquire_fn(), tag='1')
        self.assertEqual('sequence1', f1.get_name())
if __name__ == '__main__':
    unittest.main()