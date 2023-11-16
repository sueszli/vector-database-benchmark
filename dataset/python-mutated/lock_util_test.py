"""Tests for lock_util."""
import random
import time
from absl.testing import parameterized
from tensorflow.python.platform import test
from tensorflow.python.util import lock_util

class GroupLockTest(test.TestCase, parameterized.TestCase):

    @parameterized.parameters(1, 2, 3, 5, 10)
    def testGroups(self, num_groups):
        if False:
            i = 10
            return i + 15
        lock = lock_util.GroupLock(num_groups)
        num_threads = 10
        finished = set()

        def thread_fn(thread_id):
            if False:
                return 10
            time.sleep(random.random() * 0.1)
            group_id = thread_id % num_groups
            with lock.group(group_id):
                time.sleep(random.random() * 0.1)
                self.assertGreater(lock._group_member_counts[group_id], 0)
                for (g, c) in enumerate(lock._group_member_counts):
                    if g != group_id:
                        self.assertEqual(0, c)
                finished.add(thread_id)
        threads = [self.checkedThread(target=thread_fn, args=(i,)) for i in range(num_threads)]
        for i in range(num_threads):
            threads[i].start()
        for i in range(num_threads):
            threads[i].join()
        self.assertEqual(set(range(num_threads)), finished)
if __name__ == '__main__':
    test.main()