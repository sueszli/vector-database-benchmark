from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gevent import lock
import gevent.testing as greentest
from gevent.tests import test__semaphore

class TestRLockMultiThread(test__semaphore.TestSemaphoreMultiThread):

    def _makeOne(self):
        if False:
            i = 10
            return i + 15
        return lock.RLock()

    def assertOneHasNoHub(self, sem):
        if False:
            print('Hello World!')
        self.assertIsNone(sem._block.hub)
if __name__ == '__main__':
    greentest.main()