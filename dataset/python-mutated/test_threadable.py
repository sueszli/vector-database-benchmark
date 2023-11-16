"""
Tests for L{twisted.python.threadable}.
"""
import pickle
import sys
from unittest import skipIf
try:
    import threading
except ImportError:
    threadingSkip = True
else:
    threadingSkip = False
from twisted.python import threadable
from twisted.trial.unittest import FailTest, SynchronousTestCase

class TestObject:
    synchronized = ['aMethod']
    x = -1
    y = 1

    def aMethod(self):
        if False:
            return 10
        for i in range(10):
            (self.x, self.y) = (self.y, self.x)
            self.z = self.x + self.y
            assert self.z == 0, 'z == %d, not 0 as expected' % (self.z,)
threadable.synchronize(TestObject)

class SynchronizationTests(SynchronousTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        Reduce the CPython check interval so that thread switches happen much\n        more often, hopefully exercising more possible race conditions.  Also,\n        delay actual test startup until the reactor has been started.\n        '
        self.addCleanup(sys.setswitchinterval, sys.getswitchinterval())
        sys.setswitchinterval(1e-07)

    def test_synchronizedName(self):
        if False:
            while True:
                i = 10
        '\n        The name of a synchronized method is inaffected by the synchronization\n        decorator.\n        '
        self.assertEqual('aMethod', TestObject.aMethod.__name__)

    @skipIf(threadingSkip, 'Platform does not support threads')
    def test_isInIOThread(self):
        if False:
            print('Hello World!')
        '\n        L{threadable.isInIOThread} returns C{True} if and only if it is called\n        in the same thread as L{threadable.registerAsIOThread}.\n        '
        threadable.registerAsIOThread()
        foreignResult = []
        t = threading.Thread(target=lambda : foreignResult.append(threadable.isInIOThread()))
        t.start()
        t.join()
        self.assertFalse(foreignResult[0], 'Non-IO thread reported as IO thread')
        self.assertTrue(threadable.isInIOThread(), 'IO thread reported as not IO thread')

    @skipIf(threadingSkip, 'Platform does not support threads')
    def testThreadedSynchronization(self):
        if False:
            i = 10
            return i + 15
        o = TestObject()
        errors = []

        def callMethodLots():
            if False:
                return 10
            try:
                for i in range(1000):
                    o.aMethod()
            except AssertionError as e:
                errors.append(str(e))
        threads = []
        for x in range(5):
            t = threading.Thread(target=callMethodLots)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise FailTest(errors)

    def testUnthreadedSynchronization(self):
        if False:
            i = 10
            return i + 15
        o = TestObject()
        for i in range(1000):
            o.aMethod()

class SerializationTests(SynchronousTestCase):

    @skipIf(threadingSkip, 'Platform does not support threads')
    def testPickling(self):
        if False:
            print('Hello World!')
        lock = threadable.XLock()
        lockType = type(lock)
        lockPickle = pickle.dumps(lock)
        newLock = pickle.loads(lockPickle)
        self.assertIsInstance(newLock, lockType)

    def testUnpickling(self):
        if False:
            for i in range(10):
                print('nop')
        lockPickle = b'ctwisted.python.threadable\nunpickle_lock\np0\n(tp1\nRp2\n.'
        lock = pickle.loads(lockPickle)
        newPickle = pickle.dumps(lock, 2)
        pickle.loads(newPickle)