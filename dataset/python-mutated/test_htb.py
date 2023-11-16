__version__ = '$Revision: 1.3 $'[11:-2]
from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer

class DummyClock:
    time = 0

    def set(self, when: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.time = when

    def __call__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.time

class SomeBucket(htb.Bucket):
    maxburst = 100
    rate = 2

class TestBucketBase(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._realTimeFunc = htb.time
        self.clock = DummyClock()
        htb.time = self.clock

    def tearDown(self) -> None:
        if False:
            return 10
        htb.time = self._realTimeFunc

class BucketTests(TestBucketBase):

    def testBucketSize(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Testing the size of the bucket.\n        '
        b = SomeBucket()
        fit = b.add(1000)
        self.assertEqual(100, fit)

    def testBucketDrain(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Testing the bucket's drain rate.\n        "
        b = SomeBucket()
        fit = b.add(1000)
        self.clock.set(10)
        fit = b.add(1000)
        self.assertEqual(20, fit)

    def test_bucketEmpty(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{htb.Bucket.drip} returns C{True} if the bucket is empty after that drip.\n        '
        b = SomeBucket()
        b.add(20)
        self.clock.set(9)
        empty = b.drip()
        self.assertFalse(empty)
        self.clock.set(10)
        empty = b.drip()
        self.assertTrue(empty)

class BucketNestingTests(TestBucketBase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        TestBucketBase.setUp(self)
        self.parent = SomeBucket()
        self.child1 = SomeBucket(self.parent)
        self.child2 = SomeBucket(self.parent)

    def testBucketParentSize(self) -> None:
        if False:
            while True:
                i = 10
        self.child1.add(90)
        fit = self.child2.add(90)
        self.assertEqual(10, fit)

    def testBucketParentRate(self) -> None:
        if False:
            i = 10
            return i + 15
        self.parent.rate = 1
        self.child1.add(100)
        self.clock.set(10)
        fit = self.child1.add(100)
        self.assertEqual(10, fit)

class ConsumerShaperTests(TestBucketBase):

    def setUp(self) -> None:
        if False:
            return 10
        TestBucketBase.setUp(self)
        self.underlying = DummyConsumer()
        self.bucket = SomeBucket()
        self.shaped = htb.ShapedConsumer(self.underlying, self.bucket)

    def testRate(self) -> None:
        if False:
            print('Hello World!')
        delta_t = 10
        self.bucket.add(100)
        self.shaped.write('x' * 100)
        self.clock.set(delta_t)
        self.shaped.resumeProducing()
        self.assertEqual(len(self.underlying.getvalue()), delta_t * self.bucket.rate)

    def testBucketRefs(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.bucket._refcount, 1)
        self.shaped.stopProducing()
        self.assertEqual(self.bucket._refcount, 0)