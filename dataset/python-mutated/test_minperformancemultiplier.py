from golem.environments.minperformancemultiplier import MinPerformanceMultiplier
from golem.tools.testwithdatabase import TestWithDatabase

class TestMinPerformanceMultiplier(TestWithDatabase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.min = MinPerformanceMultiplier.MIN
        self.max = MinPerformanceMultiplier.MAX

    def test_zero_when_not_set(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(0, MinPerformanceMultiplier.get())

    def test_min(self):
        if False:
            return 10
        MinPerformanceMultiplier.set(self.min)
        self.assertEqual(self.min, MinPerformanceMultiplier.get())

    def test_fractional(self):
        if False:
            while True:
                i = 10
        MinPerformanceMultiplier.set(3.1415)
        self.assertEqual(3.1415, MinPerformanceMultiplier.get())

    def test_max(self):
        if False:
            for i in range(10):
                print('nop')
        MinPerformanceMultiplier.set(self.max)
        self.assertEqual(self.max, MinPerformanceMultiplier.get())

    def test_below_min(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(Exception, 'must be within'):
            MinPerformanceMultiplier.set(self.min - 1)

    def test_above_max(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(Exception, 'must be within'):
            MinPerformanceMultiplier.set(self.max + 2)