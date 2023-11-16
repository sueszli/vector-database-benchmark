import math
from pyspark.statcounter import StatCounter
from pyspark.testing.utils import ReusedPySparkTestCase

class StatCounterTests(ReusedPySparkTestCase):

    def test_base(self):
        if False:
            i = 10
            return i + 15
        stats = self.sc.parallelize([1.0, 2.0, 3.0, 4.0]).stats()
        self.assertEqual(stats.count(), 4)
        self.assertEqual(stats.max(), 4.0)
        self.assertEqual(stats.mean(), 2.5)
        self.assertEqual(stats.min(), 1.0)
        self.assertAlmostEqual(stats.stdev(), 1.118033988749895)
        self.assertAlmostEqual(stats.sampleStdev(), 1.2909944487358056)
        self.assertEqual(stats.sum(), 10.0)
        self.assertAlmostEqual(stats.variance(), 1.25)
        self.assertAlmostEqual(stats.sampleVariance(), 1.6666666666666667)

    def test_as_dict(self):
        if False:
            while True:
                i = 10
        stats = self.sc.parallelize([1.0, 2.0, 3.0, 4.0]).stats().asDict()
        self.assertEqual(stats['count'], 4)
        self.assertEqual(stats['max'], 4.0)
        self.assertEqual(stats['mean'], 2.5)
        self.assertEqual(stats['min'], 1.0)
        self.assertAlmostEqual(stats['stdev'], 1.2909944487358056)
        self.assertEqual(stats['sum'], 10.0)
        self.assertAlmostEqual(stats['variance'], 1.6666666666666667)
        stats = self.sc.parallelize([1.0, 2.0, 3.0, 4.0]).stats().asDict(sample=True)
        self.assertEqual(stats['count'], 4)
        self.assertEqual(stats['max'], 4.0)
        self.assertEqual(stats['mean'], 2.5)
        self.assertEqual(stats['min'], 1.0)
        self.assertAlmostEqual(stats['stdev'], 1.118033988749895)
        self.assertEqual(stats['sum'], 10.0)
        self.assertAlmostEqual(stats['variance'], 1.25)

    def test_merge(self):
        if False:
            while True:
                i = 10
        stats = StatCounter([1.0, 2.0, 3.0, 4.0])
        stats.merge(5.0)
        self.assertEqual(stats.count(), 5)
        self.assertEqual(stats.max(), 5.0)
        self.assertEqual(stats.mean(), 3.0)
        self.assertEqual(stats.min(), 1.0)
        self.assertAlmostEqual(stats.stdev(), 1.414213562373095)
        self.assertAlmostEqual(stats.sampleStdev(), 1.5811388300841898)
        self.assertEqual(stats.sum(), 15.0)
        self.assertAlmostEqual(stats.variance(), 2.0)
        self.assertAlmostEqual(stats.sampleVariance(), 2.5)

    def test_merge_stats(self):
        if False:
            i = 10
            return i + 15
        stats1 = StatCounter([1.0, 2.0, 3.0, 4.0])
        stats2 = StatCounter([1.0, 2.0, 3.0, 4.0])
        stats = stats1.mergeStats(stats2)
        self.assertEqual(stats.count(), 8)
        self.assertEqual(stats.max(), 4.0)
        self.assertEqual(stats.mean(), 2.5)
        self.assertEqual(stats.min(), 1.0)
        self.assertAlmostEqual(stats.stdev(), 1.118033988749895)
        self.assertAlmostEqual(stats.sampleStdev(), 1.1952286093343936)
        self.assertEqual(stats.sum(), 20.0)
        self.assertAlmostEqual(stats.variance(), 1.25)
        self.assertAlmostEqual(stats.sampleVariance(), 1.4285714285714286)
        execution_statements = [StatCounter([1.0, 2.0]).mergeStats(StatCounter(range(1, 301))), StatCounter(range(1, 301)).mergeStats(StatCounter([1.0, 2.0]))]
        for stats in execution_statements:
            self.assertEqual(stats.count(), 302)
            self.assertEqual(stats.max(), 300.0)
            self.assertEqual(stats.min(), 1.0)
            self.assertAlmostEqual(stats.mean(), 149.51324503311)
            self.assertAlmostEqual(stats.variance(), 7596.302804701549)
            self.assertAlmostEqual(stats.sampleVariance(), 7621.539691095905)

    def test_variance_when_size_zero(self):
        if False:
            i = 10
            return i + 15
        arguments = [[], None]
        for arg in arguments:
            stats = StatCounter(arg)
            self.assertTrue(math.isnan(stats.variance()))
            self.assertTrue(math.isnan(stats.sampleVariance()))
            self.assertEqual(stats.count(), 0)
            self.assertTrue(math.isinf(stats.max()))
            self.assertTrue(math.isinf(stats.min()))
            self.assertEqual(stats.mean(), 0.0)

    def test_merge_stats_with_self(self):
        if False:
            i = 10
            return i + 15
        stats = StatCounter([1.0, 2.0, 3.0, 4.0])
        stats.mergeStats(stats)
        self.assertEqual(stats.count(), 8)
        self.assertEqual(stats.max(), 4.0)
        self.assertEqual(stats.mean(), 2.5)
        self.assertEqual(stats.min(), 1.0)
        self.assertAlmostEqual(stats.stdev(), 1.118033988749895)
        self.assertAlmostEqual(stats.sampleStdev(), 1.1952286093343936)
        self.assertEqual(stats.sum(), 20.0)
        self.assertAlmostEqual(stats.variance(), 1.25)
        self.assertAlmostEqual(stats.sampleVariance(), 1.4285714285714286)
if __name__ == '__main__':
    import unittest
    from pyspark.tests.test_statcounter import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)