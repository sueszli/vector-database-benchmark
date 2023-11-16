from pyspark.testing.utils import ReusedPySparkTestCase
from pyspark.rddsampler import RDDSampler, RDDStratifiedSampler

class RDDSamplerTests(ReusedPySparkTestCase):

    def test_rdd_sampler_func(self):
        if False:
            return 10
        rdd = self.sc.parallelize(range(20), 2)
        sample_count = rdd.mapPartitionsWithIndex(RDDSampler(False, 0.4, 10).func).count()
        self.assertGreater(sample_count, 3)
        self.assertLess(sample_count, 10)
        sample_data = rdd.mapPartitionsWithIndex(RDDSampler(True, 1, 10).func).collect()
        sample_data.sort()
        self.assertTrue(any((sample_data[i] == sample_data[i - 1] for i in range(1, len(sample_data)))))

    def test_rdd_stratified_sampler_func(self):
        if False:
            return 10
        fractions = {'a': 0.8, 'b': 0.2}
        rdd = self.sc.parallelize(fractions.keys()).cartesian(self.sc.parallelize(range(0, 100)))
        sample_data = dict(rdd.mapPartitionsWithIndex(RDDStratifiedSampler(False, fractions, 10).func, True).countByKey())
        self.assertGreater(sample_data['a'], sample_data['b'])
        self.assertGreater(sample_data['a'], 60)
        self.assertLess(sample_data['a'], 90)
        self.assertGreater(sample_data['b'], 15)
        self.assertLess(sample_data['b'], 30)
if __name__ == '__main__':
    import unittest
    from pyspark.tests.test_rddsampler import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)