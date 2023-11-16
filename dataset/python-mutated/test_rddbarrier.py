from pyspark.testing.utils import ReusedPySparkTestCase

class RDDBarrierTests(ReusedPySparkTestCase):

    def test_map_partitions(self):
        if False:
            return 10
        'Test RDDBarrier.mapPartitions'
        rdd = self.sc.parallelize(range(12), 4)
        self.assertFalse(rdd._is_barrier())
        rdd1 = rdd.barrier().mapPartitions(lambda it: it)
        self.assertTrue(rdd1._is_barrier())

    def test_map_partitions_with_index(self):
        if False:
            for i in range(10):
                print('nop')
        'Test RDDBarrier.mapPartitionsWithIndex'
        rdd = self.sc.parallelize(range(12), 4)
        self.assertFalse(rdd._is_barrier())

        def f(index, iterator):
            if False:
                while True:
                    i = 10
            yield index
        rdd1 = rdd.barrier().mapPartitionsWithIndex(f)
        self.assertTrue(rdd1._is_barrier())
        self.assertEqual(rdd1.collect(), [0, 1, 2, 3])
if __name__ == '__main__':
    import unittest
    from pyspark.tests.test_rddbarrier import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)