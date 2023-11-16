import os
import pandas as pd
from pyspark import pandas as ps
from pyspark.testing.pandasutils import PandasOnSparkTestCase, TestUtils
from pyspark.testing.sqlutils import SQLTestUtils

class SparkFrameMethodsTestsMixin:

    def test_frame_apply_negative(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'The output of the function.* pyspark.sql.DataFrame.*int'):
            ps.range(10).spark.apply(lambda scol: 1)

    def test_hint(self):
        if False:
            for i in range(10):
                print('nop')
        pdf1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'], 'value': [1, 2, 3, 5]}).set_index('lkey')
        pdf2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'], 'value': [5, 6, 7, 8]}).set_index('rkey')
        psdf1 = ps.from_pandas(pdf1)
        psdf2 = ps.from_pandas(pdf2)
        hints = ['broadcast', 'merge', 'shuffle_hash', 'shuffle_replicate_nl']
        for hint in hints:
            self.assert_eq(pdf1.merge(pdf2, left_index=True, right_index=True).sort_values(['value_x', 'value_y']), psdf1.merge(psdf2.spark.hint(hint), left_index=True, right_index=True).sort_values(['value_x', 'value_y']), almost=True)
            self.assert_eq(pdf1.merge(pdf2 + 1, left_index=True, right_index=True).sort_values(['value_x', 'value_y']), psdf1.merge((psdf2 + 1).spark.hint(hint), left_index=True, right_index=True).sort_values(['value_x', 'value_y']), almost=True)

    def test_repartition(self):
        if False:
            return 10
        psdf = ps.DataFrame({'age': [5, 5, 2, 2], 'name': ['Bob', 'Bob', 'Alice', 'Alice']})
        num_partitions = psdf.to_spark().rdd.getNumPartitions() + 1
        num_partitions += 1
        new_psdf = psdf.spark.repartition(num_partitions)
        self.assertEqual(new_psdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(psdf.sort_index(), new_psdf.sort_index())
        psdf = psdf.set_index('age')
        num_partitions += 1
        new_psdf = psdf.spark.repartition(num_partitions)
        self.assertEqual(new_psdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(psdf.sort_index(), new_psdf.sort_index())
        psdf = psdf.reset_index()
        psdf = psdf.set_index('name')
        psdf2 = psdf + 1
        num_partitions += 1
        self.assert_eq(psdf2.sort_index(), (psdf + 1).spark.repartition(num_partitions).sort_index())
        psdf = ps.DataFrame({'a': ['a', 'b', 'c']}, index=[[1, 2, 3], [4, 5, 6]])
        num_partitions = psdf.to_spark().rdd.getNumPartitions() + 1
        new_psdf = psdf.spark.repartition(num_partitions)
        self.assertEqual(new_psdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(psdf.sort_index(), new_psdf.sort_index())

    def test_coalesce(self):
        if False:
            i = 10
            return i + 15
        num_partitions = 10
        psdf = ps.DataFrame({'age': [5, 5, 2, 2], 'name': ['Bob', 'Bob', 'Alice', 'Alice']})
        psdf = psdf.spark.repartition(num_partitions)
        num_partitions -= 1
        new_psdf = psdf.spark.coalesce(num_partitions)
        self.assertEqual(new_psdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(psdf.sort_index(), new_psdf.sort_index())
        psdf = psdf.set_index('age')
        num_partitions -= 1
        new_psdf = psdf.spark.coalesce(num_partitions)
        self.assertEqual(new_psdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(psdf.sort_index(), new_psdf.sort_index())
        psdf = psdf.reset_index()
        psdf = psdf.set_index('name')
        psdf2 = psdf + 1
        num_partitions -= 1
        self.assert_eq(psdf2.sort_index(), (psdf + 1).spark.coalesce(num_partitions).sort_index())
        psdf = ps.DataFrame({'a': ['a', 'b', 'c']}, index=[[1, 2, 3], [4, 5, 6]])
        num_partitions -= 1
        psdf = psdf.spark.repartition(num_partitions)
        num_partitions -= 1
        new_psdf = psdf.spark.coalesce(num_partitions)
        self.assertEqual(new_psdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(psdf.sort_index(), new_psdf.sort_index())

    def test_checkpoint(self):
        if False:
            print('Hello World!')
        with self.temp_dir() as tmp:
            self.spark.sparkContext.setCheckpointDir(tmp)
            psdf = ps.DataFrame({'a': ['a', 'b', 'c']})
            new_psdf = psdf.spark.checkpoint()
            self.assertIsNotNone(os.listdir(tmp))
            self.assert_eq(psdf, new_psdf)

    def test_local_checkpoint(self):
        if False:
            print('Hello World!')
        psdf = ps.DataFrame({'a': ['a', 'b', 'c']})
        new_psdf = psdf.spark.local_checkpoint()
        self.assert_eq(psdf, new_psdf)

class SparkFrameMethodsTests(SparkFrameMethodsTestsMixin, PandasOnSparkTestCase, SQLTestUtils, TestUtils):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.test_frame_spark import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)