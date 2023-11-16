import pandas as pd
from pyspark.sql import functions as F
from pyspark import pandas as ps
from pyspark.testing.pandasutils import PandasOnSparkTestCase

class DefaultIndexTestsMixin:

    def test_default_index_sequence(self):
        if False:
            for i in range(10):
                print('nop')
        with ps.option_context('compute.default_index_type', 'sequence'):
            sdf = self.spark.range(1000)
            self.assert_eq(ps.DataFrame(sdf), pd.DataFrame({'id': list(range(1000))}))

    def test_default_index_distributed_sequence(self):
        if False:
            i = 10
            return i + 15
        with ps.option_context('compute.default_index_type', 'distributed-sequence'):
            sdf = self.spark.range(1000)
            self.assert_eq(ps.DataFrame(sdf), pd.DataFrame({'id': list(range(1000))}))

    def test_default_index_distributed(self):
        if False:
            return 10
        with ps.option_context('compute.default_index_type', 'distributed'):
            sdf = self.spark.range(1000)
            pdf = ps.DataFrame(sdf)._to_pandas()
            self.assertEqual(len(set(pdf.index)), len(pdf))

    def test_index_distributed_sequence_cleanup(self):
        if False:
            while True:
                i = 10
        with ps.option_context('compute.default_index_type', 'distributed-sequence'), ps.option_context('compute.ops_on_diff_frames', True):
            with ps.option_context('compute.default_index_cache', 'LOCAL_CHECKPOINT'):
                cached_rdd_ids = [rdd_id for rdd_id in self.spark._jsc.getPersistentRDDs()]
                psdf1 = self.spark.range(0, 100, 1, 10).withColumn('Key', F.col('id') % 33).pandas_api()
                psdf2 = psdf1['Key'].reset_index()
                psdf2['index'] = (psdf2.groupby(['Key']).cumcount() == 0).astype(int)
                psdf2['index'] = psdf2['index'].cumsum()
                psdf3 = ps.merge(psdf1, psdf2, how='inner', left_on=['Key'], right_on=['Key'])
                _ = len(psdf3)
                self.assertTrue(any((rdd_id not in cached_rdd_ids for rdd_id in self.spark._jsc.getPersistentRDDs())))
            for storage_level in ['NONE', 'DISK_ONLY_2', 'MEMORY_AND_DISK_SER']:
                with ps.option_context('compute.default_index_cache', storage_level):
                    cached_rdd_ids = [rdd_id for rdd_id in self.spark._jsc.getPersistentRDDs()]
                    psdf1 = self.spark.range(0, 100, 1, 10).withColumn('Key', F.col('id') % 33).pandas_api()
                    psdf2 = psdf1['Key'].reset_index()
                    psdf2['index'] = (psdf2.groupby(['Key']).cumcount() == 0).astype(int)
                    psdf2['index'] = psdf2['index'].cumsum()
                    psdf3 = ps.merge(psdf1, psdf2, how='inner', left_on=['Key'], right_on=['Key'])
                    _ = len(psdf3)
                    self.assertTrue(all((rdd_id in cached_rdd_ids for rdd_id in self.spark._jsc.getPersistentRDDs())))

class DefaultIndexTests(DefaultIndexTestsMixin, PandasOnSparkTestCase):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.test_default_index import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)