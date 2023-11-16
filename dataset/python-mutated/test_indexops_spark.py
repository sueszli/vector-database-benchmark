import pandas as pd
from pyspark.errors import AnalysisException
from pyspark.sql import functions as F
from pyspark import pandas as ps
from pyspark.testing.pandasutils import PandasOnSparkTestCase
from pyspark.testing.sqlutils import SQLTestUtils

class SparkIndexOpsMethodsTestsMixin:

    @property
    def pser(self):
        if False:
            return 10
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def psser(self):
        if False:
            i = 10
            return i + 15
        return ps.from_pandas(self.pser)

    def test_series_transform_negative(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'The output of the function.* pyspark.sql.Column.*int'):
            self.psser.spark.transform(lambda scol: 1)
        with self.assertRaisesRegex(AnalysisException, '.*UNRESOLVED_COLUMN.*`non-existent`.*'):
            self.psser.spark.transform(lambda scol: F.col('non-existent'))

    def test_multiindex_transform_negative(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(NotImplementedError, 'MultiIndex does not support spark.transform yet'):
            midx = pd.MultiIndex([['lama', 'cow', 'falcon'], ['speed', 'weight', 'length']], [[0, 0, 0, 1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 2, 1, 2, 2]])
            s = ps.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
            s.index.spark.transform(lambda scol: scol)

    def test_series_apply_negative(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'The output of the function.* pyspark.sql.Column.*int'):
            self.psser.spark.apply(lambda scol: 1)
        with self.assertRaisesRegex(AnalysisException, '.*UNRESOLVED_COLUMN.*`non-existent`.*'):
            self.psser.spark.transform(lambda scol: F.col('non-existent'))

class SparkIndexOpsMethodsTests(SparkIndexOpsMethodsTestsMixin, PandasOnSparkTestCase, SQLTestUtils):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.test_indexops_spark import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)