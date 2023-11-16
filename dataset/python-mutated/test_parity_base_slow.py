import unittest
from pyspark import pandas as ps
from pyspark.pandas.tests.indexes.test_base_slow import IndexesSlowTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils, TestUtils

class IndexesSlowParityTests(IndexesSlowTestsMixin, PandasOnSparkTestUtils, TestUtils, ReusedConnectTestCase):

    @property
    def psdf(self):
        if False:
            return 10
        return ps.from_pandas(self.pdf)
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.indexes.test_parity_base_slow import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)