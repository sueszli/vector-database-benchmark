import unittest
from pyspark import pandas as ps
from pyspark.pandas.tests.test_dataframe_conversion import DataFrameConversionTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils, TestUtils

class DataFrameConversionParityTests(DataFrameConversionTestsMixin, PandasOnSparkTestUtils, ReusedConnectTestCase, TestUtils):

    @property
    def psdf(self):
        if False:
            while True:
                i = 10
        return ps.from_pandas(self.pdf)
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.test_parity_dataframe_conversion import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)