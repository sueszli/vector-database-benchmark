import unittest
from pyspark import pandas as ps
from pyspark.pandas.tests.test_categorical import CategoricalTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils, TestUtils

class CategoricalParityTests(CategoricalTestsMixin, PandasOnSparkTestUtils, TestUtils, ReusedConnectTestCase):

    @property
    def psdf(self):
        if False:
            i = 10
            return i + 15
        return ps.from_pandas(self.pdf)
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.test_parity_categorical import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)