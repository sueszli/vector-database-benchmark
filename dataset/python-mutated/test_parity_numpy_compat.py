import unittest
import pandas as pd
from pyspark import pandas as ps
from pyspark.pandas.tests.test_numpy_compat import NumPyCompatTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class NumPyCompatParityTests(NumPyCompatTestsMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @property
    def pdf(self):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def psdf(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.pdf)
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.test_parity_numpy_compat import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)