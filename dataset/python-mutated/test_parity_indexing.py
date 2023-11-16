import unittest
import pandas as pd
from pyspark import pandas as ps
from pyspark.pandas.tests.test_indexing import BasicIndexingTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class BasicIndexingParityTests(BasicIndexingTestsMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @property
    def pdf(self):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame({'month': [1, 4, 7, 10], 'year': [2012, 2014, 2013, 2014], 'sale': [55, 40, 84, 31]})

    @property
    def psdf(self):
        if False:
            print('Hello World!')
        return ps.from_pandas(self.pdf)
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.test_parity_indexing import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)