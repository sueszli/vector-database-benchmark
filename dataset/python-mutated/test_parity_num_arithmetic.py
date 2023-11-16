import unittest
from pyspark import pandas as ps
from pyspark.pandas.tests.data_type_ops.test_num_arithmetic import ArithmeticTestsMixin
from pyspark.pandas.tests.connect.data_type_ops.testing_utils import OpsTestBase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils
from pyspark.testing.connectutils import ReusedConnectTestCase

class ArithmeticParityTests(ArithmeticTestsMixin, PandasOnSparkTestUtils, OpsTestBase, ReusedConnectTestCase):

    @property
    def psdf(self):
        if False:
            for i in range(10):
                print('nop')
        return ps.from_pandas(self.pdf)
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.data_type_ops.test_parity_num_arithmetic import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)