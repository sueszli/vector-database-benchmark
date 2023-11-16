import unittest
from pyspark.sql.tests.pandas.test_pandas_grouped_map_with_state import GroupedApplyInPandasWithStateTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class GroupedApplyInPandasWithStateTests(GroupedApplyInPandasWithStateTestsMixin, ReusedConnectTestCase):

    @unittest.skip('foreachBatch will be supported in SPARK-42944.')
    def test_apply_in_pandas_with_state_basic(self):
        if False:
            while True:
                i = 10
        super().test_apply_in_pandas_with_state_basic()

    @unittest.skip('foreachBatch will be supported in SPARK-42944.')
    def test_apply_in_pandas_with_state_basic_no_state(self):
        if False:
            while True:
                i = 10
        super().test_apply_in_pandas_with_state_basic()

    @unittest.skip('foreachBatch will be supported in SPARK-42944.')
    def test_apply_in_pandas_with_state_basic_no_state_no_data(self):
        if False:
            i = 10
            return i + 15
        super().test_apply_in_pandas_with_state_basic()

    @unittest.skip('foreachBatch will be supported in SPARK-42944.')
    def test_apply_in_pandas_with_state_basic_more_data(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_apply_in_pandas_with_state_basic()

    @unittest.skip('foreachBatch will be supported in SPARK-42944.')
    def test_apply_in_pandas_with_state_basic_fewer_data(self):
        if False:
            while True:
                i = 10
        super().test_apply_in_pandas_with_state_basic()

    @unittest.skip('foreachBatch will be supported in SPARK-42944.')
    def test_apply_in_pandas_with_state_basic_with_null(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_apply_in_pandas_with_state_basic()
if __name__ == '__main__':
    from pyspark.sql.tests.connect.test_parity_pandas_grouped_map_with_state import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)