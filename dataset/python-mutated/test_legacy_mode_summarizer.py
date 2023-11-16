import unittest
import numpy as np
from pyspark.sql import SparkSession
from pyspark.testing.connectutils import should_test_connect, connect_requirement_message
if should_test_connect:
    from pyspark.ml.connect.summarizer import summarize_dataframe

class SummarizerTestsMixin:

    def test_summarize_dataframe(self):
        if False:
            return 10
        df1 = self.spark.createDataFrame([([2.0, -1.5],), ([-3.0, 0.5],), ([1.0, 3.5],)], schema=['features'])
        df1_local = df1.toPandas()
        result = summarize_dataframe(df1, 'features', ['min', 'max', 'sum', 'mean', 'std'])
        result_local = summarize_dataframe(df1_local, 'features', ['min', 'max', 'sum', 'mean', 'std'])
        expected_result = {'min': [-3.0, -1.5], 'max': [2.0, 3.5], 'sum': [0.0, 2.5], 'mean': [0.0, 0.83333333], 'std': [2.64575131, 2.51661148]}

        def assert_dict_allclose(dict1, dict2):
            if False:
                return 10
            assert set(dict1.keys()) == set(dict2.keys())
            for key in dict1:
                np.testing.assert_allclose(dict1[key], dict2[key])
        assert_dict_allclose(result, expected_result)
        assert_dict_allclose(result_local, expected_result)

@unittest.skipIf(not should_test_connect, connect_requirement_message)
class SummarizerTests(SummarizerTestsMixin, unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.spark = SparkSession.builder.master('local[2]').getOrCreate()

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        self.spark.stop()
if __name__ == '__main__':
    from pyspark.ml.tests.connect.test_legacy_mode_summarizer import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)