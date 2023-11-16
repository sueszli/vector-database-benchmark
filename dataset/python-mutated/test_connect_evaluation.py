import unittest
from pyspark.sql import SparkSession
from pyspark.testing.connectutils import should_test_connect, connect_requirement_message
have_torcheval = True
try:
    import torcheval
except ImportError:
    have_torcheval = False
if should_test_connect:
    from pyspark.ml.tests.connect.test_legacy_mode_evaluation import EvaluationTestsMixin

@unittest.skipIf(not should_test_connect or not have_torcheval, connect_requirement_message or 'torcheval is required')
class EvaluationTestsOnConnect(EvaluationTestsMixin, unittest.TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.spark = SparkSession.builder.remote('local[2]').getOrCreate()

    def tearDown(self) -> None:
        if False:
            while True:
                i = 10
        self.spark.stop()
if __name__ == '__main__':
    from pyspark.ml.tests.connect.test_connect_evaluation import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)