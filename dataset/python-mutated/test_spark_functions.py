from pyspark.testing.pandasutils import PandasOnSparkTestCase

class SparkFunctionsTestsMixin:

    def test_repeat(self):
        if False:
            print('Hello World!')
        pass

class SparkFunctionsTests(SparkFunctionsTestsMixin, PandasOnSparkTestCase):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.test_spark_functions import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)