import unittest
from pyspark.testing.connectutils import should_test_connect
if should_test_connect:
    from pyspark import sql
    from pyspark.sql.connect.udtf import UserDefinedTableFunction
    sql.udtf.UserDefinedTableFunction = UserDefinedTableFunction
from pyspark.sql.connect.functions import lit, udtf
from pyspark.sql.tests.test_udtf import BaseUDTFTestsMixin, UDTFArrowTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.errors.exceptions.connect import SparkConnectGrpcException

class UDTFParityTests(BaseUDTFTestsMixin, ReusedConnectTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super(UDTFParityTests, cls).setUpClass()
        cls.spark.conf.set('spark.sql.execution.pythonUDTF.arrow.enabled', 'false')

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        try:
            cls.spark.conf.unset('spark.sql.execution.pythonUDTF.arrow.enabled')
        finally:
            super(UDTFParityTests, cls).tearDownClass()

    def test_struct_output_type_casting_row(self):
        if False:
            return 10
        self.check_struct_output_type_casting_row(SparkConnectGrpcException)

    def test_udtf_with_invalid_return_type(self):
        if False:
            i = 10
            return i + 15

        @udtf(returnType='int')
        class TestUDTF:

            def eval(self, a: int):
                if False:
                    i = 10
                    return i + 15
                yield (a + 1,)
        with self.assertRaisesRegex(SparkConnectGrpcException, 'Invalid Python user-defined table function return type.'):
            TestUDTF(lit(1)).collect()

    @unittest.skip('Spark Connect does not support broadcast but the test depends on it.')
    def test_udtf_with_analyze_using_broadcast(self):
        if False:
            while True:
                i = 10
        super().test_udtf_with_analyze_using_broadcast()

    @unittest.skip('Spark Connect does not support accumulator but the test depends on it.')
    def test_udtf_with_analyze_using_accumulator(self):
        if False:
            return 10
        super().test_udtf_with_analyze_using_accumulator()

    def _add_pyfile(self, path):
        if False:
            i = 10
            return i + 15
        self.spark.addArtifacts(path, pyfile=True)

    def _add_archive(self, path):
        if False:
            i = 10
            return i + 15
        self.spark.addArtifacts(path, archive=True)

    def _add_file(self, path):
        if False:
            return 10
        self.spark.addArtifacts(path, file=True)

class ArrowUDTFParityTests(UDTFArrowTestsMixin, UDTFParityTests):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super(ArrowUDTFParityTests, cls).setUpClass()
        cls.spark.conf.set('spark.sql.execution.pythonUDTF.arrow.enabled', 'true')

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        try:
            cls.spark.conf.unset('spark.sql.execution.pythonUDTF.arrow.enabled')
        finally:
            super(ArrowUDTFParityTests, cls).tearDownClass()
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_udtf import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)