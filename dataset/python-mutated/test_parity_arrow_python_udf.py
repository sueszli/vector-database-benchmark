import unittest
from pyspark.errors import AnalysisException, PythonException
from pyspark.sql.functions import udf
from pyspark.sql.tests.connect.test_parity_udf import UDFParityTests
from pyspark.sql.tests.test_arrow_python_udf import PythonUDFArrowTestsMixin

class ArrowPythonUDFParityTests(UDFParityTests, PythonUDFArrowTestsMixin):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super(ArrowPythonUDFParityTests, cls).setUpClass()
        cls.spark.conf.set('spark.sql.execution.pythonUDF.arrow.enabled', 'true')

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        try:
            cls.spark.conf.unset('spark.sql.execution.pythonUDF.arrow.enabled')
        finally:
            super(ArrowPythonUDFParityTests, cls).tearDownClass()

    def test_named_arguments_negative(self):
        if False:
            for i in range(10):
                print('nop')

        @udf('int')
        def test_udf(a, b):
            if False:
                while True:
                    i = 10
            return a + b
        self.spark.udf.register('test_udf', test_udf)
        with self.assertRaisesRegex(AnalysisException, 'DUPLICATE_ROUTINE_PARAMETER_ASSIGNMENT.DOUBLE_NAMED_ARGUMENT_REFERENCE'):
            self.spark.sql('SELECT test_udf(a => id, a => id * 10) FROM range(2)').show()
        with self.assertRaisesRegex(AnalysisException, 'UNEXPECTED_POSITIONAL_ARGUMENT'):
            self.spark.sql('SELECT test_udf(a => id, id * 10) FROM range(2)').show()
        with self.assertRaises(PythonException):
            self.spark.sql("SELECT test_udf(c => 'x') FROM range(2)").show()
        with self.assertRaises(PythonException):
            self.spark.sql('SELECT test_udf(id, a => id * 10) FROM range(2)').show()
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_arrow_python_udf import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)