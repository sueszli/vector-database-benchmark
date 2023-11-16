from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.tests.pandas.test_pandas_udf import PandasUDFTestsMixin
from pyspark.testing.connectutils import should_test_connect, ReusedConnectTestCase
if should_test_connect:
    from pyspark.sql.connect.types import UnparsedDataType

class PandasUDFParityTests(PandasUDFTestsMixin, ReusedConnectTestCase):

    def test_udf_wrong_arg(self):
        if False:
            i = 10
            return i + 15
        self.check_udf_wrong_arg()

    def test_pandas_udf_decorator_with_return_type_string(self):
        if False:
            i = 10
            return i + 15

        @pandas_udf('v double', PandasUDFType.GROUPED_MAP)
        def foo(x):
            if False:
                while True:
                    i = 10
            return x
        self.assertEqual(foo.returnType, UnparsedDataType('v double'))
        self.assertEqual(foo.evalType, PandasUDFType.GROUPED_MAP)

        @pandas_udf(returnType='double', functionType=PandasUDFType.SCALAR)
        def foo(x):
            if False:
                while True:
                    i = 10
            return x
        self.assertEqual(foo.returnType, UnparsedDataType('double'))
        self.assertEqual(foo.evalType, PandasUDFType.SCALAR)

    def test_pandas_udf_basic_with_return_type_string(self):
        if False:
            return 10
        udf = pandas_udf(lambda x: x, 'double', PandasUDFType.SCALAR)
        self.assertEqual(udf.returnType, UnparsedDataType('double'))
        self.assertEqual(udf.evalType, PandasUDFType.SCALAR)
        udf = pandas_udf(lambda x: x, 'v double', PandasUDFType.GROUPED_MAP)
        self.assertEqual(udf.returnType, UnparsedDataType('v double'))
        self.assertEqual(udf.evalType, PandasUDFType.GROUPED_MAP)
        udf = pandas_udf(lambda x: x, 'v double', functionType=PandasUDFType.GROUPED_MAP)
        self.assertEqual(udf.returnType, UnparsedDataType('v double'))
        self.assertEqual(udf.evalType, PandasUDFType.GROUPED_MAP)
        udf = pandas_udf(lambda x: x, returnType='v double', functionType=PandasUDFType.GROUPED_MAP)
        self.assertEqual(udf.returnType, UnparsedDataType('v double'))
        self.assertEqual(udf.evalType, PandasUDFType.GROUPED_MAP)
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_pandas_udf import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)