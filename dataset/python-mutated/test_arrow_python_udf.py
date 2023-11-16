import unittest
from pyspark.errors import PythonException
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.tests.test_udf import BaseUDFTestsMixin
from pyspark.testing.sqlutils import have_pandas, have_pyarrow, pandas_requirement_message, pyarrow_requirement_message, ReusedSQLTestCase
from pyspark.rdd import PythonEvalType

@unittest.skipIf(not have_pandas or not have_pyarrow, pandas_requirement_message or pyarrow_requirement_message)
class PythonUDFArrowTestsMixin(BaseUDFTestsMixin):

    @unittest.skip('Unrelated test, and it fails when it runs duplicatedly.')
    def test_broadcast_in_udf(self):
        if False:
            print('Hello World!')
        super(PythonUDFArrowTests, self).test_broadcast_in_udf()

    @unittest.skip('Unrelated test, and it fails when it runs duplicatedly.')
    def test_register_java_function(self):
        if False:
            i = 10
            return i + 15
        super(PythonUDFArrowTests, self).test_register_java_function()

    @unittest.skip('Unrelated test, and it fails when it runs duplicatedly.')
    def test_register_java_udaf(self):
        if False:
            while True:
                i = 10
        super(PythonUDFArrowTests, self).test_register_java_udaf()

    def test_complex_input_types(self):
        if False:
            return 10
        row = self.spark.range(1).selectExpr('array(1, 2, 3) as array', "map('a', 'b') as map", 'struct(1, 2) as struct').select(udf(lambda x: str(x))('array'), udf(lambda x: str(x))('map'), udf(lambda x: str(x))('struct')).first()
        self.assertEquals(row[0], '[1, 2, 3]')
        self.assertEquals(row[1], "{'a': 'b'}")
        self.assertEquals(row[2], 'Row(col1=1, col2=2)')

    def test_use_arrow(self):
        if False:
            i = 10
            return i + 15
        row_true = self.spark.range(1).selectExpr('array(1, 2, 3) as array').select(udf(lambda x: str(x), useArrow=True)('array')).first()
        row_none = self.spark.range(1).selectExpr('array(1, 2, 3) as array').select(udf(lambda x: str(x), useArrow=None)('array')).first()
        self.assertEquals(row_true[0], row_none[0])
        row_false = self.spark.range(1).selectExpr('array(1, 2, 3) as array').select(udf(lambda x: str(x), useArrow=False)('array')).first()
        self.assertEquals(row_false[0], '[1, 2, 3]')

    def test_eval_type(self):
        if False:
            return 10
        self.assertEquals(udf(lambda x: str(x), useArrow=True).evalType, PythonEvalType.SQL_ARROW_BATCHED_UDF)
        self.assertEquals(udf(lambda x: str(x), useArrow=False).evalType, PythonEvalType.SQL_BATCHED_UDF)

    def test_register(self):
        if False:
            return 10
        df = self.spark.range(1).selectExpr('array(1, 2, 3) as array')
        str_repr_func = self.spark.udf.register('str_repr', udf(lambda x: str(x), useArrow=True))
        self.assertEquals(df.selectExpr('str_repr(array) AS str_id').first()[0], '[1, 2, 3]')
        self.assertListEqual(df.selectExpr('str_repr(array) AS str_id').collect(), df.select(str_repr_func('array').alias('str_id')).collect())

    def test_nested_array_input(self):
        if False:
            i = 10
            return i + 15
        df = self.spark.range(1).selectExpr('array(array(1, 2), array(3, 4)) as nested_array')
        self.assertEquals(df.select(udf(lambda x: str(x), returnType='string', useArrow=True)('nested_array')).first()[0], '[[1, 2], [3, 4]]')

    def test_type_coercion_string_to_numeric(self):
        if False:
            i = 10
            return i + 15
        df_int_value = self.spark.createDataFrame(['1', '2'], schema='string')
        df_floating_value = self.spark.createDataFrame(['1.1', '2.2'], schema='string')
        int_ddl_types = ['tinyint', 'smallint', 'int', 'bigint']
        floating_ddl_types = ['double', 'float']
        for ddl_type in int_ddl_types:
            res = df_int_value.select(udf(lambda x: x, ddl_type)('value').alias('res'))
            self.assertEquals(res.collect(), [Row(res=1), Row(res=2)])
            self.assertEquals(res.dtypes[0][1], ddl_type)
        floating_results = [[Row(res=1.1), Row(res=2.2)], [Row(res=1.100000023841858), Row(res=2.200000047683716)]]
        for (ddl_type, floating_res) in zip(floating_ddl_types, floating_results):
            res = df_int_value.select(udf(lambda x: x, ddl_type)('value').alias('res'))
            self.assertEquals(res.collect(), [Row(res=1.0), Row(res=2.0)])
            self.assertEquals(res.dtypes[0][1], ddl_type)
            res = df_floating_value.select(udf(lambda x: x, ddl_type)('value').alias('res'))
            self.assertEquals(res.collect(), floating_res)
            self.assertEquals(res.dtypes[0][1], ddl_type)
        with self.assertRaises(PythonException):
            df_floating_value.select(udf(lambda x: x, 'int')('value').alias('res')).collect()
        with self.assertRaises(PythonException):
            df_int_value.select(udf(lambda x: x, 'decimal')('value').alias('res')).collect()
        with self.assertRaises(PythonException):
            df_floating_value.select(udf(lambda x: x, 'decimal')('value').alias('res')).collect()

class PythonUDFArrowTests(PythonUDFArrowTestsMixin, ReusedSQLTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super(PythonUDFArrowTests, cls).setUpClass()
        cls.spark.conf.set('spark.sql.execution.pythonUDF.arrow.enabled', 'true')

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        try:
            cls.spark.conf.unset('spark.sql.execution.pythonUDF.arrow.enabled')
        finally:
            super(PythonUDFArrowTests, cls).tearDownClass()
if __name__ == '__main__':
    from pyspark.sql.tests.test_arrow_python_udf import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)