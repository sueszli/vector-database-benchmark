import unittest
from pyflink.table import DataTypes
from pyflink.table.udf import TableFunction, udtf, ScalarFunction, udf
from pyflink.table.expressions import col
from pyflink.testing import source_sink_utils
from pyflink.testing.test_case_utils import PyFlinkStreamTableTestCase, PyFlinkBatchTableTestCase

class UserDefinedTableFunctionTests(object):

    def test_table_function(self):
        if False:
            while True:
                i = 10
        self.t_env.execute_sql("\n            CREATE TABLE Results_test_table_function(\n                a BIGINT,\n                b BIGINT,\n                c BIGINT\n            ) WITH ('connector'='test-sink')")
        multi_emit = udtf(MultiEmit(), result_types=[DataTypes.BIGINT(), DataTypes.BIGINT()])
        multi_num = udf(MultiNum(), result_type=DataTypes.BIGINT())
        t = self.t_env.from_elements([(1, 1, 3), (2, 1, 6), (3, 2, 9)], ['a', 'b', 'c'])
        t = t.join_lateral(multi_emit((t.a + t.a) / 2, multi_num(t.b)).alias('x', 'y'))
        t = t.left_outer_join_lateral(condition_multi_emit(t.x, t.y).alias('m')).select(t.x, t.y, col('m'))
        t = t.left_outer_join_lateral(identity(t.m).alias('n')).select(t.x, t.y, col('n'))
        t.execute_insert('Results_test_table_function').wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 0, null]', '+I[1, 1, null]', '+I[2, 0, null]', '+I[2, 1, null]', '+I[3, 0, 0]', '+I[3, 0, 1]', '+I[3, 0, 2]', '+I[3, 1, 1]', '+I[3, 1, 2]', '+I[3, 2, 2]', '+I[3, 3, null]'])

    def test_table_function_with_sql_query(self):
        if False:
            i = 10
            return i + 15
        self.t_env.execute_sql("\n            CREATE TABLE Results_test_table_function_with_sql_query(\n                a BIGINT,\n                b BIGINT,\n                c BIGINT\n            ) WITH ('connector'='test-sink')")
        self.t_env.create_temporary_system_function('multi_emit', udtf(MultiEmit(), result_types=[DataTypes.BIGINT(), DataTypes.BIGINT()]))
        t = self.t_env.from_elements([(1, 1, 3), (2, 1, 6), (3, 2, 9)], ['a', 'b', 'c'])
        self.t_env.create_temporary_view('MyTable', t)
        t = self.t_env.sql_query('SELECT a, x, y FROM MyTable LEFT JOIN LATERAL TABLE(multi_emit(a, b)) as T(x, y) ON TRUE')
        t.execute_insert('Results_test_table_function_with_sql_query').wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 1, 0]', '+I[2, 2, 0]', '+I[3, 3, 0]', '+I[3, 3, 1]'])

class PyFlinkStreamUserDefinedFunctionTests(UserDefinedTableFunctionTests, PyFlinkStreamTableTestCase):

    def test_execute_from_json_plan(self):
        if False:
            i = 10
            return i + 15
        tmp_dir = self.tempdir
        data = ['1,1', '3,2', '2,1']
        source_path = tmp_dir + '/test_execute_from_json_plan_input.csv'
        sink_path = tmp_dir + '/test_execute_from_json_plan_out'
        with open(source_path, 'w') as fd:
            for ele in data:
                fd.write(ele + '\n')
        source_table = "\n            CREATE TABLE source_table (\n                a BIGINT,\n                b BIGINT\n            ) WITH (\n                'connector' = 'filesystem',\n                'path' = '%s',\n                'format' = 'csv'\n            )\n        " % source_path
        self.t_env.execute_sql(source_table)
        self.t_env.execute_sql("\n            CREATE TABLE sink_table (\n                a BIGINT,\n                b BIGINT,\n                c BIGINT\n            ) WITH (\n                'connector' = 'filesystem',\n                'path' = '%s',\n                'format' = 'csv'\n            )\n        " % sink_path)
        self.t_env.create_temporary_system_function('multi_emit2', udtf(MultiEmit(), result_types=[DataTypes.BIGINT(), DataTypes.BIGINT()]))
        json_plan = self.t_env._j_tenv.compilePlanSql('INSERT INTO sink_table SELECT a, x, y FROM source_table LEFT JOIN LATERAL TABLE(multi_emit2(a, b)) as T(x, y) ON TRUE')
        from py4j.java_gateway import get_method
        get_method(json_plan.execute(), 'await')()
        import glob
        lines = [line.strip() for file in glob.glob(sink_path + '/*') for line in open(file, 'r')]
        lines.sort()
        self.assertEqual(lines, ['1,1,0', '2,2,0', '3,3,0', '3,3,1'])

class PyFlinkBatchUserDefinedFunctionTests(UserDefinedTableFunctionTests, PyFlinkBatchTableTestCase):
    pass

class PyFlinkEmbeddedThreadTests(UserDefinedTableFunctionTests, PyFlinkStreamTableTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(PyFlinkEmbeddedThreadTests, self).setUp()
        self.t_env.get_config().set('python.execution-mode', 'thread')

class MultiEmit(TableFunction, unittest.TestCase):

    def open(self, function_context):
        if False:
            for i in range(10):
                print('nop')
        self.counter_sum = 0

    def eval(self, x, y):
        if False:
            return 10
        self.counter_sum += y
        for i in range(y):
            yield (x, i)

@udtf(result_types=['bigint'])
def identity(x):
    if False:
        i = 10
        return i + 15
    if x is not None:
        from pyflink.common import Row
        return Row(x)

@udtf(input_types=[DataTypes.BIGINT(), DataTypes.BIGINT()], result_types=DataTypes.BIGINT())
def condition_multi_emit(x, y):
    if False:
        for i in range(10):
            print('nop')
    if x == 3:
        return range(y, x)

class MultiNum(ScalarFunction):

    def eval(self, x):
        if False:
            i = 10
            return i + 15
        return x * 2
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)