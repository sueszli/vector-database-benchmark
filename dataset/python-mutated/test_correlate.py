from pyflink.table import expressions as expr
from pyflink.testing.test_case_utils import PyFlinkUTTestCase

class CorrelateTests(PyFlinkUTTestCase):

    def test_join_lateral(self):
        if False:
            return 10
        t_env = self.t_env
        t_env.create_java_temporary_system_function('split', 'org.apache.flink.table.utils.TestingFunctions$TableFunc1')
        source = t_env.from_elements([('1', '1#3#5#7'), ('2', '2#4#6#8')], ['id', 'words'])
        result = source.join_lateral(expr.call('split', source.words).alias('word'))
        query_operation = result._j_table.getQueryOperation()
        self.assertEqual('INNER', query_operation.getJoinType().toString())
        self.assertTrue(query_operation.isCorrelated())
        self.assertEqual('true', query_operation.getCondition().toString())

    def test_join_lateral_with_join_predicate(self):
        if False:
            while True:
                i = 10
        t_env = self.t_env
        t_env.create_java_temporary_system_function('split', 'org.apache.flink.table.utils.TestingFunctions$TableFunc1')
        source = t_env.from_elements([('1', '1#3#5#7'), ('2', '2#4#6#8')], ['id', 'words'])
        result = source.join_lateral(expr.call('split', source.words).alias('word'), expr.col('id') == expr.col('word'))
        query_operation = result._j_table.getQueryOperation()
        self.assertEqual('INNER', query_operation.getJoinType().toString())
        self.assertTrue(query_operation.isCorrelated())
        self.assertEqual('equals(id, word)', query_operation.getCondition().toString())

    def test_left_outer_join_lateral(self):
        if False:
            i = 10
            return i + 15
        t_env = self.t_env
        t_env.create_java_temporary_system_function('split', 'org.apache.flink.table.utils.TestingFunctions$TableFunc1')
        source = t_env.from_elements([('1', '1#3#5#7'), ('2', '2#4#6#8')], ['id', 'words'])
        result = source.left_outer_join_lateral(expr.call('split', source.words).alias('word'))
        query_operation = result._j_table.getQueryOperation()
        self.assertEqual('LEFT_OUTER', query_operation.getJoinType().toString())
        self.assertTrue(query_operation.isCorrelated())
        self.assertEqual('true', query_operation.getCondition().toString())

    def test_left_outer_join_lateral_with_join_predicate(self):
        if False:
            print('Hello World!')
        t_env = self.t_env
        t_env.create_java_temporary_system_function('split', 'org.apache.flink.table.utils.TestingFunctions$TableFunc1')
        source = t_env.from_elements([('1', '1#3#5#7'), ('2', '2#4#6#8')], ['id', 'words'])
        result = source.left_outer_join_lateral(expr.call('split', source.words).alias('word'), expr.lit(True))
        query_operation = result._j_table.getQueryOperation()
        self.assertEqual('LEFT_OUTER', query_operation.getJoinType().toString())
        self.assertTrue(query_operation.isCorrelated())
        self.assertEqual('true', query_operation.getCondition().toString())