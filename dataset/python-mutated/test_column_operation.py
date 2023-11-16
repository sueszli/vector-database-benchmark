from pyflink.testing.test_case_utils import PyFlinkStreamTableTestCase

class StreamTableColumnsOperationTests(PyFlinkStreamTableTestCase):

    def test_add_columns(self):
        if False:
            i = 10
            return i + 15
        t = self.t_env.from_elements([(1, 'Hi', 'Hello')], ['a', 'b', 'c'])
        result = t.select(t.a).add_columns((t.a + 1).alias('b'), (t.a + 2).alias('c'))
        query_operation = result._j_table.getQueryOperation()
        self.assertEqual('[a, plus(a, 1), plus(a, 2)]', query_operation.getProjectList().toString())

    def test_add_or_replace_columns(self):
        if False:
            while True:
                i = 10
        t = self.t_env.from_elements([(1, 'Hi', 'Hello')], ['a', 'b', 'c'])
        result = t.select(t.a).add_or_replace_columns((t.a + 1).alias('b'), (t.a + 2).alias('a'))
        query_operation = result._j_table.getQueryOperation()
        self.assertEqual('[plus(a, 2), plus(a, 1)]', query_operation.getProjectList().toString())

    def test_rename_columns(self):
        if False:
            return 10
        t = self.t_env.from_elements([(1, 'Hi', 'Hello')], ['a', 'b', 'c'])
        result = t.select(t.a, t.b, t.c).rename_columns(t.a.alias('d'), t.c.alias('f'), t.b.alias('e'))
        resolved_schema = result._j_table.getQueryOperation().getResolvedSchema()
        self.assertEqual(['d', 'e', 'f'], list(resolved_schema.getColumnNames()))

    def test_drop_columns(self):
        if False:
            return 10
        t = self.t_env.from_elements([(1, 'Hi', 'Hello')], ['a', 'b', 'c'])
        result = t.drop_columns(t.a, t.c)
        query_operation = result._j_table.getQueryOperation()
        self.assertEqual('[b]', query_operation.getProjectList().toString())
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)