from pyflink.table.table_schema import TableSchema
from pyflink.table.types import DataTypes
from pyflink.testing.test_case_utils import PyFlinkStreamTableTestCase

class StreamTableSchemaTests(PyFlinkStreamTableTestCase):

    def test_print_schema(self):
        if False:
            i = 10
            return i + 15
        t = self.t_env.from_elements([(1, 'Hi', 'Hello')], ['a', 'b', 'c'])
        result = t.group_by(t.c).select(t.a.sum, t.c.alias('b'))
        result.print_schema()

    def test_get_schema(self):
        if False:
            return 10
        t = self.t_env.from_elements([(1, 'Hi', 'Hello')], ['a', 'b', 'c'])
        result = t.group_by(t.c).select(t.a.sum.alias('a'), t.c.alias('b'))
        schema = result.get_schema()
        assert schema == TableSchema(['a', 'b'], [DataTypes.BIGINT(), DataTypes.STRING()])
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)