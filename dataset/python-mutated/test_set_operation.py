from pyflink.table import TableEnvironment, EnvironmentSettings
from pyflink.testing.test_case_utils import PyFlinkTestCase

class StreamTableSetOperationTests(PyFlinkTestCase):
    data1 = [(1, 'Hi', 'Hello')]
    data2 = [(3, 'Hello', 'Hello')]
    schema = ['a', 'b', 'c']

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.t_env = TableEnvironment.create(EnvironmentSettings.in_batch_mode())

    def test_minus(self):
        if False:
            return 10
        t_env = self.t_env
        t1 = t_env.from_elements(self.data1, self.schema)
        t2 = t_env.from_elements(self.data2, self.schema)
        result = t1.minus(t2)
        self.assertEqual('MINUS', result._j_table.getQueryOperation().getType().toString())
        self.assertFalse(result._j_table.getQueryOperation().isAll())

    def test_minus_all(self):
        if False:
            print('Hello World!')
        t_env = self.t_env
        t1 = t_env.from_elements(self.data1, self.schema)
        t2 = t_env.from_elements(self.data2, self.schema)
        result = t1.minus_all(t2)
        self.assertEqual('MINUS', result._j_table.getQueryOperation().getType().toString())
        self.assertTrue(result._j_table.getQueryOperation().isAll())

    def test_union(self):
        if False:
            i = 10
            return i + 15
        t_env = self.t_env
        t1 = t_env.from_elements(self.data1, self.schema)
        t2 = t_env.from_elements(self.data2, self.schema)
        result = t1.union(t2)
        self.assertEqual('UNION', result._j_table.getQueryOperation().getType().toString())
        self.assertFalse(result._j_table.getQueryOperation().isAll())

    def test_union_all(self):
        if False:
            for i in range(10):
                print('nop')
        t_env = self.t_env
        t1 = t_env.from_elements(self.data1, self.schema)
        t2 = t_env.from_elements(self.data2, self.schema)
        result = t1.union_all(t2)
        self.assertEqual('UNION', result._j_table.getQueryOperation().getType().toString())
        self.assertTrue(result._j_table.getQueryOperation().isAll())

    def test_intersect(self):
        if False:
            return 10
        t_env = self.t_env
        t1 = t_env.from_elements(self.data1, self.schema)
        t2 = t_env.from_elements(self.data2, self.schema)
        result = t1.intersect(t2)
        self.assertEqual('INTERSECT', result._j_table.getQueryOperation().getType().toString())
        self.assertFalse(result._j_table.getQueryOperation().isAll())

    def test_intersect_all(self):
        if False:
            while True:
                i = 10
        t_env = self.t_env
        t1 = t_env.from_elements(self.data1, self.schema)
        t2 = t_env.from_elements(self.data2, self.schema)
        result = t1.intersect_all(t2)
        self.assertEqual('INTERSECT', result._j_table.getQueryOperation().getType().toString())
        self.assertTrue(result._j_table.getQueryOperation().isAll())
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)