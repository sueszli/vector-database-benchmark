from pyflink.testing.test_case_utils import PyFlinkStreamTableTestCase

class StreamTableDistinctTests(PyFlinkStreamTableTestCase):

    def test_distinct(self):
        if False:
            i = 10
            return i + 15
        t = self.t_env.from_elements([(1, 'Hi', 'Hello')], ['a', 'b', 'c'])
        result = t.distinct()
        query_operation = result._j_table.getQueryOperation()
        self.assertEqual('DistinctQueryOperation', query_operation.getClass().getSimpleName())
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)