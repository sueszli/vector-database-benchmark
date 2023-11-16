from pyflink.testing.test_case_utils import PythonAPICompletenessTestCase, PyFlinkTestCase
from pyflink.table import expressions

class ExpressionsCompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :module:`pyflink.table.expressions` is consistent with
    Java `org.apache.flink.table.api.Expressions`.
    """

    @classmethod
    def python_class(cls):
        if False:
            print('Hello World!')
        return expressions

    @classmethod
    def java_class(cls):
        if False:
            return 10
        return 'org.apache.flink.table.api.Expressions'

    @classmethod
    def java_method_name(cls, python_method_name):
        if False:
            i = 10
            return i + 15
        return {'and_': 'and', 'or_': 'or', 'not_': 'not', 'range_': 'range', 'map_': 'map'}.get(python_method_name, python_method_name)

    @classmethod
    def excluded_methods(cls):
        if False:
            print('Hello World!')
        return {'$'}
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)