from pyflink.testing.test_case_utils import PythonAPICompletenessTestCase, PyFlinkTestCase
from pyflink.table import Expression

class ExpressionCompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`Expression` is consistent with
    Java `org.apache.flink.table.api.ApiExpression`.
    """

    @classmethod
    def python_class(cls):
        if False:
            while True:
                i = 10
        return Expression

    @classmethod
    def java_class(cls):
        if False:
            while True:
                i = 10
        return 'org.apache.flink.table.api.ApiExpression'

    @classmethod
    def excluded_methods(cls):
        if False:
            while True:
                i = 10
        return {'asSummaryString', 'accept', 'toExpr', 'getChildren', 'and', 'or', 'not', 'isGreater', 'isGreaterOrEqual', 'isLess', 'isLessOrEqual', 'isEqual', 'isNotEqual', 'plus', 'minus', 'dividedBy', 'times', 'mod', 'power'}

    @classmethod
    def java_method_name(cls, python_method_name):
        if False:
            print('Hello World!')
        return {'alias': 'as', 'in_': 'in'}.get(python_method_name, python_method_name)
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)