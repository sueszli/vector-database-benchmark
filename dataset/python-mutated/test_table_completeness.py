from pyflink.testing.test_case_utils import PythonAPICompletenessTestCase, PyFlinkTestCase
from pyflink.table import Table

class TableAPICompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`Table` is consistent with
    Java `org.apache.flink.table.api.Table`.
    """

    @classmethod
    def python_class(cls):
        if False:
            while True:
                i = 10
        return Table

    @classmethod
    def java_class(cls):
        if False:
            i = 10
            return i + 15
        return 'org.apache.flink.table.api.Table'

    @classmethod
    def excluded_methods(cls):
        if False:
            print('Hello World!')
        return {'createTemporalTableFunction', 'getQueryOperation', 'getResolvedSchema', 'insertInto', 'printExplain'}

    @classmethod
    def java_method_name(cls, python_method_name):
        if False:
            while True:
                i = 10
        "\n        Due to 'as' is python keyword, so we use 'alias'\n        in Python API corresponding 'as' in Java API.\n\n        :param python_method_name:\n        :return:\n        "
        return {'alias': 'as'}.get(python_method_name, python_method_name)
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)