from pyflink.testing.test_case_utils import PythonAPICompletenessTestCase, PyFlinkTestCase
from pyflink.table import TableEnvironment

class EnvironmentAPICompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`TableEnvironment` is consistent with
    Java `org.apache.flink.table.api.TableEnvironment`.
    """

    @classmethod
    def python_class(cls):
        if False:
            return 10
        return TableEnvironment

    @classmethod
    def java_class(cls):
        if False:
            while True:
                i = 10
        return 'org.apache.flink.table.api.TableEnvironment'

    @classmethod
    def excluded_methods(cls):
        if False:
            i = 10
            return i + 15
        return {'getCompletionHints', 'fromValues', 'loadPlan', 'compilePlanSql', 'executePlan', 'explainPlan', 'createCatalog'}

    @classmethod
    def java_method_name(cls, python_method_name):
        if False:
            print('Hello World!')
        "\n        Due to 'from' is python keyword, so we use 'from_path'\n        in Python API corresponding 'from' in Java API.\n\n        :param python_method_name:\n        :return:\n        "
        py_func_to_java_method_dict = {'from_path': 'from', 'from_descriptor': 'from', 'create_java_function': 'create_function'}
        return py_func_to_java_method_dict.get(python_method_name, python_method_name)
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)