from pyflink.testing.test_case_utils import PythonAPICompletenessTestCase, PyFlinkTestCase
from pyflink.table import TableConfig

class TableConfigCompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`TableConfig` is consistent with
    Java `org.apache.flink.table.api.TableConfig`.
    """

    @classmethod
    def python_class(cls):
        if False:
            i = 10
            return i + 15
        return TableConfig

    @classmethod
    def java_class(cls):
        if False:
            for i in range(10):
                print('nop')
        return 'org.apache.flink.table.api.TableConfig'

    @classmethod
    def excluded_methods(cls):
        if False:
            print('Hello World!')
        return {'getPlannerConfig', 'setPlannerConfig', 'addJobParameter', 'setRootConfiguration', 'getRootConfiguration', 'getOptional'}

    @classmethod
    def java_method_name(cls, python_method_name):
        if False:
            for i in range(10):
                print('nop')
        return {'get_local_timezone': 'get_local_time_zone', 'set_local_timezone': 'set_local_time_zone'}.get(python_method_name, python_method_name)
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)