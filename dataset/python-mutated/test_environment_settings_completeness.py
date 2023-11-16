from pyflink.table import EnvironmentSettings
from pyflink.testing.test_case_utils import PythonAPICompletenessTestCase, PyFlinkTestCase

class EnvironmentSettingsCompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`EnvironmentSettings` is consistent with
    Java `org.apache.flink.table.api.EnvironmentSettings`.
    """

    @classmethod
    def python_class(cls):
        if False:
            while True:
                i = 10
        return EnvironmentSettings

    @classmethod
    def java_class(cls):
        if False:
            while True:
                i = 10
        return 'org.apache.flink.table.api.EnvironmentSettings'

    @classmethod
    def excluded_methods(cls):
        if False:
            for i in range(10):
                print('nop')
        return {'getPlanner', 'getExecutor', 'getUserClassLoader', 'getCatalogStore'}

class EnvironmentSettingsBuilderCompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`EnvironmentSettings.Builder` is consistent with
    Java `org.apache.flink.table.api.EnvironmentSettings$Builder`.
    """

    @classmethod
    def python_class(cls):
        if False:
            for i in range(10):
                print('nop')
        return EnvironmentSettings.Builder

    @classmethod
    def java_class(cls):
        if False:
            return 10
        return 'org.apache.flink.table.api.EnvironmentSettings$Builder'

    @classmethod
    def excluded_methods(cls):
        if False:
            print('Hello World!')
        return {'withClassLoader', 'withCatalogStore'}
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)