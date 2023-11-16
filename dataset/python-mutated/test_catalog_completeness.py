from pyflink.testing.test_case_utils import PythonAPICompletenessTestCase, PyFlinkTestCase
from pyflink.table.catalog import Catalog, CatalogDatabase, CatalogBaseTable, CatalogPartition, CatalogFunction, CatalogColumnStatistics, CatalogPartitionSpec, ObjectPath

class CatalogAPICompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`Catalog` is consistent with
    Java `org.apache.flink.table.catalog.Catalog`.
    """

    @classmethod
    def python_class(cls):
        if False:
            for i in range(10):
                print('nop')
        return Catalog

    @classmethod
    def java_class(cls):
        if False:
            while True:
                i = 10
        return 'org.apache.flink.table.catalog.Catalog'

    @classmethod
    def excluded_methods(cls):
        if False:
            for i in range(10):
                print('nop')
        return {'open', 'close', 'getFactory', 'getTableFactory', 'getFunctionDefinitionFactory', 'listPartitionsByFilter', 'supportsManagedTable'}

class CatalogDatabaseAPICompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`CatalogDatabase` is consistent with
    Java `org.apache.flink.table.catalog.CatalogDatabase`.
    """

    @classmethod
    def python_class(cls):
        if False:
            for i in range(10):
                print('nop')
        return CatalogDatabase

    @classmethod
    def java_class(cls):
        if False:
            return 10
        return 'org.apache.flink.table.catalog.CatalogDatabase'

class CatalogBaseTableAPICompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`CatalogBaseTable` is consistent with
    Java `org.apache.flink.table.catalog.CatalogBaseTable`.
    """

    @classmethod
    def python_class(cls):
        if False:
            for i in range(10):
                print('nop')
        return CatalogBaseTable

    @classmethod
    def java_class(cls):
        if False:
            i = 10
            return i + 15
        return 'org.apache.flink.table.catalog.CatalogBaseTable'

    @classmethod
    def excluded_methods(cls):
        if False:
            i = 10
            return i + 15
        return {'getUnresolvedSchema', 'getTableKind'}

class CatalogFunctionAPICompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`CatalogFunction` is consistent with
    Java `org.apache.flink.table.catalog.CatalogFunction`.
    """

    @classmethod
    def python_class(cls):
        if False:
            i = 10
            return i + 15
        return CatalogFunction

    @classmethod
    def java_class(cls):
        if False:
            i = 10
            return i + 15
        return 'org.apache.flink.table.catalog.CatalogFunction'

    @classmethod
    def excluded_methods(cls):
        if False:
            for i in range(10):
                print('nop')
        return {'getFunctionResources'}

class CatalogPartitionAPICompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`CatalogPartition` is consistent with
    Java `org.apache.flink.table.catalog.CatalogPartition`.
    """

    @classmethod
    def python_class(cls):
        if False:
            i = 10
            return i + 15
        return CatalogPartition

    @classmethod
    def java_class(cls):
        if False:
            print('Hello World!')
        return 'org.apache.flink.table.catalog.CatalogPartition'

class ObjectPathAPICompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`ObjectPath` is consistent with
    Java `org.apache.flink.table.catalog.ObjectPath`.
    """

    @classmethod
    def python_class(cls):
        if False:
            i = 10
            return i + 15
        return ObjectPath

    @classmethod
    def java_class(cls):
        if False:
            print('Hello World!')
        return 'org.apache.flink.table.catalog.ObjectPath'

class CatalogPartitionSpecAPICompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`CatalogPartitionSpec` is consistent with
    Java `org.apache.flink.table.catalog.CatalogPartitionSpec`.
    """

    @classmethod
    def python_class(cls):
        if False:
            return 10
        return CatalogPartitionSpec

    @classmethod
    def java_class(cls):
        if False:
            print('Hello World!')
        return 'org.apache.flink.table.catalog.CatalogPartitionSpec'

class CatalogColumnStatisticsAPICompletenessTests(PythonAPICompletenessTestCase, PyFlinkTestCase):
    """
    Tests whether the Python :class:`CatalogColumnStatistics` is consistent with
    Java `org.apache.flink.table.catalog.CatalogColumnStatistics`.
    """

    @classmethod
    def python_class(cls):
        if False:
            print('Hello World!')
        return CatalogColumnStatistics

    @classmethod
    def java_class(cls):
        if False:
            i = 10
            return i + 15
        return 'org.apache.flink.table.catalog.stats.CatalogColumnStatistics'
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)