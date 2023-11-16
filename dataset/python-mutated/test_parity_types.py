import unittest
from pyspark.sql.tests.test_types import TypesTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class TypesParityTests(TypesTestsMixin, ReusedConnectTestCase):

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_apply_schema(self):
        if False:
            return 10
        super().test_apply_schema()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_apply_schema_to_dict_and_rows(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_apply_schema_to_dict_and_rows()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_apply_schema_to_row(self):
        if False:
            while True:
                i = 10
        super().test_apply_schema_to_dict_and_rows()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_create_dataframe_schema_mismatch(self):
        if False:
            while True:
                i = 10
        super().test_create_dataframe_schema_mismatch()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_infer_array_element_type_empty(self):
        if False:
            i = 10
            return i + 15
        super().test_infer_array_element_type_empty()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_infer_array_element_type_with_struct(self):
        if False:
            while True:
                i = 10
        super().test_infer_array_element_type_with_struct()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_infer_array_merge_element_types_with_rdd(self):
        if False:
            while True:
                i = 10
        super().test_infer_array_merge_element_types_with_rdd()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_infer_binary_type(self):
        if False:
            while True:
                i = 10
        super().test_infer_binary_type()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_infer_long_type(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_infer_long_type()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_infer_nested_dict_as_struct_with_rdd(self):
        if False:
            return 10
        super().test_infer_nested_dict_as_struct_with_rdd()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_infer_nested_schema(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_infer_nested_schema()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_infer_schema(self):
        if False:
            return 10
        super().test_infer_schema()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_infer_schema_to_local(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_infer_schema_to_local()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_infer_schema_upcast_int_to_string(self):
        if False:
            return 10
        super().test_infer_schema_upcast_int_to_string()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_rdd_with_udt(self):
        if False:
            i = 10
            return i + 15
        super().test_rdd_with_udt()

    @unittest.skip('Requires JVM access.')
    def test_udt(self):
        if False:
            print('Hello World!')
        super().test_udt()
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_types import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)