import unittest
from pyspark.sql.tests.test_arrow import ArrowTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class ArrowParityTests(ArrowTestsMixin, ReusedConnectTestCase, PandasOnSparkTestUtils):

    @unittest.skip('Spark Connect does not support Spark Context but the test depends on that.')
    def test_createDataFrame_empty_partition(self):
        if False:
            i = 10
            return i + 15
        super().test_createDataFrame_empty_partition()

    @unittest.skip('Spark Connect does not support fallback.')
    def test_createDataFrame_fallback_disabled(self):
        if False:
            while True:
                i = 10
        super().test_createDataFrame_fallback_disabled()

    @unittest.skip('Spark Connect does not support fallback.')
    def test_createDataFrame_fallback_enabled(self):
        if False:
            print('Hello World!')
        super().test_createDataFrame_fallback_enabled()

    def test_createDataFrame_with_incorrect_schema(self):
        if False:
            while True:
                i = 10
        self.check_createDataFrame_with_incorrect_schema()

    def test_createDataFrame_with_map_type(self):
        if False:
            while True:
                i = 10
        self.check_createDataFrame_with_map_type(True)

    def test_createDataFrame_with_ndarray(self):
        if False:
            i = 10
            return i + 15
        self.check_createDataFrame_with_ndarray(True)

    def test_createDataFrame_with_single_data_type(self):
        if False:
            i = 10
            return i + 15
        self.check_createDataFrame_with_single_data_type()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_no_partition_frame(self):
        if False:
            while True:
                i = 10
        super().test_no_partition_frame()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_no_partition_toPandas(self):
        if False:
            return 10
        super().test_no_partition_toPandas()

    def test_pandas_self_destruct(self):
        if False:
            i = 10
            return i + 15
        df = self.spark.range(100).select('id', 'id', 'id')
        with self.sql_conf({'spark.sql.execution.arrow.pyspark.selfDestruct.enabled': True}):
            self_destruct_pdf = df.toPandas()
        with self.sql_conf({'spark.sql.execution.arrow.pyspark.selfDestruct.enabled': False}):
            no_self_destruct_pdf = df.toPandas()
        self.assert_eq(self_destruct_pdf, no_self_destruct_pdf)

    def test_propagates_spark_exception(self):
        if False:
            return 10
        self.check_propagates_spark_exception()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_toPandas_batch_order(self):
        if False:
            print('Hello World!')
        super().test_toPandas_batch_order()

    def test_toPandas_empty_df_arrow_enabled(self):
        if False:
            while True:
                i = 10
        self.check_toPandas_empty_df_arrow_enabled(True)

    def test_create_data_frame_to_pandas_timestamp_ntz(self):
        if False:
            while True:
                i = 10
        self.check_create_data_frame_to_pandas_timestamp_ntz(True)

    def test_create_data_frame_to_pandas_day_time_internal(self):
        if False:
            while True:
                i = 10
        self.check_create_data_frame_to_pandas_day_time_internal(True)

    def test_toPandas_respect_session_timezone(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_toPandas_respect_session_timezone(True)

    def test_toPandas_with_array_type(self):
        if False:
            print('Hello World!')
        self.check_toPandas_with_array_type(True)

    @unittest.skip('Spark Connect does not support fallback.')
    def test_toPandas_fallback_disabled(self):
        if False:
            i = 10
            return i + 15
        super().test_toPandas_fallback_disabled()

    @unittest.skip('Spark Connect does not support fallback.')
    def test_toPandas_fallback_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_toPandas_fallback_enabled()

    def test_toPandas_with_map_type(self):
        if False:
            print('Hello World!')
        self.check_toPandas_with_map_type(True)

    def test_toPandas_with_map_type_nulls(self):
        if False:
            i = 10
            return i + 15
        self.check_toPandas_with_map_type_nulls(True)

    def test_createDataFrame_with_array_type(self):
        if False:
            return 10
        self.check_createDataFrame_with_array_type(True)

    def test_createDataFrame_with_int_col_names(self):
        if False:
            return 10
        self.check_createDataFrame_with_int_col_names(True)

    def test_timestamp_nat(self):
        if False:
            print('Hello World!')
        self.check_timestamp_nat(True)

    def test_toPandas_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_toPandas_error(True)

    def test_toPandas_duplicate_field_names(self):
        if False:
            while True:
                i = 10
        self.check_toPandas_duplicate_field_names(True)

    def test_createDataFrame_duplicate_field_names(self):
        if False:
            print('Hello World!')
        self.check_createDataFrame_duplicate_field_names(True)

    def test_toPandas_empty_columns(self):
        if False:
            i = 10
            return i + 15
        self.check_toPandas_empty_columns(True)

    def test_createDataFrame_nested_timestamp(self):
        if False:
            while True:
                i = 10
        self.check_createDataFrame_nested_timestamp(True)

    def test_toPandas_nested_timestamp(self):
        if False:
            i = 10
            return i + 15
        self.check_toPandas_nested_timestamp(True)

    def test_createDataFrame_udt(self):
        if False:
            print('Hello World!')
        self.check_createDataFrame_udt(True)

    def test_toPandas_udt(self):
        if False:
            return 10
        self.check_toPandas_udt(True)

    def test_create_dataframe_namedtuples(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_create_dataframe_namedtuples(True)
if __name__ == '__main__':
    from pyspark.sql.tests.connect.test_parity_arrow import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)