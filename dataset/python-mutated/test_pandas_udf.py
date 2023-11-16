import datetime
import decimal
import pytz
from pyflink.common import Row
from pyflink.table import DataTypes
from pyflink.table.tests.test_udf import SubtractOne
from pyflink.table.udf import udf
from pyflink.testing import source_sink_utils
from pyflink.testing.test_case_utils import PyFlinkBatchTableTestCase, PyFlinkStreamTableTestCase, PyFlinkTestCase

class PandasUDFTests(PyFlinkTestCase):

    def test_non_exist_func_type(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, "The func_type must be one of 'general, pandas'"):
            udf(lambda i: i + 1, result_type=DataTypes.BIGINT(), func_type='non-exist')

class PandasUDFITTests(object):

    def test_basic_functionality(self):
        if False:
            i = 10
            return i + 15
        add_one = udf(lambda i: i + 1, result_type=DataTypes.BIGINT(), func_type='pandas')
        subtract_one = udf(SubtractOne(), DataTypes.BIGINT(), DataTypes.BIGINT())
        sink_table_ddl = "\n        CREATE TABLE Results_test_basic_functionality(\n            a BIGINT,\n            b BIGINT,\n            c BIGINT,\n            d BIGINT\n        ) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        t = self.t_env.from_elements([(1, 2, 3), (2, 5, 6), (3, 1, 9)], ['a', 'b', 'c'])
        t.where(add_one(t.b) <= 3).select(t.a, t.b + 1, add(t.a + 1, subtract_one(t.c)) + 2, add(add_one(t.a), 1)).execute_insert('Results_test_basic_functionality').wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 3, 6, 3]', '+I[3, 2, 14, 5]'])

    def test_all_data_types(self):
        if False:
            print('Hello World!')
        import pandas as pd
        import numpy as np

        @udf(result_type=DataTypes.TINYINT(), func_type='pandas')
        def tinyint_func(tinyint_param):
            if False:
                return 10
            assert isinstance(tinyint_param, pd.Series)
            assert isinstance(tinyint_param[0], np.int8), 'tinyint_param of wrong type %s !' % type(tinyint_param[0])
            return tinyint_param

        @udf(result_type=DataTypes.SMALLINT(), func_type='pandas')
        def smallint_func(smallint_param):
            if False:
                return 10
            assert isinstance(smallint_param, pd.Series)
            assert isinstance(smallint_param[0], np.int16), 'smallint_param of wrong type %s !' % type(smallint_param[0])
            assert smallint_param[0] == 32767, 'smallint_param of wrong value %s' % smallint_param
            return smallint_param

        @udf(result_type=DataTypes.INT(), func_type='pandas')
        def int_func(int_param):
            if False:
                print('Hello World!')
            assert isinstance(int_param, pd.Series)
            assert isinstance(int_param[0], np.int32), 'int_param of wrong type %s !' % type(int_param[0])
            assert int_param[0] == -2147483648, 'int_param of wrong value %s' % int_param
            return int_param

        @udf(result_type=DataTypes.BIGINT(), func_type='pandas')
        def bigint_func(bigint_param):
            if False:
                for i in range(10):
                    print('nop')
            assert isinstance(bigint_param, pd.Series)
            assert isinstance(bigint_param[0], np.int64), 'bigint_param of wrong type %s !' % type(bigint_param[0])
            return bigint_param

        @udf(result_type=DataTypes.BOOLEAN(), func_type='pandas')
        def boolean_func(boolean_param):
            if False:
                i = 10
                return i + 15
            assert isinstance(boolean_param, pd.Series)
            assert isinstance(boolean_param[0], np.bool_), 'boolean_param of wrong type %s !' % type(boolean_param[0])
            return boolean_param

        @udf(result_type=DataTypes.FLOAT(), func_type='pandas')
        def float_func(float_param):
            if False:
                while True:
                    i = 10
            assert isinstance(float_param, pd.Series)
            assert isinstance(float_param[0], np.float32), 'float_param of wrong type %s !' % type(float_param[0])
            return float_param

        @udf(result_type=DataTypes.DOUBLE(), func_type='pandas')
        def double_func(double_param):
            if False:
                return 10
            assert isinstance(double_param, pd.Series)
            assert isinstance(double_param[0], np.float64), 'double_param of wrong type %s !' % type(double_param[0])
            return double_param

        @udf(result_type=DataTypes.STRING(), func_type='pandas')
        def varchar_func(varchar_param):
            if False:
                return 10
            assert isinstance(varchar_param, pd.Series)
            assert isinstance(varchar_param[0], str), 'varchar_param of wrong type %s !' % type(varchar_param[0])
            return varchar_param

        @udf(result_type=DataTypes.BYTES(), func_type='pandas')
        def varbinary_func(varbinary_param):
            if False:
                for i in range(10):
                    print('nop')
            assert isinstance(varbinary_param, pd.Series)
            assert isinstance(varbinary_param[0], bytes), 'varbinary_param of wrong type %s !' % type(varbinary_param[0])
            return varbinary_param

        @udf(result_type=DataTypes.DECIMAL(38, 18), func_type='pandas')
        def decimal_func(decimal_param):
            if False:
                i = 10
                return i + 15
            assert isinstance(decimal_param, pd.Series)
            assert isinstance(decimal_param[0], decimal.Decimal), 'decimal_param of wrong type %s !' % type(decimal_param[0])
            return decimal_param

        @udf(result_type=DataTypes.DATE(), func_type='pandas')
        def date_func(date_param):
            if False:
                i = 10
                return i + 15
            assert isinstance(date_param, pd.Series)
            assert isinstance(date_param[0], datetime.date), 'date_param of wrong type %s !' % type(date_param[0])
            return date_param

        @udf(result_type=DataTypes.TIME(), func_type='pandas')
        def time_func(time_param):
            if False:
                return 10
            assert isinstance(time_param, pd.Series)
            assert isinstance(time_param[0], datetime.time), 'time_param of wrong type %s !' % type(time_param[0])
            return time_param
        timestamp_value = datetime.datetime(1970, 1, 2, 0, 0, 0, 123000)

        @udf(result_type=DataTypes.TIMESTAMP(3), func_type='pandas')
        def timestamp_func(timestamp_param):
            if False:
                for i in range(10):
                    print('nop')
            assert isinstance(timestamp_param, pd.Series)
            assert isinstance(timestamp_param[0], datetime.datetime), 'timestamp_param of wrong type %s !' % type(timestamp_param[0])
            assert timestamp_param[0] == timestamp_value, 'timestamp_param is wrong value %s, should be %s!' % (timestamp_param[0], timestamp_value)
            return timestamp_param

        def array_func(array_param):
            if False:
                return 10
            assert isinstance(array_param, pd.Series)
            assert isinstance(array_param[0], np.ndarray), 'array_param of wrong type %s !' % type(array_param[0])
            return array_param
        array_str_func = udf(array_func, result_type=DataTypes.ARRAY(DataTypes.STRING()), func_type='pandas')
        array_timestamp_func = udf(array_func, result_type=DataTypes.ARRAY(DataTypes.TIMESTAMP(3)), func_type='pandas')
        array_int_func = udf(array_func, result_type=DataTypes.ARRAY(DataTypes.INT()), func_type='pandas')

        @udf(result_type=DataTypes.ARRAY(DataTypes.STRING()), func_type='pandas')
        def nested_array_func(nested_array_param):
            if False:
                print('Hello World!')
            assert isinstance(nested_array_param, pd.Series)
            assert isinstance(nested_array_param[0], np.ndarray), 'nested_array_param of wrong type %s !' % type(nested_array_param[0])
            return pd.Series(nested_array_param[0])
        row_type = DataTypes.ROW([DataTypes.FIELD('f1', DataTypes.INT()), DataTypes.FIELD('f2', DataTypes.STRING()), DataTypes.FIELD('f3', DataTypes.TIMESTAMP(3)), DataTypes.FIELD('f4', DataTypes.ARRAY(DataTypes.INT()))])

        @udf(result_type=row_type, func_type='pandas')
        def row_func(row_param):
            if False:
                while True:
                    i = 10
            assert isinstance(row_param, pd.DataFrame)
            assert isinstance(row_param.f1, pd.Series)
            assert isinstance(row_param.f1[0], np.int32), 'row_param.f1 of wrong type %s !' % type(row_param.f1[0])
            assert isinstance(row_param.f2, pd.Series)
            assert isinstance(row_param.f2[0], str), 'row_param.f2 of wrong type %s !' % type(row_param.f2[0])
            assert isinstance(row_param.f3, pd.Series)
            assert isinstance(row_param.f3[0], datetime.datetime), 'row_param.f3 of wrong type %s !' % type(row_param.f3[0])
            assert isinstance(row_param.f4, pd.Series)
            assert isinstance(row_param.f4[0], np.ndarray), 'row_param.f4 of wrong type %s !' % type(row_param.f4[0])
            return row_param
        map_type = DataTypes.MAP(DataTypes.STRING(False), DataTypes.STRING())

        @udf(result_type=map_type, func_type='pandas')
        def map_func(map_param):
            if False:
                while True:
                    i = 10
            assert isinstance(map_param, pd.Series)
            return map_param

        @udf(result_type=DataTypes.BINARY(5), func_type='pandas')
        def binary_func(binary_param):
            if False:
                return 10
            assert isinstance(binary_param, pd.Series)
            assert isinstance(binary_param[0], bytes), 'binary_param of wrong type %s !' % type(binary_param[0])
            assert len(binary_param[0]) == 5
            return binary_param
        sink_table_ddl = "\n            CREATE TABLE Results_test_all_data_types(\n                a TINYINT,\n                b SMALLINT,\n                c INT,\n                d BIGINT,\n                e BOOLEAN,\n                f BOOLEAN,\n                g FLOAT,\n                h DOUBLE,\n                i STRING,\n                j StRING,\n                k BYTES,\n                l DECIMAL(38, 18),\n                m DECIMAL(38, 18),\n                n DATE,\n                o TIME,\n                p TIMESTAMP(3),\n                q ARRAY<STRING>,\n                r ARRAY<TIMESTAMP(3)>,\n                s ARRAY<INT>,\n                t ARRAY<STRING>,\n                u ROW<f1 INT, f2 STRING, f3 TIMESTAMP(3), f4 ARRAY<INT>>,\n                v MAP<STRING, STRING>,\n                w BINARY(5)\n            ) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        t = self.t_env.from_elements([(1, 32767, -2147483648, 1, True, False, 1.0, 1.0, 'hello', '中文', bytearray(b'flink'), decimal.Decimal('1000000000000000000.05'), decimal.Decimal('1000000000000000000.05999999999999999899999999999'), datetime.date(2014, 9, 13), datetime.time(hour=1, minute=0, second=1), timestamp_value, ['hello', '中文', None], [timestamp_value], [1, 2], [['hello', '中文', None]], Row(1, 'hello', timestamp_value, [1, 2]), {'1': 'hello', '2': 'world'}, bytearray(b'flink'))], DataTypes.ROW([DataTypes.FIELD('a', DataTypes.TINYINT()), DataTypes.FIELD('b', DataTypes.SMALLINT()), DataTypes.FIELD('c', DataTypes.INT()), DataTypes.FIELD('d', DataTypes.BIGINT()), DataTypes.FIELD('e', DataTypes.BOOLEAN()), DataTypes.FIELD('f', DataTypes.BOOLEAN()), DataTypes.FIELD('g', DataTypes.FLOAT()), DataTypes.FIELD('h', DataTypes.DOUBLE()), DataTypes.FIELD('i', DataTypes.STRING()), DataTypes.FIELD('j', DataTypes.STRING()), DataTypes.FIELD('k', DataTypes.BYTES()), DataTypes.FIELD('l', DataTypes.DECIMAL(38, 18)), DataTypes.FIELD('m', DataTypes.DECIMAL(38, 18)), DataTypes.FIELD('n', DataTypes.DATE()), DataTypes.FIELD('o', DataTypes.TIME()), DataTypes.FIELD('p', DataTypes.TIMESTAMP(3)), DataTypes.FIELD('q', DataTypes.ARRAY(DataTypes.STRING())), DataTypes.FIELD('r', DataTypes.ARRAY(DataTypes.TIMESTAMP(3))), DataTypes.FIELD('s', DataTypes.ARRAY(DataTypes.INT())), DataTypes.FIELD('t', DataTypes.ARRAY(DataTypes.ARRAY(DataTypes.STRING()))), DataTypes.FIELD('u', row_type), DataTypes.FIELD('v', map_type), DataTypes.FIELD('w', DataTypes.BINARY(5))]))
        t.select(tinyint_func(t.a), smallint_func(t.b), int_func(t.c), bigint_func(t.d), boolean_func(t.e), boolean_func(t.f), float_func(t.g), double_func(t.h), varchar_func(t.i), varchar_func(t.j), varbinary_func(t.k), decimal_func(t.l), decimal_func(t.m), date_func(t.n), time_func(t.o), timestamp_func(t.p), array_str_func(t.q), array_timestamp_func(t.r), array_int_func(t.s), nested_array_func(t.t), row_func(t.u), map_func(t.v), binary_func(t.w)).execute_insert('Results_test_all_data_types').wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1, 32767, -2147483648, 1, true, false, 1.0, 1.0, hello, 中文, [102, 108, 105, 110, 107], 1000000000000000000.050000000000000000, 1000000000000000000.059999999999999999, 2014-09-13, 01:00:01, 1970-01-02T00:00:00.123, [hello, 中文, null], [1970-01-02T00:00:00.123], [1, 2], [hello, 中文, null], +I[1, hello, 1970-01-02T00:00:00.123, [1, 2]], {1=hello, 2=world}, [102, 108, 105, 110, 107]]'])

    def test_invalid_pandas_udf(self):
        if False:
            return 10

        @udf(result_type=DataTypes.INT(), func_type='pandas')
        def length_mismatch(i):
            if False:
                print('Hello World!')
            return i[1:]

        @udf(result_type=DataTypes.INT(), func_type='pandas')
        def result_type_not_series(i):
            if False:
                print('Hello World!')
            return i.iloc[0]
        t = self.t_env.from_elements([(1, 2, 3), (2, 5, 6), (3, 1, 9)], ['a', 'b', 'c'])
        msg = "The result length '0' of Pandas UDF 'length_mismatch' is not equal to the input length '1'"
        from py4j.protocol import Py4JJavaError
        with self.assertRaisesRegex(Py4JJavaError, expected_regex=msg):
            t.select(length_mismatch(t.a)).to_pandas()
        msg = "The result type of Pandas UDF 'result_type_not_series' must be pandas.Series or pandas.DataFrame, got <class 'numpy.int64'>"
        from py4j.protocol import Py4JJavaError
        with self.assertRaisesRegex(Py4JJavaError, expected_regex=msg):
            t.select(result_type_not_series(t.a)).to_pandas()

    def test_data_types(self):
        if False:
            for i in range(10):
                print('nop')
        import pandas as pd
        timezone = self.t_env.get_config().get_local_timezone()
        local_datetime = pytz.timezone(timezone).localize(datetime.datetime(1970, 1, 2, 0, 0, 0, 123000))

        @udf(result_type=DataTypes.TIMESTAMP_WITH_LOCAL_TIME_ZONE(3), func_type='pandas')
        def local_zoned_timestamp_func(local_zoned_timestamp_param):
            if False:
                print('Hello World!')
            assert isinstance(local_zoned_timestamp_param, pd.Series)
            assert isinstance(local_zoned_timestamp_param[0], datetime.datetime), 'local_zoned_timestamp_param of wrong type %s !' % type(local_zoned_timestamp_param[0])
            assert local_zoned_timestamp_param[0] == local_datetime, 'local_zoned_timestamp_param is wrong value %s, %s!' % (local_zoned_timestamp_param[0], local_datetime)
            return local_zoned_timestamp_param
        sink_table_ddl = "\n        CREATE TABLE Results_test_data_types(a TIMESTAMP_LTZ(3)) WITH ('connector'='test-sink')\n        "
        self.t_env.execute_sql(sink_table_ddl)
        t = self.t_env.from_elements([(local_datetime,)], DataTypes.ROW([DataTypes.FIELD('a', DataTypes.TIMESTAMP_WITH_LOCAL_TIME_ZONE(3))]))
        t.select(local_zoned_timestamp_func(local_zoned_timestamp_func(t.a))).execute_insert('Results_test_data_types').wait()
        actual = source_sink_utils.results()
        self.assert_equals(actual, ['+I[1970-01-02T00:00:00.123Z]'])

class BatchPandasUDFITTests(PandasUDFITTests, PyFlinkBatchTableTestCase):
    pass

class StreamPandasUDFITTests(PandasUDFITTests, PyFlinkStreamTableTestCase):
    pass

@udf(result_type=DataTypes.BIGINT(), func_type='pandas')
def add(i, j):
    if False:
        i = 10
        return i + 15
    return i + j
if __name__ == '__main__':
    import unittest
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)