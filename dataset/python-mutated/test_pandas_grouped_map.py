import datetime
import unittest
from collections import OrderedDict
from decimal import Decimal
from typing import cast
from pyspark.sql import Row
from pyspark.sql.functions import array, explode, col, lit, udf, sum, pandas_udf, PandasUDFType, window
from pyspark.sql.types import IntegerType, DoubleType, ArrayType, BinaryType, ByteType, LongType, DecimalType, ShortType, FloatType, StringType, BooleanType, StructType, StructField, NullType, MapType, YearMonthIntervalType
from pyspark.errors import PythonException, PySparkTypeError
from pyspark.testing.sqlutils import ReusedSQLTestCase, have_pandas, have_pyarrow, pandas_requirement_message, pyarrow_requirement_message
from pyspark.testing.utils import QuietTest
if have_pandas:
    import pandas as pd
    from pandas.testing import assert_frame_equal
if have_pyarrow:
    import pyarrow as pa

@unittest.skipIf(not have_pandas or not have_pyarrow, cast(str, pandas_requirement_message or pyarrow_requirement_message))
class GroupedApplyInPandasTestsMixin:

    @property
    def data(self):
        if False:
            print('Hello World!')
        return self.spark.range(10).withColumn('vs', array([lit(i) for i in range(20, 30)])).withColumn('v', explode(col('vs'))).drop('vs')

    def test_supported_types(self):
        if False:
            for i in range(10):
                print('nop')
        values = [1, 2, 3, 4, 5, 1.1, 2.2, Decimal(1.123), [1, 2, 2], True, 'hello', bytearray([1, 2]), None]
        output_fields = [('id', IntegerType()), ('byte', ByteType()), ('short', ShortType()), ('int', IntegerType()), ('long', LongType()), ('float', FloatType()), ('double', DoubleType()), ('decim', DecimalType(10, 3)), ('array', ArrayType(IntegerType())), ('bool', BooleanType()), ('str', StringType()), ('bin', BinaryType()), ('null', NullType())]
        output_schema = StructType([StructField(*x) for x in output_fields])
        df = self.spark.createDataFrame([values], schema=output_schema)
        udf1 = pandas_udf(lambda pdf: pdf.assign(byte=pdf.byte * 2, short=pdf.short * 2, int=pdf.int * 2, long=pdf.long * 2, float=pdf.float * 2, double=pdf.double * 2, decim=pdf.decim * 2, bool=False if pdf.bool else True, str=pdf.str + 'there', array=pdf.array, bin=pdf.bin, null=pdf.null), output_schema, PandasUDFType.GROUPED_MAP)
        udf2 = pandas_udf(lambda _, pdf: pdf.assign(byte=pdf.byte * 2, short=pdf.short * 2, int=pdf.int * 2, long=pdf.long * 2, float=pdf.float * 2, double=pdf.double * 2, decim=pdf.decim * 2, bool=False if pdf.bool else True, str=pdf.str + 'there', array=pdf.array, bin=pdf.bin, null=pdf.null), output_schema, PandasUDFType.GROUPED_MAP)
        udf3 = pandas_udf(lambda key, pdf: pdf.assign(id=key[0], byte=pdf.byte * 2, short=pdf.short * 2, int=pdf.int * 2, long=pdf.long * 2, float=pdf.float * 2, double=pdf.double * 2, decim=pdf.decim * 2, bool=False if pdf.bool else True, str=pdf.str + 'there', array=pdf.array, bin=pdf.bin, null=pdf.null), output_schema, PandasUDFType.GROUPED_MAP)
        result1 = df.groupby('id').apply(udf1).sort('id').toPandas()
        expected1 = df.toPandas().groupby('id').apply(udf1.func).reset_index(drop=True)
        result2 = df.groupby('id').apply(udf2).sort('id').toPandas()
        expected2 = expected1
        result3 = df.groupby('id').apply(udf3).sort('id').toPandas()
        expected3 = expected1
        assert_frame_equal(expected1, result1)
        assert_frame_equal(expected2, result2)
        assert_frame_equal(expected3, result3)

    def test_array_type_correct(self):
        if False:
            while True:
                i = 10
        df = self.data.withColumn('arr', array(col('id'))).repartition(1, 'id')
        output_schema = StructType([StructField('id', LongType()), StructField('v', IntegerType()), StructField('arr', ArrayType(LongType()))])
        udf = pandas_udf(lambda pdf: pdf, output_schema, PandasUDFType.GROUPED_MAP)
        result = df.groupby('id').apply(udf).sort('id').toPandas()
        expected = df.toPandas().groupby('id').apply(udf.func).reset_index(drop=True)
        assert_frame_equal(expected, result)

    def test_register_grouped_map_udf(self):
        if False:
            i = 10
            return i + 15
        with QuietTest(self.sc):
            self.check_register_grouped_map_udf()

    def check_register_grouped_map_udf(self):
        if False:
            while True:
                i = 10
        foo_udf = pandas_udf(lambda x: x, 'id long', PandasUDFType.GROUPED_MAP)
        with self.assertRaises(PySparkTypeError) as pe:
            self.spark.catalog.registerFunction('foo_udf', foo_udf)
        self.check_error(exception=pe.exception, error_class='INVALID_UDF_EVAL_TYPE', message_parameters={'eval_type': 'SQL_BATCHED_UDF, SQL_ARROW_BATCHED_UDF, SQL_SCALAR_PANDAS_UDF, SQL_SCALAR_PANDAS_ITER_UDF or SQL_GROUPED_AGG_PANDAS_UDF'})

    def test_decorator(self):
        if False:
            for i in range(10):
                print('nop')
        df = self.data

        @pandas_udf('id long, v int, v1 double, v2 long', PandasUDFType.GROUPED_MAP)
        def foo(pdf):
            if False:
                print('Hello World!')
            return pdf.assign(v1=pdf.v * pdf.id * 1.0, v2=pdf.v + pdf.id)
        result = df.groupby('id').apply(foo).sort('id').toPandas()
        expected = df.toPandas().groupby('id').apply(foo.func).reset_index(drop=True)
        assert_frame_equal(expected, result)

    def test_coerce(self):
        if False:
            for i in range(10):
                print('nop')
        df = self.data
        foo = pandas_udf(lambda pdf: pdf, 'id long, v double', PandasUDFType.GROUPED_MAP)
        result = df.groupby('id').apply(foo).sort('id').toPandas()
        expected = df.toPandas().groupby('id').apply(foo.func).reset_index(drop=True)
        expected = expected.assign(v=expected.v.astype('float64'))
        assert_frame_equal(expected, result)

    def test_complex_groupby(self):
        if False:
            print('Hello World!')
        df = self.data

        @pandas_udf('id long, v int, norm double', PandasUDFType.GROUPED_MAP)
        def normalize(pdf):
            if False:
                while True:
                    i = 10
            v = pdf.v
            return pdf.assign(norm=(v - v.mean()) / v.std())
        result = df.groupby(col('id') % 2 == 0).apply(normalize).sort('id', 'v').toPandas()
        pdf = df.toPandas()
        expected = pdf.groupby(pdf['id'] % 2 == 0, as_index=False).apply(normalize.func)
        expected = expected.sort_values(['id', 'v']).reset_index(drop=True)
        expected = expected.assign(norm=expected.norm.astype('float64'))
        assert_frame_equal(expected, result)

    def test_empty_groupby(self):
        if False:
            for i in range(10):
                print('nop')
        df = self.data

        @pandas_udf('id long, v int, norm double', PandasUDFType.GROUPED_MAP)
        def normalize(pdf):
            if False:
                for i in range(10):
                    print('nop')
            v = pdf.v
            return pdf.assign(norm=(v - v.mean()) / v.std())
        result = df.groupby().apply(normalize).sort('id', 'v').toPandas()
        pdf = df.toPandas()
        expected = normalize.func(pdf)
        expected = expected.sort_values(['id', 'v']).reset_index(drop=True)
        expected = expected.assign(norm=expected.norm.astype('float64'))
        assert_frame_equal(expected, result)

    def test_apply_in_pandas_not_returning_pandas_dataframe(self):
        if False:
            return 10
        with QuietTest(self.sc):
            self.check_apply_in_pandas_not_returning_pandas_dataframe()

    def check_apply_in_pandas_not_returning_pandas_dataframe(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(PythonException, 'Return type of the user-defined function should be pandas.DataFrame, but is tuple.'):
            self._test_apply_in_pandas(lambda key, pdf: key)

    @staticmethod
    def stats_with_column_names(key, pdf):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame([(pdf.v.mean(),) + key], columns=['mean', 'id'])

    @staticmethod
    def stats_with_no_column_names(key, pdf):
        if False:
            print('Hello World!')
        return pd.DataFrame([key + (pdf.v.mean(),)])

    def test_apply_in_pandas_returning_column_names(self):
        if False:
            print('Hello World!')
        self._test_apply_in_pandas(GroupedApplyInPandasTestsMixin.stats_with_column_names)

    def test_apply_in_pandas_returning_no_column_names(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_apply_in_pandas(GroupedApplyInPandasTestsMixin.stats_with_no_column_names)

    def test_apply_in_pandas_returning_column_names_sometimes(self):
        if False:
            print('Hello World!')

        def stats(key, pdf):
            if False:
                i = 10
                return i + 15
            if key[0] % 2:
                return GroupedApplyInPandasTestsMixin.stats_with_column_names(key, pdf)
            else:
                return GroupedApplyInPandasTestsMixin.stats_with_no_column_names(key, pdf)
        self._test_apply_in_pandas(stats)

    def test_apply_in_pandas_returning_wrong_column_names(self):
        if False:
            while True:
                i = 10
        with QuietTest(self.sc):
            self.check_apply_in_pandas_returning_wrong_column_names()

    def check_apply_in_pandas_returning_wrong_column_names(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(PythonException, 'Column names of the returned pandas.DataFrame do not match specified schema. Missing: mean. Unexpected: median, std.\n'):
            self._test_apply_in_pandas(lambda key, pdf: pd.DataFrame([key + (pdf.v.median(), pdf.v.std())], columns=['id', 'median', 'std']))

    def test_apply_in_pandas_returning_no_column_names_and_wrong_amount(self):
        if False:
            for i in range(10):
                print('nop')
        with QuietTest(self.sc):
            self.check_apply_in_pandas_returning_no_column_names_and_wrong_amount()

    def check_apply_in_pandas_returning_no_column_names_and_wrong_amount(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(PythonException, "Number of columns of the returned pandas.DataFrame doesn't match specified schema. Expected: 2 Actual: 3\n"):
            self._test_apply_in_pandas(lambda key, pdf: pd.DataFrame([key + (pdf.v.mean(), pdf.v.std())]))

    def test_apply_in_pandas_returning_empty_dataframe(self):
        if False:
            return 10
        self._test_apply_in_pandas_returning_empty_dataframe(pd.DataFrame())

    def test_apply_in_pandas_returning_incompatible_type(self):
        if False:
            return 10
        with QuietTest(self.sc):
            self.check_apply_in_pandas_returning_incompatible_type()

    def check_apply_in_pandas_returning_incompatible_type(self):
        if False:
            for i in range(10):
                print('nop')
        for safely in [True, False]:
            with self.subTest(convertToArrowArraySafely=safely), self.sql_conf({'spark.sql.execution.pandas.convertToArrowArraySafely': safely}):
                with self.subTest(convert='string to double'):
                    expected = "ValueError: Exception thrown when converting pandas.Series \\(object\\) with name 'mean' to Arrow Array \\(double\\)."
                    if safely:
                        expected = expected + ' It can be caused by overflows or other unsafe conversions warned by Arrow. Arrow safe type check can be disabled by using SQL config `spark.sql.execution.pandas.convertToArrowArraySafely`.'
                    with self.assertRaisesRegex(PythonException, expected + '\n'):
                        self._test_apply_in_pandas(lambda key, pdf: pd.DataFrame([key + (str(pdf.v.mean()),)]), output_schema='id long, mean double')
                with self.subTest(convert='double to string'):
                    with self.assertRaisesRegex(PythonException, "TypeError: Exception thrown when converting pandas.Series \\(float64\\) with name 'mean' to Arrow Array \\(string\\).\\n"):
                        self._test_apply_in_pandas(lambda key, pdf: pd.DataFrame([key + (pdf.v.mean(),)]), output_schema='id long, mean string')

    def test_datatype_string(self):
        if False:
            while True:
                i = 10
        df = self.data
        foo_udf = pandas_udf(lambda pdf: pdf.assign(v1=pdf.v * pdf.id * 1.0, v2=pdf.v + pdf.id), 'id long, v int, v1 double, v2 long', PandasUDFType.GROUPED_MAP)
        result = df.groupby('id').apply(foo_udf).sort('id').toPandas()
        expected = df.toPandas().groupby('id').apply(foo_udf.func).reset_index(drop=True)
        assert_frame_equal(expected, result)

    def test_wrong_return_type(self):
        if False:
            for i in range(10):
                print('nop')
        with QuietTest(self.sc):
            self.check_wrong_return_type()

    def check_wrong_return_type(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(NotImplementedError, 'Invalid return type.*grouped map Pandas UDF.*ArrayType.*YearMonthIntervalType'):
            pandas_udf(lambda pdf: pdf, StructType().add('id', LongType()).add('v', ArrayType(YearMonthIntervalType())), PandasUDFType.GROUPED_MAP)

    def test_wrong_args(self):
        if False:
            print('Hello World!')
        with QuietTest(self.sc):
            self.check_wrong_args()

    def check_wrong_args(self):
        if False:
            i = 10
            return i + 15
        df = self.data
        with self.assertRaisesRegex(ValueError, 'Invalid udf'):
            df.groupby('id').apply(lambda x: x)
        with self.assertRaisesRegex(ValueError, 'Invalid udf'):
            df.groupby('id').apply(udf(lambda x: x, DoubleType()))
        with self.assertRaisesRegex(ValueError, 'Invalid udf'):
            df.groupby('id').apply(sum(df.v))
        with self.assertRaisesRegex(ValueError, 'Invalid udf'):
            df.groupby('id').apply(df.v + 1)
        with self.assertRaisesRegex(ValueError, 'Invalid function'):
            df.groupby('id').apply(pandas_udf(lambda : 1, StructType([StructField('d', DoubleType())])))
        with self.assertRaisesRegex(ValueError, 'Invalid udf'):
            df.groupby('id').apply(pandas_udf(lambda x, y: x, DoubleType()))
        with self.assertRaisesRegex(ValueError, 'Invalid udf.*GROUPED_MAP'):
            df.groupby('id').apply(pandas_udf(lambda x, y: x, DoubleType(), PandasUDFType.SCALAR))

    def test_unsupported_types(self):
        if False:
            for i in range(10):
                print('nop')
        with QuietTest(self.sc):
            self.check_unsupported_types()

    def check_unsupported_types(self):
        if False:
            print('Hello World!')
        common_err_msg = 'Invalid return type.*grouped map Pandas UDF.*'
        unsupported_types = [StructField('array_struct', ArrayType(YearMonthIntervalType())), StructField('map', MapType(StringType(), YearMonthIntervalType()))]
        for unsupported_type in unsupported_types:
            with self.subTest(unsupported_type=unsupported_type.name):
                schema = StructType([StructField('id', LongType(), True), unsupported_type])
                with self.assertRaisesRegex(NotImplementedError, common_err_msg):
                    pandas_udf(lambda x: x, schema, PandasUDFType.GROUPED_MAP)

    def test_timestamp_dst(self):
        if False:
            for i in range(10):
                print('nop')
        dt = [datetime.datetime(2015, 11, 1, 0, 30), datetime.datetime(2015, 11, 1, 1, 30), datetime.datetime(2015, 11, 1, 2, 30)]
        df = self.spark.createDataFrame(dt, 'timestamp').toDF('time')
        foo_udf = pandas_udf(lambda pdf: pdf, 'time timestamp', PandasUDFType.GROUPED_MAP)
        result = df.groupby('time').apply(foo_udf).sort('time')
        assert_frame_equal(df.toPandas(), result.toPandas())

    def test_udf_with_key(self):
        if False:
            for i in range(10):
                print('nop')
        import numpy as np
        df = self.data
        pdf = df.toPandas()

        def foo1(key, pdf):
            if False:
                for i in range(10):
                    print('nop')
            assert type(key) == tuple
            assert type(key[0]) == np.int64
            return pdf.assign(v1=key[0], v2=pdf.v * key[0], v3=pdf.v * pdf.id, v4=pdf.v * pdf.id.mean())

        def foo2(key, pdf):
            if False:
                return 10
            assert type(key) == tuple
            assert type(key[0]) == np.int64
            assert type(key[1]) == np.int32
            return pdf.assign(v1=key[0], v2=key[1], v3=pdf.v * key[0], v4=pdf.v + key[1])

        def foo3(key, pdf):
            if False:
                for i in range(10):
                    print('nop')
            assert type(key) == tuple
            assert len(key) == 0
            return pdf.assign(v1=pdf.v * pdf.id)
        udf1 = pandas_udf(foo1, 'id long, v int, v1 long, v2 int, v3 long, v4 double', PandasUDFType.GROUPED_MAP)
        udf2 = pandas_udf(foo2, 'id long, v int, v1 long, v2 int, v3 int, v4 int', PandasUDFType.GROUPED_MAP)
        udf3 = pandas_udf(foo3, 'id long, v int, v1 long', PandasUDFType.GROUPED_MAP)
        result1 = df.groupby('id').apply(udf1).sort('id', 'v').toPandas()
        expected1 = pdf.groupby('id', as_index=False).apply(lambda x: udf1.func((x.id.iloc[0],), x)).sort_values(['id', 'v']).reset_index(drop=True)
        assert_frame_equal(expected1, result1)
        result2 = df.groupby(df.id % 2).apply(udf1).sort('id', 'v').toPandas()
        expected2 = pdf.groupby(pdf.id % 2, as_index=False).apply(lambda x: udf1.func((x.id.iloc[0] % 2,), x)).sort_values(['id', 'v']).reset_index(drop=True)
        assert_frame_equal(expected2, result2)
        result3 = df.groupby(df.id, df.v % 2).apply(udf2).sort('id', 'v').toPandas()
        expected3 = pdf.groupby([pdf.id, pdf.v % 2], as_index=False).apply(lambda x: udf2.func((x.id.iloc[0], (x.v % 2).iloc[0]), x)).sort_values(['id', 'v']).reset_index(drop=True)
        assert_frame_equal(expected3, result3)
        result4 = df.groupby().apply(udf3).sort('id', 'v').toPandas()
        expected4 = udf3.func((), pdf)
        assert_frame_equal(expected4, result4)

    def test_column_order(self):
        if False:
            i = 10
            return i + 15
        with QuietTest(self.sc):
            self.check_column_order()

    def check_column_order(self):
        if False:
            while True:
                i = 10

        def rename_pdf(pdf, names):
            if False:
                for i in range(10):
                    print('nop')
            pdf.rename(columns={old: new for (old, new) in zip(pd_result.columns, names)}, inplace=True)
        df = self.data
        grouped_df = df.groupby('id')
        grouped_pdf = df.toPandas().groupby('id', as_index=False)

        def change_col_order(pdf):
            if False:
                i = 10
                return i + 15
            return pd.DataFrame.from_dict(OrderedDict([('id', pdf.id), ('u', pdf.v * 2), ('v', pdf.v)]))
        ordered_udf = pandas_udf(change_col_order, 'id long, v int, u int', PandasUDFType.GROUPED_MAP)
        result = grouped_df.apply(ordered_udf).sort('id', 'v').select('id', 'u', 'v').toPandas()
        pd_result = grouped_pdf.apply(change_col_order)
        expected = pd_result.sort_values(['id', 'v']).reset_index(drop=True)
        assert_frame_equal(expected, result)

        def range_col_order(pdf):
            if False:
                print('Hello World!')
            return pd.DataFrame(list(zip(pdf.id, pdf.v * 3, pdf.v)), dtype='int64')
        range_udf = pandas_udf(range_col_order, 'id long, u long, v long', PandasUDFType.GROUPED_MAP)
        result = grouped_df.apply(range_udf).sort('id', 'v').select('id', 'u', 'v').toPandas()
        pd_result = grouped_pdf.apply(range_col_order)
        rename_pdf(pd_result, ['id', 'u', 'v'])
        expected = pd_result.sort_values(['id', 'v']).reset_index(drop=True)
        assert_frame_equal(expected, result)

        def int_index(pdf):
            if False:
                while True:
                    i = 10
            return pd.DataFrame(OrderedDict([(0, pdf.id), (1, pdf.v * 4), (2, pdf.v)]))
        int_index_udf = pandas_udf(int_index, 'id long, u int, v int', PandasUDFType.GROUPED_MAP)
        result = grouped_df.apply(int_index_udf).sort('id', 'v').select('id', 'u', 'v').toPandas()
        pd_result = grouped_pdf.apply(int_index)
        rename_pdf(pd_result, ['id', 'u', 'v'])
        expected = pd_result.sort_values(['id', 'v']).reset_index(drop=True)
        assert_frame_equal(expected, result)

        @pandas_udf('id long, v int', PandasUDFType.GROUPED_MAP)
        def column_name_typo(pdf):
            if False:
                i = 10
                return i + 15
            return pd.DataFrame({'iid': pdf.id, 'v': pdf.v})

        @pandas_udf('id long, v decimal', PandasUDFType.GROUPED_MAP)
        def invalid_positional_types(pdf):
            if False:
                return 10
            return pd.DataFrame([(1, datetime.date(2020, 10, 5))])
        with self.sql_conf({'spark.sql.execution.pandas.convertToArrowArraySafely': False}):
            with self.assertRaisesRegex(PythonException, 'Column names of the returned pandas.DataFrame do not match specified schema. Missing: id. Unexpected: iid.\n'):
                grouped_df.apply(column_name_typo).collect()
            with self.assertRaisesRegex(Exception, '[D|d]ecimal.*got.*date'):
                grouped_df.apply(invalid_positional_types).collect()

    def test_positional_assignment_conf(self):
        if False:
            print('Hello World!')
        with self.sql_conf({'spark.sql.legacy.execution.pandas.groupedMap.assignColumnsByName': False}):

            @pandas_udf('a string, b float', PandasUDFType.GROUPED_MAP)
            def foo(_):
                if False:
                    while True:
                        i = 10
                return pd.DataFrame([('hi', 1)], columns=['x', 'y'])
            df = self.data
            result = df.groupBy('id').apply(foo).select('a', 'b').collect()
            for r in result:
                self.assertEqual(r.a, 'hi')
                self.assertEqual(r.b, 1)

    def test_self_join_with_pandas(self):
        if False:
            for i in range(10):
                print('nop')

        @pandas_udf('key long, col string', PandasUDFType.GROUPED_MAP)
        def dummy_pandas_udf(df):
            if False:
                i = 10
                return i + 15
            return df[['key', 'col']]
        df = self.spark.createDataFrame([Row(key=1, col='A'), Row(key=1, col='B'), Row(key=2, col='C')])
        df_with_pandas = df.groupBy('key').apply(dummy_pandas_udf)
        res = df_with_pandas.alias('temp0').join(df_with_pandas.alias('temp1'), col('temp0.key') == col('temp1.key'))
        self.assertEqual(res.count(), 5)

    def test_mixed_scalar_udfs_followed_by_groupby_apply(self):
        if False:
            return 10
        df = self.spark.range(0, 10).toDF('v1')
        df = df.withColumn('v2', udf(lambda x: x + 1, 'int')(df['v1'])).withColumn('v3', pandas_udf(lambda x: x + 2, 'int')(df['v1']))
        result = df.groupby().apply(pandas_udf(lambda x: pd.DataFrame([x.sum().sum()]), 'sum int', PandasUDFType.GROUPED_MAP))
        self.assertEqual(result.collect()[0]['sum'], 165)

    def test_grouped_with_empty_partition(self):
        if False:
            i = 10
            return i + 15
        data = [Row(id=1, x=2), Row(id=1, x=3), Row(id=2, x=4)]
        expected = [Row(id=1, x=5), Row(id=1, x=5), Row(id=2, x=4)]
        num_parts = len(data) + 1
        df = self.spark.createDataFrame(self.sc.parallelize(data, numSlices=num_parts))
        f = pandas_udf(lambda pdf: pdf.assign(x=pdf['x'].sum()), 'id long, x int', PandasUDFType.GROUPED_MAP)
        result = df.groupBy('id').apply(f).collect()
        self.assertEqual(result, expected)

    def test_grouped_over_window(self):
        if False:
            print('Hello World!')
        data = [(0, 1, '2018-03-10T00:00:00+00:00', [0]), (1, 2, '2018-03-11T00:00:00+00:00', [0]), (2, 2, '2018-03-12T00:00:00+00:00', [0]), (3, 3, '2018-03-15T00:00:00+00:00', [0]), (4, 3, '2018-03-16T00:00:00+00:00', [0]), (5, 3, '2018-03-17T00:00:00+00:00', [0]), (6, 3, '2018-03-21T00:00:00+00:00', [0])]
        expected = {0: [0], 1: [1, 2], 2: [1, 2], 3: [3, 4, 5], 4: [3, 4, 5], 5: [3, 4, 5], 6: [6]}
        df = self.spark.createDataFrame(data, ['id', 'group', 'ts', 'result'])
        df = df.select(col('id'), col('group'), col('ts').cast('timestamp'), col('result'))

        def f(pdf):
            if False:
                i = 10
                return i + 15
            pdf['result'] = [pdf['id']] * len(pdf)
            return pdf
        result = df.groupby('group', window('ts', '5 days')).applyInPandas(f, df.schema).select('id', 'result').orderBy('id').collect()
        self.assertListEqual([Row(id=key, result=val) for (key, val) in expected.items()], result)

    def test_grouped_over_window_with_key(self):
        if False:
            return 10
        data = [(0, 1, '2018-03-10T00:00:00+00:00', [0]), (1, 2, '2018-03-11T00:00:00+00:00', [0]), (2, 2, '2018-03-12T00:00:00+00:00', [0]), (3, 3, '2018-03-15T00:00:00+00:00', [0]), (4, 3, '2018-03-16T00:00:00+00:00', [0]), (5, 3, '2018-03-17T00:00:00+00:00', [0]), (6, 3, '2018-03-21T00:00:00+00:00', [0])]
        timezone = self.spark.conf.get('spark.sql.session.timeZone')
        expected_window = [{key: pd.Timestamp(ts).tz_localize(datetime.timezone.utc).tz_convert(timezone).tz_localize(None) for (key, ts) in w.items()} for w in [{'start': datetime.datetime(2018, 3, 10, 0, 0), 'end': datetime.datetime(2018, 3, 15, 0, 0)}, {'start': datetime.datetime(2018, 3, 15, 0, 0), 'end': datetime.datetime(2018, 3, 20, 0, 0)}, {'start': datetime.datetime(2018, 3, 20, 0, 0), 'end': datetime.datetime(2018, 3, 25, 0, 0)}]]
        expected_key = {0: (1, expected_window[0]), 1: (2, expected_window[0]), 2: (2, expected_window[0]), 3: (3, expected_window[1]), 4: (3, expected_window[1]), 5: (3, expected_window[1]), 6: (3, expected_window[2])}
        expected = {0: [1], 1: [2, 2], 2: [2, 2], 3: [3, 3, 3], 4: [3, 3, 3], 5: [3, 3, 3], 6: [3]}
        df = self.spark.createDataFrame(data, ['id', 'group', 'ts', 'result'])
        df = df.select(col('id'), col('group'), col('ts').cast('timestamp'), col('result'))

        def f(key, pdf):
            if False:
                return 10
            group = key[0]
            window_range = key[1]
            for (_, i) in pdf.id.items():
                assert expected_key[i][0] == group, '{} != {}'.format(expected_key[i][0], group)
                assert expected_key[i][1] == window_range, '{} != {}'.format(expected_key[i][1], window_range)
            return pdf.assign(result=[[group] * len(pdf)] * len(pdf))
        result = df.groupby('group', window('ts', '5 days')).applyInPandas(f, df.schema).select('id', 'result').orderBy('id').collect()
        self.assertListEqual([Row(id=key, result=val) for (key, val) in expected.items()], result)

    def test_case_insensitive_grouping_column(self):
        if False:
            return 10

        def my_pandas_udf(pdf):
            if False:
                return 10
            return pdf.assign(score=0.5)
        df = self.spark.createDataFrame([[1, 1]], ['column', 'score'])
        row = df.groupby('COLUMN').applyInPandas(my_pandas_udf, schema='column integer, score float').first()
        self.assertEqual(row.asDict(), Row(column=1, score=0.5).asDict())

    def _test_apply_in_pandas(self, f, output_schema='id long, mean double'):
        if False:
            i = 10
            return i + 15
        df = self.data
        result = df.groupby('id').applyInPandas(f, schema=output_schema).sort('id', 'mean').toPandas()
        expected = df.select('id').distinct().withColumn('mean', lit(24.5)).toPandas()
        assert_frame_equal(expected, result)

    def _test_apply_in_pandas_returning_empty_dataframe(self, empty_df):
        if False:
            while True:
                i = 10
        'Tests some returned DataFrames are empty.'
        df = self.data

        def stats(key, pdf):
            if False:
                for i in range(10):
                    print('nop')
            if key[0] % 2 == 0:
                return GroupedApplyInPandasTestsMixin.stats_with_no_column_names(key, pdf)
            return empty_df
        result = df.groupby('id').applyInPandas(stats, schema='id long, mean double').sort('id', 'mean').collect()
        actual_ids = {row[0] for row in result}
        expected_ids = {row[0] for row in self.data.collect() if row[0] % 2 == 0}
        self.assertSetEqual(expected_ids, actual_ids)
        self.assertEqual(len(expected_ids), len(result))
        for row in result:
            self.assertEqual(24.5, row[1])

    def _test_apply_in_pandas_returning_empty_dataframe_error(self, empty_df, error):
        if False:
            i = 10
            return i + 15
        with QuietTest(self.sc):
            with self.assertRaisesRegex(PythonException, error):
                self._test_apply_in_pandas_returning_empty_dataframe(empty_df)

class GroupedApplyInPandasTests(GroupedApplyInPandasTestsMixin, ReusedSQLTestCase):
    pass
if __name__ == '__main__':
    from pyspark.sql.tests.pandas.test_pandas_grouped_map import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)