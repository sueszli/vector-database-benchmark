import unittest
from typing import cast
from pyspark.sql.functions import array, explode, col, lit, udf, pandas_udf, sum
from pyspark.sql.types import ArrayType, DoubleType, LongType, StructType, StructField, YearMonthIntervalType, Row
from pyspark.sql.window import Window
from pyspark.errors import IllegalArgumentException, PythonException
from pyspark.testing.sqlutils import ReusedSQLTestCase, have_pandas, have_pyarrow, pandas_requirement_message, pyarrow_requirement_message
from pyspark.testing.utils import QuietTest
if have_pandas:
    import pandas as pd
    from pandas.testing import assert_frame_equal
if have_pyarrow:
    import pyarrow as pa

@unittest.skipIf(not have_pandas or not have_pyarrow, cast(str, pandas_requirement_message or pyarrow_requirement_message))
class CogroupedApplyInPandasTestsMixin:

    @property
    def data1(self):
        if False:
            while True:
                i = 10
        return self.spark.range(10).withColumn('ks', array([lit(i) for i in range(20, 30)])).withColumn('k', explode(col('ks'))).withColumn('v', col('k') * 10).drop('ks')

    @property
    def data2(self):
        if False:
            i = 10
            return i + 15
        return self.spark.range(10).withColumn('ks', array([lit(i) for i in range(20, 30)])).withColumn('k', explode(col('ks'))).withColumn('v2', col('k') * 100).drop('ks')

    def test_simple(self):
        if False:
            print('Hello World!')
        self._test_merge(self.data1, self.data2)

    def test_left_group_empty(self):
        if False:
            print('Hello World!')
        left = self.data1.where(col('id') % 2 == 0)
        self._test_merge(left, self.data2)

    def test_right_group_empty(self):
        if False:
            i = 10
            return i + 15
        right = self.data2.where(col('id') % 2 == 0)
        self._test_merge(self.data1, right)

    def test_different_schemas(self):
        if False:
            print('Hello World!')
        right = self.data2.withColumn('v3', lit('a'))
        self._test_merge(self.data1, right, output_schema='id long, k int, v int, v2 int, v3 string')

    def test_different_keys(self):
        if False:
            while True:
                i = 10
        left = self.data1
        right = self.data2

        def merge_pandas(lft, rgt):
            if False:
                i = 10
                return i + 15
            return pd.merge(lft.rename(columns={'id2': 'id'}), rgt, on=['id', 'k'])
        result = left.withColumnRenamed('id', 'id2').groupby('id2').cogroup(right.groupby('id')).applyInPandas(merge_pandas, 'id long, k int, v int, v2 int').sort(['id', 'k']).toPandas()
        left = left.toPandas()
        right = right.toPandas()
        expected = pd.merge(left, right, on=['id', 'k']).sort_values(by=['id', 'k'])
        assert_frame_equal(expected, result)

    def test_complex_group_by(self):
        if False:
            print('Hello World!')
        left = pd.DataFrame.from_dict({'id': [1, 2, 3], 'k': [5, 6, 7], 'v': [9, 10, 11]})
        right = pd.DataFrame.from_dict({'id': [11, 12, 13], 'k': [5, 6, 7], 'v2': [90, 100, 110]})
        left_gdf = self.spark.createDataFrame(left).groupby(col('id') % 2 == 0)
        right_gdf = self.spark.createDataFrame(right).groupby(col('id') % 2 == 0)

        def merge_pandas(lft, rgt):
            if False:
                i = 10
                return i + 15
            return pd.merge(lft[['k', 'v']], rgt[['k', 'v2']], on=['k'])
        result = left_gdf.cogroup(right_gdf).applyInPandas(merge_pandas, 'k long, v long, v2 long').sort(['k']).toPandas()
        expected = pd.DataFrame.from_dict({'k': [5, 6, 7], 'v': [9, 10, 11], 'v2': [90, 100, 110]})
        assert_frame_equal(expected, result)

    def test_empty_group_by(self):
        if False:
            while True:
                i = 10
        self._test_merge(self.data1, self.data2, by=[])

    def test_different_group_key_cardinality(self):
        if False:
            while True:
                i = 10
        with QuietTest(self.sc):
            self.check_different_group_key_cardinality()

    def check_different_group_key_cardinality(self):
        if False:
            print('Hello World!')
        left = self.data1
        right = self.data2

        def merge_pandas(lft, _):
            if False:
                for i in range(10):
                    print('nop')
            return lft
        with self.assertRaisesRegex(IllegalArgumentException, 'requirement failed: Cogroup keys must have same size: 2 != 1'):
            left.groupby('id', 'k').cogroup(right.groupby('id')).applyInPandas(merge_pandas, 'id long, k int, v int')

    def test_apply_in_pandas_not_returning_pandas_dataframe(self):
        if False:
            return 10
        with QuietTest(self.sc):
            self.check_apply_in_pandas_not_returning_pandas_dataframe()

    def check_apply_in_pandas_not_returning_pandas_dataframe(self):
        if False:
            i = 10
            return i + 15
        self._test_merge_error(fn=lambda lft, rgt: lft.size + rgt.size, error_class=PythonException, error_message_regex='Return type of the user-defined function should be pandas.DataFrame, but is int.')

    def test_apply_in_pandas_returning_column_names(self):
        if False:
            return 10
        self._test_merge(fn=lambda lft, rgt: pd.merge(lft, rgt, on=['id', 'k']))

    def test_apply_in_pandas_returning_no_column_names(self):
        if False:
            for i in range(10):
                print('nop')

        def merge_pandas(lft, rgt):
            if False:
                while True:
                    i = 10
            res = pd.merge(lft, rgt, on=['id', 'k'])
            res.columns = range(res.columns.size)
            return res
        self._test_merge(fn=merge_pandas)

    def test_apply_in_pandas_returning_column_names_sometimes(self):
        if False:
            for i in range(10):
                print('nop')

        def merge_pandas(lft, rgt):
            if False:
                i = 10
                return i + 15
            res = pd.merge(lft, rgt, on=['id', 'k'])
            if 0 in lft['id'] and lft['id'][0] % 2 == 0:
                return res
            res.columns = range(res.columns.size)
            return res
        self._test_merge(fn=merge_pandas)

    def test_apply_in_pandas_returning_wrong_column_names(self):
        if False:
            i = 10
            return i + 15
        with QuietTest(self.sc):
            self.check_apply_in_pandas_returning_wrong_column_names()

    def check_apply_in_pandas_returning_wrong_column_names(self):
        if False:
            i = 10
            return i + 15

        def merge_pandas(lft, rgt):
            if False:
                i = 10
                return i + 15
            if 0 in lft['id'] and lft['id'][0] % 2 == 0:
                lft['add'] = 0
            if 0 in rgt['id'] and rgt['id'][0] % 3 == 0:
                rgt['more'] = 1
            return pd.merge(lft, rgt, on=['id', 'k'])
        self._test_merge_error(fn=merge_pandas, error_class=PythonException, error_message_regex='Column names of the returned pandas.DataFrame do not match specified schema. Unexpected: add, more.\n')

    def test_apply_in_pandas_returning_no_column_names_and_wrong_amount(self):
        if False:
            return 10
        with QuietTest(self.sc):
            self.check_apply_in_pandas_returning_no_column_names_and_wrong_amount()

    def check_apply_in_pandas_returning_no_column_names_and_wrong_amount(self):
        if False:
            for i in range(10):
                print('nop')

        def merge_pandas(lft, rgt):
            if False:
                i = 10
                return i + 15
            if 0 in lft['id'] and lft['id'][0] % 2 == 0:
                lft[3] = 0
            if 0 in rgt['id'] and rgt['id'][0] % 3 == 0:
                rgt[3] = 1
            res = pd.merge(lft, rgt, on=['id', 'k'])
            res.columns = range(res.columns.size)
            return res
        self._test_merge_error(fn=merge_pandas, error_class=PythonException, error_message_regex="Number of columns of the returned pandas.DataFrame doesn't match specified schema. Expected: 4 Actual: 6\n")

    def test_apply_in_pandas_returning_empty_dataframe(self):
        if False:
            return 10

        def merge_pandas(lft, rgt):
            if False:
                print('Hello World!')
            if 0 in lft['id'] and lft['id'][0] % 2 == 0:
                return pd.DataFrame()
            if 0 in rgt['id'] and rgt['id'][0] % 3 == 0:
                return pd.DataFrame()
            return pd.merge(lft, rgt, on=['id', 'k'])
        self._test_merge_empty(fn=merge_pandas)

    def test_apply_in_pandas_returning_incompatible_type(self):
        if False:
            return 10
        with QuietTest(self.sc):
            self.check_apply_in_pandas_returning_incompatible_type()

    def check_apply_in_pandas_returning_incompatible_type(self):
        if False:
            return 10
        for safely in [True, False]:
            with self.subTest(convertToArrowArraySafely=safely), self.sql_conf({'spark.sql.execution.pandas.convertToArrowArraySafely': safely}):
                with self.subTest(convert='string to double'):
                    expected = "ValueError: Exception thrown when converting pandas.Series \\(object\\) with name 'k' to Arrow Array \\(double\\)."
                    if safely:
                        expected = expected + ' It can be caused by overflows or other unsafe conversions warned by Arrow. Arrow safe type check can be disabled by using SQL config `spark.sql.execution.pandas.convertToArrowArraySafely`.'
                    self._test_merge_error(fn=lambda lft, rgt: pd.DataFrame({'id': [1], 'k': ['2.0']}), output_schema='id long, k double', error_class=PythonException, error_message_regex=expected)
                with self.subTest(convert='double to string'):
                    expected = "TypeError: Exception thrown when converting pandas.Series \\(float64\\) with name 'k' to Arrow Array \\(string\\).\\n"
                    self._test_merge_error(fn=lambda lft, rgt: pd.DataFrame({'id': [1], 'k': [2.0]}), output_schema='id long, k string', error_class=PythonException, error_message_regex=expected)

    def test_mixed_scalar_udfs_followed_by_cogrouby_apply(self):
        if False:
            while True:
                i = 10
        df = self.spark.range(0, 10).toDF('v1')
        df = df.withColumn('v2', udf(lambda x: x + 1, 'int')(df['v1'])).withColumn('v3', pandas_udf(lambda x: x + 2, 'int')(df['v1']))
        result = df.groupby().cogroup(df.groupby()).applyInPandas(lambda x, y: pd.DataFrame([(x.sum().sum(), y.sum().sum())]), 'sum1 int, sum2 int').collect()
        self.assertEqual(result[0]['sum1'], 165)
        self.assertEqual(result[0]['sum2'], 165)

    def test_with_key_left(self):
        if False:
            i = 10
            return i + 15
        self._test_with_key(self.data1, self.data1, isLeft=True)

    def test_with_key_right(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_with_key(self.data1, self.data1, isLeft=False)

    def test_with_key_left_group_empty(self):
        if False:
            i = 10
            return i + 15
        left = self.data1.where(col('id') % 2 == 0)
        self._test_with_key(left, self.data1, isLeft=True)

    def test_with_key_right_group_empty(self):
        if False:
            for i in range(10):
                print('nop')
        right = self.data1.where(col('id') % 2 == 0)
        self._test_with_key(self.data1, right, isLeft=False)

    def test_with_key_complex(self):
        if False:
            print('Hello World!')

        def left_assign_key(key, lft, _):
            if False:
                return 10
            return lft.assign(key=key[0])
        result = self.data1.groupby(col('id') % 2 == 0).cogroup(self.data2.groupby(col('id') % 2 == 0)).applyInPandas(left_assign_key, 'id long, k int, v int, key boolean').sort(['id', 'k']).toPandas()
        expected = self.data1.toPandas()
        expected = expected.assign(key=expected.id % 2 == 0)
        assert_frame_equal(expected, result)

    def test_wrong_return_type(self):
        if False:
            while True:
                i = 10
        with QuietTest(self.sc):
            self.check_wrong_return_type()

    def check_wrong_return_type(self):
        if False:
            print('Hello World!')
        self._test_merge_error(fn=lambda l, r: l, output_schema=StructType().add('id', LongType()).add('v', ArrayType(YearMonthIntervalType())), error_class=NotImplementedError, error_message_regex='Invalid return type.*ArrayType.*YearMonthIntervalType')

    def test_wrong_args(self):
        if False:
            return 10
        with QuietTest(self.sc):
            self.check_wrong_args()

    def check_wrong_args(self):
        if False:
            for i in range(10):
                print('nop')
        self.__test_merge_error(fn=lambda : 1, output_schema=StructType([StructField('d', DoubleType())]), error_class=ValueError, error_message_regex='Invalid function')

    def test_case_insensitive_grouping_column(self):
        if False:
            print('Hello World!')
        df1 = self.spark.createDataFrame([(1, 1)], ('column', 'value'))
        row = df1.groupby('ColUmn').cogroup(df1.groupby('COLUMN')).applyInPandas(lambda r, l: r + l, 'column long, value long').first()
        self.assertEqual(row.asDict(), Row(column=2, value=2).asDict())
        df2 = self.spark.createDataFrame([(1, 1)], ('column', 'value'))
        row = df1.groupby('ColUmn').cogroup(df2.groupby('COLUMN')).applyInPandas(lambda r, l: r + l, 'column long, value long').first()
        self.assertEqual(row.asDict(), Row(column=2, value=2).asDict())

    def test_self_join(self):
        if False:
            while True:
                i = 10
        df = self.spark.createDataFrame([(1, 1)], ('column', 'value'))
        row = df.groupby('ColUmn').cogroup(df.groupby('COLUMN')).applyInPandas(lambda r, l: r + l, 'column long, value long')
        row = row.join(row).first()
        self.assertEqual(row.asDict(), Row(column=2, value=2).asDict())

    def test_with_window_function(self):
        if False:
            i = 10
            return i + 15
        ids = 2
        days = 100
        vals = 10000
        parts = 10
        id_df = self.spark.range(ids)
        day_df = self.spark.range(days).withColumnRenamed('id', 'day')
        vals_df = self.spark.range(vals).withColumnRenamed('id', 'value')
        df = id_df.join(day_df).join(vals_df)
        left_df = df.withColumnRenamed('value', 'left').repartition(parts).cache()
        right_df = df.select(col('id').alias('id'), col('day').alias('day'), col('value').alias('right')).repartition(parts).cache()
        window = Window.partitionBy('day', 'id')
        left_grouped_df = left_df.groupBy('id', 'day')
        right_grouped_df = right_df.withColumn('day_sum', sum(col('day')).over(window)).groupBy('id', 'day')

        def cogroup(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
            if False:
                return 10
            return pd.DataFrame([{'id': left['id'][0] if not left.empty else right['id'][0] if not right.empty else None, 'day': left['day'][0] if not left.empty else right['day'][0] if not right.empty else None, 'lefts': len(left.index), 'rights': len(right.index)}])
        df = left_grouped_df.cogroup(right_grouped_df).applyInPandas(cogroup, schema='id long, day long, lefts integer, rights integer')
        actual = df.orderBy('id', 'day').take(days)
        self.assertEqual(actual, [Row(0, day, vals, vals) for day in range(days)])

    @staticmethod
    def _test_with_key(left, right, isLeft):
        if False:
            for i in range(10):
                print('nop')

        def right_assign_key(key, lft, rgt):
            if False:
                return 10
            return lft.assign(key=key[0]) if isLeft else rgt.assign(key=key[0])
        result = left.groupby('id').cogroup(right.groupby('id')).applyInPandas(right_assign_key, 'id long, k int, v int, key long').toPandas()
        expected = left.toPandas() if isLeft else right.toPandas()
        expected = expected.assign(key=expected.id)
        assert_frame_equal(expected, result)

    def _test_merge_empty(self, fn):
        if False:
            print('Hello World!')
        left = self.data1.toPandas()
        right = self.data2.toPandas()
        expected = pd.merge(left[left['id'] % 2 != 0], right[right['id'] % 3 != 0], on=['id', 'k']).sort_values(by=['id', 'k'])
        self._test_merge(self.data1, self.data2, fn=fn, expected=expected)

    def _test_merge(self, left=None, right=None, by=['id'], fn=lambda lft, rgt: pd.merge(lft, rgt, on=['id', 'k']), output_schema='id long, k int, v int, v2 int', expected=None):
        if False:
            print('Hello World!')

        def fn_with_key(_, lft, rgt):
            if False:
                for i in range(10):
                    print('nop')
            return fn(lft, rgt)
        with self.subTest('without key'):
            self.__test_merge(left, right, by, fn, output_schema, expected)
        with self.subTest('with key'):
            self.__test_merge(left, right, by, fn_with_key, output_schema, expected)

    def __test_merge(self, left=None, right=None, by=['id'], fn=lambda lft, rgt: pd.merge(lft, rgt, on=['id', 'k']), output_schema='id long, k int, v int, v2 int', expected=None):
        if False:
            i = 10
            return i + 15
        left = self.data1 if left is None else left
        right = self.data2 if right is None else right
        result = left.groupby(*by).cogroup(right.groupby(*by)).applyInPandas(fn, output_schema).sort(['id', 'k']).toPandas()
        left = left.toPandas()
        right = right.toPandas()
        expected = pd.merge(left, right, on=['id', 'k']).sort_values(by=['id', 'k']) if expected is None else expected
        assert_frame_equal(expected, result)

    def _test_merge_error(self, error_class, error_message_regex, left=None, right=None, by=['id'], fn=lambda lft, rgt: pd.merge(lft, rgt, on=['id', 'k']), output_schema='id long, k int, v int, v2 int'):
        if False:
            while True:
                i = 10

        def fn_with_key(_, lft, rgt):
            if False:
                while True:
                    i = 10
            return fn(lft, rgt)
        with self.subTest('without key'):
            self.__test_merge_error(left=left, right=right, by=by, fn=fn, output_schema=output_schema, error_class=error_class, error_message_regex=error_message_regex)
        with self.subTest('with key'):
            self.__test_merge_error(left=left, right=right, by=by, fn=fn_with_key, output_schema=output_schema, error_class=error_class, error_message_regex=error_message_regex)

    def __test_merge_error(self, error_class, error_message_regex, left=None, right=None, by=['id'], fn=lambda lft, rgt: pd.merge(lft, rgt, on=['id', 'k']), output_schema='id long, k int, v int, v2 int'):
        if False:
            return 10
        with self.assertRaisesRegex(error_class, error_message_regex):
            self.__test_merge(left, right, by, fn, output_schema)

class CogroupedApplyInPandasTests(CogroupedApplyInPandasTestsMixin, ReusedSQLTestCase):
    pass
if __name__ == '__main__':
    from pyspark.sql.tests.pandas.test_pandas_cogrouped_map import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)