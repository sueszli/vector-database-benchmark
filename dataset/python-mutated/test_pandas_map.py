import os
import shutil
import tempfile
import time
import unittest
from typing import cast
from pyspark.sql import Row
from pyspark.sql.functions import col, encode, lit
from pyspark.errors import PythonException
from pyspark.testing.sqlutils import ReusedSQLTestCase, have_pandas, have_pyarrow, pandas_requirement_message, pyarrow_requirement_message
from pyspark.testing.utils import QuietTest
if have_pandas:
    import pandas as pd

@unittest.skipIf(not have_pandas or not have_pyarrow, cast(str, pandas_requirement_message or pyarrow_requirement_message))
class MapInPandasTestsMixin:

    @staticmethod
    def identity_dataframes_iter(*columns: str):
        if False:
            i = 10
            return i + 15

        def func(iterator):
            if False:
                for i in range(10):
                    print('nop')
            for pdf in iterator:
                assert isinstance(pdf, pd.DataFrame)
                assert pdf.columns.tolist() == list(columns)
                yield pdf
        return func

    @staticmethod
    def identity_dataframes_wo_column_names_iter(*columns: str):
        if False:
            print('Hello World!')

        def func(iterator):
            if False:
                return 10
            for pdf in iterator:
                assert isinstance(pdf, pd.DataFrame)
                assert pdf.columns.tolist() == list(columns)
                yield pdf.rename(columns=list(pdf.columns).index)
        return func

    @staticmethod
    def dataframes_and_empty_dataframe_iter(*columns: str):
        if False:
            return 10

        def func(iterator):
            if False:
                for i in range(10):
                    print('nop')
            for pdf in iterator:
                yield pdf
            yield pd.DataFrame([], columns=list(columns))
        return func

    def test_map_in_pandas(self):
        if False:
            i = 10
            return i + 15
        df = self.spark.range(10, numPartitions=3)
        actual = df.mapInPandas(self.identity_dataframes_iter('id'), 'id long').collect()
        expected = df.collect()
        self.assertEqual(actual, expected)
        df = self.spark.range(10, numPartitions=3)
        actual = df.mapInPandas(lambda it: [pdf for pdf in it], 'id long').collect()
        expected = df.collect()
        self.assertEqual(actual, expected)

    def test_multiple_columns(self):
        if False:
            print('Hello World!')
        data = [(1, 'foo'), (2, None), (3, 'bar'), (4, 'bar')]
        df = self.spark.createDataFrame(data, 'a int, b string')

        def func(iterator):
            if False:
                return 10
            for pdf in iterator:
                assert isinstance(pdf, pd.DataFrame)
                assert [d.name for d in list(pdf.dtypes)] == ['int32', 'object']
                yield pdf
        actual = df.mapInPandas(func, df.schema).collect()
        expected = df.collect()
        self.assertEqual(actual, expected)

    def test_large_variable_types(self):
        if False:
            return 10
        with self.sql_conf({'spark.sql.execution.arrow.useLargeVarTypes': True}):

            def func(iterator):
                if False:
                    for i in range(10):
                        print('nop')
                for pdf in iterator:
                    assert isinstance(pdf, pd.DataFrame)
                    yield pdf
            df = self.spark.range(10, numPartitions=3).select(col('id').cast('string').alias('str')).withColumn('bin', encode(col('str'), 'utf8'))
            actual = df.mapInPandas(func, 'str string, bin binary').collect()
            expected = df.collect()
            self.assertEqual(actual, expected)

    def test_no_column_names(self):
        if False:
            i = 10
            return i + 15
        data = [(1, 'foo'), (2, None), (3, 'bar'), (4, 'bar')]
        df = self.spark.createDataFrame(data, 'a int, b string')

        def func(iterator):
            if False:
                for i in range(10):
                    print('nop')
            for pdf in iterator:
                yield pdf.rename(columns=list(pdf.columns).index)
        actual = df.mapInPandas(func, df.schema).collect()
        expected = df.collect()
        self.assertEqual(actual, expected)

    def test_different_output_length(self):
        if False:
            return 10

        def func(iterator):
            if False:
                print('Hello World!')
            for _ in iterator:
                yield pd.DataFrame({'a': list(range(100))})
        df = self.spark.range(10)
        actual = df.repartition(1).mapInPandas(func, 'a long').collect()
        self.assertEqual(set((r.a for r in actual)), set(range(100)))

    def test_other_than_dataframe_iter(self):
        if False:
            print('Hello World!')
        with QuietTest(self.sc):
            self.check_other_than_dataframe_iter()

    def check_other_than_dataframe_iter(self):
        if False:
            while True:
                i = 10

        def no_iter(_):
            if False:
                i = 10
                return i + 15
            return 1

        def bad_iter_elem(_):
            if False:
                return 10
            return iter([1])
        with self.assertRaisesRegex(PythonException, 'Return type of the user-defined function should be iterator of pandas.DataFrame, but is int.'):
            self.spark.range(10, numPartitions=3).mapInPandas(no_iter, 'a int').count()
        with self.assertRaisesRegex(PythonException, 'Return type of the user-defined function should be iterator of pandas.DataFrame, but is iterator of int.'):
            self.spark.range(10, numPartitions=3).mapInPandas(bad_iter_elem, 'a int').count()

    def test_dataframes_with_other_column_names(self):
        if False:
            while True:
                i = 10
        with QuietTest(self.sc):
            self.check_dataframes_with_other_column_names()

    def check_dataframes_with_other_column_names(self):
        if False:
            for i in range(10):
                print('nop')

        def dataframes_with_other_column_names(iterator):
            if False:
                i = 10
                return i + 15
            for pdf in iterator:
                yield pdf.rename(columns={'id': 'iid'})
        with self.assertRaisesRegex(PythonException, 'PySparkRuntimeError: \\[RESULT_COLUMNS_MISMATCH_FOR_PANDAS_UDF\\] Column names of the returned pandas.DataFrame do not match specified schema. Missing: id. Unexpected: iid.\n'):
            self.spark.range(10, numPartitions=3).withColumn('value', lit(0)).mapInPandas(dataframes_with_other_column_names, 'id int, value int').collect()

    def test_dataframes_with_duplicate_column_names(self):
        if False:
            i = 10
            return i + 15
        with QuietTest(self.sc):
            self.check_dataframes_with_duplicate_column_names()

    def check_dataframes_with_duplicate_column_names(self):
        if False:
            return 10

        def dataframes_with_other_column_names(iterator):
            if False:
                i = 10
                return i + 15
            for pdf in iterator:
                yield pdf.rename(columns={'id2': 'id'})
        with self.assertRaisesRegex(PythonException, 'PySparkRuntimeError: \\[RESULT_COLUMNS_MISMATCH_FOR_PANDAS_UDF\\] Column names of the returned pandas.DataFrame do not match specified schema. Missing: id2.\n'):
            self.spark.range(10, numPartitions=3).withColumn('id2', lit(0)).withColumn('value', lit(1)).mapInPandas(dataframes_with_other_column_names, 'id int, id2 long, value int').collect()

    def test_dataframes_with_less_columns(self):
        if False:
            return 10
        with QuietTest(self.sc):
            self.check_dataframes_with_less_columns()

    def check_dataframes_with_less_columns(self):
        if False:
            print('Hello World!')
        df = self.spark.range(10, numPartitions=3).withColumn('value', lit(0))
        with self.assertRaisesRegex(PythonException, 'PySparkRuntimeError: \\[RESULT_COLUMNS_MISMATCH_FOR_PANDAS_UDF\\] Column names of the returned pandas.DataFrame do not match specified schema. Missing: id2.\n'):
            f = self.identity_dataframes_iter('id', 'value')
            df.mapInPandas(f, 'id int, id2 long, value int').collect()
        with self.assertRaisesRegex(PythonException, "PySparkRuntimeError: \\[RESULT_LENGTH_MISMATCH_FOR_PANDAS_UDF\\] Number of columns of the returned pandas.DataFrame doesn't match specified schema. Expected: 3 Actual: 2\n"):
            f = self.identity_dataframes_wo_column_names_iter('id', 'value')
            df.mapInPandas(f, 'id int, id2 long, value int').collect()

    def test_dataframes_with_more_columns(self):
        if False:
            i = 10
            return i + 15
        df = self.spark.range(10, numPartitions=3).select('id', col('id').alias('value'), col('id').alias('extra'))
        expected = df.select('id', 'value').collect()
        f = self.identity_dataframes_iter('id', 'value', 'extra')
        actual = df.repartition(1).mapInPandas(f, 'id long, value long').collect()
        self.assertEqual(actual, expected)
        f = self.identity_dataframes_wo_column_names_iter('id', 'value', 'extra')
        actual = df.repartition(1).mapInPandas(f, 'id long, value long').collect()
        self.assertEqual(actual, expected)

    def test_dataframes_with_incompatible_types(self):
        if False:
            return 10
        with QuietTest(self.sc):
            self.check_dataframes_with_incompatible_types()

    def check_dataframes_with_incompatible_types(self):
        if False:
            print('Hello World!')

        def func(iterator):
            if False:
                i = 10
                return i + 15
            for pdf in iterator:
                yield pdf.assign(id=pdf['id'].apply(str))
        for safely in [True, False]:
            with self.subTest(convertToArrowArraySafely=safely), self.sql_conf({'spark.sql.execution.pandas.convertToArrowArraySafely': safely}):
                with self.subTest(convert='string to double'):
                    expected = "ValueError: Exception thrown when converting pandas.Series \\(object\\) with name 'id' to Arrow Array \\(double\\)."
                    if safely:
                        expected = expected + ' It can be caused by overflows or other unsafe conversions warned by Arrow. Arrow safe type check can be disabled by using SQL config `spark.sql.execution.pandas.convertToArrowArraySafely`.'
                    with self.assertRaisesRegex(PythonException, expected + '\n'):
                        self.spark.range(10, numPartitions=3).mapInPandas(func, 'id double').collect()
                with self.subTest(convert='double to string'):
                    with self.assertRaisesRegex(PythonException, "TypeError: Exception thrown when converting pandas.Series \\(float64\\) with name 'id' to Arrow Array \\(string\\).\\n"):
                        self.spark.range(10, numPartitions=3).select(col('id').cast('double')).mapInPandas(self.identity_dataframes_iter('id'), 'id string').collect()

    def test_empty_iterator(self):
        if False:
            print('Hello World!')

        def empty_iter(_):
            if False:
                i = 10
                return i + 15
            return iter([])
        mapped = self.spark.range(10, numPartitions=3).mapInPandas(empty_iter, 'a int, b string')
        self.assertEqual(mapped.count(), 0)

    def test_empty_dataframes(self):
        if False:
            i = 10
            return i + 15

        def empty_dataframes(_):
            if False:
                return 10
            return iter([pd.DataFrame({'a': []})])
        mapped = self.spark.range(10, numPartitions=3).mapInPandas(empty_dataframes, 'a int')
        self.assertEqual(mapped.count(), 0)

    def test_empty_dataframes_without_columns(self):
        if False:
            print('Hello World!')
        mapped = self.spark.range(10, numPartitions=3).mapInPandas(self.dataframes_and_empty_dataframe_iter(), 'id int')
        self.assertEqual(mapped.count(), 10)

    def test_empty_dataframes_with_less_columns(self):
        if False:
            i = 10
            return i + 15
        with QuietTest(self.sc):
            self.check_empty_dataframes_with_less_columns()

    def check_empty_dataframes_with_less_columns(self):
        if False:
            return 10
        with self.assertRaisesRegex(PythonException, 'PySparkRuntimeError: \\[RESULT_COLUMNS_MISMATCH_FOR_PANDAS_UDF\\] Column names of the returned pandas.DataFrame do not match specified schema. Missing: value.\n'):
            f = self.dataframes_and_empty_dataframe_iter('id')
            self.spark.range(10, numPartitions=3).withColumn('value', lit(0)).mapInPandas(f, 'id int, value int').collect()

    def test_empty_dataframes_with_more_columns(self):
        if False:
            return 10
        mapped = self.spark.range(10, numPartitions=3).mapInPandas(self.dataframes_and_empty_dataframe_iter('id', 'extra'), 'id int')
        self.assertEqual(mapped.count(), 10)

    def test_empty_dataframes_with_other_columns(self):
        if False:
            i = 10
            return i + 15
        with QuietTest(self.sc):
            self.check_empty_dataframes_with_other_columns()

    def check_empty_dataframes_with_other_columns(self):
        if False:
            for i in range(10):
                print('nop')

        def empty_dataframes_with_other_columns(iterator):
            if False:
                for i in range(10):
                    print('nop')
            for _ in iterator:
                yield pd.DataFrame({'iid': [], 'value': []})
        with self.assertRaisesRegex(PythonException, 'PySparkRuntimeError: \\[RESULT_COLUMNS_MISMATCH_FOR_PANDAS_UDF\\] Column names of the returned pandas.DataFrame do not match specified schema. Missing: id. Unexpected: iid.\n'):
            self.spark.range(10, numPartitions=3).withColumn('value', lit(0)).mapInPandas(empty_dataframes_with_other_columns, 'id int, value int').collect()

    def test_chain_map_partitions_in_pandas(self):
        if False:
            while True:
                i = 10

        def func(iterator):
            if False:
                print('Hello World!')
            for pdf in iterator:
                assert isinstance(pdf, pd.DataFrame)
                assert pdf.columns == ['id']
                yield pdf
        df = self.spark.range(10, numPartitions=3)
        actual = df.mapInPandas(func, 'id long').mapInPandas(func, 'id long').collect()
        expected = df.collect()
        self.assertEqual(actual, expected)

    def test_self_join(self):
        if False:
            while True:
                i = 10
        df1 = self.spark.range(10, numPartitions=3)
        df2 = df1.mapInPandas(lambda iter: iter, 'id long')
        actual = df2.join(df2).collect()
        expected = df1.join(df1).collect()
        self.assertEqual(sorted(actual), sorted(expected))

    def test_map_in_pandas_with_column_vector(self):
        if False:
            return 10
        path = tempfile.mkdtemp()
        shutil.rmtree(path)
        try:
            self.spark.range(0, 200000, 1, 1).write.parquet(path)

            def func(iterator):
                if False:
                    while True:
                        i = 10
                for pdf in iterator:
                    yield pd.DataFrame({'id': [0] * len(pdf)})
            for offheap in ['true', 'false']:
                with self.sql_conf({'spark.sql.columnVector.offheap.enabled': offheap}):
                    self.assertEquals(self.spark.read.parquet(path).mapInPandas(func, 'id long').head(), Row(0))
        finally:
            shutil.rmtree(path)

class MapInPandasTests(ReusedSQLTestCase, MapInPandasTestsMixin):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        ReusedSQLTestCase.setUpClass()
        cls.tz_prev = os.environ.get('TZ', None)
        tz = 'America/Los_Angeles'
        os.environ['TZ'] = tz
        time.tzset()
        cls.sc.environment['TZ'] = tz
        cls.spark.conf.set('spark.sql.session.timeZone', tz)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        del os.environ['TZ']
        if cls.tz_prev is not None:
            os.environ['TZ'] = cls.tz_prev
        time.tzset()
        ReusedSQLTestCase.tearDownClass()
if __name__ == '__main__':
    from pyspark.sql.tests.pandas.test_pandas_map import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)