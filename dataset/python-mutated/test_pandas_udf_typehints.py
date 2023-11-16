import sys
import unittest
from inspect import signature
from typing import Union, Iterator, Tuple, cast, get_type_hints
from pyspark.sql.functions import mean, lit
from pyspark.testing.sqlutils import ReusedSQLTestCase, have_pandas, have_pyarrow, pandas_requirement_message, pyarrow_requirement_message
from pyspark.sql.pandas.typehints import infer_eval_type
from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType
from pyspark.sql import Row
if have_pandas:
    import pandas as pd
    import numpy as np
    from pandas.testing import assert_frame_equal

@unittest.skipIf(not have_pandas or not have_pyarrow, cast(str, pandas_requirement_message or pyarrow_requirement_message))
class PandasUDFTypeHintsTests(ReusedSQLTestCase):

    def test_type_annotation_scalar(self):
        if False:
            i = 10
            return i + 15

        def func(col: pd.Series) -> pd.Series:
            if False:
                while True:
                    i = 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: pd.DataFrame, col1: pd.Series) -> pd.DataFrame:
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: pd.DataFrame, *args: pd.Series) -> pd.Series:
            if False:
                i = 10
                return i + 15
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: pd.Series, *args: pd.Series, **kwargs: pd.DataFrame) -> pd.Series:
            if False:
                while True:
                    i = 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: pd.Series, *, col2: pd.DataFrame) -> pd.DataFrame:
            if False:
                i = 10
                return i + 15
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: Union[pd.Series, pd.DataFrame], *, col2: pd.DataFrame) -> pd.Series:
            if False:
                return 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

    def test_type_annotation_scalar_iter(self):
        if False:
            i = 10
            return i + 15

        def func(iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
            if False:
                i = 10
                return i + 15
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR_ITER)

        def func(iter: Iterator[Tuple[pd.DataFrame, pd.Series]]) -> Iterator[pd.DataFrame]:
            if False:
                return 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR_ITER)

        def func(iter: Iterator[Tuple[pd.DataFrame, ...]]) -> Iterator[pd.Series]:
            if False:
                return 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR_ITER)

        def func(iter: Iterator[Tuple[Union[pd.DataFrame, pd.Series], ...]]) -> Iterator[pd.Series]:
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR_ITER)

    @unittest.skipIf(sys.version_info < (3, 9), 'Type hinting generics require Python 3.9.')
    def test_type_annotation_tuple_generics(self):
        if False:
            return 10

        def func(iter: Iterator[tuple[pd.DataFrame, pd.Series]]) -> Iterator[pd.DataFrame]:
            if False:
                print('Hello World!')
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR_ITER)

        def func(iter: Iterator[tuple[pd.DataFrame, ...]]) -> Iterator[pd.Series]:
            if False:
                print('Hello World!')
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR_ITER)

        def func(iter: Iterator[tuple[Union[pd.DataFrame, pd.Series], ...]]) -> Iterator[pd.Series]:
            if False:
                while True:
                    i = 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR_ITER)

    def test_type_annotation_group_agg(self):
        if False:
            for i in range(10):
                print('nop')

        def func(col: pd.Series) -> str:
            if False:
                return 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.GROUPED_AGG)

        def func(col: pd.DataFrame, col1: pd.Series) -> int:
            if False:
                return 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.GROUPED_AGG)

        def func(col: pd.DataFrame, *args: pd.Series) -> Row:
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.GROUPED_AGG)

        def func(col: pd.Series, *args: pd.Series, **kwargs: pd.DataFrame) -> str:
            if False:
                return 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.GROUPED_AGG)

        def func(col: pd.Series, *, col2: pd.DataFrame) -> float:
            if False:
                return 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.GROUPED_AGG)

        def func(col: Union[pd.Series, pd.DataFrame], *, col2: pd.DataFrame) -> float:
            if False:
                while True:
                    i = 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.GROUPED_AGG)

    def test_type_annotation_negative(self):
        if False:
            i = 10
            return i + 15

        def func(col: str) -> pd.Series:
            if False:
                while True:
                    i = 10
            pass
        self.assertRaisesRegex(NotImplementedError, 'Unsupported signature.*str', infer_eval_type, signature(func), get_type_hints(func))

        def func(col: pd.DataFrame, col1: int) -> pd.DataFrame:
            if False:
                print('Hello World!')
            pass
        self.assertRaisesRegex(NotImplementedError, 'Unsupported signature.*int', infer_eval_type, signature(func), get_type_hints(func))

        def func(col: Union[pd.DataFrame, str], col1: int) -> pd.DataFrame:
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.assertRaisesRegex(NotImplementedError, 'Unsupported signature.*str', infer_eval_type, signature(func), get_type_hints(func))

        def func(col: pd.Series) -> Tuple[pd.DataFrame]:
            if False:
                print('Hello World!')
            pass
        self.assertRaisesRegex(NotImplementedError, 'Unsupported signature.*Tuple', infer_eval_type, signature(func), get_type_hints(func))

        def func(col, *args: pd.Series) -> pd.Series:
            if False:
                return 10
            pass
        self.assertRaisesRegex(ValueError, 'should be specified.*Series', infer_eval_type, signature(func), get_type_hints(func))

        def func(col: pd.Series, *args: pd.Series, **kwargs: pd.DataFrame):
            if False:
                return 10
            pass
        self.assertRaisesRegex(ValueError, 'should be specified.*Series', infer_eval_type, signature(func), get_type_hints(func))

        def func(col: pd.Series, *, col2) -> pd.DataFrame:
            if False:
                print('Hello World!')
            pass
        self.assertRaisesRegex(ValueError, 'should be specified.*Series', infer_eval_type, signature(func), get_type_hints(func))

    def test_scalar_udf_type_hint(self):
        if False:
            while True:
                i = 10
        df = self.spark.range(10).selectExpr('id', 'id as v')

        def plus_one(v: Union[pd.Series, pd.DataFrame]) -> pd.Series:
            if False:
                for i in range(10):
                    print('nop')
            return v + 1
        plus_one = pandas_udf('long')(plus_one)
        actual = df.select(plus_one(df.v).alias('plus_one'))
        expected = df.selectExpr('(v + 1) as plus_one')
        assert_frame_equal(expected.toPandas(), actual.toPandas())

    def test_scalar_iter_udf_type_hint(self):
        if False:
            while True:
                i = 10
        df = self.spark.range(10).selectExpr('id', 'id as v')

        def plus_one(itr: Iterator[pd.Series]) -> Iterator[pd.Series]:
            if False:
                return 10
            for s in itr:
                yield (s + 1)
        plus_one = pandas_udf('long')(plus_one)
        actual = df.select(plus_one(df.v).alias('plus_one'))
        expected = df.selectExpr('(v + 1) as plus_one')
        assert_frame_equal(expected.toPandas(), actual.toPandas())

    def test_group_agg_udf_type_hint(self):
        if False:
            while True:
                i = 10
        df = self.spark.range(10).selectExpr('id', 'id as v')

        def weighted_mean(v: pd.Series, w: pd.Series) -> np.float64:
            if False:
                i = 10
                return i + 15
            return np.average(v, weights=w)
        weighted_mean = pandas_udf('double')(weighted_mean)
        actual = df.groupby('id').agg(weighted_mean(df.v, lit(1.0))).sort('id')
        expected = df.groupby('id').agg(mean(df.v).alias('weighted_mean(v, 1.0)')).sort('id')
        assert_frame_equal(expected.toPandas(), actual.toPandas())

    def test_ignore_type_hint_in_group_apply_in_pandas(self):
        if False:
            for i in range(10):
                print('nop')
        df = self.spark.range(10)

        def pandas_plus_one(v: pd.DataFrame) -> pd.DataFrame:
            if False:
                while True:
                    i = 10
            return v + 1
        actual = df.groupby('id').applyInPandas(pandas_plus_one, schema=df.schema).sort('id')
        expected = df.selectExpr('id + 1 as id')
        assert_frame_equal(expected.toPandas(), actual.toPandas())

    def test_ignore_type_hint_in_cogroup_apply_in_pandas(self):
        if False:
            return 10
        df = self.spark.range(10)

        def pandas_plus_one(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
            if False:
                print('Hello World!')
            return left + 1
        actual = df.groupby('id').cogroup(self.spark.range(10).groupby('id')).applyInPandas(pandas_plus_one, schema=df.schema).sort('id')
        expected = df.selectExpr('id + 1 as id')
        assert_frame_equal(expected.toPandas(), actual.toPandas())

    def test_ignore_type_hint_in_map_in_pandas(self):
        if False:
            for i in range(10):
                print('nop')
        df = self.spark.range(10)

        def pandas_plus_one(iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            if False:
                print('Hello World!')
            return map(lambda v: v + 1, iter)
        actual = df.mapInPandas(pandas_plus_one, schema=df.schema)
        expected = df.selectExpr('id + 1 as id')
        assert_frame_equal(expected.toPandas(), actual.toPandas())

    def test_string_type_annotation(self):
        if False:
            while True:
                i = 10

        def func(col: 'pd.Series') -> 'pd.Series':
            if False:
                while True:
                    i = 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: 'pd.DataFrame', col1: 'pd.Series') -> 'pd.DataFrame':
            if False:
                while True:
                    i = 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: 'pd.DataFrame', *args: 'pd.Series') -> 'pd.Series':
            if False:
                print('Hello World!')
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: 'pd.Series', *args: 'pd.Series', **kwargs: 'pd.DataFrame') -> 'pd.Series':
            if False:
                i = 10
                return i + 15
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: 'pd.Series', *, col2: 'pd.DataFrame') -> 'pd.DataFrame':
            if False:
                print('Hello World!')
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: Union['pd.Series', 'pd.DataFrame'], *, col2: 'pd.DataFrame') -> 'pd.Series':
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)

        def func(col: 'Union[pd.Series, pd.DataFrame]', *, col2: 'pd.DataFrame') -> 'pd.Series':
            if False:
                return 10
            pass
        self.assertEqual(infer_eval_type(signature(func), get_type_hints(func)), PandasUDFType.SCALAR)
if __name__ == '__main__':
    from pyspark.sql.tests.pandas.test_pandas_udf_typehints import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)