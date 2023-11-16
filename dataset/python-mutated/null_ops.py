from typing import Any, Union
from pandas.api.types import CategoricalDtype, is_list_like
from pyspark.pandas._typing import Dtype, IndexOpsLike
from pyspark.pandas.data_type_ops.base import DataTypeOps, _as_bool_type, _as_categorical_type, _as_other_type, _as_string_type, _sanitize_list_like
from pyspark.pandas._typing import SeriesOrIndex
from pyspark.pandas.typedef import pandas_on_spark_type
from pyspark.sql.types import BooleanType, StringType
from pyspark.sql.utils import pyspark_column_op
from pyspark.pandas.base import IndexOpsMixin

class NullOps(DataTypeOps):
    """
    The class for binary operations of pandas-on-Spark objects with Spark type: NullType.
    """

    @property
    def pretty_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'nulls'

    def eq(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            print('Hello World!')
        if not isinstance(right, IndexOpsMixin) and is_list_like(right):
            return super().eq(left, right)
        return pyspark_column_op('__eq__', left, right, fillna=False)

    def ne(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            print('Hello World!')
        _sanitize_list_like(right)
        return pyspark_column_op('__ne__', left, right, fillna=True)

    def lt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            while True:
                i = 10
        _sanitize_list_like(right)
        return pyspark_column_op('__lt__', left, right, fillna=False)

    def le(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            i = 10
            return i + 15
        _sanitize_list_like(right)
        return pyspark_column_op('__le__', left, right, fillna=False)

    def ge(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            while True:
                i = 10
        _sanitize_list_like(right)
        return pyspark_column_op('__ge__', left, right, fillna=False)

    def gt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            print('Hello World!')
        _sanitize_list_like(right)
        return pyspark_column_op('__gt__', left, right, fillna=False)

    def astype(self, index_ops: IndexOpsLike, dtype: Union[str, type, Dtype]) -> IndexOpsLike:
        if False:
            for i in range(10):
                print('nop')
        (dtype, spark_type) = pandas_on_spark_type(dtype)
        if isinstance(dtype, CategoricalDtype):
            return _as_categorical_type(index_ops, dtype, spark_type)
        elif isinstance(spark_type, BooleanType):
            return _as_bool_type(index_ops, dtype)
        elif isinstance(spark_type, StringType):
            return _as_string_type(index_ops, dtype)
        else:
            return _as_other_type(index_ops, dtype, spark_type)