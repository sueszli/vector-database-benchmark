from typing import Any, Union, cast
from pandas.api.types import CategoricalDtype
from pyspark.pandas.base import column_op, IndexOpsMixin
from pyspark.pandas._typing import Dtype, IndexOpsLike, SeriesOrIndex
from pyspark.pandas.data_type_ops.base import DataTypeOps, _as_categorical_type, _as_other_type, _as_string_type, _sanitize_list_like
from pyspark.pandas.typedef import pandas_on_spark_type
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType, BooleanType, StringType
from pyspark.sql.utils import pyspark_column_op

class BinaryOps(DataTypeOps):
    """
    The class for binary operations of pandas-on-Spark objects with BinaryType.
    """

    @property
    def pretty_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'binaries'

    def add(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            while True:
                i = 10
        _sanitize_list_like(right)
        if isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, BinaryType):
            return column_op(F.concat)(left, right)
        elif isinstance(right, bytes):
            return column_op(F.concat)(left, F.lit(right))
        else:
            raise TypeError('Concatenation can not be applied to %s and the given type.' % self.pretty_name)

    def radd(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            print('Hello World!')
        _sanitize_list_like(right)
        if isinstance(right, bytes):
            return cast(SeriesOrIndex, left._with_new_scol(F.concat(F.lit(right), left.spark.column)))
        else:
            raise TypeError('Concatenation can not be applied to %s and the given type.' % self.pretty_name)

    def lt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            while True:
                i = 10
        _sanitize_list_like(right)
        return pyspark_column_op('__lt__', left, right)

    def le(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            i = 10
            return i + 15
        _sanitize_list_like(right)
        return pyspark_column_op('__le__', left, right)

    def ge(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            return 10
        _sanitize_list_like(right)
        return pyspark_column_op('__ge__', left, right)

    def gt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            i = 10
            return i + 15
        _sanitize_list_like(right)
        return pyspark_column_op('__gt__', left, right)

    def astype(self, index_ops: IndexOpsLike, dtype: Union[str, type, Dtype]) -> IndexOpsLike:
        if False:
            print('Hello World!')
        (dtype, spark_type) = pandas_on_spark_type(dtype)
        if isinstance(dtype, CategoricalDtype):
            return _as_categorical_type(index_ops, dtype, spark_type)
        elif isinstance(spark_type, BooleanType):
            return index_ops.astype(str).astype(bool)
        elif isinstance(spark_type, StringType):
            return _as_string_type(index_ops, dtype)
        else:
            return _as_other_type(index_ops, dtype, spark_type)