from typing import Any, Union, cast
from pandas.api.types import CategoricalDtype
from pyspark.pandas._typing import Dtype, IndexOpsLike, SeriesOrIndex
from pyspark.pandas.base import column_op, IndexOpsMixin
from pyspark.pandas.data_type_ops.base import DataTypeOps, _as_bool_type, _as_categorical_type, _as_other_type, _as_string_type, _sanitize_list_like
from pyspark.pandas.typedef import pandas_on_spark_type
from pyspark.sql import functions as F, Column
from pyspark.sql.types import ArrayType, BooleanType, NumericType, StringType

class ArrayOps(DataTypeOps):
    """
    The class for binary operations of pandas-on-Spark objects with ArrayType.
    """

    @property
    def pretty_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'arrays'

    def add(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            for i in range(10):
                print('nop')
        _sanitize_list_like(right)
        if not isinstance(right, IndexOpsMixin) or (isinstance(right, IndexOpsMixin) and (not isinstance(right.spark.data_type, ArrayType))):
            raise TypeError('Concatenation can not be applied to %s and the given type.' % self.pretty_name)
        left_type = cast(ArrayType, left.spark.data_type).elementType
        right_type = right.spark.data_type.elementType
        if left_type != right_type and (not (isinstance(left_type, NumericType) and isinstance(right_type, NumericType))):
            raise TypeError('Concatenation can only be applied to %s of the same type' % self.pretty_name)
        return column_op(F.concat)(left, right)

    def lt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            while True:
                i = 10
        from pyspark.pandas.base import column_op
        _sanitize_list_like(right)
        return column_op(Column.__lt__)(left, right)

    def le(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            while True:
                i = 10
        from pyspark.pandas.base import column_op
        _sanitize_list_like(right)
        return column_op(Column.__le__)(left, right)

    def ge(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            i = 10
            return i + 15
        from pyspark.pandas.base import column_op
        _sanitize_list_like(right)
        return column_op(Column.__ge__)(left, right)

    def gt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            return 10
        from pyspark.pandas.base import column_op
        _sanitize_list_like(right)
        return column_op(Column.__gt__)(left, right)

    def astype(self, index_ops: IndexOpsLike, dtype: Union[str, type, Dtype]) -> IndexOpsLike:
        if False:
            print('Hello World!')
        (dtype, spark_type) = pandas_on_spark_type(dtype)
        if isinstance(dtype, CategoricalDtype):
            return _as_categorical_type(index_ops, dtype, spark_type)
        elif isinstance(spark_type, BooleanType):
            return _as_bool_type(index_ops, dtype)
        elif isinstance(spark_type, StringType):
            return _as_string_type(index_ops, dtype)
        else:
            return _as_other_type(index_ops, dtype, spark_type)

class MapOps(DataTypeOps):
    """
    The class for binary operations of pandas-on-Spark objects with MapType.
    """

    @property
    def pretty_name(self) -> str:
        if False:
            print('Hello World!')
        return 'maps'

class StructOps(DataTypeOps):
    """
    The class for binary operations of pandas-on-Spark objects with StructType.
    """

    @property
    def pretty_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'structs'

    def lt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            i = 10
            return i + 15
        from pyspark.pandas.base import column_op
        _sanitize_list_like(right)
        return column_op(Column.__lt__)(left, right)

    def le(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            print('Hello World!')
        from pyspark.pandas.base import column_op
        _sanitize_list_like(right)
        return column_op(Column.__le__)(left, right)

    def ge(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            while True:
                i = 10
        from pyspark.pandas.base import column_op
        _sanitize_list_like(right)
        return column_op(Column.__ge__)(left, right)

    def gt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            return 10
        from pyspark.pandas.base import column_op
        _sanitize_list_like(right)
        return column_op(Column.__gt__)(left, right)