from typing import cast, Any, Union
import pandas as pd
import numpy as np
from pandas.api.types import is_list_like, CategoricalDtype
from pyspark.pandas._typing import Dtype, IndexOpsLike, SeriesOrIndex
from pyspark.pandas.base import IndexOpsMixin
from pyspark.pandas.data_type_ops.base import _sanitize_list_like, DataTypeOps
from pyspark.pandas.typedef import pandas_on_spark_type
from pyspark.sql import functions as F
from pyspark.sql.utils import pyspark_column_op

class CategoricalOps(DataTypeOps):
    """
    The class for binary operations of pandas-on-Spark objects with categorical types.
    """

    @property
    def pretty_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'categoricals'

    def restore(self, col: pd.Series) -> pd.Series:
        if False:
            i = 10
            return i + 15
        'Restore column when to_pandas.'
        return pd.Series(pd.Categorical.from_codes(col.replace(np.nan, -1).astype(int), categories=cast(CategoricalDtype, self.dtype).categories, ordered=cast(CategoricalDtype, self.dtype).ordered))

    def prepare(self, col: pd.Series) -> pd.Series:
        if False:
            return 10
        'Prepare column when from_pandas.'
        return col.cat.codes

    def astype(self, index_ops: IndexOpsLike, dtype: Union[str, type, Dtype]) -> IndexOpsLike:
        if False:
            i = 10
            return i + 15
        (dtype, _) = pandas_on_spark_type(dtype)
        if isinstance(dtype, CategoricalDtype) and (dtype.categories is None or index_ops.dtype == dtype):
            return index_ops.copy()
        return _to_cat(index_ops).astype(dtype)

    def eq(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            while True:
                i = 10
        _sanitize_list_like(right)
        return _compare(left, right, '__eq__', is_equality_comparison=True)

    def ne(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            i = 10
            return i + 15
        _sanitize_list_like(right)
        return _compare(left, right, '__ne__', is_equality_comparison=True)

    def lt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            return 10
        _sanitize_list_like(right)
        return _compare(left, right, '__lt__')

    def le(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            for i in range(10):
                print('nop')
        _sanitize_list_like(right)
        return _compare(left, right, '__le__')

    def gt(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            for i in range(10):
                print('nop')
        _sanitize_list_like(right)
        return _compare(left, right, '__gt__')

    def ge(self, left: IndexOpsLike, right: Any) -> SeriesOrIndex:
        if False:
            while True:
                i = 10
        _sanitize_list_like(right)
        return _compare(left, right, '__ge__')

def _compare(left: IndexOpsLike, right: Any, func_name: str, *, is_equality_comparison: bool=False) -> SeriesOrIndex:
    if False:
        return 10
    '\n    Compare a Categorical operand `left` to `right` with the given Spark Column function.\n\n    Parameters\n    ----------\n    left: A Categorical operand\n    right: The other operand to compare with\n    func_name: The Spark Column function name to apply\n    is_equality_comparison: True if it is equality comparison, ie. == or !=. False by default.\n\n    Returns\n    -------\n    SeriesOrIndex\n    '
    if isinstance(right, IndexOpsMixin) and isinstance(right.dtype, CategoricalDtype):
        if not is_equality_comparison:
            if not cast(CategoricalDtype, left.dtype).ordered:
                raise TypeError('Unordered Categoricals can only compare equality or not.')
        if hash(left.dtype) != hash(right.dtype):
            raise TypeError("Categoricals can only be compared if 'categories' are the same.")
        if cast(CategoricalDtype, left.dtype).ordered:
            return pyspark_column_op(func_name, left, right)
        else:
            return pyspark_column_op(func_name, _to_cat(left), _to_cat(right))
    elif not is_list_like(right):
        categories = cast(CategoricalDtype, left.dtype).categories
        if right not in categories:
            raise TypeError('Cannot compare a Categorical with a scalar, which is not a category.')
        right_code = categories.get_loc(right)
        return pyspark_column_op(func_name, left, right_code)
    else:
        raise TypeError('Cannot compare a Categorical with the given type.')

def _to_cat(index_ops: IndexOpsLike) -> IndexOpsLike:
    if False:
        i = 10
        return i + 15
    categories = cast(CategoricalDtype, index_ops.dtype).categories
    if len(categories) == 0:
        scol = F.lit(None)
    else:
        scol = F.lit(None)
        for (code, category) in reversed(list(enumerate(categories))):
            scol = F.when(index_ops.spark.column == F.lit(code), F.lit(category)).otherwise(scol)
    return index_ops._with_new_scol(scol)