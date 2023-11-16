"""Top level APIs for defining vectorized UDFs.

Warning: This is an experimental module and API here can change without notice.

DO NOT USE DIRECTLY.
"""
from __future__ import annotations
import functools
from typing import TYPE_CHECKING, Any
import numpy as np
import ibis.expr.datatypes as dt
import ibis.legacy.udf.validate as v
from ibis.expr.operations import AnalyticVectorizedUDF, ElementWiseVectorizedUDF, ReductionVectorizedUDF
if TYPE_CHECKING:
    import pandas as pd

def _coerce_to_dict(data: list | np.ndarray | pd.Series, output_type: dt.Struct, index: pd.Index | None=None) -> tuple:
    if False:
        for i in range(10):
            print('nop')
    'Coerce the following shapes to a tuple.\n\n    - [](`list`)\n    - `np.ndarray`\n    - `pd.Series`\n    '
    return dict(zip(output_type.names, data))

def _coerce_to_np_array(data: list | np.ndarray | pd.Series, output_type: dt.Struct, index: pd.Index | None=None) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    'Coerce the following shapes to an np.ndarray.\n\n    - [](`list`)\n    - `np.ndarray`\n    - `pd.Series`\n    '
    return np.array(data)

def _coerce_to_series(data: list | np.ndarray | pd.Series, output_type: dt.DataType, original_index: pd.Index | None=None) -> pd.Series:
    if False:
        print('Hello World!')
    'Coerce the following shapes to a Series.\n\n    This method does NOT always return a new Series. If a Series is\n    passed in, this method will return the original object.\n\n    - [](`list`)\n    - `np.ndarray`\n    - `pd.Series`\n\n    Note:\n\n    Parameters\n    ----------\n    data\n        Input\n    output_type\n        The type of the output\n    original_index\n        Optional parameter containing the index of the output\n\n    Returns\n    -------\n    pd.Series\n        Output Series\n    '
    import pandas as pd
    if isinstance(data, (list, np.ndarray)):
        result = pd.Series(data)
    elif isinstance(data, pd.Series):
        result = data
    else:
        return data
    if original_index is not None:
        result.index = original_index
    return result

def _coerce_to_dataframe(data: Any, output_type: dt.Struct, original_index: pd.Index | None=None) -> pd.DataFrame:
    if False:
        return 10
    'Coerce the following shapes to a DataFrame.\n\n    This method does NOT always return a new DataFrame. If a DataFrame is\n    passed in, this method will return the original object.\n\n    The following shapes are allowed:\n\n    - A list/tuple of Series\n    - A list/tuple np.ndarray\n    - A list/tuple of scalars\n    - A Series of list/tuple\n    - pd.DataFrame\n\n    Note:\n\n    Parameters\n    ----------\n    data\n        Input\n    output_type\n        A Struct containing the names and types of the output\n    original_index\n        Optional parameter containing the index of the output\n\n    Returns\n    -------\n    pd.DataFrame\n        Output DataFrame\n\n    Examples\n    --------\n    >>> import pandas as pd\n    >>> _coerce_to_dataframe(\n    ...     pd.DataFrame({"a": [1, 2, 3]}), dt.Struct(dict(b="int32"))\n    ... )  # noqa: E501\n       b\n    0  1\n    1  2\n    2  3\n    >>> _coerce_to_dataframe(\n    ...     pd.Series([[1, 2, 3]]), dt.Struct(dict.fromkeys("abc", "int32"))\n    ... )  # noqa: E501\n       a  b  c\n    0  1  2  3\n    >>> _coerce_to_dataframe(\n    ...     pd.Series([range(3), range(3)]), dt.Struct(dict.fromkeys("abc", "int32"))\n    ... )  # noqa: E501\n       a  b  c\n    0  0  1  2\n    1  0  1  2\n    >>> _coerce_to_dataframe(\n    ...     [pd.Series(x) for x in [1, 2, 3]], dt.Struct(dict.fromkeys("abc", "int32"))\n    ... )  # noqa: E501\n       a  b  c\n    0  1  2  3\n    >>> _coerce_to_dataframe(\n    ...     [1, 2, 3], dt.Struct(dict.fromkeys("abc", "int32"))\n    ... )  # noqa: E501\n       a  b  c\n    0  1  2  3\n    '
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        result = data
    elif isinstance(data, pd.Series):
        if not len(data):
            result = data.to_frame()
        else:
            num_cols = len(data.iloc[0])
            series = [data.apply(lambda t, i=i: t[i]) for i in range(num_cols)]
            result = pd.concat(series, axis=1)
    elif isinstance(data, (tuple, list, np.ndarray)):
        if isinstance(data[0], pd.Series):
            result = pd.concat(data, axis=1)
        elif isinstance(data[0], np.ndarray):
            result = pd.concat([pd.Series(v) for v in data], axis=1)
        else:
            result = pd.concat([pd.Series([v]) for v in data], axis=1)
    else:
        raise ValueError(f'Cannot coerce to DataFrame: {data}')
    result.columns = output_type.names
    if original_index is not None:
        result.index = original_index
    return result

class UserDefinedFunction:
    """Class representing a user defined function.

    This class Implements __call__ that returns an ibis expr for the
    UDF.
    """

    def __init__(self, func, func_type, input_type, output_type):
        if False:
            while True:
                i = 10
        v.validate_input_type(input_type, func)
        v.validate_output_type(output_type)
        self.func = func
        self.func_type = func_type
        self.input_type = list(map(dt.dtype, input_type))
        self.output_type = dt.dtype(output_type)
        self.coercion_fn = self._get_coercion_function()

    def _get_coercion_function(self):
        if False:
            return 10
        'Return the appropriate function to coerce the result of the UDF.'
        if self.output_type.is_struct():
            if self.func_type is ElementWiseVectorizedUDF or self.func_type is AnalyticVectorizedUDF:
                return _coerce_to_dataframe
            else:
                return _coerce_to_dict
        elif self.func_type is ElementWiseVectorizedUDF or self.func_type is AnalyticVectorizedUDF:
            return _coerce_to_series
        elif self.output_type.is_array():
            return _coerce_to_np_array
        else:
            return None

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')

        @functools.wraps(self.func)
        def func(*args):
            if False:
                while True:
                    i = 10
            saved_index = getattr(args[0], 'index', None)
            result = self.func(*args, **kwargs)
            if self.coercion_fn:
                result = self.coercion_fn(result, self.output_type, saved_index)
            return result
        op = self.func_type(func=func, func_args=args, input_type=self.input_type, return_type=self.output_type)
        return op.to_expr()

def _udf_decorator(node_type, input_type, output_type):
    if False:
        while True:
            i = 10

    def wrapper(func):
        if False:
            while True:
                i = 10
        return UserDefinedFunction(func, node_type, input_type, output_type)
    return wrapper

def analytic(input_type, output_type):
    if False:
        i = 10
        return i + 15
    'Define an analytic UDF that produces the same of rows as the input.\n\n    Parameters\n    ----------\n    input_type : List[ibis.expr.datatypes.DataType]\n        A list of the types found in :mod:`~ibis.expr.datatypes`. The\n        length of this list must match the number of arguments to the\n        function. Variadic arguments are not yet supported.\n    output_type : ibis.expr.datatypes.DataType\n        The return type of the function.\n\n    Examples\n    --------\n    >>> import ibis\n    >>> import ibis.expr.datatypes as dt\n    >>> from ibis.legacy.udf.vectorized import analytic\n    >>> @analytic(input_type=[dt.double], output_type=dt.double)\n    ... def zscore(series):  # note the use of aggregate functions\n    ...     return (series - series.mean()) / series.std()\n    ...\n\n    Define and use an UDF with multiple return columns:\n\n    >>> @analytic(\n    ...     input_type=[dt.double],\n    ...     output_type=dt.Struct(dict(demean="double", zscore="double")),\n    ... )\n    ... def demean_and_zscore(v):\n    ...     mean = v.mean()\n    ...     std = v.std()\n    ...     return v - mean, (v - mean) / std\n    >>>\n    >>> win = ibis.window(preceding=None, following=None, group_by="key")\n    >>> # add two columns "demean" and "zscore"\n    >>> table = table.mutate(  # quartodoc: +SKIP # doctest: +SKIP\n    ...     demean_and_zscore(table["v"]).over(win).destructure()\n    ... )\n    '
    return _udf_decorator(AnalyticVectorizedUDF, input_type, output_type)

def elementwise(input_type, output_type):
    if False:
        return 10
    'Define a UDF that operates element-wise on a Pandas Series.\n\n    Parameters\n    ----------\n    input_type : List[ibis.expr.datatypes.DataType]\n        A list of the types found in :mod:`~ibis.expr.datatypes`. The\n        length of this list must match the number of arguments to the\n        function. Variadic arguments are not yet supported.\n    output_type : ibis.expr.datatypes.DataType\n        The return type of the function.\n\n    Examples\n    --------\n    >>> import ibis\n    >>> import ibis.expr.datatypes as dt\n    >>> from ibis.legacy.udf.vectorized import elementwise\n    >>> @elementwise(input_type=[dt.string], output_type=dt.int64)\n    ... def my_string_length(series):\n    ...     return series.str.len() * 2\n    ...\n\n    Define an UDF with non-column parameters:\n\n    >>> @elementwise(input_type=[dt.string], output_type=dt.int64)\n    ... def my_string_length(series, *, times):\n    ...     return series.str.len() * times\n    ...\n\n    Define and use an UDF with multiple return columns:\n\n    >>> @elementwise(\n    ...     input_type=[dt.string],\n    ...     output_type=dt.Struct(dict(year=dt.string, monthday=dt.string)),\n    ... )\n    ... def year_monthday(date):\n    ...     return date.str.slice(0, 4), date.str.slice(4, 8)\n    >>>\n    >>> # add two columns "year" and "monthday"\n    >>> table = table.mutate(\n    ...     year_monthday(table["date"]).destructure()\n    ... )  # quartodoc: +SKIP # doctest: +SKIP\n    '
    return _udf_decorator(ElementWiseVectorizedUDF, input_type, output_type)

def reduction(input_type, output_type):
    if False:
        print('Hello World!')
    'Define a UDF reduction function that produces 1 row of output for N rows of input.\n\n    Parameters\n    ----------\n    input_type : List[ibis.expr.datatypes.DataType]\n        A list of the types found in :mod:`~ibis.expr.datatypes`. The\n        length of this list must match the number of arguments to the\n        function. Variadic arguments are not yet supported.\n    output_type : ibis.expr.datatypes.DataType\n        The return type of the function.\n\n    Examples\n    --------\n    >>> import ibis\n    >>> import ibis.expr.datatypes as dt\n    >>> from ibis.legacy.udf.vectorized import reduction\n    >>> @reduction(input_type=[dt.string], output_type=dt.int64)\n    ... def my_string_length_agg(series, **kwargs):\n    ...     return (series.str.len() * 2).sum()\n    ...\n\n    Define and use an UDF with multiple return columns:\n\n    >>> @reduction(\n    ...     input_type=[dt.double],\n    ...     output_type=dt.Struct(dict(mean="double", std="double")),\n    ... )\n    ... def mean_and_std(v):\n    ...     return v.mean(), v.std()\n    >>>\n    >>> # create aggregation columns "mean" and "std"\n    >>> table = table.group_by("key").aggregate(  # quartodoc: +SKIP # doctest: +SKIP\n    ...     mean_and_std(table["v"]).destructure()\n    ... )\n    '
    return _udf_decorator(ReductionVectorizedUDF, input_type, output_type)