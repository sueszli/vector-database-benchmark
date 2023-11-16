import _string
from typing import Any, Dict, Optional, Union, List
import inspect
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SDataFrame
from pyspark import pandas as ps
from pyspark.pandas.utils import default_session
from pyspark.pandas.frame import DataFrame
from pyspark.pandas.series import Series
from pyspark.pandas.internal import InternalFrame
from pyspark.pandas.namespace import _get_index_map
__all__ = ['sql']
from builtins import globals as builtin_globals
from builtins import locals as builtin_locals

def sql(query: str, index_col: Optional[Union[str, List[str]]]=None, globals: Optional[Dict[str, Any]]=None, locals: Optional[Dict[str, Any]]=None, **kwargs: Any) -> DataFrame:
    if False:
        while True:
            i = 10
    '\n    Execute a SQL query and return the result as a pandas-on-Spark DataFrame.\n\n    This function also supports embedding Python variables (locals, globals, and parameters)\n    in the SQL statement by wrapping them in curly braces. See examples section for details.\n\n    In addition to the locals, globals and parameters, the function will also attempt\n    to determine if the program currently runs in an IPython (or Jupyter) environment\n    and to import the variables from this environment. The variables have the same\n    precedence as globals.\n\n    The following variable types are supported:\n\n        * string\n        * int\n        * float\n        * list, tuple, range of above types\n        * pandas-on-Spark DataFrame\n        * pandas-on-Spark Series\n        * pandas DataFrame\n\n    Parameters\n    ----------\n    query : str\n        the SQL query\n    index_col : str or list of str, optional\n        Column names to be used in Spark to represent pandas-on-Spark\'s index. The index name\n        in pandas-on-Spark is ignored. By default, the index is always lost.\n\n        .. note:: If you want to preserve the index, explicitly use :func:`DataFrame.reset_index`,\n            and pass it to the SQL statement with `index_col` parameter.\n\n            For example,\n\n            >>> from pyspark.pandas import sql_processor\n            >>> # we will call \'sql_processor\' directly in doctests so decrease one level.\n            >>> sql_processor._CAPTURE_SCOPES = 2\n            >>> sql = sql_processor.sql\n            >>> psdf = ps.DataFrame({"A": [1, 2, 3], "B":[4, 5, 6]}, index=[\'a\', \'b\', \'c\'])\n            >>> psdf_reset_index = psdf.reset_index()\n            >>> sql("SELECT * FROM {psdf_reset_index}", index_col="index")\n            ... # doctest: +NORMALIZE_WHITESPACE\n                   A  B\n            index\n            a      1  4\n            b      2  5\n            c      3  6\n\n            For MultiIndex,\n\n            >>> psdf = ps.DataFrame(\n            ...     {"A": [1, 2, 3], "B": [4, 5, 6]},\n            ...     index=pd.MultiIndex.from_tuples(\n            ...         [("a", "b"), ("c", "d"), ("e", "f")], names=["index1", "index2"]\n            ...     ),\n            ... )\n            >>> psdf_reset_index = psdf.reset_index()\n            >>> sql("SELECT * FROM {psdf_reset_index}", index_col=["index1", "index2"])\n            ... # doctest: +NORMALIZE_WHITESPACE\n                           A  B\n            index1 index2\n            a      b       1  4\n            c      d       2  5\n            e      f       3  6\n\n            Also note that the index name(s) should be matched to the existing name.\n\n    globals : dict, optional\n        the dictionary of global variables, if explicitly set by the user\n    locals : dict, optional\n        the dictionary of local variables, if explicitly set by the user\n    kwargs\n        other variables that the user may want to set manually that can be referenced in the query\n\n    Returns\n    -------\n    pandas-on-Spark DataFrame\n\n    Examples\n    --------\n\n    Calling a built-in SQL function.\n\n    >>> sql("select * from range(10) where id > 7")\n       id\n    0   8\n    1   9\n\n    A query can also reference a local variable or parameter by wrapping them in curly braces:\n\n    >>> bound1 = 7\n    >>> sql("select * from range(10) where id > {bound1} and id < {bound2}", bound2=9)\n       id\n    0   8\n\n    You can also wrap a DataFrame with curly braces to query it directly. Note that when you do\n    that, the indexes, if any, automatically become top level columns.\n\n    >>> mydf = ps.range(10)\n    >>> x = range(4)\n    >>> sql("SELECT * from {mydf} WHERE id IN {x}")\n       id\n    0   0\n    1   1\n    2   2\n    3   3\n\n    Queries can also be arbitrarily nested in functions:\n\n    >>> def statement():\n    ...     mydf2 = ps.DataFrame({"x": range(2)})\n    ...     return sql("SELECT * from {mydf2}")\n    >>> statement()\n       x\n    0  0\n    1  1\n\n    Mixing pandas-on-Spark and pandas DataFrames in a join operation. Note that the index is\n    dropped.\n\n    >>> sql(\'\'\'\n    ...   SELECT m1.a, m2.b\n    ...   FROM {table1} m1 INNER JOIN {table2} m2\n    ...   ON m1.key = m2.key\n    ...   ORDER BY m1.a, m2.b\'\'\',\n    ...   table1=ps.DataFrame({"a": [1,2], "key": ["a", "b"]}),\n    ...   table2=pd.DataFrame({"b": [3,4,5], "key": ["a", "b", "b"]}))\n       a  b\n    0  1  3\n    1  2  4\n    2  2  5\n\n    Also, it is possible to query using Series.\n\n    >>> myser = ps.Series({\'a\': [1.0, 2.0, 3.0], \'b\': [15.0, 30.0, 45.0]})\n    >>> sql("SELECT * from {myser}")\n                        0\n    0     [1.0, 2.0, 3.0]\n    1  [15.0, 30.0, 45.0]\n    '
    if globals is None:
        globals = _get_ipython_scope()
    _globals = builtin_globals() if globals is None else dict(globals)
    _locals = builtin_locals() if locals is None else dict(locals)
    _dict = dict(_globals)
    _scope = _get_local_scope()
    _dict.update(_scope)
    _dict.update(_locals)
    _dict.update(kwargs)
    return SQLProcessor(_dict, query, default_session()).execute(index_col)
_CAPTURE_SCOPES = 3

def _get_local_scope() -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    try:
        return inspect.stack()[_CAPTURE_SCOPES][0].f_locals
    except Exception:
        return {}

def _get_ipython_scope() -> Dict[str, Any]:
    if False:
        return 10
    '\n    Tries to extract the dictionary of variables if the program is running\n    in an IPython notebook environment.\n    '
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell.user_ns
    except Exception:
        return None
_escape_table = [chr(x) for x in range(128)]
_escape_table[0] = '\\0'
_escape_table[ord('\\')] = '\\\\'
_escape_table[ord('\n')] = '\\n'
_escape_table[ord('\r')] = '\\r'
_escape_table[ord('\x1a')] = '\\Z'
_escape_table[ord('"')] = '\\"'
_escape_table[ord("'")] = "\\'"

def escape_sql_string(value: str) -> str:
    if False:
        print('Hello World!')
    'Escapes value without adding quotes.\n\n    >>> escape_sql_string("foo\\nbar")\n    \'foo\\\\nbar\'\n\n    >>> escape_sql_string("\'abc\'de")\n    "\\\\\'abc\\\\\'de"\n\n    >>> escape_sql_string(\'"abc"de\')\n    \'\\\\"abc\\\\"de\'\n    '
    return value.translate(_escape_table)

class SQLProcessor:

    def __init__(self, scope: Dict[str, Any], statement: str, session: SparkSession):
        if False:
            while True:
                i = 10
        self._scope = scope
        self._statement = statement
        self._temp_views: Dict[str, SDataFrame] = {}
        self._cached_vars: Dict[str, Any] = {}
        self._normalized_statement: Optional[str] = None
        self._session = session

    def execute(self, index_col: Optional[Union[str, List[str]]]) -> DataFrame:
        if False:
            while True:
                i = 10
        '\n        Returns a DataFrame for which the SQL statement has been executed by\n        the underlying SQL engine.\n\n        >>> from pyspark.pandas import sql_processor\n        >>> # we will call \'sql_processor\' directly in doctests so decrease one level.\n        >>> sql_processor._CAPTURE_SCOPES = 2\n        >>> sql = sql_processor.sql\n        >>> str0 = \'abc\'\n        >>> sql("select {str0}")\n           abc\n        0  abc\n\n        >>> str1 = \'abc"abc\'\n        >>> str2 = "abc\'abc"\n        >>> sql("select {str0}, {str1}, {str2}")\n           abc  abc"abc  abc\'abc\n        0  abc  abc"abc  abc\'abc\n\n        >>> strs = [\'a\', \'b\']\n        >>> sql("select \'a\' in {strs} as cond1, \'c\' in {strs} as cond2")\n           cond1  cond2\n        0   True  False\n        '
        blocks = _string.formatter_parser(self._statement)
        res = ''
        try:
            for (pre, inner, _, _) in blocks:
                var_next = '' if inner is None else self._convert(inner)
                res = res + pre + var_next
            self._normalized_statement = res
            sdf = self._session.sql(self._normalized_statement)
        finally:
            for v in self._temp_views:
                self._session.catalog.dropTempView(v)
        (index_spark_columns, index_names) = _get_index_map(sdf, index_col)
        return DataFrame(InternalFrame(spark_frame=sdf, index_spark_columns=index_spark_columns, index_names=index_names))

    def _convert(self, key: str) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Given a {} key, returns an equivalent SQL representation.\n        This conversion performs all the necessary escaping so that the string\n        returned can be directly injected into the SQL statement.\n        '
        if key in self._cached_vars:
            return self._cached_vars[key]
        if key not in self._scope:
            raise ValueError('The key {} in the SQL statement was not found in global, local or parameters variables'.format(key))
        var = self._scope[key]
        fillin = self._convert_var(var)
        self._cached_vars[key] = fillin
        return fillin

    def _convert_var(self, var: Any) -> Any:
        if False:
            while True:
                i = 10
        '\n        Converts a python object into a string that is legal SQL.\n        '
        if isinstance(var, (int, float)):
            return str(var)
        if isinstance(var, Series):
            return self._convert_var(var.to_dataframe())
        if isinstance(var, pd.DataFrame):
            return self._convert_var(ps.DataFrame(var))
        if isinstance(var, DataFrame):
            df_id = 'pandas_on_spark_' + str(id(var))
            if df_id not in self._temp_views:
                sdf = var._to_spark()
                sdf.createOrReplaceTempView(df_id)
                self._temp_views[df_id] = sdf
            return df_id
        if isinstance(var, str):
            return '"' + escape_sql_string(var) + '"'
        if isinstance(var, list):
            return '(' + ', '.join([self._convert_var(v) for v in var]) + ')'
        if isinstance(var, (tuple, range)):
            return self._convert_var(list(var))
        raise ValueError('Unsupported variable type {}: {}'.format(type(var).__name__, str(var)))

def _test() -> None:
    if False:
        for i in range(10):
            print('nop')
    import os
    import doctest
    import sys
    from pyspark.sql import SparkSession
    import pyspark.pandas.sql_processor
    os.chdir(os.environ['SPARK_HOME'])
    globs = pyspark.pandas.sql_processor.__dict__.copy()
    globs['ps'] = pyspark.pandas
    spark = SparkSession.builder.master('local[4]').appName('pyspark.pandas.sql_processor tests').getOrCreate()
    (failure_count, test_count) = doctest.testmod(pyspark.pandas.sql_processor, globs=globs, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    spark.stop()
    if failure_count:
        sys.exit(-1)
if __name__ == '__main__':
    _test()