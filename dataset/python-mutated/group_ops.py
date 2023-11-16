import sys
from typing import List, Union, TYPE_CHECKING, cast
import warnings
from pyspark.rdd import PythonEvalType
from pyspark.sql.column import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.streaming.state import GroupStateTimeout
from pyspark.sql.types import StructType, _parse_datatype_string
if TYPE_CHECKING:
    from pyspark.sql.pandas._typing import GroupedMapPandasUserDefinedFunction, PandasGroupedMapFunction, PandasGroupedMapFunctionWithState, PandasCogroupedMapFunction
    from pyspark.sql.group import GroupedData

class PandasGroupedOpsMixin:
    """
    Min-in for pandas grouped operations. Currently, only :class:`GroupedData`
    can use this class.
    """

    def apply(self, udf: 'GroupedMapPandasUserDefinedFunction') -> DataFrame:
        if False:
            while True:
                i = 10
        '\n        It is an alias of :meth:`pyspark.sql.GroupedData.applyInPandas`; however, it takes a\n        :meth:`pyspark.sql.functions.pandas_udf` whereas\n        :meth:`pyspark.sql.GroupedData.applyInPandas` takes a Python native function.\n\n        .. versionadded:: 2.3.0\n\n        .. versionchanged:: 3.4.0\n            Support Spark Connect.\n\n        Parameters\n        ----------\n        udf : :func:`pyspark.sql.functions.pandas_udf`\n            a grouped map user-defined function returned by\n            :func:`pyspark.sql.functions.pandas_udf`.\n\n        Notes\n        -----\n        It is preferred to use :meth:`pyspark.sql.GroupedData.applyInPandas` over this\n        API. This API will be deprecated in the future releases.\n\n        Examples\n        --------\n        >>> from pyspark.sql.functions import pandas_udf, PandasUDFType\n        >>> df = spark.createDataFrame(\n        ...     [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],\n        ...     ("id", "v"))\n        >>> @pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)  # doctest: +SKIP\n        ... def normalize(pdf):\n        ...     v = pdf.v\n        ...     return pdf.assign(v=(v - v.mean()) / v.std())\n        ...\n        >>> df.groupby("id").apply(normalize).show()  # doctest: +SKIP\n        +---+-------------------+\n        | id|                  v|\n        +---+-------------------+\n        |  1|-0.7071067811865475|\n        |  1| 0.7071067811865475|\n        |  2|-0.8320502943378437|\n        |  2|-0.2773500981126146|\n        |  2| 1.1094003924504583|\n        +---+-------------------+\n\n        See Also\n        --------\n        pyspark.sql.functions.pandas_udf\n        '
        if isinstance(udf, Column) or not hasattr(udf, 'func') or udf.evalType != PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF:
            raise ValueError('Invalid udf: the udf argument must be a pandas_udf of type GROUPED_MAP.')
        warnings.warn("It is preferred to use 'applyInPandas' over this API. This API will be deprecated in the future releases. See SPARK-28264 for more details.", UserWarning)
        return self.applyInPandas(udf.func, schema=udf.returnType)

    def applyInPandas(self, func: 'PandasGroupedMapFunction', schema: Union[StructType, str]) -> DataFrame:
        if False:
            return 10
        '\n        Maps each group of the current :class:`DataFrame` using a pandas udf and returns the result\n        as a `DataFrame`.\n\n        The function should take a `pandas.DataFrame` and return another\n        `pandas.DataFrame`. Alternatively, the user can pass a function that takes\n        a tuple of the grouping key(s) and a `pandas.DataFrame`.\n        For each group, all columns are passed together as a `pandas.DataFrame`\n        to the user-function and the returned `pandas.DataFrame` are combined as a\n        :class:`DataFrame`.\n\n        The `schema` should be a :class:`StructType` describing the schema of the returned\n        `pandas.DataFrame`. The column labels of the returned `pandas.DataFrame` must either match\n        the field names in the defined schema if specified as strings, or match the\n        field data types by position if not strings, e.g. integer indices.\n        The length of the returned `pandas.DataFrame` can be arbitrary.\n\n        .. versionadded:: 3.0.0\n\n        .. versionchanged:: 3.4.0\n            Support Spark Connect.\n\n        Parameters\n        ----------\n        func : function\n            a Python native function that takes a `pandas.DataFrame` and outputs a\n            `pandas.DataFrame`, or that takes one tuple (grouping keys) and a\n            `pandas.DataFrame` and outputs a `pandas.DataFrame`.\n        schema : :class:`pyspark.sql.types.DataType` or str\n            the return type of the `func` in PySpark. The value can be either a\n            :class:`pyspark.sql.types.DataType` object or a DDL-formatted type string.\n\n        Examples\n        --------\n        >>> import pandas as pd  # doctest: +SKIP\n        >>> from pyspark.sql.functions import pandas_udf, ceil\n        >>> df = spark.createDataFrame(\n        ...     [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],\n        ...     ("id", "v"))  # doctest: +SKIP\n        >>> def normalize(pdf):\n        ...     v = pdf.v\n        ...     return pdf.assign(v=(v - v.mean()) / v.std())\n        ...\n        >>> df.groupby("id").applyInPandas(\n        ...     normalize, schema="id long, v double").show()  # doctest: +SKIP\n        +---+-------------------+\n        | id|                  v|\n        +---+-------------------+\n        |  1|-0.7071067811865475|\n        |  1| 0.7071067811865475|\n        |  2|-0.8320502943378437|\n        |  2|-0.2773500981126146|\n        |  2| 1.1094003924504583|\n        +---+-------------------+\n\n        Alternatively, the user can pass a function that takes two arguments.\n        In this case, the grouping key(s) will be passed as the first argument and the data will\n        be passed as the second argument. The grouping key(s) will be passed as a tuple of numpy\n        data types, e.g., `numpy.int32` and `numpy.float64`. The data will still be passed in\n        as a `pandas.DataFrame` containing all columns from the original Spark DataFrame.\n        This is useful when the user does not want to hardcode grouping key(s) in the function.\n\n        >>> df = spark.createDataFrame(\n        ...     [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],\n        ...     ("id", "v"))  # doctest: +SKIP\n        >>> def mean_func(key, pdf):\n        ...     # key is a tuple of one numpy.int64, which is the value\n        ...     # of \'id\' for the current group\n        ...     return pd.DataFrame([key + (pdf.v.mean(),)])\n        ...\n        >>> df.groupby(\'id\').applyInPandas(\n        ...     mean_func, schema="id long, v double").show()  # doctest: +SKIP\n        +---+---+\n        | id|  v|\n        +---+---+\n        |  1|1.5|\n        |  2|6.0|\n        +---+---+\n\n        >>> def sum_func(key, pdf):\n        ...     # key is a tuple of two numpy.int64s, which is the values\n        ...     # of \'id\' and \'ceil(df.v / 2)\' for the current group\n        ...     return pd.DataFrame([key + (pdf.v.sum(),)])\n        ...\n        >>> df.groupby(df.id, ceil(df.v / 2)).applyInPandas(\n        ...     sum_func, schema="id long, `ceil(v / 2)` long, v double").show()  # doctest: +SKIP\n        +---+-----------+----+\n        | id|ceil(v / 2)|   v|\n        +---+-----------+----+\n        |  2|          5|10.0|\n        |  1|          1| 3.0|\n        |  2|          3| 5.0|\n        |  2|          2| 3.0|\n        +---+-----------+----+\n\n        Notes\n        -----\n        This function requires a full shuffle. All the data of a group will be loaded\n        into memory, so the user should be aware of the potential OOM risk if data is skewed\n        and certain groups are too large to fit in memory.\n\n        This API is experimental.\n\n        See Also\n        --------\n        pyspark.sql.functions.pandas_udf\n        '
        from pyspark.sql import GroupedData
        from pyspark.sql.functions import pandas_udf, PandasUDFType
        assert isinstance(self, GroupedData)
        udf = pandas_udf(func, returnType=schema, functionType=PandasUDFType.GROUPED_MAP)
        df = self._df
        udf_column = udf(*[df[col] for col in df.columns])
        jdf = self._jgd.flatMapGroupsInPandas(udf_column._jc.expr())
        return DataFrame(jdf, self.session)

    def applyInPandasWithState(self, func: 'PandasGroupedMapFunctionWithState', outputStructType: Union[StructType, str], stateStructType: Union[StructType, str], outputMode: str, timeoutConf: str) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies the given function to each group of data, while maintaining a user-defined\n        per-group state. The result Dataset will represent the flattened record returned by the\n        function.\n\n        For a streaming :class:`DataFrame`, the function will be invoked first for all input groups\n        and then for all timed out states where the input data is set to be empty. Updates to each\n        group\'s state will be saved across invocations.\n\n        The function should take parameters (key, Iterator[`pandas.DataFrame`], state) and\n        return another Iterator[`pandas.DataFrame`]. The grouping key(s) will be passed as a tuple\n        of numpy data types, e.g., `numpy.int32` and `numpy.float64`. The state will be passed as\n        :class:`pyspark.sql.streaming.state.GroupState`.\n\n        For each group, all columns are passed together as `pandas.DataFrame` to the user-function,\n        and the returned `pandas.DataFrame` across all invocations are combined as a\n        :class:`DataFrame`. Note that the user function should not make a guess of the number of\n        elements in the iterator. To process all data, the user function needs to iterate all\n        elements and process them. On the other hand, the user function is not strictly required to\n        iterate through all elements in the iterator if it intends to read a part of data.\n\n        The `outputStructType` should be a :class:`StructType` describing the schema of all\n        elements in the returned value, `pandas.DataFrame`. The column labels of all elements in\n        returned `pandas.DataFrame` must either match the field names in the defined schema if\n        specified as strings, or match the field data types by position if not strings,\n        e.g. integer indices.\n\n        The `stateStructType` should be :class:`StructType` describing the schema of the\n        user-defined state. The value of the state will be presented as a tuple, as well as the\n        update should be performed with the tuple. The corresponding Python types for\n        :class:DataType are supported. Please refer to the page\n        https://spark.apache.org/docs/latest/sql-ref-datatypes.html (Python tab).\n\n        The size of each `pandas.DataFrame` in both the input and output can be arbitrary. The\n        number of `pandas.DataFrame` in both the input and output can also be arbitrary.\n\n        .. versionadded:: 3.4.0\n\n        .. versionchanged:: 3.5.0\n            Supports Spark Connect.\n\n        Parameters\n        ----------\n        func : function\n            a Python native function to be called on every group. It should take parameters\n            (key, Iterator[`pandas.DataFrame`], state) and return Iterator[`pandas.DataFrame`].\n            Note that the type of the key is tuple and the type of the state is\n            :class:`pyspark.sql.streaming.state.GroupState`.\n        outputStructType : :class:`pyspark.sql.types.DataType` or str\n            the type of the output records. The value can be either a\n            :class:`pyspark.sql.types.DataType` object or a DDL-formatted type string.\n        stateStructType : :class:`pyspark.sql.types.DataType` or str\n            the type of the user-defined state. The value can be either a\n            :class:`pyspark.sql.types.DataType` object or a DDL-formatted type string.\n        outputMode : str\n            the output mode of the function.\n        timeoutConf : str\n            timeout configuration for groups that do not receive data for a while. valid values\n            are defined in :class:`pyspark.sql.streaming.state.GroupStateTimeout`.\n\n        Examples\n        --------\n        >>> import pandas as pd  # doctest: +SKIP\n        >>> from pyspark.sql.streaming.state import GroupStateTimeout\n        >>> def count_fn(key, pdf_iter, state):\n        ...     assert isinstance(state, GroupStateImpl)\n        ...     total_len = 0\n        ...     for pdf in pdf_iter:\n        ...         total_len += len(pdf)\n        ...     state.update((total_len,))\n        ...     yield pd.DataFrame({"id": [key[0]], "countAsString": [str(total_len)]})\n        ...\n        >>> df.groupby("id").applyInPandasWithState(\n        ...     count_fn, outputStructType="id long, countAsString string",\n        ...     stateStructType="len long", outputMode="Update",\n        ...     timeoutConf=GroupStateTimeout.NoTimeout) # doctest: +SKIP\n\n        Notes\n        -----\n        This function requires a full shuffle.\n\n        This API is experimental.\n        '
        from pyspark.sql import GroupedData
        from pyspark.sql.functions import pandas_udf
        assert isinstance(self, GroupedData)
        assert timeoutConf in [GroupStateTimeout.NoTimeout, GroupStateTimeout.ProcessingTimeTimeout, GroupStateTimeout.EventTimeTimeout]
        if isinstance(outputStructType, str):
            outputStructType = cast(StructType, _parse_datatype_string(outputStructType))
        if isinstance(stateStructType, str):
            stateStructType = cast(StructType, _parse_datatype_string(stateStructType))
        udf = pandas_udf(func, returnType=outputStructType, functionType=PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE)
        df = self._df
        udf_column = udf(*[df[col] for col in df.columns])
        jdf = self._jgd.applyInPandasWithState(udf_column._jc.expr(), self.session._jsparkSession.parseDataType(outputStructType.json()), self.session._jsparkSession.parseDataType(stateStructType.json()), outputMode, timeoutConf)
        return DataFrame(jdf, self.session)

    def cogroup(self, other: 'GroupedData') -> 'PandasCogroupedOps':
        if False:
            i = 10
            return i + 15
        '\n        Cogroups this group with another group so that we can run cogrouped operations.\n\n        .. versionadded:: 3.0.0\n\n        .. versionchanged:: 3.4.0\n            Support Spark Connect.\n\n        See :class:`PandasCogroupedOps` for the operations that can be run.\n        '
        from pyspark.sql import GroupedData
        assert isinstance(self, GroupedData)
        return PandasCogroupedOps(self, other)

class PandasCogroupedOps:
    """
    A logical grouping of two :class:`GroupedData`,
    created by :func:`GroupedData.cogroup`.

    .. versionadded:: 3.0.0

    .. versionchanged:: 3.4.0
        Support Spark Connect.

    Notes
    -----
    This API is experimental.
    """

    def __init__(self, gd1: 'GroupedData', gd2: 'GroupedData'):
        if False:
            return 10
        self._gd1 = gd1
        self._gd2 = gd2

    def applyInPandas(self, func: 'PandasCogroupedMapFunction', schema: Union[StructType, str]) -> DataFrame:
        if False:
            print('Hello World!')
        '\n        Applies a function to each cogroup using pandas and returns the result\n        as a `DataFrame`.\n\n        The function should take two `pandas.DataFrame`\\s and return another\n        `pandas.DataFrame`. Alternatively, the user can pass a function that takes\n        a tuple of the grouping key(s) and the two `pandas.DataFrame`\\s.\n        For each side of the cogroup, all columns are passed together as a\n        `pandas.DataFrame` to the user-function and the returned `pandas.DataFrame` are combined as\n        a :class:`DataFrame`.\n\n        The `schema` should be a :class:`StructType` describing the schema of the returned\n        `pandas.DataFrame`. The column labels of the returned `pandas.DataFrame` must either match\n        the field names in the defined schema if specified as strings, or match the\n        field data types by position if not strings, e.g. integer indices.\n        The length of the returned `pandas.DataFrame` can be arbitrary.\n\n        .. versionadded:: 3.0.0\n\n        .. versionchanged:: 3.4.0\n            Support Spark Connect.\n\n        Parameters\n        ----------\n        func : function\n            a Python native function that takes two `pandas.DataFrame`\\s, and\n            outputs a `pandas.DataFrame`, or that takes one tuple (grouping keys) and two\n            ``pandas.DataFrame``\\s, and outputs a ``pandas.DataFrame``.\n        schema : :class:`pyspark.sql.types.DataType` or str\n            the return type of the `func` in PySpark. The value can be either a\n            :class:`pyspark.sql.types.DataType` object or a DDL-formatted type string.\n\n        Examples\n        --------\n        >>> from pyspark.sql.functions import pandas_udf\n        >>> df1 = spark.createDataFrame(\n        ...     [(20000101, 1, 1.0), (20000101, 2, 2.0), (20000102, 1, 3.0), (20000102, 2, 4.0)],\n        ...     ("time", "id", "v1"))\n        >>> df2 = spark.createDataFrame(\n        ...     [(20000101, 1, "x"), (20000101, 2, "y")],\n        ...     ("time", "id", "v2"))\n        >>> def asof_join(l, r):\n        ...     return pd.merge_asof(l, r, on="time", by="id")\n        ...\n        >>> df1.groupby("id").cogroup(df2.groupby("id")).applyInPandas(\n        ...     asof_join, schema="time int, id int, v1 double, v2 string"\n        ... ).show()  # doctest: +SKIP\n        +--------+---+---+---+\n        |    time| id| v1| v2|\n        +--------+---+---+---+\n        |20000101|  1|1.0|  x|\n        |20000102|  1|3.0|  x|\n        |20000101|  2|2.0|  y|\n        |20000102|  2|4.0|  y|\n        +--------+---+---+---+\n\n        Alternatively, the user can define a function that takes three arguments.  In this case,\n        the grouping key(s) will be passed as the first argument and the data will be passed as the\n        second and third arguments.  The grouping key(s) will be passed as a tuple of numpy data\n        types, e.g., `numpy.int32` and `numpy.float64`. The data will still be passed in as two\n        `pandas.DataFrame` containing all columns from the original Spark DataFrames.\n\n        >>> def asof_join(k, l, r):\n        ...     if k == (1,):\n        ...         return pd.merge_asof(l, r, on="time", by="id")\n        ...     else:\n        ...         return pd.DataFrame(columns=[\'time\', \'id\', \'v1\', \'v2\'])\n        ...\n        >>> df1.groupby("id").cogroup(df2.groupby("id")).applyInPandas(\n        ...     asof_join, "time int, id int, v1 double, v2 string").show()  # doctest: +SKIP\n        +--------+---+---+---+\n        |    time| id| v1| v2|\n        +--------+---+---+---+\n        |20000101|  1|1.0|  x|\n        |20000102|  1|3.0|  x|\n        +--------+---+---+---+\n\n        Notes\n        -----\n        This function requires a full shuffle. All the data of a cogroup will be loaded\n        into memory, so the user should be aware of the potential OOM risk if data is skewed\n        and certain groups are too large to fit in memory.\n\n        This API is experimental.\n\n        See Also\n        --------\n        pyspark.sql.functions.pandas_udf\n        '
        from pyspark.sql.pandas.functions import pandas_udf
        udf = pandas_udf(func, returnType=schema, functionType=PythonEvalType.SQL_COGROUPED_MAP_PANDAS_UDF)
        all_cols = self._extract_cols(self._gd1) + self._extract_cols(self._gd2)
        udf_column = udf(*all_cols)
        jdf = self._gd1._jgd.flatMapCoGroupsInPandas(self._gd2._jgd, udf_column._jc.expr())
        return DataFrame(jdf, self._gd1.session)

    @staticmethod
    def _extract_cols(gd: 'GroupedData') -> List[Column]:
        if False:
            for i in range(10):
                print('nop')
        df = gd._df
        return [df[col] for col in df.columns]

def _test() -> None:
    if False:
        while True:
            i = 10
    import doctest
    from pyspark.sql import SparkSession
    import pyspark.sql.pandas.group_ops
    globs = pyspark.sql.pandas.group_ops.__dict__.copy()
    spark = SparkSession.builder.master('local[4]').appName('sql.pandas.group tests').getOrCreate()
    globs['spark'] = spark
    (failure_count, test_count) = doctest.testmod(pyspark.sql.pandas.group_ops, globs=globs, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.REPORT_NDIFF)
    spark.stop()
    if failure_count:
        sys.exit(-1)
if __name__ == '__main__':
    _test()