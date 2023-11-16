import string
import typing
from typing import Any, Optional, List, Tuple, Sequence, Mapping
import uuid
from py4j.java_gateway import is_instance_of
if typing.TYPE_CHECKING:
    from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit

class SQLStringFormatter(string.Formatter):
    """
    A standard ``string.Formatter`` in Python that can understand PySpark instances
    with basic Python objects. This object has to be clear after the use for single SQL
    query; cannot be reused across multiple SQL queries without cleaning.
    """

    def __init__(self, session: 'SparkSession') -> None:
        if False:
            print('Hello World!')
        self._session: 'SparkSession' = session
        self._temp_views: List[Tuple[DataFrame, str]] = []

    def get_field(self, field_name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
        if False:
            while True:
                i = 10
        (obj, first) = super(SQLStringFormatter, self).get_field(field_name, args, kwargs)
        return (self._convert_value(obj, field_name), first)

    def _convert_value(self, val: Any, field_name: str) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Converts the given value into a SQL string.\n        '
        from pyspark import SparkContext
        from pyspark.sql import Column, DataFrame
        if isinstance(val, Column):
            assert SparkContext._gateway is not None
            gw = SparkContext._gateway
            jexpr = val._jc.expr()
            if is_instance_of(gw, jexpr, 'org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute') or is_instance_of(gw, jexpr, 'org.apache.spark.sql.catalyst.expressions.AttributeReference'):
                return jexpr.sql()
            else:
                raise ValueError("%s in %s should be a plain column reference such as `df.col` or `col('column')`" % (val, field_name))
        elif isinstance(val, DataFrame):
            for (df, n) in self._temp_views:
                if df is val:
                    return n
            df_name = '_pyspark_%s' % str(uuid.uuid4()).replace('-', '')
            self._temp_views.append((val, df_name))
            val.createOrReplaceTempView(df_name)
            return df_name
        elif isinstance(val, str):
            return lit(val)._jc.expr().sql()
        else:
            return val

    def clear(self) -> None:
        if False:
            i = 10
            return i + 15
        for (_, n) in self._temp_views:
            self._session.catalog.dropTempView(n)
        self._temp_views = []