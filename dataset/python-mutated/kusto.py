import re
from datetime import datetime
from typing import Any, Optional
from sqlalchemy import types
from sqlalchemy.dialects.mssql.base import SMALLDATETIME
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec, LimitMethod
from superset.db_engine_specs.exceptions import SupersetDBAPIDatabaseError, SupersetDBAPIOperationalError, SupersetDBAPIProgrammingError
from superset.sql_parse import ParsedQuery
from superset.utils.core import GenericDataType

class KustoSqlEngineSpec(BaseEngineSpec):
    limit_method = LimitMethod.WRAP_SQL
    engine = 'kustosql'
    engine_name = 'KustoSQL'
    time_groupby_inline = True
    allows_joins = True
    allows_subqueries = True
    allows_sql_comments = False
    _time_grain_expressions = {None: '{col}', TimeGrain.SECOND: "DATEADD(second,             'DATEDIFF(second, 2000-01-01', {col}), '2000-01-01')", TimeGrain.MINUTE: 'DATEADD(minute, DATEDIFF(minute, 0, {col}), 0)', TimeGrain.FIVE_MINUTES: 'DATEADD(minute, DATEDIFF(minute, 0, {col}) / 5 * 5, 0)', TimeGrain.TEN_MINUTES: 'DATEADD(minute,             DATEDIFF(minute, 0, {col}) / 10 * 10, 0)', TimeGrain.FIFTEEN_MINUTES: 'DATEADD(minute,             DATEDIFF(minute, 0, {col}) / 15 * 15, 0)', TimeGrain.HALF_HOUR: 'DATEADD(minute, DATEDIFF(minute, 0, {col}) / 30 * 30, 0)', TimeGrain.HOUR: 'DATEADD(hour, DATEDIFF(hour, 0, {col}), 0)', TimeGrain.DAY: 'DATEADD(day, DATEDIFF(day, 0, {col}), 0)', TimeGrain.WEEK: 'DATEADD(day, -1, DATEADD(week, DATEDIFF(week, 0, {col}), 0))', TimeGrain.MONTH: 'DATEADD(month, DATEDIFF(month, 0, {col}), 0)', TimeGrain.QUARTER: 'DATEADD(quarter, DATEDIFF(quarter, 0, {col}), 0)', TimeGrain.YEAR: 'DATEADD(year, DATEDIFF(year, 0, {col}), 0)', TimeGrain.WEEK_STARTING_SUNDAY: 'DATEADD(day, -1, DATEADD(week, DATEDIFF(week, 0, {col}), 0))', TimeGrain.WEEK_STARTING_MONDAY: 'DATEADD(week, DATEDIFF(week, 0, DATEADD(day, -1, {col})), 0)'}
    type_code_map: dict[int, str] = {}
    column_type_mappings = ((re.compile('^smalldatetime.*', re.IGNORECASE), SMALLDATETIME(), GenericDataType.TEMPORAL),)

    @classmethod
    def get_dbapi_exception_mapping(cls) -> dict[type[Exception], type[Exception]]:
        if False:
            while True:
                i = 10
        import sqlalchemy_kusto.errors as kusto_exceptions
        return {kusto_exceptions.DatabaseError: SupersetDBAPIDatabaseError, kusto_exceptions.OperationalError: SupersetDBAPIOperationalError, kusto_exceptions.ProgrammingError: SupersetDBAPIProgrammingError}

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[dict[str, Any]]=None) -> Optional[str]:
        if False:
            return 10
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"CONVERT(DATE, '{dttm.date().isoformat()}', 23)"
        if isinstance(sqla_type, types.TIMESTAMP):
            datetime_formatted = dttm.isoformat(sep=' ', timespec='seconds')
            return f"CONVERT(TIMESTAMP, '{datetime_formatted}', 20)"
        if isinstance(sqla_type, SMALLDATETIME):
            datetime_formatted = dttm.isoformat(sep=' ', timespec='seconds')
            return f"CONVERT(SMALLDATETIME, '{datetime_formatted}', 20)"
        if isinstance(sqla_type, types.DateTime):
            datetime_formatted = dttm.isoformat(timespec='milliseconds')
            return f"CONVERT(DATETIME, '{datetime_formatted}', 126)"
        return None

    @classmethod
    def is_readonly_query(cls, parsed_query: ParsedQuery) -> bool:
        if False:
            while True:
                i = 10
        "Pessimistic readonly, 100% sure statement won't mutate anything"
        return parsed_query.sql.lower().startswith('select')

class KustoKqlEngineSpec(BaseEngineSpec):
    limit_method = LimitMethod.WRAP_SQL
    engine = 'kustokql'
    engine_name = 'KustoKQL'
    time_groupby_inline = True
    allows_joins = True
    allows_subqueries = True
    allows_sql_comments = False
    run_multiple_statements_as_one = True
    _time_grain_expressions = {None: '{col}', TimeGrain.SECOND: '{col}/ time(1s)', TimeGrain.MINUTE: '{col}/ time(1min)', TimeGrain.HOUR: '{col}/ time(1h)', TimeGrain.DAY: '{col}/ time(1d)', TimeGrain.MONTH: "datetime_diff('month', CreateDate,             datetime(0001-01-01 00:00:00))+1", TimeGrain.YEAR: "datetime_diff('year', CreateDate,             datetime(0001-01-01 00:00:00))+1"}
    type_code_map: dict[int, str] = {}

    @classmethod
    def get_dbapi_exception_mapping(cls) -> dict[type[Exception], type[Exception]]:
        if False:
            while True:
                i = 10
        import sqlalchemy_kusto.errors as kusto_exceptions
        return {kusto_exceptions.DatabaseError: SupersetDBAPIDatabaseError, kusto_exceptions.OperationalError: SupersetDBAPIOperationalError, kusto_exceptions.ProgrammingError: SupersetDBAPIProgrammingError}

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[dict[str, Any]]=None) -> Optional[str]:
        if False:
            print('Hello World!')
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f'datetime({dttm.date().isoformat()})'
        if isinstance(sqla_type, types.DateTime):
            return f"datetime({dttm.isoformat(timespec='microseconds')})"
        return None

    @classmethod
    def is_readonly_query(cls, parsed_query: ParsedQuery) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "\n        Pessimistic readonly, 100% sure statement won't mutate anything.\n        "
        return KustoKqlEngineSpec.is_select_query(parsed_query) or parsed_query.sql.startswith('.show')

    @classmethod
    def is_select_query(cls, parsed_query: ParsedQuery) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not parsed_query.sql.startswith('.')

    @classmethod
    def parse_sql(cls, sql: str) -> list[str]:
        if False:
            while True:
                i = 10
        '\n        Kusto supports a single query statement, but it could include sub queries\n        and variables declared via let keyword.\n        '
        return [sql]