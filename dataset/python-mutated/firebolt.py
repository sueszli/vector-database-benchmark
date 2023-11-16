from datetime import datetime
from typing import Any, Optional
from sqlalchemy import types
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec

class FireboltEngineSpec(BaseEngineSpec):
    """Engine spec for Firebolt"""
    engine = 'firebolt'
    engine_name = 'Firebolt'
    default_driver = 'firebolt'
    _time_grain_expressions = {None: '{col}', TimeGrain.SECOND: "date_trunc('second', CAST({col} AS TIMESTAMP))", TimeGrain.MINUTE: "date_trunc('minute', CAST({col} AS TIMESTAMP))", TimeGrain.HOUR: "date_trunc('hour', CAST({col} AS TIMESTAMP))", TimeGrain.DAY: "date_trunc('day', CAST({col} AS TIMESTAMP))", TimeGrain.WEEK: "date_trunc('week', CAST({col} AS TIMESTAMP))", TimeGrain.MONTH: "date_trunc('month', CAST({col} AS TIMESTAMP))", TimeGrain.QUARTER: "date_trunc('quarter', CAST({col} AS TIMESTAMP))", TimeGrain.YEAR: "date_trunc('year', CAST({col} AS TIMESTAMP))"}

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[dict[str, Any]]=None) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"CAST('{dttm.date().isoformat()}' AS DATE)"
        if isinstance(sqla_type, types.TIMESTAMP):
            return f"CAST('{dttm.isoformat(timespec='seconds')}' AS TIMESTAMP)"
        if isinstance(sqla_type, types.DateTime):
            return f"CAST('{dttm.isoformat(timespec='seconds')}' AS DATETIME)"
        return None

    @classmethod
    def epoch_to_dttm(cls) -> str:
        if False:
            i = 10
            return i + 15
        return 'from_unixtime({col})'