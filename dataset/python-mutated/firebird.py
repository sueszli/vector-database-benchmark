from datetime import datetime
from typing import Any, Optional
from sqlalchemy import types
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec, LimitMethod

class FirebirdEngineSpec(BaseEngineSpec):
    """Engine for Firebird"""
    engine = 'firebird'
    engine_name = 'Firebird'
    limit_method = LimitMethod.FETCH_MANY
    _time_grain_expressions = {None: '{col}', TimeGrain.SECOND: "CAST(CAST({col} AS DATE) || ' ' || EXTRACT(HOUR FROM {col}) || ':' || EXTRACT(MINUTE FROM {col}) || ':' || FLOOR(EXTRACT(SECOND FROM {col})) AS TIMESTAMP)", TimeGrain.MINUTE: "CAST(CAST({col} AS DATE) || ' ' || EXTRACT(HOUR FROM {col}) || ':' || EXTRACT(MINUTE FROM {col}) || ':00' AS TIMESTAMP)", TimeGrain.HOUR: "CAST(CAST({col} AS DATE) || ' ' || EXTRACT(HOUR FROM {col}) || ':00:00' AS TIMESTAMP)", TimeGrain.DAY: 'CAST({col} AS DATE)', TimeGrain.MONTH: "CAST(EXTRACT(YEAR FROM {col}) || '-' || EXTRACT(MONTH FROM {col}) || '-01' AS DATE)", TimeGrain.YEAR: "CAST(EXTRACT(YEAR FROM {col}) || '-01-01' AS DATE)"}

    @classmethod
    def epoch_to_dttm(cls) -> str:
        if False:
            print('Hello World!')
        return "DATEADD(second, {col}, CAST('00:00:00' AS TIMESTAMP))"

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: Optional[dict[str, Any]]=None) -> Optional[str]:
        if False:
            print('Hello World!')
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"CAST('{dttm.date().isoformat()}' AS DATE)"
        if isinstance(sqla_type, types.DateTime):
            dttm_formatted = dttm.isoformat(sep=' ')
            dttm_valid_precision = dttm_formatted[:len('YYYY-MM-DD HH:MM:SS.MMMM')]
            return f"CAST('{dttm_valid_precision}' AS TIMESTAMP)"
        if isinstance(sqla_type, types.Time):
            return f"CAST('{dttm.time().isoformat()}' AS TIME)"
        return None