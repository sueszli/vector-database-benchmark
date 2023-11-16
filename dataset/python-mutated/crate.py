from __future__ import annotations
from datetime import datetime
from typing import Any, TYPE_CHECKING
from sqlalchemy import types
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec
if TYPE_CHECKING:
    from superset.connectors.sqla.models import TableColumn

class CrateEngineSpec(BaseEngineSpec):
    engine = 'crate'
    engine_name = 'CrateDB'
    _time_grain_expressions = {None: '{col}', TimeGrain.SECOND: "DATE_TRUNC('second', {col})", TimeGrain.MINUTE: "DATE_TRUNC('minute', {col})", TimeGrain.HOUR: "DATE_TRUNC('hour', {col})", TimeGrain.DAY: "DATE_TRUNC('day', {col})", TimeGrain.WEEK: "DATE_TRUNC('week', {col})", TimeGrain.MONTH: "DATE_TRUNC('month', {col})", TimeGrain.QUARTER: "DATE_TRUNC('quarter', {col})", TimeGrain.YEAR: "DATE_TRUNC('year', {col})"}

    @classmethod
    def epoch_to_dttm(cls) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '{col} * 1000'

    @classmethod
    def epoch_ms_to_dttm(cls) -> str:
        if False:
            i = 10
            return i + 15
        return '{col}'

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: dict[str, Any] | None=None) -> str | None:
        if False:
            return 10
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.TIMESTAMP):
            return f'{dttm.timestamp() * 1000}'
        return None

    @classmethod
    def alter_new_orm_column(cls, orm_col: TableColumn) -> None:
        if False:
            while True:
                i = 10
        if orm_col.type == 'TIMESTAMP':
            orm_col.python_date_format = 'epoch_ms'