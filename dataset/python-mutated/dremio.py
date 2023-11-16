from __future__ import annotations
from datetime import datetime
from typing import Any, TYPE_CHECKING
from packaging.version import Version
from sqlalchemy import types
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec
if TYPE_CHECKING:
    from superset.models.core import Database
FIXED_ALIAS_IN_SELECT_VERSION = Version('24.1.0')

class DremioEngineSpec(BaseEngineSpec):
    engine = 'dremio'
    engine_name = 'Dremio'
    _time_grain_expressions = {None: '{col}', TimeGrain.SECOND: "DATE_TRUNC('second', {col})", TimeGrain.MINUTE: "DATE_TRUNC('minute', {col})", TimeGrain.HOUR: "DATE_TRUNC('hour', {col})", TimeGrain.DAY: "DATE_TRUNC('day', {col})", TimeGrain.WEEK: "DATE_TRUNC('week', {col})", TimeGrain.MONTH: "DATE_TRUNC('month', {col})", TimeGrain.QUARTER: "DATE_TRUNC('quarter', {col})", TimeGrain.YEAR: "DATE_TRUNC('year', {col})"}

    @classmethod
    def epoch_to_dttm(cls) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'TO_DATE({col})'

    @classmethod
    def get_allows_alias_in_select(cls, database: Database) -> bool:
        if False:
            while True:
                i = 10
        "\n        Dremio supports aliases in SELECT statements since version 24.1.0.\n\n        If no version is specified in the DB extra, we assume the Dremio version is post\n        24.1.0. This way, as we move forward people don't have to specify a version when\n        setting up their databases.\n        "
        version = database.get_extra().get('version')
        if version and Version(version) < FIXED_ALIAS_IN_SELECT_VERSION:
            return False
        return True

    @classmethod
    def convert_dttm(cls, target_type: str, dttm: datetime, db_extra: dict[str, Any] | None=None) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        sqla_type = cls.get_sqla_column_type(target_type)
        if isinstance(sqla_type, types.Date):
            return f"TO_DATE('{dttm.date().isoformat()}', 'YYYY-MM-DD')"
        if isinstance(sqla_type, types.TIMESTAMP):
            dttm_formatted = dttm.isoformat(sep=' ', timespec='milliseconds')
            return f"TO_TIMESTAMP('{dttm_formatted}', 'YYYY-MM-DD HH24:MI:SS.FFF')"
        return None