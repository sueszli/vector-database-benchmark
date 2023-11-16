from __future__ import annotations
from typing import TYPE_CHECKING
from superset.db_engine_specs.sqlite import SqliteEngineSpec
if TYPE_CHECKING:
    from superset.models.core import Database

class ShillelaghEngineSpec(SqliteEngineSpec):
    """Engine for shillelagh"""
    engine_name = 'Shillelagh'
    engine = 'shillelagh'
    drivers = {'apsw': 'SQLite driver'}
    default_driver = 'apsw'
    sqlalchemy_uri_placeholder = 'shillelagh://'
    allows_joins = True
    allows_subqueries = True

    @classmethod
    def get_function_names(cls, database: Database) -> list[str]:
        if False:
            i = 10
            return i + 15
        return super().get_function_names(database) + ['sleep', 'version', 'get_metadata']