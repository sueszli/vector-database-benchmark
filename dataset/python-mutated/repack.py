"""Purge repack helper."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from sqlalchemy import text
from .const import SupportedDialect
from .db_schema import ALL_TABLES
if TYPE_CHECKING:
    from . import Recorder
_LOGGER = logging.getLogger(__name__)

def repack_database(instance: Recorder) -> None:
    if False:
        while True:
            i = 10
    'Repack based on engine type.'
    assert instance.engine is not None
    dialect_name = instance.engine.dialect.name
    if dialect_name == SupportedDialect.SQLITE:
        _LOGGER.debug('Vacuuming SQL DB to free space')
        with instance.engine.connect() as conn:
            conn.execute(text('VACUUM'))
            conn.commit()
        return
    if dialect_name == SupportedDialect.POSTGRESQL:
        _LOGGER.debug('Vacuuming SQL DB to free space')
        with instance.engine.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
            conn.execute(text('VACUUM'))
            conn.commit()
        return
    if dialect_name == SupportedDialect.MYSQL:
        _LOGGER.debug('Optimizing SQL DB to free space')
        with instance.engine.connect() as conn:
            conn.execute(text(f"OPTIMIZE TABLE {','.join(ALL_TABLES)}"))
            conn.commit()
        return