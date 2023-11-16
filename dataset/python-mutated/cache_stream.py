import logging
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine
from synapse.storage.prepare_database import get_statements
logger = logging.getLogger(__name__)
CREATE_TABLE = '\nCREATE TABLE cache_invalidation_stream (\n    stream_id       BIGINT,\n    cache_func      TEXT,\n    keys            TEXT[],\n    invalidation_ts BIGINT\n);\n\nCREATE INDEX cache_invalidation_stream_id ON cache_invalidation_stream(stream_id);\n'

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        i = 10
        return i + 15
    if not isinstance(database_engine, PostgresEngine):
        return
    for statement in get_statements(CREATE_TABLE.splitlines()):
        cur.execute(statement)