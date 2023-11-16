"""
This migration rebuilds the device_lists_outbound_last_success table without duplicate
entries, and with a UNIQUE index.
"""
import logging
from io import StringIO
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine
from synapse.storage.prepare_database import execute_statements_from_stream
logger = logging.getLogger(__name__)

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        return 10
    if isinstance(database_engine, PostgresEngine):
        cur.execute("\n            SELECT 1 FROM pg_class WHERE relkind = 'i'\n            AND relname = 'device_lists_outbound_last_success_unique_idx'\n            ")
        if cur.rowcount:
            logger.info('Unique index exists on device_lists_outbound_last_success: skipping rebuild')
            return
    logger.info('Rebuilding device_lists_outbound_last_success with unique index')
    execute_statements_from_stream(cur, StringIO(_rebuild_commands))
_rebuild_commands = "\nDROP TABLE IF EXISTS device_lists_outbound_last_success_new;\n\nCREATE TABLE device_lists_outbound_last_success_new (\n    destination TEXT NOT NULL,\n    user_id TEXT NOT NULL,\n    stream_id BIGINT NOT NULL\n);\n\n-- this took about 30 seconds on matrix.org's 16 million rows.\nINSERT INTO device_lists_outbound_last_success_new\n    SELECT destination, user_id, MAX(stream_id) FROM device_lists_outbound_last_success\n    GROUP BY destination, user_id;\n\n-- and this another 30 seconds.\nCREATE UNIQUE INDEX device_lists_outbound_last_success_unique_idx\n    ON device_lists_outbound_last_success_new (destination, user_id);\n\nDROP TABLE device_lists_outbound_last_success;\n\nALTER TABLE device_lists_outbound_last_success_new\n    RENAME TO device_lists_outbound_last_success;\n"