import json
import logging
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine
from synapse.storage.prepare_database import get_statements
logger = logging.getLogger(__name__)
ALTER_TABLE = 'ALTER TABLE events ADD COLUMN origin_server_ts BIGINT;CREATE INDEX events_ts ON events(origin_server_ts, stream_ordering);'

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        return 10
    for statement in get_statements(ALTER_TABLE.splitlines()):
        cur.execute(statement)
    cur.execute('SELECT MIN(stream_ordering) FROM events')
    rows = cur.fetchall()
    min_stream_id = rows[0][0]
    cur.execute('SELECT MAX(stream_ordering) FROM events')
    rows = cur.fetchall()
    max_stream_id = rows[0][0]
    if min_stream_id is not None and max_stream_id is not None:
        progress = {'target_min_stream_id_inclusive': min_stream_id, 'max_stream_id_exclusive': max_stream_id + 1, 'rows_inserted': 0}
        progress_json = json.dumps(progress)
        sql = 'INSERT into background_updates (update_name, progress_json) VALUES (?, ?)'
        cur.execute(sql, ('event_origin_server_ts', progress_json))