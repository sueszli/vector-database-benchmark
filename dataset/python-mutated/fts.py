import json
import logging
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine, Sqlite3Engine
from synapse.storage.prepare_database import get_statements
logger = logging.getLogger(__name__)
POSTGRES_TABLE = '\nCREATE TABLE IF NOT EXISTS event_search (\n    event_id TEXT,\n    room_id TEXT,\n    sender TEXT,\n    key TEXT,\n    vector tsvector\n);\n\nCREATE INDEX event_search_fts_idx ON event_search USING gin(vector);\nCREATE INDEX event_search_ev_idx ON event_search(event_id);\nCREATE INDEX event_search_ev_ridx ON event_search(room_id);\n'
SQLITE_TABLE = 'CREATE VIRTUAL TABLE event_search USING fts4 ( event_id, room_id, sender, key, value )'

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        while True:
            i = 10
    if isinstance(database_engine, PostgresEngine):
        for statement in get_statements(POSTGRES_TABLE.splitlines()):
            cur.execute(statement)
    elif isinstance(database_engine, Sqlite3Engine):
        cur.execute(SQLITE_TABLE)
    else:
        raise Exception('Unrecognized database engine')
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
        cur.execute(sql, ('event_search', progress_json))