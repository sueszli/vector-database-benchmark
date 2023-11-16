import json
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, Sqlite3Engine

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        print('Hello World!')
    "\n    Upgrade the event_search table to use the porter tokenizer if it isn't already\n\n    Applies only for sqlite.\n    "
    if not isinstance(database_engine, Sqlite3Engine):
        return
    cur.execute('DROP TABLE event_search')
    cur.execute('\n        CREATE VIRTUAL TABLE event_search\n        USING fts4 (tokenize=porter, event_id, room_id, sender, key, value )\n        ')
    cur.execute('SELECT MIN(stream_ordering) FROM events')
    row = cur.fetchone()
    assert row is not None
    min_stream_id = row[0]
    if min_stream_id is None:
        return
    cur.execute('SELECT MAX(stream_ordering) FROM events')
    row = cur.fetchone()
    assert row is not None
    max_stream_id = row[0]
    progress = {'target_min_stream_id_inclusive': min_stream_id, 'max_stream_id_exclusive': max_stream_id + 1}
    progress_json = json.dumps(progress)
    sql = '\n    INSERT into background_updates (ordering, update_name, progress_json)\n    VALUES (?, ?, ?)\n    '
    cur.execute(sql, (7310, 'event_search', progress_json))