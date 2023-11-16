import json
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        print('Hello World!')
    'Add a bg update to populate the `state_key` and `rejection_reason` columns of `events`'
    cur.execute('SELECT MIN(stream_ordering), MAX(stream_ordering) FROM events')
    row = cur.fetchone()
    assert row is not None
    (min_stream_ordering, max_stream_ordering) = row
    if min_stream_ordering is None:
        return
    cur.execute("INSERT into background_updates (ordering, update_name, progress_json) VALUES (7203, 'events_populate_state_key_rejections', ?)", (json.dumps({'min_stream_ordering_exclusive': min_stream_ordering - 1, 'max_stream_ordering_inclusive': max_stream_ordering}),))