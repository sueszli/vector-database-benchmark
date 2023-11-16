"""
We want to stop populating 'event.content', so we need to make it nullable.

If this has to be rolled back, then the following should populate the missing data:

Postgres:

    UPDATE events SET content=(ej.json::json)->'content' FROM event_json ej
    WHERE ej.event_id = events.event_id AND
        stream_ordering < (
            SELECT stream_ordering FROM events WHERE content IS NOT NULL
            ORDER BY stream_ordering LIMIT 1
        );

    UPDATE events SET content=(ej.json::json)->'content' FROM event_json ej
    WHERE ej.event_id = events.event_id AND
        stream_ordering > (
            SELECT stream_ordering FROM events WHERE content IS NOT NULL
            ORDER BY stream_ordering DESC LIMIT 1
        );

SQLite:

    UPDATE events SET content=(
        SELECT json_extract(json,'$.content') FROM event_json ej
        WHERE ej.event_id = events.event_id
    )
    WHERE
        stream_ordering < (
            SELECT stream_ordering FROM events WHERE content IS NOT NULL
            ORDER BY stream_ordering LIMIT 1
        )
        OR stream_ordering > (
            SELECT stream_ordering FROM events WHERE content IS NOT NULL
            ORDER BY stream_ordering DESC LIMIT 1
        );

"""
import logging
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine
logger = logging.getLogger(__name__)

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(database_engine, PostgresEngine):
        cur.execute('\n            ALTER TABLE events ALTER COLUMN content DROP NOT NULL;\n        ')
        return
    cur.execute("SELECT sql FROM sqlite_master WHERE tbl_name='events' AND type='table'")
    row = cur.fetchone()
    assert row is not None
    (oldsql,) = row
    sql = oldsql.replace('content TEXT NOT NULL', 'content TEXT')
    if sql == oldsql:
        raise Exception("Couldn't find null constraint to drop in %s" % oldsql)
    logger.info("Replacing definition of 'events' with: %s", sql)
    cur.execute('PRAGMA schema_version')
    row = cur.fetchone()
    assert row is not None
    (oldver,) = row
    cur.execute('PRAGMA writable_schema=ON')
    cur.execute("UPDATE sqlite_master SET sql=? WHERE tbl_name='events' AND type='table'", (sql,))
    cur.execute('PRAGMA schema_version=%i' % (oldver + 1,))
    cur.execute('PRAGMA writable_schema=OFF')