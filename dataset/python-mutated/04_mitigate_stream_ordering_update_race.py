from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        i = 10
        return i + 15
    '\n    An attempt to mitigate a painful race between foreground and background updates\n    touching the `stream_ordering` column of the events table. More info can be found\n    at https://github.com/matrix-org/synapse/issues/15677.\n    '
    if isinstance(database_engine, PostgresEngine):
        select_sql = "\n            SELECT 1 FROM background_updates\n                WHERE update_name = 'replace_stream_ordering_column'\n        "
        cur.execute(select_sql)
        res = cur.fetchone()
        if res:
            drop_cse_sql = '\n            ALTER TABLE current_state_events DROP CONSTRAINT IF EXISTS event_stream_ordering_fkey\n            '
            cur.execute(drop_cse_sql)
            drop_lcm_sql = '\n            ALTER TABLE local_current_membership DROP CONSTRAINT IF EXISTS event_stream_ordering_fkey\n            '
            cur.execute(drop_lcm_sql)
            drop_rm_sql = '\n            ALTER TABLE room_memberships DROP CONSTRAINT IF EXISTS event_stream_ordering_fkey\n            '
            cur.execute(drop_rm_sql)
            add_cse_sql = '\n            ALTER TABLE current_state_events ADD CONSTRAINT event_stream_ordering_fkey\n            FOREIGN KEY (event_stream_ordering) REFERENCES events(stream_ordering2) NOT VALID;\n            '
            cur.execute(add_cse_sql)
            add_lcm_sql = '\n            ALTER TABLE local_current_membership ADD CONSTRAINT event_stream_ordering_fkey\n            FOREIGN KEY (event_stream_ordering) REFERENCES events(stream_ordering2) NOT VALID;\n            '
            cur.execute(add_lcm_sql)
            add_rm_sql = '\n            ALTER TABLE room_memberships ADD CONSTRAINT event_stream_ordering_fkey\n            FOREIGN KEY (event_stream_ordering) REFERENCES events(stream_ordering2) NOT VALID;\n            '
            cur.execute(add_rm_sql)