"""
This migration adds triggers to the room membership tables to enforce consistency.
Triggers cannot be expressed in .sql files, so we have to use a separate file.
"""
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine, Sqlite3Engine

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        print('Hello World!')
    if isinstance(database_engine, Sqlite3Engine):
        for table in ('current_state_events', 'local_current_membership', 'room_memberships'):
            cur.execute(f"\n                CREATE TRIGGER IF NOT EXISTS {table}_bad_event_stream_ordering\n                BEFORE INSERT ON {table}\n                FOR EACH ROW\n                BEGIN\n                    SELECT RAISE(ABORT, 'Incorrect event_stream_ordering in {table}')\n                    WHERE EXISTS (\n                        SELECT 1 FROM events\n                        WHERE events.event_id = NEW.event_id\n                           AND events.stream_ordering != NEW.event_stream_ordering\n                    );\n                END;\n                ")
    elif isinstance(database_engine, PostgresEngine):
        cur.execute("\n            CREATE OR REPLACE FUNCTION check_event_stream_ordering() RETURNS trigger AS $BODY$\n            BEGIN\n                IF EXISTS (\n                    SELECT 1 FROM events\n                    WHERE events.event_id = NEW.event_id\n                       AND events.stream_ordering != NEW.event_stream_ordering\n                ) THEN\n                    RAISE EXCEPTION 'Incorrect event_stream_ordering';\n                END IF;\n                RETURN NEW;\n            END;\n            $BODY$ LANGUAGE plpgsql;\n            ")
        for table in ('current_state_events', 'local_current_membership', 'room_memberships'):
            cur.execute(f'\n                CREATE TRIGGER check_event_stream_ordering BEFORE INSERT OR UPDATE ON {table}\n                FOR EACH ROW\n                EXECUTE PROCEDURE check_event_stream_ordering()\n                ')
    else:
        raise NotImplementedError('Unknown database engine')