"""
This migration adds triggers to the partial_state_events tables to enforce uniqueness

Triggers cannot be expressed in .sql files, so we have to use a separate file.
"""
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine, Sqlite3Engine

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        i = 10
        return i + 15
    if isinstance(database_engine, Sqlite3Engine):
        cur.execute("\n            CREATE TRIGGER IF NOT EXISTS partial_state_events_bad_room_id\n            BEFORE INSERT ON partial_state_events\n            FOR EACH ROW\n            BEGIN\n                SELECT RAISE(ABORT, 'Incorrect room_id in partial_state_events')\n                WHERE EXISTS (\n                    SELECT 1 FROM events\n                    WHERE events.event_id = NEW.event_id\n                       AND events.room_id != NEW.room_id\n                );\n            END;\n            ")
    elif isinstance(database_engine, PostgresEngine):
        cur.execute("\n            CREATE OR REPLACE FUNCTION check_partial_state_events() RETURNS trigger AS $BODY$\n            BEGIN\n                IF EXISTS (\n                    SELECT 1 FROM events\n                    WHERE events.event_id = NEW.event_id\n                       AND events.room_id != NEW.room_id\n                ) THEN\n                    RAISE EXCEPTION 'Incorrect room_id in partial_state_events';\n                END IF;\n                RETURN NEW;\n            END;\n            $BODY$ LANGUAGE plpgsql;\n            ")
        cur.execute('\n            CREATE TRIGGER check_partial_state_events BEFORE INSERT OR UPDATE ON partial_state_events\n            FOR EACH ROW\n            EXECUTE PROCEDURE check_partial_state_events()\n            ')
    else:
        raise NotImplementedError('Unknown database engine')