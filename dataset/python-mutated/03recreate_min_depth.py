"""
This migration handles the process of changing the type of `room_depth.min_depth` to
a BIGINT.
"""
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        return 10
    if not isinstance(database_engine, PostgresEngine):
        return
    cur.execute('ALTER TABLE room_depth ADD COLUMN min_depth2 BIGINT')
    cur.execute('\n        CREATE OR REPLACE FUNCTION populate_min_depth2() RETURNS trigger AS $BODY$\n            BEGIN\n                new.min_depth2 := new.min_depth;\n                RETURN NEW;\n            END;\n        $BODY$ LANGUAGE plpgsql\n        ')
    cur.execute('\n        CREATE TRIGGER populate_min_depth2_trigger BEFORE INSERT OR UPDATE ON room_depth\n        FOR EACH ROW\n        EXECUTE PROCEDURE populate_min_depth2()\n        ')
    cur.execute("\n       INSERT INTO background_updates (ordering, update_name, progress_json) VALUES\n            (6103, 'populate_room_depth_min_depth2', '{}')\n       ")
    cur.execute("\n        INSERT INTO background_updates (ordering, update_name, progress_json, depends_on) VALUES\n            (6103, 'replace_room_depth_min_depth', '{}', 'populate_room_depth2')\n        ")