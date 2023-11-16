"""
This migration adds foreign key constraint to `event_forward_extremities` table.
"""
from synapse.storage.background_updates import ForeignKeyConstraint, run_validate_constraint_and_delete_rows_schema_delta
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine
FORWARD_EXTREMITIES_TABLE_SCHEMA = '\n    CREATE TABLE event_forward_extremities2(\n        event_id TEXT NOT NULL,\n        room_id TEXT NOT NULL,\n        UNIQUE (event_id, room_id),\n        CONSTRAINT event_forward_extremities_event_id FOREIGN KEY (event_id) REFERENCES events (event_id) DEFERRABLE INITIALLY DEFERRED\n    )\n'

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        i = 10
        return i + 15
    run_validate_constraint_and_delete_rows_schema_delta(cur, ordering=7803, update_name='event_forward_extremities_event_id_foreign_key_constraint_update', table='event_forward_extremities', constraint_name='event_forward_extremities_event_id', constraint=ForeignKeyConstraint('events', [('event_id', 'event_id')], deferred=True), sqlite_table_name='event_forward_extremities2', sqlite_table_schema=FORWARD_EXTREMITIES_TABLE_SCHEMA)