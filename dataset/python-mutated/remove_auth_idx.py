import logging
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine
from synapse.storage.prepare_database import get_statements
logger = logging.getLogger(__name__)
DROP_INDICES = "\n-- We only ever query based on event_id\nDROP INDEX IF EXISTS state_events_room_id;\nDROP INDEX IF EXISTS state_events_type;\nDROP INDEX IF EXISTS state_events_state_key;\n\n-- room_id is indexed elsewhere\nDROP INDEX IF EXISTS current_state_events_room_id;\nDROP INDEX IF EXISTS current_state_events_state_key;\nDROP INDEX IF EXISTS current_state_events_type;\n\nDROP INDEX IF EXISTS transactions_have_ref;\n\n-- (topological_ordering, stream_ordering, room_id) seems like a strange index,\n-- and is used incredibly rarely.\nDROP INDEX IF EXISTS events_order_topo_stream_room;\n\n-- an equivalent index to this actually gets re-created in delta 41, because it\n-- turned out that deleting it wasn't a great plan :/. In any case, let's\n-- delete it here, and delta 41 will create a new one with an added UNIQUE\n-- constraint\nDROP INDEX IF EXISTS event_search_ev_idx;\n"
POSTGRES_DROP_CONSTRAINT = '\nALTER TABLE event_auth DROP CONSTRAINT IF EXISTS event_auth_event_id_auth_id_room_id_key;\n'
SQLITE_DROP_CONSTRAINT = '\nDROP INDEX IF EXISTS evauth_edges_id;\n\nCREATE TABLE IF NOT EXISTS event_auth_new(\n    event_id TEXT NOT NULL,\n    auth_id TEXT NOT NULL,\n    room_id TEXT NOT NULL\n);\n\nINSERT INTO event_auth_new\n    SELECT event_id, auth_id, room_id\n    FROM event_auth;\n\nDROP TABLE event_auth;\n\nALTER TABLE event_auth_new RENAME TO event_auth;\n\nCREATE INDEX evauth_edges_id ON event_auth(event_id);\n'

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        print('Hello World!')
    for statement in get_statements(DROP_INDICES.splitlines()):
        cur.execute(statement)
    if isinstance(database_engine, PostgresEngine):
        drop_constraint = POSTGRES_DROP_CONSTRAINT
    else:
        drop_constraint = SQLITE_DROP_CONSTRAINT
    for statement in get_statements(drop_constraint.splitlines()):
        cur.execute(statement)