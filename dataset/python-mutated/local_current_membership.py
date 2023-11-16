from synapse.config.homeserver import HomeServerConfig
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine

def run_upgrade(cur: LoggingTransaction, database_engine: BaseDatabaseEngine, config: HomeServerConfig) -> None:
    if False:
        return 10
    cur.execute("SELECT 1 FROM background_updates\n            WHERE update_name = 'current_state_events_membership'\n        ")
    current_state_membership_up_to_date = not bool(cur.fetchone())
    cur.execute('DROP INDEX local_current_membership_idx')
    cur.execute('DROP INDEX local_current_membership_room_idx')
    if current_state_membership_up_to_date:
        sql = "\n            INSERT INTO local_current_membership (room_id, user_id, event_id, membership)\n                SELECT c.room_id, state_key AS user_id, event_id, c.membership\n                FROM current_state_events AS c\n                WHERE type = 'm.room.member' AND c.membership IS NOT NULL AND state_key LIKE ?\n        "
    else:
        sql = "\n            INSERT INTO local_current_membership (room_id, user_id, event_id, membership)\n                SELECT c.room_id, state_key AS user_id, event_id, r.membership\n                FROM current_state_events AS c\n                INNER JOIN room_memberships AS r USING (event_id)\n                WHERE type = 'm.room.member' AND state_key LIKE ?\n        "
    cur.execute(sql, ('%:' + config.server.server_name,))
    cur.execute('CREATE UNIQUE INDEX local_current_membership_idx ON local_current_membership(user_id, room_id)')
    cur.execute('CREATE INDEX local_current_membership_room_idx ON local_current_membership(room_id)')

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        i = 10
        return i + 15
    cur.execute('\n        CREATE TABLE local_current_membership (\n            room_id TEXT NOT NULL,\n            user_id TEXT NOT NULL,\n            event_id TEXT NOT NULL,\n            membership TEXT NOT NULL\n        )')
    cur.execute('CREATE UNIQUE INDEX local_current_membership_idx ON local_current_membership(user_id, room_id)')
    cur.execute('CREATE INDEX local_current_membership_room_idx ON local_current_membership(room_id)')