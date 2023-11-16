"""
Forces through the `current_state_events_membership` background job so checks
for its completion can be removed.

Note the background job must still remain defined in the database class.
"""
from synapse.config.homeserver import HomeServerConfig
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine

def run_upgrade(cur: LoggingTransaction, database_engine: BaseDatabaseEngine, config: HomeServerConfig) -> None:
    if False:
        return 10
    cur.execute('SELECT update_name FROM background_updates')
    rows = cur.fetchall()
    for row in rows:
        if row[0] == 'current_state_events_membership':
            break
    else:
        return
    cur.execute('\n        UPDATE current_state_events\n        SET membership = (\n            SELECT membership FROM room_memberships\n            WHERE event_id = current_state_events.event_id\n        )\n        ')
    cur.execute("\n        DELETE FROM background_updates\n        WHERE update_name = 'current_state_events_membership'\n        ")