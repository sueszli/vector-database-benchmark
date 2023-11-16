"""
Adds a postgres SEQUENCE for generating guest user IDs.
"""
from synapse.storage.database import LoggingTransaction
from synapse.storage.databases.main.registration import find_max_generated_user_id_localpart
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        while True:
            i = 10
    if not isinstance(database_engine, PostgresEngine):
        return
    next_id = find_max_generated_user_id_localpart(cur) + 1
    cur.execute('CREATE SEQUENCE user_id_seq START WITH %s', (next_id,))