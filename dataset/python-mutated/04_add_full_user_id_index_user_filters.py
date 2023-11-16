from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, Sqlite3Engine

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(database_engine, Sqlite3Engine):
        idx_sql = '\n        CREATE UNIQUE INDEX IF NOT EXISTS user_filters_full_user_id_unique ON\n        user_filters (full_user_id, filter_id)\n        '
        cur.execute(idx_sql)