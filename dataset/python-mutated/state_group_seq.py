from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        return 10
    if isinstance(database_engine, PostgresEngine):
        cur.execute('SELECT max(id) FROM state_groups')
        row = cur.fetchone()
        assert row is not None
        if row[0] is None:
            start_val = 1
        else:
            start_val = row[0] + 1
        cur.execute('CREATE SEQUENCE state_group_id_seq START WITH %s', (start_val,))