"""
Adds a postgres SEQUENCE for generating application service transaction IDs.
"""
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        print('Hello World!')
    if isinstance(database_engine, PostgresEngine):
        cur.execute('SELECT COALESCE(max(txn_id), 0) FROM application_services_txns')
        row = cur.fetchone()
        assert row is not None
        txn_max = row[0]
        cur.execute('SELECT COALESCE(max(last_txn), 0) FROM application_services_state')
        row = cur.fetchone()
        assert row is not None
        last_txn_max = row[0]
        start_val = max(last_txn_max, txn_max) + 1
        cur.execute('CREATE SEQUENCE application_services_txn_id_seq START WITH %s', (start_val,))