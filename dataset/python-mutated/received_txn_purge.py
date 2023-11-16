import logging
from synapse.storage.database import LoggingTransaction
from synapse.storage.engines import BaseDatabaseEngine, PostgresEngine
logger = logging.getLogger(__name__)

def run_create(cur: LoggingTransaction, database_engine: BaseDatabaseEngine) -> None:
    if False:
        return 10
    if isinstance(database_engine, PostgresEngine):
        cur.execute('TRUNCATE received_transactions')
    else:
        cur.execute('DELETE FROM received_transactions')
    cur.execute('CREATE INDEX received_transactions_ts ON received_transactions(ts)')