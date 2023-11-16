from typing import List
from unittest import mock
from twisted.test.proto_helpers import MemoryReactor
from synapse.app.generic_worker import GenericWorkerServer
from synapse.server import HomeServer
from synapse.storage.database import LoggingDatabaseConnection
from synapse.storage.prepare_database import PrepareDatabaseException, prepare_database
from synapse.storage.schema import SCHEMA_VERSION
from synapse.types import JsonDict
from synapse.util import Clock
from tests.unittest import HomeserverTestCase

def fake_listdir(filepath: str) -> List[str]:
    if False:
        return 10
    '\n    A fake implementation of os.listdir which we can use to mock out the filesystem.\n\n    Args:\n        filepath: The directory to list files for.\n\n    Returns:\n        A list of files and folders in the directory.\n    '
    if filepath.endswith('full_schemas'):
        return [str(SCHEMA_VERSION)]
    return ['99_add_unicorn_to_database.sql']

class WorkerSchemaTests(HomeserverTestCase):

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            return 10
        hs = self.setup_test_homeserver(homeserver_to_use=GenericWorkerServer)
        return hs

    def default_config(self) -> JsonDict:
        if False:
            print('Hello World!')
        conf = super().default_config()
        conf['worker_app'] = 'yes'
        conf['instance_map'] = {'main': {'host': '127.0.0.1', 'port': 0}}
        return conf

    def test_rolling_back(self) -> None:
        if False:
            print('Hello World!')
        'Test that workers can start if the DB is a newer schema version'
        db_pool = self.hs.get_datastores().main.db_pool
        db_conn = LoggingDatabaseConnection(db_pool._db_pool.connect(), db_pool.engine, 'tests')
        cur = db_conn.cursor()
        cur.execute('UPDATE schema_version SET version = ?', (SCHEMA_VERSION + 1,))
        db_conn.commit()
        prepare_database(db_conn, db_pool.engine, self.hs.config)

    def test_not_upgraded_old_schema_version(self) -> None:
        if False:
            i = 10
            return i + 15
        "Test that workers don't start if the DB has an older schema version"
        db_pool = self.hs.get_datastores().main.db_pool
        db_conn = LoggingDatabaseConnection(db_pool._db_pool.connect(), db_pool.engine, 'tests')
        cur = db_conn.cursor()
        cur.execute('UPDATE schema_version SET version = ?', (SCHEMA_VERSION - 1,))
        db_conn.commit()
        with self.assertRaises(PrepareDatabaseException):
            prepare_database(db_conn, db_pool.engine, self.hs.config)

    def test_not_upgraded_current_schema_version_with_outstanding_deltas(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Test that workers don't start if the DB is on the current schema version,\n        but there are still outstanding delta migrations to run.\n        "
        db_pool = self.hs.get_datastores().main.db_pool
        db_conn = LoggingDatabaseConnection(db_pool._db_pool.connect(), db_pool.engine, 'tests')
        cur = db_conn.cursor()
        cur.execute('UPDATE schema_version SET version = ?', (SCHEMA_VERSION,))
        db_conn.commit()
        with mock.patch('os.listdir', mock.Mock(side_effect=fake_listdir)):
            with self.assertRaises(PrepareDatabaseException):
                prepare_database(db_conn, db_pool.engine, self.hs.config)