from unittest.mock import MagicMock, patch
from synapse.storage.database import make_conn
from synapse.storage.engines import PostgresEngine
from synapse.storage.engines._base import IncorrectDatabaseSetup
from tests.unittest import HomeserverTestCase
from tests.utils import USE_POSTGRES_FOR_TESTS

class UnsafeLocaleTest(HomeserverTestCase):
    if not USE_POSTGRES_FOR_TESTS:
        skip = 'Requires Postgres'

    @patch('synapse.storage.engines.postgres.PostgresEngine.get_db_locale')
    def test_unsafe_locale(self, mock_db_locale: MagicMock) -> None:
        if False:
            i = 10
            return i + 15
        mock_db_locale.return_value = ('B', 'B')
        database = self.hs.get_datastores().databases[0]
        db_conn = make_conn(database._database_config, database.engine, 'test_unsafe')
        with self.assertRaises(IncorrectDatabaseSetup):
            database.engine.check_database(db_conn)
        with self.assertRaises(IncorrectDatabaseSetup):
            database.engine.check_new_database(db_conn)
        db_conn.close()

    def test_safe_locale(self) -> None:
        if False:
            while True:
                i = 10
        database = self.hs.get_datastores().databases[0]
        assert isinstance(database.engine, PostgresEngine)
        db_conn = make_conn(database._database_config, database.engine, 'test_unsafe')
        with db_conn.cursor() as txn:
            res = database.engine.get_db_locale(txn)
        self.assertEqual(res, ('C', 'C'))
        db_conn.close()