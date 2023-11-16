import pytest
from dagster_mysql.run_storage import MySQLRunStorage
from dagster_tests.storage_tests.utils.daemon_cursor_storage import TestDaemonCursorStorage

class TestMySQLDaemonCursorStorage(TestDaemonCursorStorage):
    __test__ = True

    @pytest.fixture(scope='function', name='storage')
    def cursor_storage(self, conn_string):
        if False:
            i = 10
            return i + 15
        storage = MySQLRunStorage.create_clean_storage(conn_string)
        assert storage
        return storage