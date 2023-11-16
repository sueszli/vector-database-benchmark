from urllib.parse import urlparse
import pytest
import yaml
from dagster._core.test_utils import environ, instance_for_test
from dagster_mysql.run_storage import MySQLRunStorage
from dagster_tests.storage_tests.utils.run_storage import TestRunStorage
TestRunStorage.__test__ = False

class TestMySQLRunStorage(TestRunStorage):
    __test__ = True

    @pytest.fixture(scope='function', name='storage')
    def run_storage(self, conn_string):
        if False:
            i = 10
            return i + 15
        storage = MySQLRunStorage.create_clean_storage(conn_string)
        assert storage
        return storage

    def test_load_from_config(self, conn_string):
        if False:
            return 10
        parse_result = urlparse(conn_string)
        hostname = parse_result.hostname
        port = parse_result.port
        url_cfg = f'\n          run_storage:\n            module: dagster_mysql.run_storage\n            class: MySQLRunStorage\n            config:\n              mysql_url: mysql+mysqlconnector://test:test@{hostname}:{port}/test\n        '
        explicit_cfg = f'\n          run_storage:\n            module: dagster_mysql.run_storage\n            class: MySQLRunStorage\n            config:\n              mysql_db:\n                username: test\n                password: test\n                hostname: {hostname}\n                db_name: test\n                port: {port}\n        '
        with environ({'TEST_MYSQL_PASSWORD': 'test'}):
            env_cfg = f'\n            run_storage:\n              module: dagster_mysql.run_storage\n              class: MySQLRunStorage\n              config:\n                mysql_db:\n                  username: test\n                  password:\n                    env: TEST_MYSQL_PASSWORD\n                  hostname: {hostname}\n                  db_name: test\n                  port: {port}\n            '
            with instance_for_test(overrides=yaml.safe_load(url_cfg)) as from_url_instance:
                with instance_for_test(overrides=yaml.safe_load(explicit_cfg)) as from_explicit_instance:
                    assert from_url_instance._run_storage.mysql_url == from_explicit_instance._run_storage.mysql_url
                with instance_for_test(overrides=yaml.safe_load(env_cfg)) as from_env_instance:
                    assert from_url_instance._run_storage.mysql_url == from_env_instance._run_storage.mysql_url