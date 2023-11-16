from unittest import mock
from superset.db_engine_specs import get_engine_spec
from superset.db_engine_specs.databricks import DatabricksNativeEngineSpec
from tests.integration_tests.db_engine_specs.base_tests import TestDbEngineSpec
from tests.integration_tests.fixtures.certificates import ssl_certificate
from tests.integration_tests.fixtures.database import default_db_extra

class TestDatabricksDbEngineSpec(TestDbEngineSpec):

    def test_get_engine_spec(self):
        if False:
            return 10
        '\n        DB Eng Specs (databricks): Test "databricks" in engine spec\n        '
        assert get_engine_spec('databricks', 'connector').engine == 'databricks'
        assert get_engine_spec('databricks', 'pyodbc').engine == 'databricks'
        assert get_engine_spec('databricks', 'pyhive').engine == 'databricks'

    def test_extras_without_ssl(self):
        if False:
            i = 10
            return i + 15
        db = mock.Mock()
        db.extra = default_db_extra
        db.server_cert = None
        extras = DatabricksNativeEngineSpec.get_extra_params(db)
        assert extras == {'engine_params': {'connect_args': {'_user_agent_entry': 'Apache Superset', 'http_headers': [('User-Agent', 'Apache Superset')]}}, 'metadata_cache_timeout': {}, 'metadata_params': {}, 'schemas_allowed_for_file_upload': []}

    def test_extras_with_ssl_custom(self):
        if False:
            i = 10
            return i + 15
        db = mock.Mock()
        db.extra = default_db_extra.replace('"engine_params": {}', '"engine_params": {"connect_args": {"ssl": "1"}}')
        db.server_cert = ssl_certificate
        extras = DatabricksNativeEngineSpec.get_extra_params(db)
        connect_args = extras['engine_params']['connect_args']
        assert connect_args['ssl'] == '1'