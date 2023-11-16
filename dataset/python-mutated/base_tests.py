from tests.integration_tests.test_app import app
from tests.integration_tests.base_tests import SupersetTestCase
from superset.db_engine_specs.base import BaseEngineSpec
from superset.models.core import Database

class TestDbEngineSpec(SupersetTestCase):

    def sql_limit_regex(self, sql, expected_sql, engine_spec_class=BaseEngineSpec, limit=1000, force=False):
        if False:
            i = 10
            return i + 15
        main = Database(database_name='test_database', sqlalchemy_uri='sqlite://')
        limited = engine_spec_class.apply_limit_to_sql(sql, limit, main, force)
        self.assertEqual(expected_sql, limited)