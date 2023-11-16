import datetime
from pyflink.common import Configuration
from pyflink.table import TableConfig, SqlDialect
from pyflink.testing.test_case_utils import PyFlinkTestCase

class TableConfigTests(PyFlinkTestCase):

    def test_get_set_idle_state_retention_time(self):
        if False:
            while True:
                i = 10
        table_config = TableConfig.get_default()
        table_config.set_idle_state_retention_time(datetime.timedelta(days=1), datetime.timedelta(days=2))
        self.assertEqual(3 * 24 * 3600 * 1000 / 2, table_config.get_max_idle_state_retention_time())
        self.assertEqual(24 * 3600 * 1000, table_config.get_min_idle_state_retention_time())

    def test_get_set_idle_state_rentention(self):
        if False:
            while True:
                i = 10
        table_config = TableConfig.get_default()
        table_config.set_idle_state_retention(datetime.timedelta(days=1))
        self.assertEqual(datetime.timedelta(days=1), table_config.get_idle_state_retention())

    def test_get_set_local_timezone(self):
        if False:
            return 10
        table_config = TableConfig.get_default()
        table_config.set_local_timezone('Asia/Shanghai')
        timezone = table_config.get_local_timezone()
        self.assertEqual(timezone, 'Asia/Shanghai')

    def test_get_set_max_generated_code_length(self):
        if False:
            return 10
        table_config = TableConfig.get_default()
        table_config.set_max_generated_code_length(32000)
        max_generated_code_length = table_config.get_max_generated_code_length()
        self.assertEqual(max_generated_code_length, 32000)

    def test_get_configuration(self):
        if False:
            while True:
                i = 10
        table_config = TableConfig.get_default()
        table_config.set('k1', 'v1')
        self.assertEqual(table_config.get('k1', ''), 'v1')

    def test_add_configuration(self):
        if False:
            return 10
        table_config = TableConfig.get_default()
        configuration = Configuration()
        configuration.set_string('k1', 'v1')
        table_config.add_configuration(configuration)
        self.assertEqual(table_config.get('k1', ''), 'v1')

    def test_get_set_sql_dialect(self):
        if False:
            for i in range(10):
                print('nop')
        table_config = TableConfig.get_default()
        sql_dialect = table_config.get_sql_dialect()
        self.assertEqual(sql_dialect, SqlDialect.DEFAULT)
        table_config.set_sql_dialect(SqlDialect.HIVE)
        sql_dialect = table_config.get_sql_dialect()
        self.assertEqual(sql_dialect, SqlDialect.HIVE)