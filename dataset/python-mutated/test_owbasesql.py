import unittest
from unittest.mock import Mock
from collections import OrderedDict
from types import SimpleNamespace
from Orange.data import Table
from Orange.data.sql.backend import Backend
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.owbasesql import OWBaseSql
from Orange.data.sql.backend.base import BackendError
USERNAME = 'UN'
PASSWORD = 'PASS'

class BrokenBackend(Backend):

    def __init__(self, connection_params):
        if False:
            while True:
                i = 10
        super().__init__(connection_params)
        raise BackendError('Error connecting to DB.')

class TestableSqlWidget(OWBaseSql):
    name = 'SQL'

    def __init__(self):
        if False:
            while True:
                i = 10
        self.mocked_backend = Mock()
        super().__init__()

    def get_backend(self):
        if False:
            i = 10
            return i + 15
        return self.mocked_backend

    def get_table(self) -> Table:
        if False:
            i = 10
            return i + 15
        return Table('iris')

    @staticmethod
    def _credential_manager(_, __):
        if False:
            i = 10
            return i + 15
        return SimpleNamespace(username=USERNAME, password=PASSWORD)

class TestOWBaseSql(WidgetTest):

    def setUp(self):
        if False:
            return 10
        (self.host, self.port, self.db) = ('host', 'port', 'DB')
        settings = {'host': self.host, 'port': self.port, 'database': self.db, 'schema': ''}
        self.widget = self.create_widget(TestableSqlWidget, stored_settings=settings)

    def test_connect(self):
        if False:
            return 10
        self.widget.mocked_backend.assert_called_once_with({'host': 'host', 'port': 'port', 'database': self.db, 'user': USERNAME, 'password': PASSWORD})
        self.assertDictEqual(self.widget.database_desc, OrderedDict((('Host', 'host'), ('Port', 'port'), ('Database', self.db), ('User name', USERNAME))))

    def test_connection_error(self):
        if False:
            return 10
        self.widget.get_backend = Mock(return_value=BrokenBackend)
        self.widget.connectbutton.click()
        self.assertTrue(self.widget.Error.connection.is_shown())
        self.assertIsNone(self.widget.database_desc)

    def test_output(self):
        if False:
            print('Hello World!')
        self.widget.open_table()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNotNone(self.widget.data_desc_table)

    def test_output_error(self):
        if False:
            i = 10
            return i + 15
        self.widget.get_table = lambda : None
        self.widget.open_table()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNone(self.widget.data_desc_table)

    def test_missing_database_parameter(self):
        if False:
            while True:
                i = 10
        self.widget.open_table()
        self.widget.databasetext.setText('')
        self.widget.mocked_backend.reset_mock()
        self.widget.connectbutton.click()
        self.widget.mocked_backend.assert_not_called()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertIsNone(self.widget.data_desc_table)
        self.assertFalse(self.widget.Error.connection.is_shown())

    def test_report(self):
        if False:
            for i in range(10):
                print('nop')
        self.widget.report_button.click()
        self.widget.open_table()
        self.widget.report_button.click()
        self.widget.databasetext.setText('')
        self.widget.connectbutton.click()
        self.widget.report_button.click()
if __name__ == '__main__':
    unittest.main()