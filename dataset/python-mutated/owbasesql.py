from typing import Type
from collections import OrderedDict
from AnyQt.QtWidgets import QLineEdit, QSizePolicy
from Orange.data import Table
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import BackendError
from Orange.widgets import gui, report
from Orange.widgets.credentials import CredentialManager
from Orange.widgets.settings import Setting
from Orange.widgets.utils.signals import Output
from Orange.widgets.widget import OWWidget, Msg

class OWBaseSql(OWWidget, openclass=True):
    """Base widget for connecting to a database.
    Override `get_backend` when subclassing to get corresponding backend.
    """

    class Outputs:
        data = Output('Data', Table)

    class Error(OWWidget.Error):
        connection = Msg('{}')
    want_main_area = False
    resizing_enabled = False
    host = Setting(None)
    port = Setting(None)
    database = Setting(None)
    schema = Setting(None)
    username = ''
    password = ''

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.backend = None
        self.data_desc_table = None
        self.database_desc = None
        self._setup_gui()
        self.connect()

    def _setup_gui(self):
        if False:
            print('Hello World!')
        self.controlArea.setMinimumWidth(360)
        vbox = gui.vBox(self.controlArea, 'Server')
        self.serverbox = gui.vBox(vbox)
        self.servertext = QLineEdit(self.serverbox)
        self.servertext.setPlaceholderText('Server')
        self.servertext.setToolTip('Server')
        self.servertext.editingFinished.connect(self._load_credentials)
        if self.host:
            self.servertext.setText(self.host if not self.port else '{}:{}'.format(self.host, self.port))
        self.serverbox.layout().addWidget(self.servertext)
        self.databasetext = QLineEdit(self.serverbox)
        self.databasetext.setPlaceholderText('Database[/Schema]')
        self.databasetext.setToolTip('Database or optionally Database/Schema')
        if self.database:
            self.databasetext.setText(self.database if not self.schema else '{}/{}'.format(self.database, self.schema))
        self.serverbox.layout().addWidget(self.databasetext)
        self.usernametext = QLineEdit(self.serverbox)
        self.usernametext.setPlaceholderText('Username')
        self.usernametext.setToolTip('Username')
        self.serverbox.layout().addWidget(self.usernametext)
        self.passwordtext = QLineEdit(self.serverbox)
        self.passwordtext.setPlaceholderText('Password')
        self.passwordtext.setToolTip('Password')
        self.passwordtext.setEchoMode(QLineEdit.Password)
        self.serverbox.layout().addWidget(self.passwordtext)
        self._load_credentials()
        self.connectbutton = gui.button(self.serverbox, self, 'Connect', callback=self.connect)
        self.connectbutton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def _load_credentials(self):
        if False:
            while True:
                i = 10
        self._parse_host_port()
        cm = self._credential_manager(self.host, self.port)
        self.username = cm.username
        self.password = cm.password
        if self.username:
            self.usernametext.setText(self.username)
        if self.password:
            self.passwordtext.setText(self.password)

    def _save_credentials(self):
        if False:
            print('Hello World!')
        cm = self._credential_manager(self.host, self.port)
        cm.username = self.username or ''
        cm.password = self.password or ''

    @staticmethod
    def _credential_manager(host, port):
        if False:
            while True:
                i = 10
        return CredentialManager('SQL Table: {}:{}'.format(host, port))

    def _parse_host_port(self):
        if False:
            return 10
        hostport = self.servertext.text().split(':')
        self.host = hostport[0]
        self.port = hostport[1] if len(hostport) == 2 else None

    def _check_db_settings(self):
        if False:
            return 10
        self._parse_host_port()
        (self.database, _, self.schema) = self.databasetext.text().partition('/')
        self.username = self.usernametext.text() or None
        self.password = self.passwordtext.text() or None

    def connect(self):
        if False:
            return 10
        self.clear()
        self._check_db_settings()
        if not self.host or not self.database:
            return
        try:
            backend = self.get_backend()
            if backend is None:
                return
            self.backend = backend(dict(host=self.host, port=self.port, database=self.database, user=self.username, password=self.password))
            self.on_connection_success()
        except BackendError as err:
            self.on_connection_error(err)

    def get_backend(self) -> Type[Backend]:
        if False:
            print('Hello World!')
        '\n        Derived widgets should override this to get corresponding backend.\n\n        Returns\n        -------\n        backend: Type[Backend]\n        '
        raise NotImplementedError

    def on_connection_success(self):
        if False:
            for i in range(10):
                print('nop')
        self._save_credentials()
        self.database_desc = OrderedDict((('Host', self.host), ('Port', self.port), ('Database', self.database), ('User name', self.username)))

    def on_connection_error(self, err):
        if False:
            return 10
        error = str(err).split('\n')[0]
        self.Error.connection(error)

    def open_table(self):
        if False:
            i = 10
            return i + 15
        data = self.get_table()
        self.data_desc_table = data
        self.Outputs.data.send(data)

    def get_table(self) -> Table:
        if False:
            print('Hello World!')
        '\n        Derived widgets should override this to get corresponding table.\n\n        Returns\n        -------\n        table: Table\n        '
        raise NotImplementedError

    def clear(self):
        if False:
            while True:
                i = 10
        self.Error.connection.clear()
        self.database_desc = None
        self.data_desc_table = None
        self.Outputs.data.send(None)

    def send_report(self):
        if False:
            print('Hello World!')
        if not self.database_desc:
            self.report_paragraph('No database connection.')
            return
        self.report_items('Database', self.database_desc)
        if self.data_desc_table:
            self.report_items('Data', report.describe_data(self.data_desc_table))