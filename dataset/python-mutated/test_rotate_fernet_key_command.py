from __future__ import annotations
from unittest import mock
import pytest
from cryptography.fernet import Fernet
from airflow.cli import cli_parser
from airflow.cli.commands import rotate_fernet_key_command
from airflow.hooks.base import BaseHook
from airflow.models import Connection, Variable
from airflow.utils.session import provide_session
from tests.test_utils.config import conf_vars
from tests.test_utils.db import clear_db_connections, clear_db_variables
pytestmark = pytest.mark.db_test

class TestRotateFernetKeyCommand:

    @classmethod
    def setup_class(cls):
        if False:
            return 10
        cls.parser = cli_parser.get_parser()

    def setup_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        clear_db_connections(add_default_connections_back=False)
        clear_db_variables()

    def teardown_method(self) -> None:
        if False:
            i = 10
            return i + 15
        clear_db_connections(add_default_connections_back=False)
        clear_db_variables()

    @provide_session
    def test_should_rotate_variable(self, session):
        if False:
            print('Hello World!')
        fernet_key1 = Fernet.generate_key()
        fernet_key2 = Fernet.generate_key()
        var1_key = f'{__file__}_var1'
        var2_key = f'{__file__}_var2'
        with conf_vars({('core', 'fernet_key'): ''}), mock.patch('airflow.models.crypto._fernet', None):
            Variable.set(key=var1_key, value='value')
        with conf_vars({('core', 'fernet_key'): fernet_key1.decode()}), mock.patch('airflow.models.crypto._fernet', None):
            Variable.set(key=var2_key, value='value')
        with conf_vars({('core', 'fernet_key'): f'{fernet_key2.decode()},{fernet_key1.decode()}'}), mock.patch('airflow.models.crypto._fernet', None):
            args = self.parser.parse_args(['rotate-fernet-key'])
            rotate_fernet_key_command.rotate_fernet_key(args)
        with conf_vars({('core', 'fernet_key'): fernet_key2.decode()}), mock.patch('airflow.models.crypto._fernet', None):
            var1 = session.query(Variable).filter(Variable.key == var1_key).first()
            assert Variable.get(key=var1_key) == 'value'
            assert var1._val == 'value'
            assert Variable.get(key=var2_key) == 'value'

    @provide_session
    def test_should_rotate_connection(self, session):
        if False:
            print('Hello World!')
        fernet_key1 = Fernet.generate_key()
        fernet_key2 = Fernet.generate_key()
        var1_key = f'{__file__}_var1'
        var2_key = f'{__file__}_var2'
        with conf_vars({('core', 'fernet_key'): ''}), mock.patch('airflow.models.crypto._fernet', None):
            session.add(Connection(conn_id=var1_key, uri='mysql://user:pass@localhost'))
            session.commit()
        with conf_vars({('core', 'fernet_key'): fernet_key1.decode()}), mock.patch('airflow.models.crypto._fernet', None):
            session.add(Connection(conn_id=var2_key, uri='mysql://user:pass@localhost'))
            session.commit()
        with conf_vars({('core', 'fernet_key'): f'{fernet_key2.decode()},{fernet_key1.decode()}'}), mock.patch('airflow.models.crypto._fernet', None):
            args = self.parser.parse_args(['rotate-fernet-key'])
            rotate_fernet_key_command.rotate_fernet_key(args)
        with conf_vars({('core', 'fernet_key'): fernet_key2.decode()}), mock.patch('airflow.models.crypto._fernet', None):
            conn1: Connection = BaseHook.get_connection(var1_key)
            assert conn1.password == 'pass'
            assert conn1._password == 'pass'
            assert BaseHook.get_connection(var2_key).password == 'pass'