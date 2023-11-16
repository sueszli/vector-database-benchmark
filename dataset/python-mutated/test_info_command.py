from __future__ import annotations
import contextlib
import importlib
import logging
import os
from io import StringIO
import pytest
from rich.console import Console
from airflow.cli import cli_parser
from airflow.cli.commands import info_command
from airflow.config_templates import airflow_local_settings
from airflow.logging_config import configure_logging
from airflow.version import version as airflow_version
from tests.test_utils.config import conf_vars

def capture_show_output(instance):
    if False:
        return 10
    console = Console()
    with console.capture() as capture:
        instance.info(console)
    return capture.get()

class TestPiiAnonymizer:

    def setup_method(self) -> None:
        if False:
            i = 10
            return i + 15
        self.instance = info_command.PiiAnonymizer()

    def test_should_remove_pii_from_path(self):
        if False:
            i = 10
            return i + 15
        home_path = os.path.expanduser('~/airflow/config')
        assert '${HOME}/airflow/config' == self.instance.process_path(home_path)

    @pytest.mark.parametrize('before, after', [('postgresql+psycopg2://postgres:airflow@postgres/airflow', 'postgresql+psycopg2://p...s:PASSWORD@postgres/airflow'), ('postgresql+psycopg2://postgres@postgres/airflow', 'postgresql+psycopg2://p...s@postgres/airflow'), ('postgresql+psycopg2://:airflow@postgres/airflow', 'postgresql+psycopg2://:PASSWORD@postgres/airflow'), ('postgresql+psycopg2://postgres/airflow', 'postgresql+psycopg2://postgres/airflow')])
    def test_should_remove_pii_from_url(self, before, after):
        if False:
            print('Hello World!')
        assert after == self.instance.process_url(before)

class TestAirflowInfo:

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.parser = cli_parser.get_parser()

    @classmethod
    def teardown_class(cls) -> None:
        if False:
            i = 10
            return i + 15
        for handler_ref in logging._handlerList[:]:
            logging._removeHandlerRef(handler_ref)
        importlib.reload(airflow_local_settings)
        configure_logging()

    @staticmethod
    def unique_items(items):
        if False:
            while True:
                i = 10
        return {i[0] for i in items}

    @conf_vars({('core', 'executor'): 'TEST_EXECUTOR', ('core', 'dags_folder'): 'TEST_DAGS_FOLDER', ('core', 'plugins_folder'): 'TEST_PLUGINS_FOLDER', ('logging', 'base_log_folder'): 'TEST_LOG_FOLDER', ('database', 'sql_alchemy_conn'): 'postgresql+psycopg2://postgres:airflow@postgres/airflow', ('logging', 'remote_logging'): 'True', ('logging', 'remote_base_log_folder'): 's3://logs-name'})
    def test_airflow_info(self):
        if False:
            print('Hello World!')
        importlib.reload(airflow_local_settings)
        configure_logging()
        instance = info_command.AirflowInfo(info_command.NullAnonymizer())
        expected = {'executor', 'version', 'task_logging_handler', 'plugins_folder', 'base_log_folder', 'remote_base_log_folder', 'dags_folder', 'sql_alchemy_conn'}
        assert self.unique_items(instance._airflow_info) == expected

    def test_system_info(self):
        if False:
            print('Hello World!')
        instance = info_command.AirflowInfo(info_command.NullAnonymizer())
        expected = {'uname', 'architecture', 'OS', 'python_location', 'locale', 'python_version'}
        assert self.unique_items(instance._system_info) == expected

    def test_paths_info(self):
        if False:
            while True:
                i = 10
        instance = info_command.AirflowInfo(info_command.NullAnonymizer())
        expected = {'airflow_on_path', 'airflow_home', 'system_path', 'python_path'}
        assert self.unique_items(instance._paths_info) == expected

    def test_tools_info(self):
        if False:
            i = 10
            return i + 15
        instance = info_command.AirflowInfo(info_command.NullAnonymizer())
        expected = {'cloud_sql_proxy', 'gcloud', 'git', 'kubectl', 'mysql', 'psql', 'sqlite3', 'ssh'}
        assert self.unique_items(instance._tools_info) == expected

    @pytest.mark.db_test
    @conf_vars({('database', 'sql_alchemy_conn'): 'postgresql+psycopg2://postgres:airflow@postgres/airflow'})
    def test_show_info(self):
        if False:
            for i in range(10):
                print('nop')
        with contextlib.redirect_stdout(StringIO()) as stdout:
            info_command.show_info(self.parser.parse_args(['info']))
        output = stdout.getvalue()
        assert airflow_version in output
        assert 'postgresql+psycopg2://postgres:airflow@postgres/airflow' in output

    @pytest.mark.db_test
    @conf_vars({('database', 'sql_alchemy_conn'): 'postgresql+psycopg2://postgres:airflow@postgres/airflow'})
    def test_show_info_anonymize(self):
        if False:
            while True:
                i = 10
        with contextlib.redirect_stdout(StringIO()) as stdout:
            info_command.show_info(self.parser.parse_args(['info', '--anonymize']))
        output = stdout.getvalue()
        assert airflow_version in output
        assert 'postgresql+psycopg2://p...s:PASSWORD@postgres/airflow' in output

@pytest.fixture()
def setup_parser():
    if False:
        return 10
    yield cli_parser.get_parser()

class TestInfoCommandMockHttpx:

    @conf_vars({('database', 'sql_alchemy_conn'): 'postgresql+psycopg2://postgres:airflow@postgres/airflow'})
    def test_show_info_anonymize_fileio(self, httpx_mock, setup_parser):
        if False:
            for i in range(10):
                print('nop')
        httpx_mock.add_response(url='https://file.io', method='post', json={'success': True, 'key': 'f9U3zs3I', 'link': 'https://file.io/TEST', 'expiry': '14 days'}, status_code=200)
        with contextlib.redirect_stdout(StringIO()) as stdout:
            info_command.show_info(setup_parser.parse_args(['info', '--file-io']))
        assert 'https://file.io/TEST' in stdout.getvalue()