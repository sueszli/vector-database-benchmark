from __future__ import annotations
import json
from contextlib import redirect_stdout
from io import StringIO
import pytest
from airflow import models, settings
from airflow.cli import cli_parser
from airflow.cli.commands import pool_command
from airflow.models import Pool
from airflow.settings import Session
from airflow.utils.db import add_default_pool_if_not_exists
pytestmark = pytest.mark.db_test

class TestCliPools:

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.dagbag = models.DagBag(include_examples=True)
        cls.parser = cli_parser.get_parser()
        settings.configure_orm()
        cls.session = Session
        cls._cleanup()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self._cleanup()

    @staticmethod
    def _cleanup(session=None):
        if False:
            i = 10
            return i + 15
        if session is None:
            session = Session()
        session.query(Pool).filter(Pool.pool != Pool.DEFAULT_POOL_NAME).delete()
        session.commit()
        add_default_pool_if_not_exists()
        session.close()

    def test_pool_list(self):
        if False:
            while True:
                i = 10
        pool_command.pool_set(self.parser.parse_args(['pools', 'set', 'foo', '1', 'test']))
        with redirect_stdout(StringIO()) as stdout:
            pool_command.pool_list(self.parser.parse_args(['pools', 'list']))
        assert 'foo' in stdout.getvalue()

    def test_pool_list_with_args(self):
        if False:
            while True:
                i = 10
        pool_command.pool_list(self.parser.parse_args(['pools', 'list', '--output', 'json']))

    def test_pool_create(self):
        if False:
            while True:
                i = 10
        pool_command.pool_set(self.parser.parse_args(['pools', 'set', 'foo', '1', 'test']))
        assert self.session.query(Pool).count() == 2

    def test_pool_update_deferred(self):
        if False:
            return 10
        pool_command.pool_set(self.parser.parse_args(['pools', 'set', 'foo', '1', 'test']))
        assert self.session.query(Pool).filter(Pool.pool == 'foo').first().include_deferred is False
        pool_command.pool_set(self.parser.parse_args(['pools', 'set', 'foo', '1', 'test', '--include-deferred']))
        assert self.session.query(Pool).filter(Pool.pool == 'foo').first().include_deferred is True
        pool_command.pool_set(self.parser.parse_args(['pools', 'set', 'foo', '1', 'test']))
        assert self.session.query(Pool).filter(Pool.pool == 'foo').first().include_deferred is False

    def test_pool_get(self):
        if False:
            while True:
                i = 10
        pool_command.pool_set(self.parser.parse_args(['pools', 'set', 'foo', '1', 'test']))
        pool_command.pool_get(self.parser.parse_args(['pools', 'get', 'foo']))

    def test_pool_delete(self):
        if False:
            while True:
                i = 10
        pool_command.pool_set(self.parser.parse_args(['pools', 'set', 'foo', '1', 'test']))
        pool_command.pool_delete(self.parser.parse_args(['pools', 'delete', 'foo']))
        assert self.session.query(Pool).count() == 1

    def test_pool_import_nonexistent(self):
        if False:
            while True:
                i = 10
        with pytest.raises(SystemExit):
            pool_command.pool_import(self.parser.parse_args(['pools', 'import', 'nonexistent.json']))

    def test_pool_import_invalid_json(self, tmp_path):
        if False:
            while True:
                i = 10
        invalid_pool_import_file_path = tmp_path / 'pools_import_invalid.json'
        with open(invalid_pool_import_file_path, mode='w') as file:
            file.write('not valid json')
        with pytest.raises(SystemExit):
            pool_command.pool_import(self.parser.parse_args(['pools', 'import', str(invalid_pool_import_file_path)]))

    def test_pool_import_invalid_pools(self, tmp_path):
        if False:
            i = 10
            return i + 15
        invalid_pool_import_file_path = tmp_path / 'pools_import_invalid.json'
        pool_config_input = {'foo': {'description': 'foo_test', 'include_deferred': False}}
        with open(invalid_pool_import_file_path, mode='w') as file:
            json.dump(pool_config_input, file)
        with pytest.raises(SystemExit):
            pool_command.pool_import(self.parser.parse_args(['pools', 'import', str(invalid_pool_import_file_path)]))

    def test_pool_import_backwards_compatibility(self, tmp_path):
        if False:
            while True:
                i = 10
        pool_import_file_path = tmp_path / 'pools_import.json'
        pool_config_input = {'foo': {'description': 'foo_test', 'slots': 1}}
        with open(pool_import_file_path, mode='w') as file:
            json.dump(pool_config_input, file)
        pool_command.pool_import(self.parser.parse_args(['pools', 'import', str(pool_import_file_path)]))
        assert self.session.query(Pool).filter(Pool.pool == 'foo').first().include_deferred is False

    def test_pool_import_export(self, tmp_path):
        if False:
            print('Hello World!')
        pool_import_file_path = tmp_path / 'pools_import.json'
        pool_export_file_path = tmp_path / 'pools_export.json'
        pool_config_input = {'foo': {'description': 'foo_test', 'slots': 1, 'include_deferred': True}, 'default_pool': {'description': 'Default pool', 'slots': 128, 'include_deferred': False}, 'baz': {'description': 'baz_test', 'slots': 2, 'include_deferred': False}}
        with open(pool_import_file_path, mode='w') as file:
            json.dump(pool_config_input, file)
        pool_command.pool_import(self.parser.parse_args(['pools', 'import', str(pool_import_file_path)]))
        pool_command.pool_export(self.parser.parse_args(['pools', 'export', str(pool_export_file_path)]))
        with open(pool_export_file_path) as file:
            pool_config_output = json.load(file)
            assert pool_config_input == pool_config_output, 'Input and output pool files are not same'