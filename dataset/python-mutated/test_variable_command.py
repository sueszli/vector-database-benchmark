from __future__ import annotations
import os
from contextlib import redirect_stdout
from io import StringIO
import pytest
from sqlalchemy import select
from airflow import models
from airflow.cli import cli_parser
from airflow.cli.commands import variable_command
from airflow.models import Variable
from airflow.utils.session import create_session
from tests.test_utils.db import clear_db_variables
pytestmark = pytest.mark.db_test

class TestCliVariables:

    @classmethod
    def setup_class(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.dagbag = models.DagBag(include_examples=True)
        cls.parser = cli_parser.get_parser()

    def setup_method(self):
        if False:
            print('Hello World!')
        clear_db_variables()

    def teardown_method(self):
        if False:
            return 10
        clear_db_variables()

    def test_variables_set(self):
        if False:
            while True:
                i = 10
        'Test variable_set command'
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'foo', 'bar']))
        assert Variable.get('foo') is not None
        with pytest.raises(KeyError):
            Variable.get('foo1')

    def test_variables_set_with_description(self):
        if False:
            print('Hello World!')
        'Test variable_set command with optional description argument'
        expected_var_desc = 'foo_bar_description'
        var_key = 'foo'
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', var_key, 'bar', '--description', expected_var_desc]))
        assert Variable.get(var_key) == 'bar'
        with create_session() as session:
            actual_var_desc = session.scalar(select(Variable.description).where(Variable.key == var_key))
            assert actual_var_desc == expected_var_desc
        with pytest.raises(KeyError):
            Variable.get('foo1')

    def test_variables_get(self):
        if False:
            i = 10
            return i + 15
        Variable.set('foo', {'foo': 'bar'}, serialize_json=True)
        with redirect_stdout(StringIO()) as stdout:
            variable_command.variables_get(self.parser.parse_args(['variables', 'get', 'foo']))
            assert '{\n  "foo": "bar"\n}\n' == stdout.getvalue()

    def test_get_variable_default_value(self):
        if False:
            for i in range(10):
                print('nop')
        with redirect_stdout(StringIO()) as stdout:
            variable_command.variables_get(self.parser.parse_args(['variables', 'get', 'baz', '--default', 'bar']))
            assert 'bar\n' == stdout.getvalue()

    def test_get_variable_missing_variable(self):
        if False:
            while True:
                i = 10
        with pytest.raises(SystemExit):
            variable_command.variables_get(self.parser.parse_args(['variables', 'get', 'no-existing-VAR']))

    def test_variables_set_different_types(self):
        if False:
            print('Hello World!')
        'Test storage of various data types'
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'dict', '{"foo": "oops"}']))
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'list', '["oops"]']))
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'str', 'hello string']))
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'int', '42']))
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'float', '42.0']))
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'true', 'true']))
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'false', 'false']))
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'null', 'null']))
        variable_command.variables_export(self.parser.parse_args(['variables', 'export', 'variables_types.json']))
        variable_command.variables_import(self.parser.parse_args(['variables', 'import', 'variables_types.json']))
        assert {'foo': 'oops'} == Variable.get('dict', deserialize_json=True)
        assert ['oops'] == Variable.get('list', deserialize_json=True)
        assert 'hello string' == Variable.get('str')
        assert 42 == Variable.get('int', deserialize_json=True)
        assert 42.0 == Variable.get('float', deserialize_json=True)
        assert Variable.get('true', deserialize_json=True) is True
        assert Variable.get('false', deserialize_json=True) is False
        assert Variable.get('null', deserialize_json=True) is None
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'list', '["airflow"]']))
        variable_command.variables_import(self.parser.parse_args(['variables', 'import', 'variables_types.json', '--action-on-existing-key', 'skip']))
        assert ['airflow'] == Variable.get('list', deserialize_json=True)
        with pytest.raises(SystemExit):
            variable_command.variables_import(self.parser.parse_args(['variables', 'import', 'variables_types.json', '--action-on-existing-key', 'fail']))
        os.remove('variables_types.json')

    def test_variables_list(self):
        if False:
            return 10
        'Test variable_list command'
        variable_command.variables_list(self.parser.parse_args(['variables', 'list']))

    def test_variables_delete(self):
        if False:
            while True:
                i = 10
        'Test variable_delete command'
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'foo', 'bar']))
        variable_command.variables_delete(self.parser.parse_args(['variables', 'delete', 'foo']))
        with pytest.raises(KeyError):
            Variable.get('foo')

    def test_variables_import(self):
        if False:
            for i in range(10):
                print('nop')
        'Test variables_import command'
        with pytest.raises(SystemExit, match='Invalid variables file'):
            variable_command.variables_import(self.parser.parse_args(['variables', 'import', os.devnull]))

    def test_variables_export(self):
        if False:
            i = 10
            return i + 15
        'Test variables_export command'
        variable_command.variables_export(self.parser.parse_args(['variables', 'export', os.devnull]))

    def test_variables_isolation(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        'Test isolation of variables'
        path1 = tmp_path / 'testfile1'
        path2 = tmp_path / 'testfile2'
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'foo', '{"foo":"bar"}']))
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'bar', 'original']))
        variable_command.variables_export(self.parser.parse_args(['variables', 'export', os.fspath(path1)]))
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'bar', 'updated']))
        variable_command.variables_set(self.parser.parse_args(['variables', 'set', 'foo', '{"foo":"oops"}']))
        variable_command.variables_delete(self.parser.parse_args(['variables', 'delete', 'foo']))
        variable_command.variables_import(self.parser.parse_args(['variables', 'import', os.fspath(path1)]))
        assert 'original' == Variable.get('bar')
        assert '{\n  "foo": "bar"\n}' == Variable.get('foo')
        variable_command.variables_export(self.parser.parse_args(['variables', 'export', os.fspath(path2)]))
        assert path1.read_text() == path2.read_text()