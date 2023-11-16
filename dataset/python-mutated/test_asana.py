from __future__ import annotations
import os
from unittest.mock import patch
import pytest
from asana import Client
from airflow.models import Connection
from airflow.providers.asana.hooks.asana import AsanaHook

class TestAsanaHook:
    """
    Tests for AsanaHook Asana client retrieval
    """

    def test_asana_client_retrieved(self):
        if False:
            while True:
                i = 10
        '\n        Test that we successfully retrieve an Asana client given a Connection with complete information.\n        :return: None\n        '
        with patch.object(AsanaHook, 'get_connection', return_value=Connection(conn_type='asana', password='test')):
            hook = AsanaHook()
        client = hook.get_conn()
        assert type(client) == Client

    def test_missing_password_raises(self):
        if False:
            return 10
        '\n        Test that the Asana hook raises an exception if password not provided in connection.\n        :return: None\n        '
        with patch.object(AsanaHook, 'get_connection', return_value=Connection(conn_type='asana')):
            hook = AsanaHook()
        with pytest.raises(ValueError):
            hook.get_conn()

    def test_merge_create_task_parameters_default_project(self):
        if False:
            return 10
        '\n        Test that merge_create_task_parameters correctly merges the default and method parameters when we\n        do not override the default project.\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__project": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'name': 'test', 'projects': ['1']}
        assert hook._merge_create_task_parameters('test', {}) == expected_merged_params

    def test_merge_create_task_parameters_specified_project(self):
        if False:
            while True:
                i = 10
        '\n        Test that merge_create_task_parameters correctly merges the default and method parameters when we\n        override the default project.\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__project": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'name': 'test', 'projects': ['1', '2']}
        assert hook._merge_create_task_parameters('test', {'projects': ['1', '2']}) == expected_merged_params

    def test_merge_create_task_parameters_specified_workspace(self):
        if False:
            return 10
        '\n        Test that merge_create_task_parameters correctly merges the default and method parameters when we\n        do not override the default workspace.\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__workspace": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'name': 'test', 'workspace': '1'}
        assert hook._merge_create_task_parameters('test', {}) == expected_merged_params

    def test_merge_create_task_parameters_default_project_overrides_default_workspace(self):
        if False:
            return 10
        '\n        Test that merge_create_task_parameters uses the default project over the default workspace\n        if it is available\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__workspace": "1", "extra__asana__project": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'name': 'test', 'projects': ['1']}
        assert hook._merge_create_task_parameters('test', {}) == expected_merged_params

    def test_merge_create_task_parameters_specified_project_overrides_default_workspace(self):
        if False:
            while True:
                i = 10
        '\n        Test that merge_create_task_parameters uses the method parameter project over the default workspace\n        if it is available\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__workspace": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'name': 'test', 'projects': ['2']}
        assert hook._merge_create_task_parameters('test', {'projects': ['2']}) == expected_merged_params

    def test_merge_find_task_parameters_default_project(self):
        if False:
            return 10
        '\n        Test that merge_find_task_parameters correctly merges the default and method parameters when we\n        do not override the default project.\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__project": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'project': '1'}
        assert hook._merge_find_task_parameters({}) == expected_merged_params

    def test_merge_find_task_parameters_specified_project(self):
        if False:
            return 10
        '\n        Test that merge_find_task_parameters correctly merges the default and method parameters when we\n        do override the default project.\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__project": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'project': '2'}
        assert hook._merge_find_task_parameters({'project': '2'}) == expected_merged_params

    def test_merge_find_task_parameters_default_workspace(self):
        if False:
            while True:
                i = 10
        '\n        Test that merge_find_task_parameters correctly merges the default and method parameters when we\n        do not override the default workspace.\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__workspace": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'workspace': '1', 'assignee': '1'}
        assert hook._merge_find_task_parameters({'assignee': '1'}) == expected_merged_params

    def test_merge_find_task_parameters_specified_workspace(self):
        if False:
            return 10
        '\n        Test that merge_find_task_parameters correctly merges the default and method parameters when we\n        do override the default workspace.\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__workspace": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'workspace': '2', 'assignee': '1'}
        assert hook._merge_find_task_parameters({'workspace': '2', 'assignee': '1'}) == expected_merged_params

    def test_merge_find_task_parameters_default_project_overrides_workspace(self):
        if False:
            while True:
                i = 10
        '\n        Test that merge_find_task_parameters uses the default project over the workspace if it is available\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__workspace": "1", "extra__asana__project": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'project': '1'}
        assert hook._merge_find_task_parameters({}) == expected_merged_params

    def test_merge_find_task_parameters_specified_project_overrides_workspace(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that merge_find_task_parameters uses the method parameter project over the default workspace\n        if it is available\n        :return: None\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__workspace": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'project': '2'}
        assert hook._merge_find_task_parameters({'project': '2'}) == expected_merged_params

    def test_merge_project_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that default workspace is used if not overridden\n        :return:\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__workspace": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'workspace': '1', 'name': 'name'}
        assert hook._merge_project_parameters({'name': 'name'}) == expected_merged_params

    def test_merge_project_parameters_override(self):
        if False:
            return 10
        '\n        Tests that default workspace is successfully overridden\n        :return:\n        '
        conn = Connection(conn_type='asana', password='test', extra='{"extra__asana__workspace": "1"}')
        with patch.object(AsanaHook, 'get_connection', return_value=conn):
            hook = AsanaHook()
        expected_merged_params = {'workspace': '2'}
        assert hook._merge_project_parameters({'workspace': '2'}) == expected_merged_params

    @pytest.mark.parametrize('uri', [pytest.param('a://?extra__asana__workspace=abc&extra__asana__project=abc', id='prefix'), pytest.param('a://?workspace=abc&project=abc', id='no-prefix')])
    def test_backcompat_prefix_works(self, uri):
        if False:
            print('Hello World!')
        with patch.dict(os.environ, {'AIRFLOW_CONN_MY_CONN': uri}):
            hook = AsanaHook('my_conn')
            assert hook.workspace == 'abc'
            assert hook.project == 'abc'

    def test_backcompat_prefix_both_prefers_short(self):
        if False:
            return 10
        with patch.dict(os.environ, {'AIRFLOW_CONN_MY_CONN': 'a://?workspace=non-prefixed&extra__asana__workspace=prefixed'}):
            hook = AsanaHook('my_conn')
            assert hook.workspace == 'non-prefixed'