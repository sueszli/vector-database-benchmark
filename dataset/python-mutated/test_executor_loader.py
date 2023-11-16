from __future__ import annotations
from contextlib import nullcontext
from unittest import mock
import pytest
from airflow import plugins_manager
from airflow.exceptions import AirflowConfigException
from airflow.executors.executor_loader import ConnectorSource, ExecutorLoader
from tests.test_utils.config import conf_vars
TEST_PLUGIN_NAME = 'unique_plugin_name_to_avoid_collision_i_love_kitties'

class FakeExecutor:
    is_single_threaded = False

class FakeSingleThreadedExecutor:
    is_single_threaded = True

class FakePlugin(plugins_manager.AirflowPlugin):
    name = TEST_PLUGIN_NAME
    executors = [FakeExecutor]

class TestExecutorLoader:

    def setup_method(self) -> None:
        if False:
            i = 10
            return i + 15
        ExecutorLoader._default_executor = None

    def teardown_method(self) -> None:
        if False:
            i = 10
            return i + 15
        ExecutorLoader._default_executor = None

    @pytest.mark.parametrize('executor_name', ['CeleryExecutor', 'CeleryKubernetesExecutor', 'DebugExecutor', 'KubernetesExecutor', 'LocalExecutor'])
    def test_should_support_executor_from_core(self, executor_name):
        if False:
            for i in range(10):
                print('nop')
        with conf_vars({('core', 'executor'): executor_name}):
            executor = ExecutorLoader.get_default_executor()
            assert executor is not None
            assert executor_name == executor.__class__.__name__

    @mock.patch('airflow.plugins_manager.plugins', [FakePlugin()])
    @mock.patch('airflow.plugins_manager.executors_modules', None)
    def test_should_support_plugins(self):
        if False:
            print('Hello World!')
        with conf_vars({('core', 'executor'): f'{TEST_PLUGIN_NAME}.FakeExecutor'}):
            executor = ExecutorLoader.get_default_executor()
            assert executor is not None
            assert 'FakeExecutor' == executor.__class__.__name__

    def test_should_support_custom_path(self):
        if False:
            i = 10
            return i + 15
        with conf_vars({('core', 'executor'): 'tests.executors.test_executor_loader.FakeExecutor'}):
            executor = ExecutorLoader.get_default_executor()
            assert executor is not None
            assert 'FakeExecutor' == executor.__class__.__name__

    @pytest.mark.parametrize('executor_name', ['CeleryExecutor', 'CeleryKubernetesExecutor', 'DebugExecutor', 'KubernetesExecutor', 'LocalExecutor'])
    def test_should_support_import_executor_from_core(self, executor_name):
        if False:
            i = 10
            return i + 15
        with conf_vars({('core', 'executor'): executor_name}):
            (executor, import_source) = ExecutorLoader.import_default_executor_cls()
            assert executor_name == executor.__name__
            assert import_source == ConnectorSource.CORE

    @mock.patch('airflow.plugins_manager.plugins', [FakePlugin()])
    @mock.patch('airflow.plugins_manager.executors_modules', None)
    def test_should_support_import_plugins(self):
        if False:
            print('Hello World!')
        with conf_vars({('core', 'executor'): f'{TEST_PLUGIN_NAME}.FakeExecutor'}):
            (executor, import_source) = ExecutorLoader.import_default_executor_cls()
            assert 'FakeExecutor' == executor.__name__
            assert import_source == ConnectorSource.PLUGIN

    def test_should_support_import_custom_path(self):
        if False:
            while True:
                i = 10
        with conf_vars({('core', 'executor'): 'tests.executors.test_executor_loader.FakeExecutor'}):
            (executor, import_source) = ExecutorLoader.import_default_executor_cls()
            assert 'FakeExecutor' == executor.__name__
            assert import_source == ConnectorSource.CUSTOM_PATH

    @pytest.mark.db_test
    @pytest.mark.backend('mssql', 'mysql', 'postgres')
    @pytest.mark.parametrize('executor', [FakeExecutor, FakeSingleThreadedExecutor])
    def test_validate_database_executor_compatibility_general(self, monkeypatch, executor):
        if False:
            return 10
        monkeypatch.delenv('_AIRFLOW__SKIP_DATABASE_EXECUTOR_COMPATIBILITY_CHECK')
        ExecutorLoader.validate_database_executor_compatibility(executor)

    @pytest.mark.db_test
    @pytest.mark.backend('sqlite')
    @pytest.mark.parametrize(['executor', 'expectation'], [pytest.param(FakeSingleThreadedExecutor, nullcontext(), id='single-threaded'), pytest.param(FakeExecutor, pytest.raises(AirflowConfigException, match='^error: cannot use SQLite with the .+'), id='multi-threaded')])
    def test_validate_database_executor_compatibility_sqlite(self, monkeypatch, executor, expectation):
        if False:
            return 10
        monkeypatch.delenv('_AIRFLOW__SKIP_DATABASE_EXECUTOR_COMPATIBILITY_CHECK')
        with expectation:
            ExecutorLoader.validate_database_executor_compatibility(executor)