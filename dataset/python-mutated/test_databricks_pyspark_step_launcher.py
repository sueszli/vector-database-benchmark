import os
from typing import Dict, Optional
from unittest import mock
import pytest
from dagster._check import CheckError
from dagster_databricks.databricks_pyspark_step_launcher import DAGSTER_SYSTEM_ENV_VARS, DatabricksPySparkStepLauncher

@pytest.fixture
def mock_step_launcher_factory():
    if False:
        for i in range(10):
            print('nop')

    def _mocked(add_dagster_env_variables: bool=True, env_variables: Optional[dict]=None, databricks_token: Optional[str]='abc123', oauth_creds: Optional[Dict[str, str]]=None):
        if False:
            i = 10
            return i + 15
        return DatabricksPySparkStepLauncher(run_config={'some': 'config'}, permissions={'some': 'permissions'}, databricks_host='databricks.host.com', databricks_token=databricks_token, secrets_to_env_variables=[{'some': 'secret'}], staging_prefix='/a/prefix', wait_for_logs=False, max_completion_wait_time_seconds=100, env_variables=env_variables, add_dagster_env_variables=add_dagster_env_variables, local_dagster_job_package_path='some/local/path', oauth_credentials=oauth_creds)
    return _mocked

class TestCreateRemoteConfig:

    def test_given_add_dagster_env_vars_retrieves_dagster_system_vars(self, mock_step_launcher_factory, monkeypatch):
        if False:
            while True:
                i = 10
        test_env_variables = {'add': 'this'}
        test_launcher = mock_step_launcher_factory(add_dagster_env_variables=True, env_variables=test_env_variables)
        system_vars = {}
        for var in DAGSTER_SYSTEM_ENV_VARS:
            system_vars[var] = f'{var}_value'
            monkeypatch.setenv(var, f'{var}_value')
        correct_vars = dict(**system_vars, **test_env_variables)
        env_vars = test_launcher.create_remote_config()
        assert env_vars.env_variables == correct_vars

    def test_given_no_add_dagster_env_vars_no_system_vars_added(self, mock_step_launcher_factory, monkeypatch):
        if False:
            i = 10
            return i + 15
        vars_to_add = {'add': 'this'}
        test_launcher = mock_step_launcher_factory(add_dagster_env_variables=False, env_variables=vars_to_add)
        for var in DAGSTER_SYSTEM_ENV_VARS:
            monkeypatch.setenv(var, f'{var}_value')
        env_vars = test_launcher.create_remote_config()
        assert env_vars.env_variables == vars_to_add

    def test_given_no_dagster_system_vars_none_added(self, mock_step_launcher_factory):
        if False:
            return 10
        vars_to_add = {'add': 'this'}
        test_launcher = mock_step_launcher_factory(add_dagster_env_variables=True, env_variables=vars_to_add)
        for var in DAGSTER_SYSTEM_ENV_VARS:
            assert not os.getenv(var)
        env_vars = test_launcher.create_remote_config()
        assert env_vars.env_variables == vars_to_add

    @mock.patch('dagster_databricks.databricks.WorkspaceClient')
    def test_given_oauth_creds_when_accessing_legacy_clients_raises_ValueError(self, mock_workspace_client, mock_step_launcher_factory):
        if False:
            i = 10
            return i + 15
        test_launcher = mock_step_launcher_factory(databricks_token=None, oauth_creds={'client_id': 'abc123', 'client_secret': 'super-secret'})
        assert test_launcher.databricks_runner.oauth_client_id == 'abc123'
        assert test_launcher.databricks_runner.oauth_client_secret == 'super-secret'
        with pytest.raises(ValueError):
            assert test_launcher.databricks_runner.client.client

    @mock.patch('dagster_databricks.databricks.WorkspaceClient')
    def test_given_oauth_creds_and_token_raises_ValueError(self, mock_workspace_client, mock_step_launcher_factory):
        if False:
            while True:
                i = 10
        with pytest.raises(CheckError):
            mock_step_launcher_factory(databricks_token='abc123', oauth_creds={'client_id': 'abc123', 'client_secret': 'super-secret'})