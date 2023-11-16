from __future__ import annotations
import re
from typing import TYPE_CHECKING, Sequence
from unittest import mock
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.google.cloud.hooks.os_login import OSLoginHook
from tests.providers.google.cloud.utils.base_gcp_mock import mock_base_gcp_hook_default_project_id, mock_base_gcp_hook_no_default_project_id
if TYPE_CHECKING:
    from google.api_core.gapic_v1.method import _MethodDefault
    from google.api_core.retry import Retry
TEST_GCP_CONN_ID: str = 'test-gcp-conn-id'
TEST_PROJECT_ID: str = 'test-project-id'
TEST_PROJECT_ID_2: str = 'test-project-id-2'
TEST_USER: str = 'test-user'
TEST_CREDENTIALS = mock.MagicMock()
TEST_BODY: dict = mock.MagicMock()
TEST_RETRY: Retry | _MethodDefault = mock.MagicMock()
TEST_TIMEOUT: float = 4
TEST_METADATA: Sequence[tuple[str, str]] = ()
TEST_PARENT: str = 'users/test-user'

class TestOSLoginHook:

    def test_delegate_to_runtime_error(self):
        if False:
            while True:
                i = 10
        with pytest.raises(RuntimeError):
            OSLoginHook(gcp_conn_id='GCP_CONN_ID', delegate_to='delegate_to')

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        with mock.patch('airflow.providers.google.cloud.hooks.os_login.OSLoginHook.__init__', new=mock_base_gcp_hook_default_project_id):
            self.hook = OSLoginHook(gcp_conn_id='test')

    @mock.patch('airflow.providers.google.common.hooks.base_google.GoogleBaseHook.get_credentials_and_project_id', return_value=(TEST_CREDENTIALS, None))
    @mock.patch('airflow.providers.google.cloud.hooks.os_login.OSLoginHook.get_conn')
    def test_import_ssh_public_key(self, mock_get_conn, mock_get_creds_and_project_id) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.hook.import_ssh_public_key(user=TEST_USER, ssh_public_key=TEST_BODY, project_id=TEST_PROJECT_ID, retry=TEST_RETRY, timeout=TEST_TIMEOUT, metadata=TEST_METADATA)
        mock_get_conn.return_value.import_ssh_public_key.assert_called_once_with(request=dict(parent=TEST_PARENT, ssh_public_key=TEST_BODY, project_id=TEST_PROJECT_ID), retry=TEST_RETRY, timeout=TEST_TIMEOUT, metadata=TEST_METADATA)

class TestOSLoginHookWithDefaultProjectIdHook:

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        with mock.patch('airflow.providers.google.cloud.hooks.os_login.OSLoginHook.__init__', new=mock_base_gcp_hook_default_project_id):
            self.hook = OSLoginHook(gcp_conn_id='test')

    @mock.patch('airflow.providers.google.common.hooks.base_google.GoogleBaseHook.get_credentials_and_project_id', return_value=(TEST_CREDENTIALS, TEST_PROJECT_ID_2))
    @mock.patch('airflow.providers.google.cloud.hooks.os_login.OSLoginHook.get_conn')
    def test_import_ssh_public_key(self, mock_get_conn, mock_get_creds_and_project_id) -> None:
        if False:
            print('Hello World!')
        self.hook.import_ssh_public_key(user=TEST_USER, ssh_public_key=TEST_BODY, project_id=None, retry=TEST_RETRY, timeout=TEST_TIMEOUT, metadata=TEST_METADATA)
        mock_get_conn.return_value.import_ssh_public_key.assert_called_once_with(request=dict(parent=TEST_PARENT, ssh_public_key=TEST_BODY, project_id=TEST_PROJECT_ID_2), retry=TEST_RETRY, timeout=TEST_TIMEOUT, metadata=TEST_METADATA)

class TestOSLoginHookWithoutDefaultProjectIdHook:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch('airflow.providers.google.cloud.hooks.os_login.OSLoginHook.__init__', new=mock_base_gcp_hook_no_default_project_id):
            self.hook = OSLoginHook(gcp_conn_id='test')

    @mock.patch('airflow.providers.google.common.hooks.base_google.GoogleBaseHook.get_credentials_and_project_id', return_value=(TEST_CREDENTIALS, TEST_PROJECT_ID_2))
    @mock.patch('airflow.providers.google.cloud.hooks.os_login.OSLoginHook.get_conn')
    def test_import_ssh_public_key(self, mock_get_conn, mock_get_creds_and_project_id) -> None:
        if False:
            i = 10
            return i + 15
        self.hook.import_ssh_public_key(user=TEST_USER, ssh_public_key=TEST_BODY, project_id=TEST_PROJECT_ID, retry=TEST_RETRY, timeout=TEST_TIMEOUT, metadata=TEST_METADATA)
        mock_get_conn.return_value.import_ssh_public_key.assert_called_once_with(request=dict(parent=TEST_PARENT, ssh_public_key=TEST_BODY, project_id=TEST_PROJECT_ID), retry=TEST_RETRY, timeout=TEST_TIMEOUT, metadata=TEST_METADATA)
TEST_MESSAGE = re.escape('The project id must be passed either as keyword project_id parameter or as project_id extra in Google Cloud connection definition. Both are not set!')

class TestOSLoginHookMissingProjectIdHook:

    def setup_method(self):
        if False:
            print('Hello World!')
        with mock.patch('airflow.providers.google.cloud.hooks.os_login.OSLoginHook.__init__', new=mock_base_gcp_hook_no_default_project_id):
            self.hook = OSLoginHook(gcp_conn_id='test')

    @mock.patch('airflow.providers.google.common.hooks.base_google.GoogleBaseHook.get_credentials_and_project_id', return_value=(TEST_CREDENTIALS, None))
    @mock.patch('airflow.providers.google.cloud.hooks.os_login.OSLoginHook.get_conn')
    def test_import_ssh_public_key(self, mock_get_conn, mock_get_creds_and_project_id) -> None:
        if False:
            print('Hello World!')
        with pytest.raises(AirflowException, match=TEST_MESSAGE):
            self.hook.import_ssh_public_key(user=TEST_USER, ssh_public_key=TEST_BODY, project_id=None, retry=TEST_RETRY, timeout=TEST_TIMEOUT, metadata=TEST_METADATA)