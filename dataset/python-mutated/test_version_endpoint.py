from __future__ import annotations
from unittest import mock
import pytest
pytestmark = pytest.mark.db_test

class TestGetHealthTest:

    @pytest.fixture(autouse=True)
    def setup_attrs(self, minimal_app_for_api) -> None:
        if False:
            while True:
                i = 10
        '\n        Setup For XCom endpoint TC\n        '
        self.app = minimal_app_for_api
        self.client = self.app.test_client()

    @mock.patch('airflow.api_connexion.endpoints.version_endpoint.airflow.__version__', 'MOCK_VERSION')
    @mock.patch('airflow.api_connexion.endpoints.version_endpoint.get_airflow_git_version', return_value='GIT_COMMIT')
    def test_should_respond_200(self, mock_get_airflow_get_commit):
        if False:
            for i in range(10):
                print('nop')
        response = self.client.get('/api/v1/version')
        assert 200 == response.status_code
        assert {'git_version': 'GIT_COMMIT', 'version': 'MOCK_VERSION'} == response.json
        mock_get_airflow_get_commit.assert_called_once_with()