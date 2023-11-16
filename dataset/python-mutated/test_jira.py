from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
from airflow.models import Connection
from airflow.providers.atlassian.jira.hooks.jira import JiraHook
from airflow.utils import db
pytestmark = pytest.mark.db_test
jira_client_mock = Mock(name='jira_client')

class TestJiraHook:

    def setup_method(self):
        if False:
            return 10
        db.merge_conn(Connection(conn_id='jira_default', conn_type='jira', host='https://localhost/jira/', port=443, extra='{"verify": "False", "project": "AIRFLOW"}'))

    @patch('airflow.providers.atlassian.jira.hooks.jira.Jira', autospec=True, return_value=jira_client_mock)
    def test_jira_client_connection(self, jira_mock):
        if False:
            i = 10
            return i + 15
        jira_hook = JiraHook()
        assert jira_mock.called
        assert isinstance(jira_hook.client, Mock)
        assert jira_hook.client.name == jira_mock.return_value.name