import json
from unittest.mock import Mock, patch
import pytest
from superagi.helper.webhook_manager import WebHookManager
from superagi.models.webhooks import Webhooks

@pytest.fixture
def mock_session():
    if False:
        print('Hello World!')
    return Mock()

@pytest.fixture
def mock_agent_execution():
    if False:
        while True:
            i = 10
    return Mock()

@pytest.fixture
def mock_agent():
    if False:
        for i in range(10):
            print('nop')
    return Mock()

@pytest.fixture
def mock_webhook():
    if False:
        print('Hello World!')
    return Mock()

@pytest.fixture
def mock_org():
    if False:
        print('Hello World!')
    org_mock = Mock()
    org_mock.id = 'mock_org_id'
    return org_mock

def test_agent_status_change_callback(mock_session, mock_agent_execution, mock_agent, mock_org, mock_webhook):
    if False:
        return 10
    curr_status = 'NEW_STATUS'
    old_status = 'OLD_STATUS'
    mock_agent_id = 'mock_agent_id'
    mock_org_id = 'mock_org_id'
    mock_agent_execution_instance = Mock()
    mock_agent_execution_instance.agent_id = 'mock_agent_id'
    mock_agent_instance = Mock()
    mock_agent_instance.get_agent_organisation.return_value = mock_org
    mock_webhook_instance = Mock(spec=Webhooks)
    mock_webhook_instance.filters = {'status': ['PAUSED', 'RUNNING']}
    mock_session.query.return_value.filter.return_value.all.return_value = [mock_webhook_instance]
    with patch('superagi.controllers.agent_execution_config.AgentExecution.get_agent_execution_from_id', return_value=mock_agent_execution_instance), patch('superagi.models.agent.Agent.get_agent_from_id', return_value=mock_agent_instance), patch('requests.post', return_value=Mock(status_code=200)) as mock_post, patch('json.dumps') as mock_json_dumps:
        web_hook_manager = WebHookManager(mock_session)
        web_hook_manager.agent_status_change_callback(mock_agent_execution_instance, curr_status, old_status)
    assert mock_agent_execution_instance.agent_status_change_callback