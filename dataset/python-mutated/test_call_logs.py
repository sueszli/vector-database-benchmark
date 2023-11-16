import pytest
from unittest.mock import MagicMock
from superagi.models.call_logs import CallLogs

@pytest.fixture
def mock_session():
    if False:
        i = 10
        return i + 15
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = None
    return session

@pytest.mark.parametrize('agent_execution_name, agent_id, tokens_consumed, tool_used, model, org_id', [('example_execution', 1, 1, 'Test Tool', 'Test Model', 1)])
def test_create_call_logs(mock_session, agent_execution_name, agent_id, tokens_consumed, tool_used, model, org_id):
    if False:
        i = 10
        return i + 15
    call_log = CallLogs(agent_execution_name=agent_execution_name, agent_id=agent_id, tokens_consumed=tokens_consumed, tool_used=tool_used, model=model, org_id=org_id)
    mock_session.add(call_log)
    mock_session.add.assert_called_once_with(call_log)

@pytest.mark.parametrize('agent_execution_name, agent_id, tokens_consumed, tool_used, model, org_id', [('example_execution', 1, 1, 'Test Tool', 'Test Model', 1)])
def test_repr_method_call_logs(mock_session, agent_execution_name, agent_id, tokens_consumed, tool_used, model, org_id):
    if False:
        print('Hello World!')
    call_log = CallLogs(agent_execution_name=agent_execution_name, agent_id=agent_id, tokens_consumed=tokens_consumed, tool_used=tool_used, model=model, org_id=org_id)
    result = repr(call_log)
    assert result == f'CallLogs(id=None, agent_execution_name={agent_execution_name}, agent_id={agent_id}, tokens_consumed={tokens_consumed}, tool_used={tool_used}, model={model}, org_id={org_id})'