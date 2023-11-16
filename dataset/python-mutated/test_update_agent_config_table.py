import pytest
from unittest.mock import patch, Mock
from superagi.models.agent_config import AgentConfiguration
from superagi.controllers.types.agent_execution_config import AgentRunIn

def test_update_existing_toolkits():
    if False:
        while True:
            i = 10
    agent_id = 1
    updated_details = AgentRunIn(agent_workflow='test', constraints=['c1', 'c2'], toolkits=[1, 2], tools=[1, 2, 3], exit='exit', iteration_interval=1, model='test', permission_type='p', LTM_DB='LTM', max_iterations=100)
    existing_toolkits_config = Mock(spec=AgentConfiguration)
    existing_toolkits_config.key = 'toolkits'
    existing_toolkits_config.value = [3, 4]
    agent_configs = [existing_toolkits_config]
    mock_session = Mock()
    mock_session.query().filter().all.return_value = agent_configs
    result = AgentConfiguration.update_agent_configurations_table(mock_session, agent_id, updated_details)
    assert existing_toolkits_config.value == '[1, 2]'
    assert mock_session.commit.called_once()
    assert result == 'Details updated successfully'