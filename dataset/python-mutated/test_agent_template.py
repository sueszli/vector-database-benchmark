from unittest.mock import patch, MagicMock
from superagi.models.agent_template import AgentTemplate
from superagi.models.agent_template_config import AgentTemplateConfig
from fastapi.testclient import TestClient
from main import app
client = TestClient(app)

@patch('superagi.controllers.agent_template.db')
@patch('superagi.helper.auth.db')
@patch('superagi.helper.auth.get_user_organisation')
def test_edit_agent_template_success(mock_get_user_org, mock_auth_db, mock_db):
    if False:
        print('Hello World!')
    mock_agent_template = AgentTemplate(id=1, name='Test Agent Template', description='Test Description')
    mock_updated_agent_configs = {'name': 'Updated Agent Template', 'description': 'Updated Description', 'agent_configs': {'agent_workflow': "Don't Maintain Task Queue", 'goal': ['Create a simple pacman game for me.', 'Write all files properly.'], 'instruction': ['write spec', 'write code', 'improve the code', 'write test'], 'constraints': ['If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.', 'Ensure the tool and args are as per current plan and reasoning', 'Exclusively use the tools listed under "TOOLS"', 'REMEMBER to format your response as JSON, using double quotes ("") around keys and string values, and commas (,) to separate items in arrays and objects. IMPORTANTLY, to use a JSON object as a string in another JSON object, you need to escape the double quotes.'], 'tools': ['Read Email', 'Send Email', 'Write File'], 'exit': 'No exit criterion', 'iteration_interval': 500, 'model': 'gpt-4', 'max_iterations': 25, 'permission_type': 'God Mode', 'LTM_DB': 'Pinecone'}}
    mock_get_user_org.return_value = MagicMock(id=1)
    session_mock = MagicMock()
    mock_db.session = session_mock
    mock_db.session.query.return_value.filter.return_value.first.return_value = mock_agent_template
    mock_db.session.commit.return_value = None
    mock_db.session.add.return_value = None
    mock_db.session.flush.return_value = None
    mock_agent_template_config = AgentTemplateConfig(agent_template_id=1, key='goal', value=['Create a simple pacman game for me.', 'Write all files properly.'])
    response = client.put('agent_templates/update_agent_template/1', json=mock_updated_agent_configs)
    assert response.status_code == 200
    assert mock_agent_template.name == 'Updated Agent Template'
    assert mock_agent_template.description == 'Updated Description'
    assert mock_agent_template_config.key == 'goal'
    assert mock_agent_template_config.value == ['Create a simple pacman game for me.', 'Write all files properly.']
    session_mock.commit.assert_called()
    session_mock.flush.assert_called()

@patch('superagi.controllers.agent_template.db')
@patch('superagi.helper.auth.db')
@patch('superagi.helper.auth.get_user_organisation')
def test_edit_agent_template_failure(mock_get_user_org, mock_auth_db, mock_db):
    if False:
        i = 10
        return i + 15
    mock_get_user_org.return_value = MagicMock(id=1)
    session_mock = MagicMock()
    mock_db.session = session_mock
    mock_db.session.query.return_value.filter.return_value.first.return_value = None
    response = client.put('agent_templates/update_agent_template/1', json={})
    assert response.status_code == 404
    assert response.json() == {'detail': 'Agent Template not found'}
    session_mock.commit.assert_not_called()
    session_mock.flush.assert_not_called()

@patch('superagi.controllers.agent_template.db')
@patch('superagi.helper.auth.db')
@patch('superagi.helper.auth.get_user_organisation')
def test_edit_agent_template_with_new_config_success(mock_get_user_org, mock_auth_db, mock_db):
    if False:
        return 10
    mock_agent_template = AgentTemplate(id=1, name='Test Agent Template', description='Test Description')
    mock_updated_agent_configs = {'name': 'Updated Agent Template', 'description': 'Updated Description', 'agent_configs': {'new_config_key': 'New config value', 'agent_workflow': "Don't Maintain Task Queue"}}
    mock_get_user_org.return_value = MagicMock(id=1)
    session_mock = MagicMock()
    mock_db.session = session_mock
    mock_db.session.query.return_value.filter.return_value.first.return_value = mock_agent_template
    mock_db.session.commit.return_value = None
    mock_db.session.add.return_value = None
    mock_db.session.flush.return_value = None
    response = client.put('agent_templates/update_agent_template/1', json=mock_updated_agent_configs)
    assert response.status_code == 200
    assert mock_agent_template.name == 'Updated Agent Template'
    assert mock_agent_template.description == 'Updated Description'
    session_mock.commit.assert_called()
    session_mock.flush.assert_called()