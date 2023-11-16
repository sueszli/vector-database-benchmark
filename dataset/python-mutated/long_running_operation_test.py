import os
import uuid
from google.cloud.dialogflowcx_v3.services.agents.client import AgentsClient
from google.cloud.dialogflowcx_v3.types.agent import Agent, DeleteAgentRequest
import pytest
from long_running_operation import export_long_running_agent
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
pytest.AGENT_ID = None
pytest.PARENT = None

def create_agent(project_id, display_name):
    if False:
        for i in range(10):
            print('nop')
    parent = 'projects/' + project_id + '/locations/global'
    agents_client = AgentsClient()
    agent = Agent(display_name=display_name, default_language_code='en', time_zone='America/Los_Angeles')
    response = agents_client.create_agent(request={'agent': agent, 'parent': parent})
    return response

def delete_agent(name):
    if False:
        i = 10
        return i + 15
    agents_client = AgentsClient()
    agent = DeleteAgentRequest(name=name)
    agents_client.delete_agent(request=agent)

@pytest.fixture(scope='function', autouse=True)
def setup_teardown():
    if False:
        i = 10
        return i + 15
    agentName = 'temp_agent_' + str(uuid.uuid4())
    pytest.PARENT = create_agent(PROJECT_ID, agentName).name
    pytest.AGENT_ID = pytest.PARENT.split('/')[5]
    print('Created Agent in setUp')
    yield
    delete_agent(pytest.PARENT)

def test_export_agent():
    if False:
        return 10
    actualResponse = export_long_running_agent(PROJECT_ID, pytest.AGENT_ID, 'global')
    assert pytest.AGENT_ID in str(actualResponse)