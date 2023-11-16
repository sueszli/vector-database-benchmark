"""Test for create_agent"""
import os
import uuid
from google.cloud.dialogflowcx_v3.services.agents.client import AgentsClient
from google.cloud.dialogflowcx_v3.types.agent import DeleteAgentRequest
import pytest
from create_agent import create_agent
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
pytest.AGENT_PATH = ''

def delete_agent(name):
    if False:
        print('Hello World!')
    agents_client = AgentsClient()
    request = DeleteAgentRequest(name=name)
    agents_client.delete_agent(request=request)

def test_create_agent():
    if False:
        while True:
            i = 10
    agentName = f'fake_agent_{uuid.uuid4()}'
    response = create_agent(PROJECT_ID, agentName)
    delete_agent(response.name)
    assert response.display_name == agentName