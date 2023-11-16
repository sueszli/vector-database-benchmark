"""DialogFlow API Create Agent Sample"""
from google.cloud.dialogflowcx_v3.services.agents.client import AgentsClient
from google.cloud.dialogflowcx_v3.types.agent import Agent

def create_agent(project_id, display_name):
    if False:
        i = 10
        return i + 15
    parent = 'projects/' + project_id + '/locations/global'
    agents_client = AgentsClient()
    agent = Agent(display_name=display_name, default_language_code='en', time_zone='America/Los_Angeles')
    response = agents_client.create_agent(request={'agent': agent, 'parent': parent})
    return response