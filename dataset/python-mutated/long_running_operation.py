""" DialogFlow CX long running operation code snippet """
from google.cloud.dialogflowcx_v3.services.agents.client import AgentsClient
from google.cloud.dialogflowcx_v3.types.agent import ExportAgentRequest

def export_long_running_agent(project_id, agent_id, location):
    if False:
        for i in range(10):
            print('nop')
    api_endpoint = f'{location}-dialogflow.googleapis.com:443'
    client_options = {'api_endpoint': api_endpoint}
    agents_client = AgentsClient(client_options=client_options)
    export_request = ExportAgentRequest()
    export_request.name = f'projects/{project_id}/locations/{location}/agents/{agent_id}'
    operation = agents_client.export_agent(request=export_request)
    return operation.result()