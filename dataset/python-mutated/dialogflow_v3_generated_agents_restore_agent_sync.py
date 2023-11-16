from google.cloud import dialogflowcx_v3

def sample_restore_agent():
    if False:
        return 10
    client = dialogflowcx_v3.AgentsClient()
    request = dialogflowcx_v3.RestoreAgentRequest(agent_uri='agent_uri_value', name='name_value')
    operation = client.restore_agent(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)