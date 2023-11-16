from google.cloud import dialogflowcx_v3beta1

def sample_restore_agent():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.AgentsClient()
    request = dialogflowcx_v3beta1.RestoreAgentRequest(agent_uri='agent_uri_value', name='name_value')
    operation = client.restore_agent(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)