from google.cloud import dialogflow_v2

def sample_restore_agent():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.AgentsClient()
    request = dialogflow_v2.RestoreAgentRequest(agent_uri='agent_uri_value', parent='parent_value')
    operation = client.restore_agent(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)