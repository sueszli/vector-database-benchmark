from google.cloud import dialogflow_v2beta1

def sample_restore_agent():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.AgentsClient()
    request = dialogflow_v2beta1.RestoreAgentRequest(agent_uri='agent_uri_value', parent='parent_value')
    operation = client.restore_agent(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)